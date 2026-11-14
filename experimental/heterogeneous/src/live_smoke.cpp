// Manually launched, backend-neutral CUDA->HIP STFR smoke test.
// The coordinator never links CUDA or HIP; each child owns its runtime.
#include "strata_heterogeneous/cuda_worker_config.hpp"
#include "strata_heterogeneous/hip_worker_config.hpp"
#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata_heterogeneous/process.hpp"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>
#include <utility>
#include <unistd.h>

namespace hetero = strata::heterogeneous;
namespace {
struct Options {
    std::string plan;
    std::string cuda_worker;
    std::string hip_worker;
    std::vector<std::string> configs;
    std::vector<int32_t> tokens;
    std::string capture_prefix;
    uint32_t layers = 48;
    uint32_t hc = 4;
    uint32_t n_embd = 2560;
    uint64_t fingerprint = 0;
    uint32_t token = 1;
    uint32_t rows = 1;
    uint32_t keep = 0; // unspecified: commit all evaluated rows
    uint32_t windows = 1;
    int timeout_ms = 120000;
    bool run = false;
    bool shared = false;
    bool layer_taps = false;
    bool layer_zero_probe = false;
    bool stage0_phases = false;
    bool branch_probe = false;
    bool micro_probe = false;
    bool boundary_capture = false;
};
uint64_t decimal(const std::string& str, const char* key) {
    if (str.empty() || str[0] == '-' || str[0] == '+')
        throw std::invalid_argument(std::string("invalid ") + key);
    uint64_t value = 0;
    const auto result = std::from_chars(str.data(), str.data() + str.size(), value);
    if (result.ec != std::errc{} || result.ptr != str.data() + str.size())
        throw std::invalid_argument(std::string("invalid ") + key);
    return value;
}
uint32_t u32(const std::string& str, const char* key) {
    const uint64_t n = decimal(str, key);
    if (n > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(std::string("out-of-range ") + key);
    return static_cast<uint32_t>(n);
}
std::string required_value(int& i, int argc, char** argv, const char* option) {
    if (i + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + option);
    return argv[++i];
}
void usage() {
    std::cerr <<
        "usage: strata_heterogeneous_live_smoke --plan <spec> --fingerprint <id>\n"
        "  --config <cuda0.cfg> --config <hip0.cfg> --config <hip1.cfg>\n"
        "  [--cuda-worker /absolute/path] [--hip-worker /absolute/path]\n"
        "  [--layers 48] [--hc 4] [--n-embd 2560] [--token 1]\n"
        "  [--rows 1] [--keep N] [--windows 1] [--tokens 1,198,...] [--capture-prefix /absolute/path]\n"
        "  --keep: positive accepted prefix per window (default: all rows)\n"
        "  [--layer-taps] [--layer0-probe] [--stage0-phases] [--branch-probe] [--micro-probe]\n"
        "  [--boundary-capture] (requires --capture-prefix; real GPU handoff)\n"
        "  [--timeout-ms 120000] [--shared] [--run]\n"
        "Without --run: CPU-only configuration preflight. With --run: start native GPU workers.\n";
}
Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--run") o.run = true;
        else if (key == "--layer-taps") o.layer_taps = true;
        else if (key == "--layer0-probe") o.layer_zero_probe = true;
        else if (key == "--stage0-phases") o.stage0_phases = true;
        else if (key == "--branch-probe") o.branch_probe = true;
        else if (key == "--micro-probe") o.micro_probe = true;
        else if (key == "--boundary-capture") o.boundary_capture = true;
        else if (key == "--shared") o.shared = true;
        else if (key == "--plan") o.plan = required_value(i, argc, argv, "--plan");
        else if (key == "--cuda-worker") o.cuda_worker = required_value(i, argc, argv, "--cuda-worker");
        else if (key == "--hip-worker") o.hip_worker = required_value(i, argc, argv, "--hip-worker");
        else if (key == "--config") o.configs.push_back(required_value(i, argc, argv, "--config"));
        else if (key == "--capture-prefix")
            o.capture_prefix = required_value(i, argc, argv, "--capture-prefix");
        else if (key == "--tokens") {
            const std::string value = required_value(i, argc, argv, "--tokens");
            std::istringstream input(value);
            std::string item;
            while (std::getline(input, item, ',')) {
                const uint32_t token = u32(item, "tokens");
                if (token > static_cast<uint32_t>(INT32_MAX) || o.tokens.size() >= 64)
                    throw std::invalid_argument("bounded token sequence invalid");
                o.tokens.push_back(static_cast<int32_t>(token));
            }
            if (value.empty() || value.back() == ',')
                throw std::invalid_argument("invalid comma-separated tokens");
        }
        else if (key == "--fingerprint")
            o.fingerprint = decimal(required_value(i, argc, argv, "--fingerprint"), "fingerprint");
        else if (key == "--layers") o.layers = u32(required_value(i, argc, argv, "--layers"), "layers");
        else if (key == "--hc") o.hc = u32(required_value(i, argc, argv, "--hc"), "hc");
        else if (key == "--n-embd") o.n_embd = u32(required_value(i, argc, argv, "--n-embd"), "n-embd");
        else if (key == "--token") o.token = u32(required_value(i, argc, argv, "--token"), "token");
        else if (key == "--rows") o.rows = u32(required_value(i, argc, argv, "--rows"), "rows");
        else if (key == "--keep") {
            o.keep = u32(required_value(i, argc, argv, "--keep"), "keep");
            if (!o.keep) throw std::invalid_argument("--keep must be positive");
        }
        else if (key == "--windows") o.windows = u32(required_value(i, argc, argv, "--windows"), "windows");
        else if (key == "--timeout-ms") {
            const auto v = decimal(required_value(i, argc, argv, "--timeout-ms"), "timeout-ms");
            if (v < 100 || v > 600000) throw std::invalid_argument("timeout-ms outside 100..600000");
            o.timeout_ms = static_cast<int>(v);
        } else if (key == "--help") {
            usage();
            throw std::invalid_argument("help requested");
        } else throw std::invalid_argument("unknown option: " + key);
    }
    if (o.plan.empty() || !o.fingerprint || o.rows == 0 || o.rows > 8 ||
        o.windows == 0 || o.windows > 8 || o.token > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        !o.layers || !o.hc || !o.n_embd)
        throw std::invalid_argument("missing plan/fingerprint or invalid bounded smoke-test options");
    if (!o.keep) o.keep = o.rows;
    if (o.keep > o.rows) throw std::invalid_argument("--keep cannot exceed --rows");
    if (o.layer_taps && o.capture_prefix.empty())
        throw std::invalid_argument("--layer-taps requires --capture-prefix");
    if (o.layer_zero_probe && (o.capture_prefix.empty() || o.rows != 1))
        throw std::invalid_argument("--layer0-probe requires --capture-prefix and --rows 1");
    if (o.stage0_phases && (o.capture_prefix.empty() || o.rows != 1))
        throw std::invalid_argument("--stage0-phases requires --capture-prefix and --rows 1");
    if (o.branch_probe && (o.capture_prefix.empty() || o.rows != 1))
        throw std::invalid_argument("--branch-probe requires --capture-prefix and --rows 1");
    if (o.micro_probe && (o.capture_prefix.empty() || o.rows != 1))
        throw std::invalid_argument("--micro-probe requires --capture-prefix and --rows 1");
    if (o.boundary_capture && o.capture_prefix.empty())
        throw std::invalid_argument("--boundary-capture requires --capture-prefix");
    if (!o.tokens.empty() && o.tokens.size() != static_cast<size_t>(o.rows) * o.windows)
        throw std::invalid_argument("--tokens must contain exactly rows*windows token IDs");
    return o;
}
void require_executable(const std::string& path, const char* label) {
    std::error_code ec;
    if (path.empty() || path[0] != '/' ||
        !std::filesystem::is_regular_file(path, ec) || ec ||
        ::access(path.c_str(), X_OK) != 0)
        throw std::invalid_argument(std::string(label) + " must be an executable absolute file");
}
void preflight(const Options& o, const hetero::Plan& plan) {
    if (o.configs.size() != plan.stages.size())
        throw std::invalid_argument("one ordered --config is required per planned stage");
    // The last provisional window, including its discarded suffix, must fit.
    const uint64_t needed = uint64_t(o.windows - 1) * o.keep + o.rows;
    const uint64_t stride = plan.geometry.floats_per_row();
    if (stride > hetero::max_payload_bytes / (sizeof(float) * o.rows))
        throw std::invalid_argument("requested activation geometry exceeds bounded 16 MiB transport");
    std::optional<hetero::CudaWorkerConfig> common_model;
    for (size_t index = 0; index < plan.stages.size(); ++index) {
        std::string error;
        hetero::CudaWorkerConfig config;
        const bool is_hip = plan.stages[index].backend == hetero::Backend::hip;
        if (is_hip) {
            hetero::HipWorkerConfig hip;
            if (!hetero::read_hip_worker_config(o.configs[index], hip, error) ||
                !hetero::validate_hip_worker_config(hip, plan, index, error))
                throw std::runtime_error("stage " + std::to_string(index) + ": " + error);
            config = std::move(hip.fields);
        } else if (!hetero::read_cuda_worker_config(o.configs[index], config, error) ||
                   !hetero::validate_cuda_worker_config(config, plan, index, error)) {
            throw std::runtime_error("stage " + std::to_string(index) + ": " + error);
        }
        // All stages must use one canonical pack/profile/original GGUF
        // source. Capture metadata is otherwise insufficient to prove that
        // each worker computed against the same source model.
        if (common_model) {
            if (config.pack_dir != common_model->pack_dir ||
                config.profile_path != common_model->profile_path ||
                config.index_sha256 != common_model->index_sha256 ||
                config.gguf_shards != common_model->gguf_shards ||
                config.n_vocab != common_model->n_vocab ||
                config.model_fingerprint != common_model->model_fingerprint)
                throw std::invalid_argument("cross-stage Q2_0 model source identity mismatch");
        } else common_model = config;
        const bool invalid_token = o.tokens.empty()
            ? config.n_vocab <= o.token
            : std::any_of(o.tokens.begin(), o.tokens.end(),
                          [&](int32_t token) { return token >= static_cast<int32_t>(config.n_vocab); });
        if (config.version != 2 || config.max_cells < needed ||
            config.max_t < o.rows || invalid_token)
            throw std::invalid_argument("stage " + std::to_string(index) +
                ": v2, context, window or token bounds cannot serve smoke-test request");
        std::cout << "PREFLIGHT stage=" << index
                  << " backend=" << (is_hip ? "HIP" : "CUDA")
                  << " ordinal=" << config.device_ordinal
                  << " layers=[" << config.first_layer << ',' << config.end_layer << ')'
                  << " max_cells=" << config.max_cells
                  << " max_t=" << config.max_t << '\n';
    }
    if (o.run) {
        for (const auto& stage : plan.stages) {
            if (stage.backend == hetero::Backend::cuda)
                require_executable(o.cuda_worker, "CUDA worker");
            else require_executable(o.hip_worker, "HIP worker");
        }
        if (!o.capture_prefix.empty()) {
            const std::filesystem::path prefix(o.capture_prefix);
            std::error_code ec;
            if (!prefix.is_absolute() || o.capture_prefix.size() > 4096 ||
                o.capture_prefix.find('\n') != std::string::npos ||
                o.capture_prefix.find('\r') != std::string::npos ||
                !std::filesystem::is_directory(prefix.parent_path(), ec))
                throw std::invalid_argument("capture prefix must be an absolute path in an existing directory");
            if (std::filesystem::exists(o.capture_prefix + ".complete.json"))
                throw std::invalid_argument("capture already completed; use a fresh prefix");
            for (uint32_t n = 1; n <= o.windows; ++n) {
                const std::string base = o.capture_prefix + ".w" + std::to_string(n);
                if (o.layer_zero_probe &&
                    std::filesystem::exists(o.capture_prefix + ".stage0.w" +
                                            std::to_string(n) + ".stl0"))
                    throw std::invalid_argument("layer-zero probe already exists; use fresh prefix");
                if (o.stage0_phases &&
                    std::filesystem::exists(o.capture_prefix + ".stage0.w" +
                                            std::to_string(n) + ".stp0"))
                    throw std::invalid_argument("stage-zero phase record exists; use fresh prefix");
                if (o.branch_probe &&
                    std::filesystem::exists(o.capture_prefix + ".stage0.w" +
                                            std::to_string(n) + ".stb0"))
                    throw std::invalid_argument("branch probe record exists; use fresh prefix");
                if (o.micro_probe &&
                    std::filesystem::exists(o.capture_prefix + ".stage0.w" +
                                            std::to_string(n) + ".stm0"))
                    throw std::invalid_argument("microprobe record exists; use fresh prefix");
                if (o.boundary_capture) {
                    for (size_t stage = 0; stage < plan.stages.size(); ++stage) {
                        const std::string root = o.capture_prefix + ".stage" +
                            std::to_string(stage) + ".r1.w" + std::to_string(n);
                        if ((plan.stages[stage].first_layer != 0 &&
                             std::filesystem::exists(root + ".in.f32")) ||
                            (stage + 1 < plan.stages.size() &&
                             std::filesystem::exists(root + ".out.f32")))
                            throw std::invalid_argument("boundary image already exists; use a fresh prefix");
                    }
                }
                if (o.layer_taps)
                    for (size_t stage = 0; stage < plan.stages.size(); ++stage)
                        if (std::filesystem::exists(o.capture_prefix + ".stage" +
                                std::to_string(stage) + ".w" + std::to_string(n) + ".stlt"))
                            throw std::invalid_argument("layer tap already exists; use fresh prefix");
                if (std::filesystem::exists(base + ".json") ||
                    std::filesystem::exists(base + ".stfr"))
                    throw std::invalid_argument("capture output already exists; use a fresh prefix");
            }
        }
    }
}
// Strict JSON escaping is needed because model paths are recorded as metadata.
std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') { out << '\\' << c; }
        else if (c < 0x20) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << static_cast<char>(c);
    }
    out << '"';
    return out.str();
}
void capture_window(const Options& o, const hetero::Plan& plan,
                    const hetero::WindowSpec& spec,
                    const hetero::FinalStageResult& committed) {
    if (!committed.logits || committed.logits->empty())
        throw std::runtime_error("parity capture requires full final-stage FP32 logits");
    hetero::CudaWorkerConfig source;
    std::string error;
    if (!hetero::read_cuda_worker_config(o.configs[0], source, error))
        throw std::runtime_error("parity capture config: " + error);
    const std::string base = o.capture_prefix + ".w" + std::to_string(committed.window_id);
    const auto packet = hetero::encode_final_result(committed);
    std::ofstream stfr(base + ".stfr", std::ios::binary);
    if (!stfr) throw std::runtime_error("cannot create parity STFR packet");
    stfr.write(reinterpret_cast<const char*>(packet.data()),
               static_cast<std::streamsize>(packet.size()));
    stfr.flush();
    if (!stfr) throw std::runtime_error("parity STFR packet write failed");
    std::ofstream manifest(base + ".json", std::ios::binary);
    if (!manifest) throw std::runtime_error("cannot create parity window manifest");
    manifest << "{\"schema\":\"strata-heterogeneous-parity/v1\""
             << ",\"plan\":" << json_string(hetero::plan_spec(plan))
             << ",\"model_fingerprint\":" << o.fingerprint
             << ",\"plan_fingerprint\":" << committed.plan_fingerprint
             << ",\"pack_index_sha256\":" << json_string(source.index_sha256)
             << ",\"gguf_shards\":" << json_string([&]() {
                   std::string combined;
                   for (const auto& shard : source.gguf_shards) {
                       if (!combined.empty()) combined += ';';
                       combined += shard;
                   }
                   return combined;
               }())
             << ",\"profile_path\":" << json_string(source.profile_path)
             << ",\"window_id\":" << committed.window_id
             << ",\"pos0\":" << spec.pos0
             << ",\"rows\":" << committed.rows
             << ",\"n_keep\":" << o.keep
             << ",\"token_ids\":[";
    for (size_t t = 0; t < spec.token_ids.size(); ++t) {
        if (t) manifest << ',';
        manifest << spec.token_ids[t];
    }
    manifest << "]}\n";
    manifest.flush();
    if (!manifest) throw std::runtime_error("parity window manifest write failed");
}
int execute(const Options& o, const hetero::Plan& plan) {
    // No device calls in coordinator. Child workers alone see this opt-in;
    // no tap allocations/copies occur for ordinary M2.9/M2.10 tests.
    if (o.layer_taps && ::setenv("STRATA_HETERO_LAYER_TAP_PREFIX",
                                 o.capture_prefix.c_str(), 1) != 0)
        throw std::runtime_error("cannot configure native worker layer-tap capture");
    if (o.layer_zero_probe) {
        if (::setenv("STRATA_HETERO_L0_PROBE_PREFIX",
                     o.capture_prefix.c_str(), 1) != 0)
            throw std::runtime_error("cannot configure native layer-zero arithmetic probe");
    } else if (::unsetenv("STRATA_HETERO_L0_PROBE_PREFIX") != 0) {
        // Prevent a stale environment variable from silently changing a
        // non-diagnostic manually launched inference session.
        throw std::runtime_error("cannot clear inherited native layer-zero probe");
    }
    if (o.stage0_phases) {
        if (::setenv("STRATA_HETERO_STAGE0_PHASE_PREFIX",
                     o.capture_prefix.c_str(), 1) != 0)
            throw std::runtime_error("cannot configure native stage-zero phase probe");
    } else if (::unsetenv("STRATA_HETERO_STAGE0_PHASE_PREFIX") != 0) {
        throw std::runtime_error("cannot clear inherited native stage-zero phase probe");
    }
    if (o.branch_probe) {
        if (::setenv("STRATA_HETERO_BRANCH_PROBE_PREFIX",
                     o.capture_prefix.c_str(), 1) != 0)
            throw std::runtime_error("cannot configure native branch probe");
    } else if (::unsetenv("STRATA_HETERO_BRANCH_PROBE_PREFIX") != 0) {
        throw std::runtime_error("cannot clear inherited native branch probe");
    }
    if (o.micro_probe) {
        if (::setenv("STRATA_HETERO_MICRO_PROBE_PREFIX",
                     o.capture_prefix.c_str(), 1) != 0)
            throw std::runtime_error("cannot configure native microprobe");
    } else if (::unsetenv("STRATA_HETERO_MICRO_PROBE_PREFIX") != 0) {
        throw std::runtime_error("cannot clear inherited native microprobe");
    }
    // Explicit opt-in only. Clear inherited capture state for normal runs.
    if (o.boundary_capture) {
        if (::setenv("STRATA_HETERO_BOUNDARY_CAPTURE_PREFIX",
                     o.capture_prefix.c_str(), 1) != 0)
            throw std::runtime_error("cannot enable native boundary capture");
    } else if (::unsetenv("STRATA_HETERO_BOUNDARY_CAPTURE_PREFIX") != 0) {
        throw std::runtime_error("cannot clear inherited boundary capture");
    }
    auto workers = hetero::WorkerProcessGroup::launch(
        plan, o.cuda_worker, o.hip_worker, o.shared, {}, o.configs);
    hetero::FinalPipelineSession session(plan, workers.sockets(), o.timeout_ms, o.shared);
    session.initialize(); // Native children initialize their own CUDA/HIP runtimes.
    std::cout << "LIVE workers ready, mode=" << (o.shared ? "shared" : "socket") << '\n';
    for (uint32_t n = 0; n < o.windows; ++n) {
        hetero::WindowSpec spec;
        spec.pos0 = static_cast<int64_t>(uint64_t(n) * o.keep);
        if (o.tokens.empty()) spec.token_ids.assign(o.rows, static_cast<int32_t>(o.token));
        else spec.token_ids.assign(o.tokens.begin() + static_cast<size_t>(n) * o.rows,
                                   o.tokens.begin() + static_cast<size_t>(n + 1) * o.rows);
        hetero::Frame initial;
        initial.kind = hetero::Kind::data;
        initial.request_id = 1;
        initial.window_id = uint64_t(n) + 1;
        initial.plan_fingerprint = plan.fingerprint();
        initial.rows = o.rows;
        initial.geometry = plan.geometry;
        // Layer-zero verifier embeds token_ids; it intentionally ignores
        // the placeholder initial activation. Later stages receive real
        // native GPU output from the immediately preceding stage.
        initial.payload.assign(
            static_cast<size_t>(plan.geometry.floats_per_row()) * o.rows, 0.0f);
        (void)session.execute(std::move(initial), spec); // provisional only
        const hetero::FinalStageResult committed = session.commit(o.keep);
        if (committed.predicted_tokens.size() != o.rows || committed.has_nonfinite())
            throw std::runtime_error("committed head predictions malformed or nonfinite");
        if (!o.capture_prefix.empty()) capture_window(o, plan, spec, committed);
        std::cout << "LIVE PASS window=" << n + 1 << " pos0=" << spec.pos0
                  << " committed=" << o.keep << " predictions=";
        // STFR records every evaluated row; log only the accepted prefix.
        for (uint32_t t = 0; t < o.keep; ++t)
            std::cout << committed.predicted_tokens[t] << ',';
        std::cout << '\n';
    }
    std::string error;
    if (!workers.finish(error))
        throw std::runtime_error("worker shutdown failed: " + error);
    if (!o.capture_prefix.empty()) {
        std::ofstream complete(o.capture_prefix + ".complete.json", std::ios::binary);
        if (!complete) throw std::runtime_error("cannot create parity completion marker");
        complete << "{\"schema\":\"strata-heterogeneous-parity/v1\",\"windows\":"
                 << o.windows << ",\"model_fingerprint\":" << o.fingerprint
                 << ",\"plan\":" << json_string(hetero::plan_spec(plan)) << "}\n";
        complete.flush();
        if (!complete) throw std::runtime_error("parity completion marker write failed");
    }
    std::cout << "LIVE PASS: all " << o.windows
              << " window(s) committed on actual CUDA/HIP worker chain\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            usage();
            return 0;
        }
        const Options opts = parse(argc, argv);
        const hetero::Plan plan = hetero::parse_plan(
            opts.plan, opts.layers, {opts.hc, opts.n_embd}, opts.fingerprint);
        preflight(opts, plan); // always CPU-only; no runtime in coordinator.
        if (!opts.run) {
            std::cout << "PASS: CPU-only native stage preflight. "
                         "No workers or GPU runtimes started; add --run for live test.\n";
            return 0;
        }
        return execute(opts, plan);
    } catch (const std::exception& ex) {
        std::cerr << "heterogeneous live smoke: " << ex.what() << '\n';
        return 1;
    }
}
