// Manually launched, backend-neutral CUDA->HIP STFR smoke test.
// The coordinator never links CUDA or HIP; each child owns its runtime.
#include "strata_heterogeneous/cuda_worker_config.hpp"
#include "strata_heterogeneous/hip_worker_config.hpp"
#include "strata_heterogeneous/final_pipeline.hpp"
#include "strata_heterogeneous/backend_domains.hpp"
#include "strata_heterogeneous/performance.hpp"
#include "strata_heterogeneous/speculation.hpp"
#include "strata_heterogeneous/mtp_fresh_replay.hpp"
#include "strata_heterogeneous/mtp_three_round.hpp"
#include "strata_heterogeneous/mtp_generation.hpp"
#include "strata_heterogeneous/mtp_resident.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/prefill/hc_experiment.hpp"
#include "strata_heterogeneous/process.hpp"
#include "strata_heterogeneous/kv_mode.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <limits>
#include <memory>
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
    std::string source_gguf, source_index;
    std::vector<int32_t> stop_ids;
    uint32_t generate = 0, prompt_chunk = 1, native_prefill_chunk = 0;
    bool describe_source = false;
    bool resident = false;
    bool workload_mode = false;
    bool stream_tokens = false;
    bool lookup_draft = false;
    uint32_t context_limit = 4096;
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
    bool group_hip = false;
    bool hip_split_overlap = false;
    bool receiver_arm = false;
    bool cuda_mapped_egress = false;
    bool direct_verified = false;
    hetero::KvMode kv = hetero::default_kv_mode;
    bool kv_explicit = false;
    std::string mtp_probe_rt;
    uint32_t mtp_relay_rounds=0;
    bool mtp_fresh_replay=false, mtp_three_round=false, mtp_three_round_variable=false;
    bool generation_capture=false, mtp_generation=false;
    uint32_t mtp_resident_prompt_rows=0;
    bool mtp_state_trace=false, mtp_reprime_tail=false, mtp_front_trace=false, mtp_repeat_native_round=false;
    uint32_t warmup_windows = 0;
    bool layer_taps = false;
    bool layer_zero_probe = false;
    bool stage0_phases = false;
    bool branch_probe = false;
    bool micro_probe = false;
    bool boundary_capture = false;
    bool placement_capture = false;
    uint32_t prefill_parity_at = 0; // diagnostic: number of known prefix tokens before teacher-forced rows
    bool prefill_state_capture = false;
    bool prefill_first_token = false;
    bool prefill_hc_replay = false;
    bool prefill_stage_capture = false;
    bool numerical_trace = false;
    std::string prefill_stage_reference;
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
        "  [--placement-capture] (explicit one-row tokens, up to 128 windows, STFR only)\n"
        "  [--timeout-ms 120000] [--shared] [--group-native-hip] [--hip-split-overlap] [--receiver-arm] [--cuda-mapped-egress] [--direct-verified-shared] [--kv int8|f16] [--run]\n"
        "  --mtp-probe-rt ABS_PATH: native HIP MTP qualification runtime.\n"
        "  --generation-parity-capture --capture-prefix PATH: bounded actual nonresident greedy generation control.\n"
        "  --mtp-generation-qualification: same capture, with explicit --mtp-probe-rt; not general admission.\n"
        "  --mtp-resident-prompt-rows N: additionally require --resident; fixed prompt length and --generate budget, max 8 requests.\n"
        "  --mtp-repeat-native-round: one-shot same cell twice, no target step between; requires front trace.\n"
        "  --mtp-front-trace: capture native one-row draft round staged input and front intermediates. Requires state trace.\n"
        "  --mtp-state-trace: bounded exact terminal R and committed draft KV; capture-prefix required.\n"
        "  --mtp-reprime-tail: diagnostic native prefill of offered tail after first draft.\n"
        "  --mtp-fresh-replay: isolated one-row MTP reference, up to 12 windows, full capture, no relay.\n"
        "  --mtp-three-round: explicit three-relay qualification of the known seven-token prompt; 12 committed rows max.\n"
        "  --mtp-three-round-variable: requires --mtp-three-round; explicit alternate 1..7 one-row prompt.\n"
        "  --mtp-relay-rounds N: after fixed prompt windows, relay 1..8 actual native proposals and commit greedy prefixes.\n"
        "  --warmup-windows N: fixed-token timing after N warm windows; 128 evaluated tokens max, no captures.\n"
        "  --describe-source (CPU-only validated primary GGUF and limits)\n"
        "  --generate N --prompt-chunk N --stop-ids CSV --source-gguf PATH --source-index SHA256\n"
        "  --native-prefill-chunk N (grouped HIP resident native prefill, 1..2048)\n"
        "  --prefill-parity-at N --capture-prefix PATH: grouped diagnostic A/B/A, prefix N,\n"
        "    --generate 1 --native-prefill-chunk N --prompt-chunk 1, nonresident stdin tokens.\n"
        "    Captures full committed logits on the remaining 1..128 teacher-forced tokens and native bridge bytes.\n"
        "    --prefill-state-capture: also snapshot the entry CUDA committed prefix (cut 1..16 only).\n"
        "    --prefill-first-token: paired native/verifier CUDA first-token observations (cut/chunk 1 only).\n"
        "    --prefill-hc-replay-capture: capture actual plain BF16 HC weights and intermediate images.\n"
        "    --prefill-stage-capture: bounded A/B/A all physical stage R and native state.\n"
        "    --prefill-numerical-trace: read-only layer/routing trace for canonical cut 2 or 6.\n"
        "    --prefill-stage-reference PREFIX: file-fed independent native stages (stage capture only).\n"
        "  Ordinary generation reads prompt token IDs from stdin, not --tokens or diagnostic options.\n"
        "  --resident: accept repeated local requests (N followed by prompt IDs on one line).\n"
        "  --context-limit N (total tokens, <=8192); --stream-tokens (resident only).\n"
        "  --workload-mode: opt-in grouped or two-HIP-stage resident text, context <=32768, max-new <=4096; experimental.\n"
        "  --lookup-draft: verified suffix lookup (buffered only, prompt-chunk 2..8).\n"
        "Without --run: CPU-only configuration preflight. With --run: start native GPU workers.\n";
}
// Input is local, but bounded before allocation beyond the documented cap.
// The CLI passes decimal token IDs separated by ASCII whitespace.
// A bounded line reader for local resident requests and checkpoints.
// EOF is allowed only between complete commands.
bool read_command(std::string& line, size_t max_bytes = 65536) {
    line.clear();
    char c;
    while (std::cin.get(c)) {
        if (c == '\n') return true;
        if (line.size() == max_bytes)
            throw std::invalid_argument("resident command exceeds configured byte cap");
        line.push_back(c);
    }
    if (!std::cin.eof() || std::cin.bad() || !line.empty())
        throw std::runtime_error("incomplete resident command");
    return false;
}
std::vector<int32_t> prompt_ids(std::istream& in, uint32_t limit) {
    std::string word;
    std::vector<int32_t> ids;
    while (in >> word) {
        const auto n = u32(word, "prompt token");
        if (n > INT32_MAX || ids.size() >= limit)
            throw std::invalid_argument("prompt exceeds configured token bound");
        ids.push_back(static_cast<int32_t>(n));
    }
    if (ids.empty()) throw std::invalid_argument("prompt token input is empty");
    return ids;
}
std::vector<int32_t> prompt_from_stdin(uint32_t limit) {
    std::string input;
    char c;
    while (std::cin.get(c)) {
        if (input.size() == 65536)
            throw std::invalid_argument("prompt token input exceeds 64 KiB");
        input.push_back(c);
    }
    if (!std::cin.eof() || std::cin.bad())
        throw std::runtime_error("could not read prompt token input");
    std::istringstream in(input);
    return prompt_ids(in, limit);
}
Options parse(int argc, char** argv) {
    Options o;
    bool smoke_shape = false, generation_arg = false;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--token" || key == "--tokens" || key == "--rows" ||
            key == "--keep" || key == "--windows") smoke_shape = true;
        if (key == "--run") o.run = true;
        else if (key == "--layer-taps") o.layer_taps = true;
        else if (key == "--layer0-probe") o.layer_zero_probe = true;
        else if (key == "--stage0-phases") o.stage0_phases = true;
        else if (key == "--branch-probe") o.branch_probe = true;
        else if (key == "--micro-probe") o.micro_probe = true;
        else if (key == "--boundary-capture") o.boundary_capture = true;
        else if (key == "--placement-capture") o.placement_capture = true;
        else if (key == "--shared") o.shared = true;
        else if (key == "--group-native-hip") o.group_hip = true;
        else if (key == "--hip-split-overlap") o.hip_split_overlap = true;
        else if (key == "--receiver-arm") o.receiver_arm = true;
        else if (key == "--cuda-mapped-egress") o.cuda_mapped_egress = true;
        else if (key == "--direct-verified-shared") o.direct_verified = true;
        else if (key == "--kv") {
            if (o.kv_explicit) throw std::invalid_argument("duplicate --kv");
            o.kv = hetero::parse_kv_mode(required_value(i,argc,argv,"--kv"));
            o.kv_explicit = true;
        }
        else if (key == "--mtp-repeat-native-round") o.mtp_repeat_native_round=true;
        else if (key == "--mtp-front-trace") o.mtp_front_trace=true;
        else if (key == "--mtp-state-trace") o.mtp_state_trace=true;
        else if (key == "--mtp-reprime-tail") o.mtp_reprime_tail=true;
        else if (key == "--mtp-fresh-replay") o.mtp_fresh_replay=true;
        else if (key == "--mtp-three-round") o.mtp_three_round=true;
        else if (key == "--generation-parity-capture") o.generation_capture=true;
        else if (key == "--mtp-generation-qualification") o.mtp_generation=true;
        else if (key == "--mtp-resident-prompt-rows") {
            o.mtp_resident_prompt_rows=u32(required_value(i,argc,argv,"--mtp-resident-prompt-rows"),"mtp-resident-prompt-rows");
            if (!o.mtp_resident_prompt_rows || o.mtp_resident_prompt_rows>hetero::MtpGenerationPolicy::max_prompt)
                throw std::invalid_argument("resident MTP prompt rows must be 1..7");
        }
        else if (key == "--mtp-three-round-variable") o.mtp_three_round_variable=true;
        else if (key == "--mtp-relay-rounds") {
            o.mtp_relay_rounds=u32(required_value(i,argc,argv,"--mtp-relay-rounds"),"mtp-relay-rounds");
            if (!o.mtp_relay_rounds || o.mtp_relay_rounds>8) throw std::invalid_argument("MTP relay rounds must be 1..8");
        }
        else if (key == "--mtp-probe-rt") {
            o.mtp_probe_rt = required_value(i,argc,argv,"--mtp-probe-rt");
            if (o.mtp_probe_rt.empty()) throw std::invalid_argument("MTP probe runtime path is empty");
        }
        else if (key == "--warmup-windows") {
            o.warmup_windows = u32(required_value(i,argc,argv,"--warmup-windows"),"warmup-windows");
            if (!o.warmup_windows) throw std::invalid_argument("warmup-windows must be positive");
            smoke_shape = true;
        }
        else if (key == "--prefill-state-capture") o.prefill_state_capture = true;
        else if (key == "--prefill-first-token") o.prefill_first_token = true;
        else if (key == "--prefill-hc-replay-capture") o.prefill_hc_replay = true;
        else if (key == "--prefill-stage-capture") o.prefill_stage_capture = true;
        else if (key == "--prefill-numerical-trace") o.numerical_trace = true;
        else if (key == "--prefill-stage-reference") o.prefill_stage_reference =
            required_value(i,argc,argv,"--prefill-stage-reference");
        else if (key == "--describe-source") o.describe_source = true;
        else if (key == "--resident") { o.resident = true; generation_arg = true; }
        else if (key == "--workload-mode") { o.workload_mode = true; generation_arg = true; }
        else if (key == "--stream-tokens") { o.stream_tokens = true; generation_arg = true; }
        else if (key == "--lookup-draft") { o.lookup_draft = true; generation_arg = true; }
        else if (key == "--context-limit") {
            generation_arg = true;
            o.context_limit = u32(required_value(i, argc, argv, "--context-limit"), "context-limit");
            if (o.context_limit < 2 || o.context_limit > hetero::max_native_prefill_tokens)
                throw std::invalid_argument("--context-limit must be in 2..32768");
        }
        else if (key == "--generate") {
            generation_arg = true;
            o.generate = u32(required_value(i, argc, argv, "--generate"), "generate");
            if (!o.generate || o.generate > 4096)
                throw std::invalid_argument("--generate must be in 1..4096");
        } else if (key == "--prompt-chunk") {
            generation_arg = true;
            o.prompt_chunk = u32(required_value(i, argc, argv, "--prompt-chunk"), "prompt-chunk");
            if (!o.prompt_chunk || o.prompt_chunk > 8)
                throw std::invalid_argument("--prompt-chunk must be in 1..8");
        } else if (key == "--native-prefill-chunk") {
            generation_arg = true;
            o.native_prefill_chunk=u32(required_value(i,argc,argv,"--native-prefill-chunk"),"native-prefill-chunk");
            if (!o.native_prefill_chunk || o.native_prefill_chunk>hetero::max_native_prefill_chunk)
                throw std::invalid_argument("native prefill chunk outside bounded range");
        } else if (key == "--prefill-parity-at") {
            generation_arg = true;
            o.prefill_parity_at = u32(required_value(i, argc, argv, "--prefill-parity-at"), "prefill-parity-at");
            if (!o.prefill_parity_at) throw std::invalid_argument("prefill-parity-at must be positive");
        } else if (key == "--source-gguf") {
            generation_arg = true;
            o.source_gguf = required_value(i, argc, argv, "--source-gguf");
        } else if (key == "--source-index") {
            generation_arg = true;
            o.source_index = required_value(i, argc, argv, "--source-index");
        } else if (key == "--stop-ids") {
            generation_arg = true;
            const std::string value = required_value(i, argc, argv, "--stop-ids");
            std::istringstream in(value);
            std::string item;
            while (std::getline(in, item, ',')) {
                const uint32_t id = u32(item, "stop-ids");
                if (id > INT32_MAX || o.stop_ids.size() >= 16)
                    throw std::invalid_argument("invalid bounded stop IDs");
                o.stop_ids.push_back(static_cast<int32_t>(id));
            }
            if (value.empty() || value.back() == ',')
                throw std::invalid_argument("invalid comma-separated stop IDs");
        }
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
                if (token > static_cast<uint32_t>(INT32_MAX) || o.tokens.size() >= 128)
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
        o.windows == 0 || o.windows > ((o.placement_capture || o.warmup_windows) ? 128u : (o.mtp_fresh_replay ? hetero::mtp_fresh_replay_max : 8u)) || o.token > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
        !o.layers || !o.hc || !o.n_embd)
        throw std::invalid_argument("missing plan/fingerprint or invalid bounded smoke-test options");
    if (o.placement_capture && (o.generate || o.describe_source || o.resident || o.rows != 1 ||
        o.tokens.empty() || o.capture_prefix.empty() || o.layer_taps || o.layer_zero_probe ||
        o.stage0_phases || o.branch_probe || o.micro_probe || o.boundary_capture))
        throw std::invalid_argument("placement capture requires explicit one-row tokens, full STFR and no other probes");
    if (!o.placement_capture && !o.warmup_windows && !o.generate && o.tokens.size() > 64)
        throw std::invalid_argument("ordinary smoke token sequence exceeds 64");
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
    if (o.describe_source && (o.run || generation_arg || smoke_shape))
        throw std::invalid_argument("--describe-source is a CPU-only standalone mode");
    if (generation_arg && !o.generate)
        throw std::invalid_argument("generation options require --generate");
    if (o.generate || o.describe_source) {
        if (smoke_shape || (!o.capture_prefix.empty() && !o.prefill_parity_at && !o.generation_capture) || o.layer_taps ||
            o.layer_zero_probe || o.stage0_phases || o.branch_probe ||
            o.micro_probe || o.boundary_capture)
            throw std::invalid_argument("generation/source modes cannot use smoke diagnostic options");
    }
    if (o.mtp_generation && !o.generation_capture)
        throw std::invalid_argument("native MTP generation requires explicit bounded parity capture");
    if (o.mtp_resident_prompt_rows && (!o.resident || !o.mtp_generation || !o.generation_capture ||
        !hetero::MtpGenerationPolicy{o.mtp_resident_prompt_rows,o.generate,o.stop_ids}.valid()))
        throw std::invalid_argument("resident MTP requires explicit fixed bounded native generation qualification");
    if (o.generation_capture && (!o.generate || o.capture_prefix.empty() || !o.group_hip || !o.hip_split_overlap ||
        (o.resident && !o.mtp_resident_prompt_rows) || o.stream_tokens || o.lookup_draft || o.prompt_chunk!=1 || o.context_limit!=2048 ||
        o.receiver_arm || o.cuda_mapped_egress || o.native_prefill_chunk || o.prefill_parity_at || o.prefill_state_capture ||
        o.placement_capture || o.warmup_windows || o.mtp_fresh_replay || o.mtp_three_round || o.mtp_three_round_variable ||
        o.mtp_relay_rounds || o.mtp_reprime_tail || o.mtp_repeat_native_round ||
        (o.mtp_generation ? (o.mtp_probe_rt.empty() || o.mtp_state_trace!=o.mtp_front_trace)
                          : (!o.mtp_probe_rt.empty() || o.mtp_state_trace || o.mtp_front_trace))))
        throw std::invalid_argument("generation parity requires bounded nonresident one-row greedy grouped HIP overlap, no other probes");
    if (o.mtp_fresh_replay && (o.mtp_probe_rt.empty() || o.tokens.empty() ||
        !hetero::mtp_probe_window_count(o.windows,o.mtp_relay_rounds,true) ||
        !hetero::mtp_fresh_replay_shape(o.rows,o.keep,!o.capture_prefix.empty(),o.mtp_repeat_native_round) ||
        o.generate || o.resident || o.describe_source || o.warmup_windows || o.placement_capture ||
        o.layer_taps || o.layer_zero_probe || o.stage0_phases || o.branch_probe ||
        o.micro_probe || o.boundary_capture || o.mtp_reprime_tail))
        throw std::invalid_argument("MTP fresh replay requires 1..12 explicit one-row full commits, capture, no relay or repeat");
    if (o.mtp_three_round_variable && !o.mtp_three_round)
        throw std::invalid_argument("variable prefix requires explicit MTP three-round diagnostic");
    if (o.mtp_three_round && (o.mtp_probe_rt.empty() || o.capture_prefix.empty() ||
        !hetero::mtp_three_shape(o.windows,o.mtp_relay_rounds,o.mtp_fresh_replay,o.mtp_repeat_native_round,o.mtp_reprime_tail,o.mtp_three_round_variable) ||
        o.rows!=1 || o.keep!=1 || o.tokens.size()!=o.windows ||
        (!o.mtp_three_round_variable &&
         o.tokens!=std::vector<int32_t>(hetero::mtp_three_prompt.begin(),hetero::mtp_three_prompt.end())) ||
        o.generate || o.resident || o.describe_source || o.warmup_windows || o.placement_capture ||
        o.layer_taps || o.layer_zero_probe || o.stage0_phases || o.branch_probe ||
        o.micro_probe || o.boundary_capture || o.mtp_state_trace!=o.mtp_front_trace))
        throw std::invalid_argument("MTP three-round diagnostic requires qualified fixed or explicit bounded variable prompt, three relays, capture and either both traces or neither");
    if (o.mtp_state_trace && (o.mtp_probe_rt.empty() || o.capture_prefix.empty() ||
        (!o.mtp_three_round && !o.mtp_generation && (o.mtp_relay_rounds>1 || uint64_t(o.windows-1)*o.keep+o.rows+5ull*o.mtp_relay_rounds>16))))
        throw std::invalid_argument("MTP state trace requires fresh capture, at most 16 cells and one relay");
    if (o.mtp_repeat_native_round && (!o.mtp_front_trace || o.mtp_relay_rounds))
        throw std::invalid_argument("MTP repeat requires one-shot front trace");
    if (o.mtp_front_trace && !o.mtp_state_trace)
        throw std::invalid_argument("MTP front trace requires bounded state trace");
    if (o.mtp_reprime_tail && (!o.mtp_state_trace || o.mtp_relay_rounds!=1))
        throw std::invalid_argument("MTP tail reprime requires explicit state trace and one relay");
    if (o.mtp_relay_rounds && (o.mtp_probe_rt.empty() || o.capture_prefix.empty()))
        throw std::invalid_argument("MTP relay requires explicit MTP probe and full-logit capture prefix");
    if (!o.mtp_probe_rt.empty() && (!o.group_hip || !o.hip_split_overlap || o.receiver_arm || (o.generate && !o.mtp_generation) ||
        o.describe_source || o.warmup_windows || o.native_prefill_chunk || (!o.mtp_generation && o.tokens.empty()) ||
        !std::filesystem::path(o.mtp_probe_rt).is_absolute() || o.mtp_probe_rt.size()>4096 ||
        o.mtp_probe_rt.find_first_of("\r\n")!=std::string::npos))
        throw std::invalid_argument("MTP probe requires explicit fixed tokens, absolute runtime path, grouped HIP overlap, nonresident and no receiver arm");
    if (o.direct_verified && (!o.shared || !o.group_hip || !o.hip_split_overlap ||
        o.receiver_arm || o.cuda_mapped_egress || o.native_prefill_chunk || o.lookup_draft ||
        o.describe_source || o.boundary_capture || o.layer_taps || o.layer_zero_probe ||
        o.stage0_phases || o.branch_probe || o.micro_probe || o.prefill_parity_at ||
        o.prefill_state_capture || o.placement_capture || o.generation_capture ||
        !o.mtp_probe_rt.empty()))
        throw std::invalid_argument("direct verified shared requires grouped HIP overlap, serial prefill, no external prearm or probes");
    if (o.cuda_mapped_egress && (!o.receiver_arm || o.native_prefill_chunk || o.boundary_capture ||
        o.prefill_parity_at || o.describe_source))
        throw std::invalid_argument("mapped CUDA egress requires receiver arm, serial prefill and no device boundary capture");
    if (o.warmup_windows && (!o.group_hip || o.generate || o.describe_source || o.capture_prefix.size() ||
        o.layer_taps || o.layer_zero_probe || o.stage0_phases || o.branch_probe || o.micro_probe ||
        o.boundary_capture || o.placement_capture || o.warmup_windows >= o.windows ||
        uint64_t(o.rows)*o.windows > 128 || o.tokens.size() != uint64_t(o.rows)*o.windows))
        throw std::invalid_argument("timed smoke requires grouped fixed tokens, 1..127 warm windows and <=128 evaluated tokens, no probes/captures");
    if (o.receiver_arm && (!o.hip_split_overlap || !o.group_hip || o.native_prefill_chunk || o.describe_source))
        throw std::invalid_argument("receiver arm requires explicit grouped HIP overlap and serial prefill");
    if (o.hip_split_overlap && (!o.group_hip || o.native_prefill_chunk || o.describe_source))
        throw std::invalid_argument("HIP split overlap qualification requires grouped native HIP and serial prefill");
    // Label explicit common-weight diagnostics without changing native loader defaults.
    const char* weight_mode = std::getenv("STRATA_HETERO_HC_WEIGHT_CONTRACT");
    if (weight_mode && *weight_mode) {
        const std::string mode(weight_mode);
        const char* q8 = std::getenv("STRATA_HC_Q8");
        if ((mode != "bf16" && mode != "q8") || !q8 ||
            std::string(q8) != (mode == "bf16" ? "0" : "1") ||
            !(o.placement_capture || o.prefill_stage_capture || o.warmup_windows) ||
            o.resident || !o.mtp_probe_rt.empty())
            throw std::invalid_argument("explicit HC weight contract requires matching STRATA_HC_Q8 and bounded diagnostic");
    }
    const auto hc_exp = strata::prefill::hc_experiment_config();
    const char* global_bf16x2 = std::getenv("STRATA_PREFILL_BF16X2");
    if (!hc_exp.valid() || (hc_exp.requested() &&
        (!o.prefill_stage_capture || !o.prefill_state_capture || o.resident ||
         !o.mtp_probe_rt.empty() || o.prefill_hc_replay || o.prefill_first_token ||
         (global_bf16x2 && std::atoi(global_bf16x2) != 0) ||
         !((o.prefill_parity_at == 1 && o.native_prefill_chunk == 1) ||
           (o.prefill_parity_at == 2 && (o.native_prefill_chunk == 1 || o.native_prefill_chunk == 2)) ||
           (o.prefill_parity_at == 6 && (o.native_prefill_chunk == 3 || o.native_prefill_chunk == 6))))))
        throw std::invalid_argument("HC experiment requires one canonical bounded stage/state A/B/A, no MTP and no global BF16X2");
    if (o.numerical_trace && (!o.prefill_stage_capture || !o.prefill_stage_reference.empty() ||
        !((o.prefill_parity_at==2 && (o.native_prefill_chunk==1 || o.native_prefill_chunk==2)) ||
          (o.prefill_parity_at==6 && (o.native_prefill_chunk==3 || o.native_prefill_chunk==6)))))
        throw std::invalid_argument("numerical trace requires wire stage capture at cut 2 or 6");
    if (!o.prefill_stage_reference.empty() && !o.prefill_stage_capture)
        throw std::invalid_argument("stage reference requires stage capture");
    if (o.prefill_stage_capture && (!o.prefill_state_capture || !o.prefill_parity_at ||
        o.prefill_parity_at>16 || !o.native_prefill_chunk || o.native_prefill_chunk>o.prefill_parity_at ||
        o.prefill_first_token || o.prefill_hc_replay || !o.mtp_probe_rt.empty()))
        throw std::invalid_argument("stage capture requires bounded A/B/A state capture without HC/MTP probes");
    if (o.prefill_hc_replay && !o.prefill_first_token)
        throw std::invalid_argument("HC replay capture requires first-token A/B/A");
    if (o.prefill_first_token && (!o.prefill_state_capture || o.prefill_parity_at!=1 || o.native_prefill_chunk!=1))
        throw std::invalid_argument("first-token localization requires A/B/A cut/chunk 1 and state capture");
    if (o.prefill_state_capture && (!o.prefill_parity_at || o.prefill_parity_at > 16))
        throw std::invalid_argument("prefill state capture requires explicit A/B/A and cut 1..16");
    if (o.prefill_parity_at && (o.generate != 1 || !o.group_hip || o.resident || o.stream_tokens ||
        o.lookup_draft || o.prompt_chunk != 1 || !o.native_prefill_chunk || o.capture_prefix.empty()))
        throw std::invalid_argument("prefill parity requires grouped, nonresident --generate 1, native chunk, one-row control and capture prefix");
    if (o.native_prefill_chunk && (!o.group_hip || (!o.resident && !o.prefill_parity_at) || !o.generate))
        throw std::invalid_argument("native bulk prefill requires grouped HIP resident generation or explicit parity diagnostic");
    if (o.stream_tokens && !o.resident)
        throw std::invalid_argument("--stream-tokens requires --resident");
    if (o.lookup_draft && (o.stream_tokens || o.prompt_chunk < 2))
        throw std::invalid_argument("--lookup-draft requires buffered generation and chunk >= 2");
    if (!o.workload_mode && (o.context_limit > 8192 || o.generate > 256))
        throw std::invalid_argument("extended context or completion requires --workload-mode");
    if (o.workload_mode && (!o.resident || !o.generate || o.prefill_parity_at ||
        o.generation_capture || !o.mtp_probe_rt.empty() || o.receiver_arm || o.hip_split_overlap))
        throw std::invalid_argument("workload mode needs resident ordinary text, without MTP or diagnostic overlap");
    if (o.generate) {
        if (o.generate >= o.context_limit)
            throw std::invalid_argument("completion budget leaves no prompt context");
        if (o.source_gguf.empty() || o.source_index.empty() || o.stop_ids.empty())
            throw std::invalid_argument("generation requires validated source and stop IDs");
        if (!o.resident) {
            o.tokens = prompt_from_stdin(o.context_limit);
            if (o.generation_capture) {
                if (o.tokens.size()>hetero::MtpGenerationPolicy::max_prompt ||
                    !hetero::MtpGenerationPolicy{static_cast<unsigned>(o.tokens.size()),o.generate,o.stop_ids}.valid())
                    throw std::invalid_argument("bounded generation requires 1..7 prompt rows and at most 12 fed rows");
            }
            if (uint64_t(o.tokens.size()) + o.generate > o.context_limit)
                throw std::invalid_argument("prompt and completion exceed context limit");
            if (o.numerical_trace || hc_exp.requested()) {
                const std::vector<int32_t> expected{248045,846,198,2523,513,32767,264};
                if(o.tokens.size()!=o.prefill_parity_at+1 ||
                   !std::equal(o.tokens.begin(),o.tokens.end(),expected.begin()))
                    throw std::invalid_argument("numerical trace / HC experiment requires canonical teacher-forced sequence");
            }
            if (o.prefill_first_token && o.tokens.size()!=2)
                throw std::invalid_argument("first-token localization requires one prefix and one forced token");
            if (o.prefill_parity_at && (o.prefill_parity_at >= o.tokens.size() ||
                o.tokens.size() - o.prefill_parity_at > 128))
                throw std::invalid_argument("prefill parity requires a nonempty prefix and 1..128 teacher-forced rows");
        }
    } else if (!o.tokens.empty() && o.tokens.size() != static_cast<size_t>(o.rows) * o.windows) {
        throw std::invalid_argument("--tokens must contain exactly rows*windows token IDs");
    }
    // Existing tightly bound diagnostics have FP16 reference artifacts.
    // Keep them negative-capable without changing their historical semantics.
    const bool legacy_f16 = !o.mtp_probe_rt.empty() || o.generation_capture ||
        o.prefill_parity_at || o.prefill_state_capture || o.prefill_stage_capture ||
        o.prefill_hc_replay || o.numerical_trace || o.layer_taps || o.layer_zero_probe ||
        o.stage0_phases || o.branch_probe || o.micro_probe || o.boundary_capture ||
        o.placement_capture;
    if (legacy_f16) {
        if (o.kv_explicit && o.kv != hetero::KvMode::f16)
            throw std::invalid_argument("legacy parity/MTP diagnostics require --kv f16 until separately qualified");
        o.kv = hetero::KvMode::f16;
    }
    return o;
}
void require_executable(const std::string& path, const char* label) {
    std::error_code ec;
    if (path.empty() || path[0] != '/' ||
        !std::filesystem::is_regular_file(path, ec) || ec ||
        ::access(path.c_str(), X_OK) != 0)
        throw std::invalid_argument(std::string(label) + " must be an executable absolute file");
}
std::string json_string(const std::string& value);
struct GenerationLimits { uint32_t vocab, max_cells, max_t; };
GenerationLimits preflight(const Options& o, const hetero::Plan& plan) {
    if (o.direct_verified && !hetero::direct_verified_plan(plan))
        throw std::invalid_argument("direct verified shared supports established mixed placements only");
    const bool reverse = hetero::plan_spec(plan) == "hip:0:0-18,hip:1:18-36,cuda:0:36-48";
    const bool hip_pair = plan.stages.size()==2 &&
        plan.stages[0].backend==hetero::Backend::hip && plan.stages[1].backend==hetero::Backend::hip;
    if (o.workload_mode && !o.group_hip && !hip_pair)
        throw std::invalid_argument("ungrouped workload mode is restricted to two HIP stages");
    if (hip_pair && (o.hip_split_overlap || o.receiver_arm || o.cuda_mapped_egress ||
        !o.mtp_probe_rt.empty() || o.prefill_parity_at || !o.capture_prefix.empty() ||
        o.generation_capture || o.placement_capture || o.layer_taps || o.boundary_capture ||
        o.layer_zero_probe || o.stage0_phases || o.branch_probe || o.micro_probe))
        throw std::invalid_argument("HIP-only pair supports ordinary inference only; diagnostic overlap, captures and MTP remain gated");
    // Reverse topology first qualifies the existing per-stage serial worker path.
    // Do not advertise grouped bulk prefill or MTP until their protocols are implemented.
    if (reverse && (o.group_hip || o.native_prefill_chunk || o.prefill_parity_at ||
                    o.hip_split_overlap || o.receiver_arm || o.cuda_mapped_egress ||
                    !o.mtp_probe_rt.empty()))
        throw std::invalid_argument("reverse topology currently supports ungrouped serial workers only");
    if (o.prefill_stage_capture) {
        if (hetero::plan_spec(plan)!="cuda:0:0-12,hip:0:12-30,hip:1:30-48" ||
            plan.geometry.hc!=4 || plan.geometry.n_embd!=2560 || o.context_limit!=2048)
            throw std::invalid_argument("stage capture requires original physical plan and context");
        if (!o.prefill_stage_reference.empty()) {
            const std::filesystem::path ref(o.prefill_stage_reference);
            if (!ref.is_absolute() || ref==std::filesystem::path(o.capture_prefix) ||
                o.prefill_stage_reference.size()>4096 || o.prefill_stage_reference.find_first_of("\r\n")!=std::string::npos ||
                !std::filesystem::is_regular_file(o.prefill_stage_reference+".prefill.json"))
                throw std::invalid_argument("stage reference requires completed distinct absolute source prefix");
            for (const auto& part:ref) if (part=="." || part=="..")
                throw std::invalid_argument("stage reference may not use dot segments");
        }
    }
    if (o.prefill_hc_replay && (hetero::plan_spec(plan)!="cuda:0:0-12,hip:0:12-30,hip:1:30-48" ||
        plan.geometry.hc!=4 || plan.geometry.n_embd!=2560 || o.context_limit!=2048))
        throw std::invalid_argument("HC replay capture requires original physical plan and context");
    if (o.group_hip) {
        (void)hetero::worker_domains(plan, true);
        if (o.layer_taps || o.layer_zero_probe || o.stage0_phases || o.branch_probe ||
            o.micro_probe || o.placement_capture || o.boundary_capture)
            throw std::invalid_argument("grouped HIP currently supports ordinary generation/smoke only");
    }
    if (o.configs.size() != plan.stages.size())
        throw std::invalid_argument("one ordered --config is required per planned stage");
    // The last provisional window, including its discarded suffix, must fit.
    const uint64_t needed = o.workload_mode ? uint64_t(o.context_limit)
        : o.resident ? uint64_t(o.generate) + 1
        : o.generate ? uint64_t(o.tokens.size()) + o.generate
         : o.mtp_three_round ? hetero::mtp_three_provisional_max
         : uint64_t(o.windows - 1) * o.keep + o.rows + 5ull*o.mtp_relay_rounds;
    const uint32_t rows = o.mtp_generation ? 5u : (o.resident || o.lookup_draft) ? o.prompt_chunk
        : o.generate ? std::min(o.prompt_chunk, static_cast<uint32_t>(o.tokens.size())) :
          std::max(o.rows,o.mtp_relay_rounds?5u:1u);
    const uint64_t stride = plan.geometry.floats_per_row();
    if (stride > hetero::max_payload_bytes / (sizeof(float) * rows))
        throw std::invalid_argument("requested activation geometry exceeds bounded 16 MiB transport");
    std::optional<hetero::CudaWorkerConfig> common_model;
    uint32_t max_cells = UINT32_MAX, max_t = UINT32_MAX;
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
        if (hip_pair && config.max_cache_slots != uint64_t(config.end_layer-config.first_layer)*512)
            throw std::invalid_argument("HIP-only experiment requires explicit full routed expert residency");
        if (common_model) {
            if (config.pack_dir != common_model->pack_dir ||
                config.profile_path != common_model->profile_path ||
                config.index_sha256 != common_model->index_sha256 ||
                config.gguf_shards != common_model->gguf_shards ||
                config.n_vocab != common_model->n_vocab ||
                config.model_fingerprint != common_model->model_fingerprint)
                throw std::invalid_argument("cross-stage Q2_0 model source identity mismatch");
        } else common_model = config;
        if ((o.generation_capture || reverse) && (config.index_sha256!=hetero::mtp_qualification_index ||
            config.max_cells!=2048 || config.max_t!=8 || config.n_vocab!=248320 ||
            config.max_cache_slots!=uint64_t(config.end_layer-config.first_layer)*512 ||
            config.free_reserve_bytes!=536870912ull || config.arena_budget_bytes))
            throw std::invalid_argument("generation qualification requires the previously approved unchanged model and resource config");
        if (o.direct_verified && (config.max_cells != 2048 || config.max_t != 8 ||
            config.max_cache_slots != uint64_t(config.end_layer-config.first_layer)*512))
            throw std::invalid_argument("direct verified qualification requires 2K context, max_t=8 and full residency");
        const bool invalid_token = o.tokens.empty()
            ? config.n_vocab <= o.token
            : std::any_of(o.tokens.begin(), o.tokens.end(),
                          [&](int32_t token) { return token >= static_cast<int32_t>(config.n_vocab); });
        const bool invalid_stop = std::any_of(o.stop_ids.begin(), o.stop_ids.end(),
            [&](int32_t token) { return static_cast<uint32_t>(token) >= config.n_vocab; });
        if (config.version != 2 || config.max_cells < needed ||
            config.max_t < rows || invalid_token || invalid_stop)
            throw std::invalid_argument("stage " + std::to_string(index) +
                ": v2, context, window or token bounds cannot serve smoke-test request");
        max_cells = std::min(max_cells, config.max_cells);
        max_t = std::min(max_t, config.max_t);
        if (o.generate && (uint64_t(rows) * config.n_vocab + 2ull * rows + 1) >
                hetero::final_result_max_payload_bytes / sizeof(float))
            throw std::invalid_argument("prompt chunk logits exceed STFR payload cap");
        std::cout << "PREFLIGHT stage=" << index
                  << " backend=" << (is_hip ? "HIP" : "CUDA")
                  << " ordinal=" << config.device_ordinal
                  << " layers=[" << config.first_layer << ',' << config.end_layer << ')'
                  << " max_cells=" << config.max_cells
                  << " max_t=" << config.max_t << " kv=" << hetero::kv_mode_name(o.kv) << '\n';
    }
    if (!common_model || common_model->gguf_shards.empty())
        throw std::invalid_argument("native model source is missing");
    if (o.generate && (o.source_gguf != common_model->gguf_shards.front() ||
                       o.source_index != common_model->index_sha256))
        throw std::invalid_argument("generation tokenizer source differs from validated worker source");
    if (o.generation_capture && hetero::plan_spec(plan)!="cuda:0:0-12,hip:0:12-30,hip:1:30-48")
        throw std::invalid_argument("generation qualification requires the approved 12/18/18 plan");
    if (o.mtp_generation) {
        std::string error;
        if (!hetero::validate_mtp_qualification_runtime(o.mtp_probe_rt,common_model->index_sha256,error))
            throw std::runtime_error(error);
    }
    if (o.describe_source) {
        std::cout << "HETERO_SOURCE {\"gguf\":" << json_string(common_model->gguf_shards.front())
                  << ",\"pack_dir\":" << json_string(common_model->pack_dir)
                  << ",\"pack_index_sha256\":" << json_string(common_model->index_sha256)
                  << ",\"vocab\":" << common_model->n_vocab
                  << ",\"max_cells\":" << max_cells << ",\"max_t\":" << max_t
                  << ",\"kv\":\"" << hetero::kv_mode_name(o.kv) << "\""
                  << ",\"plan_fingerprint\":" << plan.fingerprint() << "}\n";
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
            if (std::filesystem::exists(o.capture_prefix + ".resident.json") ||
                std::filesystem::exists(o.capture_prefix + ".complete.json") ||
                std::filesystem::exists(o.capture_prefix + ".prefill.json"))
                throw std::invalid_argument("capture already completed; use a fresh prefix");
            if (o.prefill_parity_at || o.generation_capture) {
                // All potentially emitted files must be fresh; mode identities are not legacy smoke window IDs.
                const std::string stem = prefix.filename().string() + ".";
                for (const auto& entry : std::filesystem::directory_iterator(prefix.parent_path())) {
                    const auto name = entry.path().filename().string();
                    if (name.compare(0, stem.size(), stem) == 0)
                        throw std::invalid_argument("prefill parity capture exists; use a fresh prefix");
                }
            }
            for (uint32_t n = 1; n <= o.windows+o.mtp_relay_rounds; ++n) {
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
    return {common_model->n_vocab, max_cells, max_t};
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
             << ",\"kv_mode\":" << json_string(hetero::kv_mode_name(o.kv))
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
int execute(const Options& o, const hetero::Plan& plan, GenerationLimits limits) {
    // No device calls in coordinator. Child workers alone see this opt-in;
    // no tap allocations/copies occur for ordinary M2.9/M2.10 tests.
    if (o.layer_taps && ::setenv("STRATA_HETERO_LAYER_TAP_PREFIX",
                                 o.capture_prefix.c_str(), 1) != 0)
        throw std::runtime_error("cannot configure native worker layer-tap capture");
    if (o.generate && ::unsetenv("STRATA_HETERO_LAYER_TAP_PREFIX") != 0)
        throw std::runtime_error("cannot clear inherited native layer-tap capture");
    if (o.numerical_trace ? (::setenv("STRATA_HETERO_NUMERICAL_TRACE_PREFIX",o.capture_prefix.c_str(),1)!=0 ||
                             ::setenv("STRATA_HETERO_NUMERICAL_TRACE_CUT",std::to_string(o.prefill_parity_at).c_str(),1)!=0)
                          : (::unsetenv("STRATA_HETERO_NUMERICAL_TRACE_PREFIX")!=0 ||
                             ::unsetenv("STRATA_HETERO_NUMERICAL_TRACE_CUT")!=0))
        throw std::runtime_error("cannot configure numerical trace");
    if (o.numerical_trace) {
        for(const char* flag:{"STRATA_CKPT_ASYNC","STRATA_CKPT_GATHER","STRATA_CKPT_VRAM_SLOTS","STRATA_PARK_ASYNC",
                              "STRATA_DBG_ROUTE_OVERRIDE","STRATA_ROUTE_PRIOR","STRATA_ROUTE_RESIDENT"}) {
            const char* v=std::getenv(flag);
            if(v && *v && std::strcmp(v,"0")) throw std::invalid_argument(std::string("numerical trace refuses ")+flag);
        }
    }
    if (o.prefill_stage_capture ? ::setenv("STRATA_HETERO_NATIVE_STAGE_CAPTURE",o.capture_prefix.c_str(),1)!=0
                                : ::unsetenv("STRATA_HETERO_NATIVE_STAGE_CAPTURE")!=0)
        throw std::runtime_error("cannot configure native stage capture");
    if (!o.prefill_stage_reference.empty() ? ::setenv("STRATA_HETERO_NATIVE_STAGE_REFERENCE",o.prefill_stage_reference.c_str(),1)!=0
                                            : ::unsetenv("STRATA_HETERO_NATIVE_STAGE_REFERENCE")!=0)
        throw std::runtime_error("cannot configure native stage reference");
    if (o.prefill_hc_replay ? ::setenv("STRATA_HETERO_PREFILL_HC_REPLAY","1",1)!=0
                            : ::unsetenv("STRATA_HETERO_PREFILL_HC_REPLAY")!=0)
        throw std::runtime_error("cannot configure HC replay capture");
    if (o.prefill_first_token ? ::setenv("STRATA_HETERO_PREFILL_FIRST_TOKEN","1",1)!=0
                              : ::unsetenv("STRATA_HETERO_PREFILL_FIRST_TOKEN")!=0)
        throw std::runtime_error("cannot configure first-token localization");
    if (o.layer_zero_probe || o.prefill_first_token) {
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
    if (o.boundary_capture || o.prefill_parity_at) {
        const std::string verifier_prefix = o.prefill_parity_at ? o.capture_prefix + ".verifier" : o.capture_prefix;
        if (::setenv("STRATA_HETERO_BOUNDARY_CAPTURE_PREFIX",
                     verifier_prefix.c_str(), 1) != 0)
            throw std::runtime_error("cannot enable native boundary capture");
    } else if (::unsetenv("STRATA_HETERO_BOUNDARY_CAPTURE_PREFIX") != 0) {
        throw std::runtime_error("cannot clear inherited boundary capture");
    }
    // This diagnostic only captures the cross-vendor prompt residual, never a grouped decode probe.
    if (o.prefill_parity_at) {
        if (::setenv("STRATA_HETERO_PREFILL_CAPTURE_PREFIX", o.capture_prefix.c_str(), 1) != 0)
            throw std::runtime_error("cannot enable native prefill boundary readback");
    } else if (::unsetenv("STRATA_HETERO_PREFILL_CAPTURE_PREFIX") != 0) {
        throw std::runtime_error("cannot clear inherited native prefill capture");
    }
    if (o.prefill_state_capture) {
        if (::setenv("STRATA_HETERO_PREFILL_STATE_PREFIX", o.capture_prefix.c_str(), 1) != 0 ||
            ::setenv("STRATA_HETERO_PREFILL_STATE_AT", std::to_string(o.prefill_parity_at).c_str(), 1) != 0)
            throw std::runtime_error("cannot configure CUDA prefix state capture");
    } else if (::unsetenv("STRATA_HETERO_PREFILL_STATE_PREFIX") != 0 ||
               ::unsetenv("STRATA_HETERO_PREFILL_STATE_AT") != 0) {
        throw std::runtime_error("cannot clear inherited CUDA prefix state capture");
    }
    if (o.hip_split_overlap) {
        if (const char* wait = std::getenv("STRATA_SPLIT_WAIT_MS")) {
            const auto ms = u32(wait, "STRATA_SPLIT_WAIT_MS");
            if (!ms || ms > 120000) throw std::invalid_argument("split overlap requires bounded native handoff wait (1..120000 ms)");
        }
        if (const char* drop = std::getenv("STRATA_TEST_HANDOFF_DROP"))
            if (std::string(drop) != "0") throw std::invalid_argument("native handoff fault injection is not authorized");
    }
    if (::setenv("STRATA_HETERO_HIP_SPLIT_OVERLAP", o.hip_split_overlap ? "1" : "0", 1) != 0)
        throw std::runtime_error("cannot configure explicit HIP split overlap mode");
    if (::setenv("STRATA_HETERO_MTP_STATE_PREFIX",o.mtp_state_trace?o.capture_prefix.c_str():"",1)!=0 ||
        ::setenv("STRATA_HETERO_MTP_REPRIME_TAIL",o.mtp_reprime_tail?"1":"0",1)!=0 ||
        ::setenv("STRATA_HETERO_MTP_FRONT_TRACE",o.mtp_front_trace?"1":"0",1)!=0 ||
        ::setenv("STRATA_HETERO_MTP_REPEAT_NATIVE_ROUND",o.mtp_repeat_native_round?"1":"0",1)!=0 ||
        ::setenv("STRATA_HETERO_MTP_FRESH_REPLAY",o.mtp_fresh_replay?"1":"0",1)!=0 ||
        ::setenv("STRATA_HETERO_MTP_THREE_ROUND",o.mtp_three_round?"1":"0",1)!=0 ||
        ::setenv("STRATA_HETERO_MTP_THREE_ROUND_VARIABLE",o.mtp_three_round_variable?"1":"0",1)!=0)
        throw std::runtime_error("cannot configure MTP state trace");
    std::string generation_stops;
    if (o.mtp_generation) for (auto id:o.stop_ids) {
        if (!generation_stops.empty()) generation_stops+=',';
        generation_stops+=std::to_string(id);
    }
    if (::setenv("STRATA_HETERO_MTP_GENERATE_MAX_NEW",std::to_string(o.mtp_generation?o.generate:0u).c_str(),1)!=0 ||
        ::setenv("STRATA_HETERO_MTP_GENERATE_STOPS",generation_stops.c_str(),1)!=0 ||
        ::setenv("STRATA_HETERO_MTP_RESIDENT_GENERATION",o.mtp_resident_prompt_rows?"1":"0",1)!=0)
        throw std::runtime_error("cannot configure explicit MTP generation mode");
    if (!o.mtp_probe_rt.empty()) {
        if (::setenv("STRATA_HETERO_MTP_PROBE_RT",o.mtp_probe_rt.c_str(),1)!=0 ||
            ::setenv("STRATA_HETERO_MTP_PROBE_WINDOWS",std::to_string(o.mtp_resident_prompt_rows?o.mtp_resident_prompt_rows:o.mtp_generation?o.tokens.size():o.windows).c_str(),1)!=0 ||
            ::setenv("STRATA_HETERO_MTP_RELAY_ROUNDS",std::to_string(o.mtp_relay_rounds).c_str(),1)!=0)
            throw std::runtime_error("cannot configure one-shot MTP probe");
    } else if (::unsetenv("STRATA_HETERO_MTP_PROBE_RT")!=0 || ::unsetenv("STRATA_HETERO_MTP_PROBE_WINDOWS")!=0 ||
               ::unsetenv("STRATA_HETERO_MTP_RELAY_ROUNDS")!=0)
        throw std::runtime_error("cannot clear inherited MTP probe mode");
    if (::setenv("STRATA_HETERO_DIRECT_VERIFIED", o.direct_verified ? "1" : "0", 1) != 0)
        throw std::runtime_error("cannot configure explicit direct verified mode");
    if (::setenv("STRATA_HETERO_CUDA_MAPPED_EGRESS", o.cuda_mapped_egress ? "1" : "0", 1) != 0)
        throw std::runtime_error("cannot configure explicit mapped CUDA egress mode");
    if (::setenv("STRATA_HETERO_RECEIVER_ARM", o.receiver_arm ? "1" : "0", 1) != 0)
        throw std::runtime_error("cannot configure explicit receiver arm mode");
    const bool compact = hetero::performance_flag("STRATA_HETERO_COMPACT_RESULT");
    const bool fanout = hetero::performance_flag("STRATA_HETERO_FANOUT_CONTROL");
    (void)hetero::performance_flag("STRATA_HETERO_FAST_FENCE");
    (void)hetero::performance_flag("STRATA_HETERO_PERF_TRACE");
    if (o.warmup_windows && (hetero::performance_trace() || hetero::performance_flag("STRATA_HETERO_OVERLAP_TRACE") ||
        hetero::performance_flag("STRATA_HETERO_WINDOW_TRACE") || std::getenv("STRATA_VERIFY_TRACE")))
        throw std::invalid_argument("timed smoke requires tracing disabled (unset STRATA_VERIFY_TRACE)");
    if (compact && (!o.generate || o.prefill_parity_at || o.generation_capture || o.receiver_arm || o.direct_verified || !o.mtp_probe_rt.empty()))
        throw std::invalid_argument("compact head results are not supported in diagnostic modes");
    // Separate readiness-barrier wait from worker I/O and GPU fence deadlines.
    if (::setenv("STRATA_HETERO_STARTUP_WAIT_MS", std::to_string(o.timeout_ms).c_str(), 1) != 0)
        throw std::runtime_error("cannot configure worker startup wait");
    auto workers = hetero::WorkerProcessGroup::launch(
        plan, o.cuda_worker, o.hip_worker, o.shared, {}, o.configs, o.group_hip, o.timeout_ms, o.kv);
    // For this diagnostic only, drain by EOF even on post-launch exceptions;
    // if a worker cannot exit, report its PID rather than escalating to SIGKILL.
    struct DiagnosticDrain {
        hetero::WorkerProcessGroup& group;
        bool enabled;
        int timeout;
        ~DiagnosticDrain() noexcept {
            if (!enabled || !group.size()) return;
            try {
                std::string error;
                if (!group.finish(error,timeout,false)) std::cerr << "MTP diagnostic drain: " << error << '\n';
            } catch (...) {
                std::cerr << "MTP diagnostic drain could not finish\n";
            }
        }
    } diagnostic_drain{workers,true,o.timeout_ms};
    hetero::FinalPipelineSession session(plan, workers.sockets(), o.timeout_ms, o.shared, o.resident || o.prefill_parity_at,
                                         fanout, compact, o.group_hip, o.receiver_arm, o.mtp_relay_rounds!=0 || o.mtp_generation,o.mtp_generation,o.mtp_resident_prompt_rows!=0,o.direct_verified);
    session.initialize(); // Native children initialize their own CUDA/HIP runtimes.
    std::cout << "LIVE workers ready, mode=" << (o.shared ? "shared" : "socket")
              << " processes=" << workers.size() << " grouped_hip=" << o.group_hip
              << " hip_split_overlap=" << o.hip_split_overlap << " receiver_arm=" << o.receiver_arm
              << " cuda_mapped_egress=" << o.cuda_mapped_egress
              << " direct_verified_shared=" << o.direct_verified
              << " kv=" << hetero::kv_mode_name(o.kv) << '\n';
    // Both smoke and generation use the exact same committed real-worker path.
    uint64_t request_id = 1;
    auto initial_frame = [&](const hetero::WindowSpec& spec, uint64_t id) {
        hetero::Frame initial;
        initial.kind = hetero::Kind::data;
        initial.request_id = request_id;
        initial.window_id = id;
        initial.plan_fingerprint = plan.fingerprint();
        initial.rows = static_cast<uint32_t>(spec.token_ids.size());
        initial.geometry = plan.geometry;
        // Layer zero embeds the real tokens; only this ignored placeholder is zero.
        initial.payload.assign(static_cast<size_t>(plan.geometry.floats_per_row()) *
                               initial.rows, 0.0f);
        return initial;
    };
    auto run_window = [&](const hetero::WindowSpec& spec, uint64_t id, uint32_t keep) {
        session.execute_discard(initial_frame(spec, id), spec);
        auto committed = session.commit(keep);
        if (committed.predicted_tokens.size() != spec.token_ids.size() || committed.has_nonfinite())
            throw std::runtime_error("committed head predictions malformed or nonfinite");
        return committed;
    };
    if (o.prefill_parity_at) {
        // One physical topology, identical fixed inputs, explicit fresh reset before each pass.
        // Never feed a prediction back or stop on a generated token in this diagnostic.
        const size_t cut = o.prefill_parity_at;
        const std::vector<int32_t> prefix(o.tokens.begin(), o.tokens.begin() + cut);
        const char* modes[] = {"serial-a", "native", "serial-b"};
        for (size_t pass = 0; pass < 3; ++pass) {
            request_id = pass + 1;
            session.begin_request(request_id);
            uint64_t window = 0;
            if (pass == 1) {
                session.native_prefill(prefix, o.native_prefill_chunk, !o.prefill_stage_reference.empty());
                window = 1;
            } else {
                for (size_t pos = 0; pos < cut; ++pos) {
                    hetero::WindowSpec spec{static_cast<int64_t>(pos), {o.tokens[pos]}};
                    (void)run_window(spec, ++window, 1);
                }
            }
            Options capture = o;
            capture.capture_prefix += std::string(".") + modes[pass];
            for (size_t pos = cut; pos < o.tokens.size(); ++pos) {
                hetero::WindowSpec spec{static_cast<int64_t>(pos), {o.tokens[pos]}};
                const auto committed = run_window(spec, ++window, 1);
                capture_window(capture, plan, spec, committed);
            }
            std::cout << "PREFILL_PARITY_CAPTURE mode=" << modes[pass]
                      << " prefix=" << cut << " rows=" << o.tokens.size() - cut << std::endl;
        }
        std::string error;
        if (!workers.finish(error,o.timeout_ms,false)) throw std::runtime_error("parity worker shutdown failed: " + error);
        // Completion is evidence of capture, not numerical parity. Publish only after all ACKs and cleanup.
        std::ofstream complete(o.capture_prefix + ".prefill.json", std::ios::binary);
        if (!complete) throw std::runtime_error("cannot create prefill parity completion marker");
        complete << "{\"schema\":\"strata-heterogeneous-prefill-parity/v1\""
                 << ",\"plan\":" << json_string(hetero::plan_spec(plan))
                 << ",\"model_fingerprint\":" << o.fingerprint
                 << ",\"prefix_rows\":" << cut << ",\"chunk\":" << o.native_prefill_chunk
                 << ",\"state_capture\":" << (o.prefill_state_capture ? "true" : "false")
                 << ",\"first_token_capture\":" << (o.prefill_first_token ? "true" : "false")
                 << ",\"hc_replay_capture\":" << (o.prefill_hc_replay ? "true" : "false");
        if (const char* weight = std::getenv("STRATA_HETERO_HC_WEIGHT_CONTRACT"); weight && *weight)
            complete << ",\"hc_weight_contract\":" << json_string(weight);
        if (o.numerical_trace) complete << ",\"numerical_trace\":true";
        const auto hc_exp = strata::prefill::hc_experiment_config();
        if (hc_exp.requested())
            complete << ",\"hc_experiment\":{\"mode\":"
                     << json_string(strata::prefill::hc_experiment_name(hc_exp.mode))
                     << ",\"backend\":"
                     << json_string(strata::prefill::hc_experiment_name(hc_exp.backend)) << '}';
        if (o.prefill_stage_capture)
            complete << ",\"stage_capture\":true"
                     << ",\"stage_reference\":" << json_string(o.prefill_stage_reference)
                     << ",\"transport\":" << json_string(o.shared?"shared":"socket");
        complete << ",\"hc\":" << o.hc << ",\"n_embd\":" << o.n_embd << ",\"token_ids\":[";
        for (size_t pos = 0; pos < o.tokens.size(); ++pos) {
            if (pos) complete << ',';
            complete << o.tokens[pos];
        }
        complete << "]}\n";
        complete.flush();
        if (!complete) throw std::runtime_error("prefill parity completion write failed");
        return 0;
    }
    // Emitted only after successful all-stage commit and result validation.
    auto trace_committed = [&](const char* phase, const hetero::WindowSpec& spec,
                               uint64_t window, uint32_t keep) {
        if (!hetero::performance_trace()) return;
        std::fprintf(stderr, "HETERO_PERF_COMMITTED request=%llu window=%llu phase=%s rows=%zu keep=%u pos0=%lld\n",
                     static_cast<unsigned long long>(request_id),
                     static_cast<unsigned long long>(window), phase, spec.token_ids.size(),
                     keep, static_cast<long long>(spec.pos0));
    };
    std::vector<int32_t> generated, final_mtp_offer;
    uint64_t generation_windows=0;
    int32_t generation_next=-1;
    int64_t generation_fed=0;
    bool stopped = false, cancelled = false, input_closed = false;
    int32_t stop_token = -1;
    auto generate = [&](const std::vector<int32_t>& prompt, uint32_t max_new) {
        generated.clear();
        final_mtp_offer.clear();
        generation_windows=0; generation_next=-1; generation_fed=0;
        stopped = cancelled = false;
        stop_token = -1;
        uint64_t window = 0;
        int64_t pos = 0;
        int32_t next = -1;
        const auto mtp_policy=o.mtp_generation ?
            hetero::MtpGenerationPolicy{static_cast<unsigned>(prompt.size()),max_new,o.stop_ids} : hetero::MtpGenerationPolicy{};
        auto capture_generation=[&](const hetero::WindowSpec& spec,const hetero::FinalStageResult& committed,uint32_t keep) {
            if (!o.generation_capture) return;
            auto capture=o; capture.keep=keep;
            if (o.mtp_resident_prompt_rows) capture.capture_prefix+=".r"+std::to_string(request_id);
            capture_window(capture,plan,spec,committed);
            generation_windows=window;
            generation_next=committed.predicted_tokens.at(keep-1);
            generation_fed=spec.pos0+keep;
        };
        // The coordinator only yields control after all stages acknowledged
        // commit. EOF/cancel here cannot leave a provisional GPU window.
        auto checkpoint = [&]() {
            if (!o.resident) return true;
            std::cout << "HETERO_PROGRESS " << request_id << ' ' << window
                      << ' ' << pos;
            if (o.stream_tokens) {
                // Only committed predictions, never intermediate prompt rows or stop IDs.
                const bool available = pos >= static_cast<int64_t>(prompt.size()) &&
                    std::find(o.stop_ids.begin(), o.stop_ids.end(), next) == o.stop_ids.end();
                std::cout << ' ' << (available ? next : -1);
            }
            std::cout << std::endl;
            std::string command;
            if (!read_command(command)) { input_closed = true; cancelled = true; return false; }
            if (command == "cancel") { cancelled = true; return false; }
            if (command != "continue") throw std::invalid_argument("expected continue or cancel");
            return true;
        };
        // Vanilla: bulk-prefill all but final prompt token, then verifier/head.
        size_t prompt_offset=0;
        if (o.native_prefill_chunk && prompt.size()>1) {
            std::vector<int32_t> prefix(prompt.begin(),prompt.end()-1);
            const bool trace_prefix=hetero::performance_trace();
            using PrefixClock=std::chrono::steady_clock;
            const auto prefix_start=trace_prefix ? PrefixClock::now() : PrefixClock::time_point{};
            session.native_prefill(prefix,o.native_prefill_chunk);
            if (trace_prefix) {
                const auto prefix_end=PrefixClock::now();
                const auto start_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(prefix_start.time_since_epoch()).count();
                const auto end_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(prefix_end.time_since_epoch()).count();
                std::fprintf(stderr,"HETERO_NATIVE_PREFILL_TIMING request=%llu rows=%zu start_ns=%lld end_ns=%lld\n",
                             static_cast<unsigned long long>(request_id),prefix.size(),
                             static_cast<long long>(start_ns),static_cast<long long>(end_ns));
            }
            window=1; pos=static_cast<int64_t>(prefix.size()); prompt_offset=prefix.size();
            std::fprintf(stderr,"HETERO_NATIVE_PREFILL_COMMITTED request=%llu rows=%zu chunk=%u\n",
                         static_cast<unsigned long long>(request_id),prefix.size(),o.native_prefill_chunk);
            if (!checkpoint()) return;
        }
        for (size_t offset = prompt_offset; offset < prompt.size();) {
            const size_t count = std::min<size_t>(o.prompt_chunk, prompt.size() - offset);
            hetero::WindowSpec spec;
            spec.pos0 = pos;
            spec.token_ids.assign(prompt.begin() + offset, prompt.begin() + offset + count);
            const auto committed=run_window(spec, ++window, static_cast<uint32_t>(count));
            next=committed.predicted_tokens.back();
            capture_generation(spec,committed,static_cast<uint32_t>(count));
            trace_committed("prefill", spec, window, static_cast<uint32_t>(count));
            offset += count;
            pos += static_cast<int64_t>(count);
            if (!checkpoint()) return;
        }
        // Reuse Strata's existing request-local suffix lookup. The history
        // contains only committed inputs. The already-known next token is
        // temporarily appended for lookup, but never accepted as a draft.
        std::unique_ptr<strata::spec::SuffixDrafter> lookup;
        if (o.lookup_draft) {
            lookup = std::make_unique<strata::spec::SuffixDrafter>(3, 32, o.context_limit);
            lookup->append(prompt.data(), prompt.size());
        }
        uint64_t proposed = 0, accepted = 0, lookup_windows = 0;
        auto is_stop = [&](int32_t token) {
            return std::find(o.stop_ids.begin(), o.stop_ids.end(), token) != o.stop_ids.end();
        };
        auto report_lookup = [&] {
            if (lookup)
                std::cerr << "HETERO_LOOKUP request=" << request_id
                          << " proposed=" << proposed << " accepted=" << accepted
                          << " multirow_windows=" << lookup_windows << '\n';
        };
        while (generated.size() < max_new) {
            if (is_stop(next)) {
                stopped = true;
                stop_token = next;
                break;
            }
            generated.push_back(next);
            if (generated.size() == max_new) break;
            hetero::WindowSpec spec;
            spec.pos0 = pos;
            spec.token_ids = {next};
            if (o.mtp_generation) {
                spec=session.request_mtp_proposal();
                if (spec.pos0!=pos || spec.token_ids.empty() || spec.token_ids.front()!=next ||
                    !mtp_policy.live_window(pos,spec.token_ids))
                    throw std::runtime_error("native MTP offer violates stop/length/anchor policy");
            } else if (lookup) {
                // Leave room for the final un-fed output token and never
                // evaluate a proposed stop as input. Same bounds as prefill.
                const int cap = static_cast<int>(std::min<size_t>(
                    o.prompt_chunk, max_new - generated.size())) - 1;
                int32_t drafts[8]{};
                const int count = lookup->propose_after(&next, 1, cap, drafts);
                if (count < 0 || count > cap)
                    throw std::runtime_error("suffix drafter exceeded bounded window");
                for (int i = 0; i < count && !is_stop(drafts[i]); ++i)
                    spec.token_ids.push_back(drafts[i]);
            }
            uint32_t keep = 1;
            if (spec.token_ids.size() > 1 || o.mtp_generation) {
                if (lookup) ++lookup_windows;
                proposed += spec.token_ids.size() - 1;
                const auto provisional = session.execute(initial_frame(spec, ++window), spec);
                if (provisional.has_nonfinite())
                    throw std::runtime_error("nonfinite speculative window");
                keep = hetero::greedy_accepted_prefix(spec.token_ids, provisional.predicted_tokens);
                const auto committed = session.commit(keep);
                if (committed.predicted_tokens.size() != spec.token_ids.size() || committed.has_nonfinite())
                    throw std::runtime_error("invalid committed speculative window");
                next = committed.predicted_tokens[keep - 1];
                // Publish accepted draft rows only AFTER all stages ACK.
                generated.insert(generated.end(), spec.token_ids.begin() + 1,
                                 spec.token_ids.begin() + keep);
                accepted += keep - 1;
                capture_generation(spec,committed,keep);
                if (o.mtp_generation) std::cout << "HETERO_MTP_GENERATION_COMMITTED window=" << window
                    << " pos0=" << pos << " proposed=" << spec.token_ids.size()-1 << " accepted=" << keep-1
                    << " next=" << next << '\n';
            } else {
                const auto committed=run_window(spec, ++window, 1);
                next=committed.predicted_tokens.back();
                capture_generation(spec,committed,1);
            }
            trace_committed("decode", spec, window, keep);
            if (lookup) lookup->append(spec.token_ids.data(), keep);
            pos += keep;
            if (!checkpoint()) { generated.clear(); report_lookup(); return; }
        }
        if (o.generation_capture) generation_next=next;
        if (o.mtp_generation) {
            // After the terminal committed tail, request one diagnostic draft,
            // never submit it to target. Applies even to immediate EOS/length.
            if (!mtp_policy.can_draft(pos) || mtp_policy.live(pos,next))
                throw std::runtime_error("MTP generation terminated outside stop/length policy");
            const auto offer=session.request_mtp_proposal();
            if (offer.pos0!=pos || offer.token_ids.size()<2 || offer.token_ids.front()!=next)
                throw std::runtime_error("invalid terminal MTP diagnostic offer");
            final_mtp_offer=offer.token_ids;
            std::cerr << "HETERO_MTP_GENERATION_SUMMARY proposed=" << proposed << " accepted=" << accepted << '\n';
        }
        report_lookup();
    };
    auto publish = [&](size_t prompt_size) {
        std::ostringstream out;
        out << "HETERO_GENERATION {\"prompt_tokens\":" << prompt_size;
        if (o.resident) out << ",\"request_id\":" << request_id << ",\"stop_token\":" << stop_token;
        out << ",\"finish_reason\":\"" << (cancelled ? "cancelled" : stopped ? "stop" : "length")
            << "\",\"token_ids\":[";
        for (size_t i = 0; i < generated.size(); ++i) {
            if (i) out << ',';
            out << generated[i];
        }
        out << "]}\n";
        std::cout << out.str() << std::flush;
    };
    // Request captures are complete after ACKs; resident qualification also
    // requires the separate session marker after clean worker shutdown.
    auto complete_generation=[&](const std::vector<int32_t>& prompt,uint32_t max_new) {
        if (!o.generation_capture) return;
        const auto prefix=o.mtp_resident_prompt_rows?o.capture_prefix+".r"+std::to_string(request_id):o.capture_prefix;
        std::ofstream complete(prefix+".complete.json",std::ios::binary);
        if (!complete) throw std::runtime_error("cannot create generation capture completion");
        auto ids=[&](const std::vector<int32_t>& values) {
            complete << '[';
            for (size_t i=0;i<values.size();++i) { if (i) complete << ','; complete << values[i]; }
            complete << ']';
        };
        complete << "{\"schema\":\"strata-heterogeneous-parity/v1\",\"generation_capture\":true"
                 << ",\"mtp_generation\":" << (o.mtp_generation?"true":"false")
                 << ",\"mtp_resident\":" << (o.mtp_resident_prompt_rows?"true":"false")
                 << ",\"request_id\":" << request_id << ",\"fed_rows\":" << generation_fed
                 << ",\"windows\":" << generation_windows
                 << ",\"plan\":" << json_string(hetero::plan_spec(plan))
                 << ",\"model_fingerprint\":" << o.fingerprint
                 << ",\"max_new\":" << max_new << ",\"prompt_ids\":";
        ids(prompt);
        complete << ",\"stop_ids\":"; ids(o.stop_ids);
        complete << ",\"token_ids\":"; ids(generated);
        complete << ",\"finish_reason\":\"" << (cancelled?"cancelled":stopped?"stop":"length")
                 << "\",\"stop_token\":" << stop_token << ",\"next\":" << generation_next
                 << ",\"final_offer\":"; ids(final_mtp_offer);
        complete << "}\n";
        complete.flush();
        if (!complete) throw std::runtime_error("generation capture completion write failed");
    };
    uint64_t completed_requests=0;
    bool resident_quit=false;
    std::vector<int64_t> measured_ns;
    if (o.resident) {
        // Negotiate actual reset support before announcing readiness.
        session.begin_request(request_id);
        std::cout << "HETERO_READY " << request_id;
        if (o.stream_tokens) std::cout << " STREAM";
        if (o.lookup_draft) std::cout << " LOOKUP";
        std::cout << std::endl;
        bool first = true;
        std::string line;
        const size_t command_cap = o.workload_mode && o.context_limit > 8192 ? 384 * 1024 : 65536;
        while (!input_closed && read_command(line, command_cap)) {
            if (line == "quit") { resident_quit=true; break; }
            std::istringstream in(line);
            std::string count;
            in >> count;
            const uint32_t max_new = u32(count, "resident max-new-tokens");
            const auto prompt = prompt_ids(in, o.context_limit);
            if (!max_new || max_new > o.generate ||
                uint64_t(prompt.size()) + max_new > limits.max_cells ||
                uint64_t(prompt.size()) + max_new > o.context_limit ||
                std::any_of(prompt.begin(), prompt.end(), [&](int32_t id) {
                    return static_cast<uint32_t>(id) >= limits.vocab;
                }))
                throw std::invalid_argument("resident request exceeds validated model limits");
            if (o.mtp_resident_prompt_rows &&
                (!hetero::mtp_resident_shape({o.mtp_resident_prompt_rows,o.generate,o.stop_ids},prompt.size(),max_new) ||
                 !hetero::mtp_resident_next_request(completed_requests,completed_requests+1)))
                throw std::invalid_argument("resident MTP request exceeds fixed prompt/budget or session request cap");
            if (!first) {
                if (request_id == UINT64_MAX) throw std::runtime_error("resident request IDs exhausted");
                session.begin_request(++request_id);
            }
            first = false;
            if (o.mtp_resident_prompt_rows)
                std::cout << "HETERO_MTP_RESIDENT_REQUEST request=" << request_id << std::endl;
            generate(prompt, max_new);
            if (!input_closed) {
                if (o.mtp_resident_prompt_rows) {
                    complete_generation(prompt,max_new);
                    ++completed_requests;
                }
                publish(prompt.size());
            }
        }
    } else if (o.generate) {
        generate(o.tokens, o.generate);
    } else {
        for (uint32_t n = 0; n < o.windows; ++n) {
            hetero::WindowSpec spec;
            spec.pos0 = static_cast<int64_t>(uint64_t(n) * o.keep);
            if (o.tokens.empty()) spec.token_ids.assign(o.rows, static_cast<int32_t>(o.token));
            else spec.token_ids.assign(o.tokens.begin() + static_cast<size_t>(n) * o.rows,
                                       o.tokens.begin() + static_cast<size_t>(n + 1) * o.rows);
            const auto start = std::chrono::steady_clock::now();
            const auto committed = run_window(spec, uint64_t(n) + 1, o.keep);
            if (o.warmup_windows && n >= o.warmup_windows)
                measured_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count());
            if (!o.capture_prefix.empty()) capture_window(o, plan, spec, committed);
            std::cout << "LIVE PASS window=" << n + 1 << " pos0=" << spec.pos0
                      << " committed=" << o.keep << " predictions=";
            // STFR records every evaluated row; log only the accepted prefix.
            for (uint32_t t = 0; t < o.keep; ++t)
                std::cout << committed.predicted_tokens[t] << ',';
            std::cout << '\n';
        }
        for(uint32_t round=0;round<o.mtp_relay_rounds;++round) {
            const auto spec=session.request_mtp_proposal();
            const uint64_t id=uint64_t(o.windows)+round+1;
            if (o.mtp_three_round && !hetero::mtp_three_window_fits(spec.pos0,spec.token_ids.size(),round,o.windows)) {
                std::cerr << "MTP_THREE_BOUND_REFUSED phase=offer round=" << round+1 << " pos0=" << spec.pos0 << '\n';
                std::string drain;
                if (!workers.finish(drain,o.timeout_ms,false)) std::cerr << "MTP diagnostic drain: " << drain << '\n';
                return 1;
            }
            const auto provisional=session.execute(initial_frame(spec,id),spec);
            if(provisional.has_nonfinite()) throw std::runtime_error("nonfinite MTP verification");
            const uint32_t keep=hetero::greedy_accepted_prefix(spec.token_ids,provisional.predicted_tokens);
            if (o.mtp_three_round && !hetero::mtp_three_commit_fits(spec.pos0,keep,round,o.windows)) {
                // No commit frame has been sent. Workers are waiting for commit;
                // EOF abandons the provisional window and drains the session.
                std::cerr << "MTP_THREE_BOUND_REFUSED phase=precommit round=" << round+1
                          << " pos0=" << spec.pos0 << " greedy_keep=" << keep << '\n';
                std::string drain;
                if (!workers.finish(drain,o.timeout_ms,false)) std::cerr << "MTP diagnostic drain: " << drain << '\n';
                return 1;
            }
            const auto committed=session.commit(keep);
            auto capture=o; capture.keep=keep;
            capture_window(capture,plan,spec,committed);
            std::cout << "HETERO_MTP_RELAY_COMMITTED window=" << id << " pos0=" << spec.pos0
                      << " proposed=" << spec.token_ids.size()-1 << " accepted=" << keep-1
                      << " next=" << committed.predicted_tokens.at(keep-1) << '\n';
        }
    }
    std::string error;
    if (!((o.mtp_three_round || o.generation_capture || o.direct_verified) ? workers.finish(error,o.timeout_ms,false) : workers.finish(error)))
        throw std::runtime_error("worker shutdown failed: " + error);
    if (o.resident) {
        if (o.mtp_resident_prompt_rows) {
            if (!resident_quit || !completed_requests || input_closed)
                throw std::runtime_error("resident MTP qualification requires explicit quit after completed requests");
            std::ofstream marker(o.capture_prefix+".resident.json",std::ios::binary);
            if (!marker) throw std::runtime_error("cannot create resident MTP session completion");
            marker << "{\"schema\":\"strata-mtp-resident/v1\",\"requests\":" << completed_requests
                   << ",\"prompt_rows\":" << o.mtp_resident_prompt_rows << ",\"max_new\":" << o.generate
                   << ",\"transport\":" << json_string(o.shared?"shared":"socket")
                   << ",\"traced\":" << (o.mtp_state_trace?"true":"false")
                   << ",\"plan\":" << json_string(hetero::plan_spec(plan))
                   << ",\"model_fingerprint\":" << o.fingerprint << "}\n";
            marker.flush();
            if (!marker) throw std::runtime_error("resident MTP session completion write failed");
        }
        std::cout << "HETERO_CLOSED\n" << std::flush;
        return 0;
    }
    if (o.generate) {
        complete_generation(o.tokens,o.generate);
        publish(o.tokens.size());
        return 0;
    }
    if (!o.capture_prefix.empty()) {
        std::ofstream complete(o.capture_prefix + ".complete.json", std::ios::binary);
        if (!complete) throw std::runtime_error("cannot create parity completion marker");
        complete << "{\"schema\":\"strata-heterogeneous-parity/v1\",\"windows\":"
                 << o.windows+o.mtp_relay_rounds << ",\"placement_capture\":" << (o.placement_capture ? "true" : "false")
                 << ",\"mtp_fresh_replay\":" << (o.mtp_fresh_replay ? "true" : "false")
                 << ",\"mtp_three_round\":" << (o.mtp_three_round ? "true" : "false")
                 << ",\"mtp_three_round_variable\":" << (o.mtp_three_round_variable ? "true" : "false")
                 << ",\"mtp_prompt_windows\":" << (o.mtp_three_round ? o.windows : 0u)
                 << ",\"model_fingerprint\":" << o.fingerprint
                 << ",\"plan\":" << json_string(hetero::plan_spec(plan));
        if (const char* weight = std::getenv("STRATA_HETERO_HC_WEIGHT_CONTRACT"); weight && *weight)
            complete << ",\"hc_weight_contract\":" << json_string(weight);
        complete << "}\n";
        complete.flush();
        if (!complete) throw std::runtime_error("parity completion marker write failed");
    }
    if (o.warmup_windows) {
        // Report only after all commits and successful worker shutdown.
        // Kept teacher-forced rows are NOT accepted generated output tokens.
        std::cout << "HETERO_FIXED_WINDOW_TIMING {\"schema\":\"strata-fixed-window-timing/v1\","
                  << "\"plan_fingerprint\":" << plan.fingerprint()
                  << ",\"model_fingerprint\":" << plan.model_fingerprint
                  << ",\"windows\":" << o.windows
                  << ",\"warmup_windows\":" << o.warmup_windows
                  << ",\"rows\":" << o.rows << ",\"keep\":" << o.keep
                  << ",\"shared\":" << (o.shared ? "true" : "false")
                  << ",\"hip_overlap\":" << (o.hip_split_overlap ? "true" : "false")
                  << ",\"receiver_arm\":" << (o.receiver_arm ? "true" : "false")
                  << ",\"cuda_mapped_egress\":" << (o.cuda_mapped_egress ? "true" : "false")
                  << ",\"direct_verified_shared\":" << (o.direct_verified ? "true" : "false")
                  << ",\"token_ids\":[";
        for (size_t i=0;i<o.tokens.size();++i) {
            if (i) std::cout << ',';
            std::cout << o.tokens[i];
        }
        std::cout << "],\"elapsed_ns\":[";
        for (size_t i=0;i<measured_ns.size();++i) {
            if (i) std::cout << ',';
            std::cout << measured_ns[i];
        }
        std::cout << "]}\n";
    }
    std::cout << "LIVE PASS: all " << o.windows+o.mtp_relay_rounds
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
        const auto limits = preflight(opts, plan); // always CPU-only; no runtime in coordinator.
        if (!opts.run) {
            std::cout << "PASS: CPU-only native stage preflight. "
                         "No workers or GPU runtimes started; add --run for live test.\n";
            return 0;
        }
        return execute(opts, plan, limits);
    } catch (const std::exception& ex) {
        std::cerr << "heterogeneous live smoke: " << ex.what() << '\n';
        return 1;
    }
}
