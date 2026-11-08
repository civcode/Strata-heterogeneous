#include "strata_heterogeneous/stage_weights.hpp"

#include <charconv>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::heterogeneous {
namespace {

template <typename T>
T decimal(const std::string& token, const char* field) {
    if (token.empty()) throw std::invalid_argument(std::string("empty ") + field);
    T value{};
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
        throw std::invalid_argument(std::string("invalid ") + field + ": " + token);
    return value;
}

uint64_t aligned(uint64_t bytes, uint64_t alignment) {
    if (!alignment || bytes > std::numeric_limits<uint64_t>::max() - (alignment - 1))
        throw std::overflow_error("weight arena alignment overflows uint64");
    const uint64_t n = (bytes + alignment - 1) / alignment;
    if (n > std::numeric_limits<uint64_t>::max() / alignment)
        throw std::overflow_error("weight arena size overflows uint64");
    return n * alignment;
}

bool layer_number(const std::string& name, uint32_t total_layers, uint32_t& layer) {
    if (name.compare(0, 4, "blk.") != 0) return false;
    const size_t dot = name.find('.', 4);
    if (dot == std::string::npos || dot == 4 || dot + 1 >= name.size())
        throw std::invalid_argument("malformed blk.<layer>.<tensor> name: " + name);
    const auto n = decimal<uint32_t>(name.substr(4, dot - 4), "block layer number");
    if (std::to_string(n) != name.substr(4, dot - 4))
        throw std::invalid_argument("noncanonical block layer number: " + name);
    if (n >= total_layers)
        throw std::invalid_argument("block tensor has layer outside the model: " + name);
    layer = n;
    return true;
}

} // namespace

bool plan_stage_weights(const Plan& plan, size_t stage_index,
                        const std::string& index_path, StageWeightPlan& out, std::string& error) {
    try {
        plan.validate();
        if (stage_index >= plan.stages.size()) throw std::invalid_argument("stage index out of range");
        if (index_path.empty()) throw std::invalid_argument("pack index path is empty");
        std::ifstream file(index_path);
        if (!file) throw std::runtime_error("cannot open pack index: " + index_path);
        StageWeightPlan work;
        const Stage& stage = plan.stages.at(stage_index);
        std::set<std::string> names;
        std::set<uint32_t> covered;
        uint64_t alignment = 0;
        uint64_t declared_tensors = 0;
        bool header = false;
        bool started = false;
        uint64_t previous_end = 0;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) throw std::invalid_argument("blank index row");
            // The real engine uses a fixed 1024-byte fgets buffer; reject a
            // line which could be truncated by that loader.
            if (line.size() >= 1023) throw std::invalid_argument("pack index row exceeds loader line limit");
            if (line.front() == '#') {
                if (line.rfind("# align ", 0) == 0) {
                    if (header || started) throw std::invalid_argument("duplicate or misplaced pack header");
                    std::istringstream in(line);
                    std::string hash, align_key, pool_key, tensors_key, a, p, t, extra;
                    if (!(in >> hash >> align_key >> a >> pool_key >> p >> tensors_key >> t) ||
                        (in >> extra) || hash != "#" || align_key != "align" ||
                        pool_key != "pool" || tensors_key != "tensors")
                        throw std::invalid_argument("invalid pack header");
                    alignment = decimal<uint64_t>(a, "pack alignment");
                    work.full_pool_bytes = decimal<uint64_t>(p, "pack pool bytes");
                    declared_tensors = decimal<uint64_t>(t, "pack tensor count");
                    if (!alignment || alignment > (1u << 20) || !work.full_pool_bytes || !declared_tensors)
                        throw std::invalid_argument("pack header contains invalid sizes");
                    header = true;
                }
                continue;
            }
            if (!header) throw std::invalid_argument("weight row precedes pack header");
            started = true;
            std::istringstream in(line);
            std::vector<std::string> fields;
            std::string token;
            while (in >> token) fields.push_back(token);
            if (fields.size() != 19 || fields[0].size() > 255)
                throw std::invalid_argument("pack row must have exactly 19 columns and a short name");
            if (!names.insert(fields[0]).second)
                throw std::invalid_argument("duplicate tensor name: " + fields[0]);
            // Use exactly the indexed byte sizes, never recompute tensor
            // planes or quantization geometry. Confirm the columns needed by
            // WeightTable's native 19-column parser are decimal and in range.
            const int file_id = decimal<int>(fields[1], "file id");
            const int kind = decimal<int>(fields[2], "tensor kind");
            const uint64_t src_off = decimal<uint64_t>(fields[3], "source offset");
            const uint64_t src_bytes = decimal<uint64_t>(fields[4], "source bytes");
            const uint64_t off = decimal<uint64_t>(fields[5], "destination offset");
            const uint64_t bytes = decimal<uint64_t>(fields[6], "destination bytes");
            const int64_t ne0 = decimal<int64_t>(fields[7], "ne0");
            const int64_t ne1 = decimal<int64_t>(fields[8], "ne1");
            for (int i = 9; i <= 13; ++i) (void)decimal<int>(fields[i], "quantization field");
            for (int i = 14; i <= 16; ++i) (void)decimal<uint64_t>(fields[i], "plane bytes");
            for (int i = 17; i <= 18; ++i) (void)decimal<int>(fields[i], "pack flag");
            if (file_id < 0 || file_id > 3 || kind < 0 || kind > 5 || ne0 <= 0 || ne1 < 0 ||
                src_bytes > std::numeric_limits<uint64_t>::max() - src_off ||
                off % alignment != 0 || off < previous_end ||
                off > work.full_pool_bytes || bytes > work.full_pool_bytes - off)
                throw std::invalid_argument("pack row has invalid file, kind, dimensions or pool extent");
            previous_end = off + bytes;
            ++work.total_tensors;
            uint32_t layer = 0;
            const bool block = layer_number(fields[0], plan.total_layers, layer);
            const bool belongs = !block || (layer >= stage.first_layer && layer < stage.end_layer);
            if (!belongs) {
                work.skip.insert(fields[0]);
                continue;
            }
            if (!bytes) throw std::invalid_argument("selected tensor has no loadable bytes: " + fields[0]);
            if (block) { covered.insert(layer); ++work.stage_layer_tensors; }
            const uint64_t padded = aligned(bytes, alignment);
            if (padded > std::numeric_limits<uint64_t>::max() - work.selected_pool_bytes)
                throw std::overflow_error("selected weight pool size overflow");
            work.selected_pool_bytes += padded;
            ++work.selected_tensors;
        }
        if (!file.eof() && file.fail()) throw std::runtime_error("pack index read failed");
        if (!header || !started || declared_tensors != work.total_tensors)
            throw std::invalid_argument("pack index missing rows or tensor count mismatch");
        if (covered.size() != static_cast<size_t>(stage.end_layer - stage.first_layer))
            throw std::invalid_argument("pack index missing a tensor for one or more owned layers");
        if (!work.selected_pool_bytes || work.selected_pool_bytes > work.full_pool_bytes)
            throw std::invalid_argument("selected pool exceeds pack header pool");
        out = std::move(work);
        error.clear();
        return true;
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }
}
} // namespace strata::heterogeneous
