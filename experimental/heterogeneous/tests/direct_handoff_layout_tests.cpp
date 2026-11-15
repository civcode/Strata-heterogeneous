#include "strata_heterogeneous/direct_handoff_layout.hpp"
#include <stdexcept>
#include <limits>
#include <set>
using namespace strata::heterogeneous;

void check(bool ok) { if (!ok) throw std::runtime_error("direct handoff layout check failed"); }

int main() {
    for (size_t rows = 1; rows <= 8; ++rows) {
        const auto l = DirectHandoffLayout::make(4, 2560, rows);
        check(l.r_words == 10240 * rows && l.bo_words == 2560 * rows && l.inj_words == 4 * rows);
        check(l.payload_bytes == 51216 * rows && l.payload_bytes <= l.ready_offset);
        check(l.ready_offset % 4096 == 0 && l.error_offset - l.ready_offset == 64);
        check(l.extent % 4096 == 0 && l.error_offset + sizeof(uint32_t) <= l.extent);
        check(l.words() * sizeof(float) == l.payload_bytes);
        std::set<float> pattern;
        for (size_t i = 0; i < l.words(); ++i) pattern.insert(direct_probe_value(65536, i));
        check(pattern.size() == l.words());
        for (size_t i : {size_t(0), l.r_words, l.r_words + l.bo_words, l.words()-1})
            check(direct_probe_value(1, i) != direct_probe_value(2, i));
    }
    const size_t huge = std::numeric_limits<size_t>::max();
    size_t rejected = 0;
    for (auto hc : {size_t(0), size_t(3), size_t(4), huge})
        for (auto n : {size_t(0), size_t(5), size_t(2560), huge})
            for (auto r : {size_t(0), size_t(1), size_t(9), huge}) {
                try {
                    (void)DirectHandoffLayout::make(hc, n, r);
                    check(hc == 4 && n == 2560 && r == 1);
                } catch (const std::invalid_argument&) { ++rejected; }
            }
    check(rejected == 63);
    try { (void)DirectHandoffLayout::make(4, 4*1024*1024, 8); check(false); }
    catch (const std::invalid_argument&) {}
}
