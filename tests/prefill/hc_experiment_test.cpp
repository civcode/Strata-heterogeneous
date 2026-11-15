#include "strata/prefill/hc_experiment.hpp"
#include <cassert>
#include <initializer_list>
using namespace strata::prefill;
int main() {
    using M = HcExperimentMode;
    using B = HcExperimentBackend;
    assert(parse_hc_experiment(nullptr, nullptr).valid());
    assert(!parse_hc_experiment(nullptr, nullptr).requested());
    assert(parse_hc_experiment("off", "all").effective(B::cuda) == M::off);
    assert(!parse_hc_experiment("typo", "all").valid());
    assert(!parse_hc_experiment("fp32", "typo").valid());
    assert(!parse_hc_experiment("off", "cuda").valid());
    for (auto backend : {B::cuda, B::hip}) {
        for (auto mode : {M::bf16x2, M::fp32}) {
            const auto all = parse_hc_experiment(hc_experiment_name(mode), "all");
            assert(all.requested() && all.effective(backend) == mode);
            const auto one = parse_hc_experiment(hc_experiment_name(mode), hc_experiment_name(backend));
            assert(one.requested() && one.effective(backend) == mode);
            assert(one.effective(backend == B::cuda ? B::hip : B::cuda) == M::off);
            assert(one.effective(B::invalid) == M::off);
        }
    }
}
