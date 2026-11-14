#include "strata/prefill/moe_fused.hpp"
#include <cstddef>

using namespace strata::prefill::fused;
static_assert(kBatchRoom == kMaxBatch, "old CUDA toolchains require bounded kernel arguments");
static_assert(sizeof(WideBatch) < 4096, "leave room for the remaining kernel arguments");
int main() { return 0; }
