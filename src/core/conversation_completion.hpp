#pragma once
#include <cuda_runtime.h>
#include <exception>
#include "strata/core/transfer_terminal.hpp"

namespace strata::core::conversation_detail {
// A reported issue/record error is not proof of rejected submission. On the error path,
// positively drain the stream before either operand's owner may unwind. Unknown completion
// must reach the process boundary HERE, not after stack-owned copy operands have been destroyed.
inline void drain_copy_or_exit(cudaStream_t stream) {
    const auto completed = cudaStreamSynchronize(stream);
    if (completed != cudaSuccess)
        core::transfer_terminal("conversation copy completion", cudaGetErrorString(completed));
}
inline cudaError_t finish_copy(cudaError_t submitted, cudaStream_t stream, bool queued = false) {
    if (!queued || submitted != cudaSuccess) drain_copy_or_exit(stream);
    return submitted; // draining does not turn the failed operation into a successful restore
}
// A void join has no channel for publishing an invalid async image. Never merely log the
// worker's error and let restore/eviction use it. Callers hold their existing join mutex.
template<class Future> inline void join_copy_or_exit(Future& pending, const char* context) {
    // An async worker may throw after queuing a transfer. A future exception is not
    // evidence of GPU completion, so never unwind the image or its transfer owners.
    try {
        const auto e = pending.get();
        if (e != cudaSuccess) core::transfer_terminal(context, cudaGetErrorString(e));
    } catch (const std::exception& ex) {
        core::transfer_terminal(context, ex.what());
    } catch (...) {
        core::transfer_terminal(context, "unknown asynchronous worker exception");
    }
}
} // namespace strata::core::conversation_detail
