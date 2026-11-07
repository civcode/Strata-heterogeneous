#pragma once
#include <cuda_runtime.h>
#include <thread>

namespace strata::core {
// A query error is neither completion nor another pending poll. Caller must reject
// the operation and preserve its owners (Q36 uses its existing process-fatal CK).
inline cudaError_t wait_event_completion(cudaEvent_t event) {
    cudaError_t status;
    while ((status = cudaEventQuery(event)) == cudaErrorNotReady) std::this_thread::yield();
    return status;
}
} // namespace strata::core
