#pragma once

#include "strata/core/transfer_terminal.hpp"

#include <cuda_runtime.h>

namespace strata::core {

// A later layer-split stage owns streams, events, caches, and allocations on its
// own GPU. An early return may unwind after its caller's OnDevice scope has
// restored a different GPU. Keep the owner selected until *after* the stage's
// member destructors run. An OnDevice local in ~GpuStage is too short-lived:
// its destructor would restore the caller before those members are destroyed.
//
// 'armed' is set immediately after the device preflight succeeds, before the
// first stage allocation. A failed preflight must not initialize another GPU
// context merely to destroy an as-yet-empty stage.
template<class Stage>
struct LayerSplitStageDelete {
    void operator()(Stage* stage) const noexcept {
        if (!stage) return;
        if (!stage->cleanup_armed) {
            delete stage;
            return;
        }
        const int owner = stage->dev;
        if (owner < 0)
            transfer_terminal("layer split stage cleanup owner", "invalid device ordinal");
        int previous = -1;
        const cudaError_t queried = cudaGetDevice(&previous);
        if (queried != cudaSuccess)
            transfer_terminal("layer split stage cleanup get device", cudaGetErrorString(queried));
        const bool switched = previous != owner;
        if (switched) {
            const cudaError_t selected = cudaSetDevice(owner);
            if (selected != cudaSuccess)
                transfer_terminal("layer split stage cleanup select device", cudaGetErrorString(selected));
        }
        // Failed creation/copy APIs can have already submitted work. Do not
        // unwind the member allocations unless completion is established.
        const cudaError_t completed = cudaDeviceSynchronize();
        if (completed != cudaSuccess)
            transfer_terminal("layer split stage cleanup completion", cudaGetErrorString(completed));
        delete stage;  // stage and all of its GPU-owning subobjects on owner
        if (switched) {
            const cudaError_t restored = cudaSetDevice(previous);
            if (restored != cudaSuccess)
                transfer_terminal("layer split stage cleanup restore device", cudaGetErrorString(restored));
        }
    }
};

} // namespace strata::core
