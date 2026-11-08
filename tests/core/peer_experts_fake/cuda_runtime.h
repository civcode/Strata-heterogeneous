#pragma once
#include <cstdint>

using cudaStream_t = void*;
using cudaEvent_t = void*;
enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorNotReady = 34,
    cudaErrorUnknown = 999
};
namespace fake {
inline int current_device = 0;
inline int get_calls = 0, set_calls = 0, stream_calls = 0;
inline int device_sync_calls = 0, event_query_calls = 0, event_wait_calls = 0;
inline bool fail_get = false, fail_set = false, fail_stream = false;
inline bool fail_device_sync = false, fail_event_query = false;
inline bool fail_event_wait = false, event_not_ready = false;
inline void reset() {
    current_device = get_calls = set_calls = stream_calls = 0;
    device_sync_calls = event_query_calls = event_wait_calls = 0;
    fail_get = fail_set = fail_stream = fail_device_sync = false;
    fail_event_query = fail_event_wait = event_not_ready = false;
}
}
inline cudaError_t cudaGetDevice(int* device) {
    ++fake::get_calls;
    if (fake::fail_get) return cudaErrorUnknown;
    *device = fake::current_device;
    return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int device) {
    ++fake::set_calls;
    if (fake::fail_set) return cudaErrorUnknown;
    fake::current_device = device;
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
    ++fake::stream_calls;
    return fake::fail_stream ? cudaErrorUnknown : cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() {
    ++fake::device_sync_calls;
    return fake::fail_device_sync ? cudaErrorUnknown : cudaSuccess;
}
inline cudaError_t cudaEventQuery(cudaEvent_t) {
    ++fake::event_query_calls;
    return fake::fail_event_query ? cudaErrorUnknown
           : fake::event_not_ready ? cudaErrorNotReady : cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) {
    ++fake::event_wait_calls;
    return fake::fail_event_wait ? cudaErrorUnknown : cudaSuccess;
}
inline const char* cudaGetErrorString(cudaError_t error) {
    return error == cudaSuccess ? "success" : "injected error";
}
