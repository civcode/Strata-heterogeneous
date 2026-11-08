#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 999 };
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaDeviceSynchronize(void);
const char* cudaGetErrorString(cudaError_t status);
cudaError_t cudaFree(void* ptr);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaEventDestroy(cudaEvent_t event);
#ifdef __cplusplus
}
#endif
