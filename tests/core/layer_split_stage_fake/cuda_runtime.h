#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 999 };
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaDeviceSynchronize(void);
const char* cudaGetErrorString(cudaError_t status);

#ifdef __cplusplus
}
#endif
