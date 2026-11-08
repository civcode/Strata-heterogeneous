#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 999 };
cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaFree(void*);
const char* cudaGetErrorString(cudaError_t);
#ifdef __cplusplus
}
#endif
