#pragma once
#ifdef __cplusplus
extern "C" {
#endif
typedef int cudaError_t;
typedef void* cudaEvent_t;
enum { cudaSuccess = 0, cudaErrorUnknown = 999 };
cudaError_t cudaGetDevice(int* d);
cudaError_t cudaSetDevice(int d);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaFree(void* p);
cudaError_t cudaEventDestroy(cudaEvent_t event);
const char* cudaGetErrorString(cudaError_t e);
#ifdef __cplusplus
}
#endif
