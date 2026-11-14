#pragma once

enum cudaError_t { cudaSuccess = 0, cudaErrorUnknown = 999 };
using cudaStream_t = void*;

cudaError_t cudaGetDevice(int*);
cudaError_t cudaSetDevice(int);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaDeviceSynchronize();
const char* cudaGetErrorString(cudaError_t);
