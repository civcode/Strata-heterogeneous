#pragma once
// CPU-only CUDA-shaped API for the remote optimization teardown error contract.
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
constexpr cudaError_t cudaErrorUnknown = 999;

cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaDeviceSynchronize();
const char* cudaGetErrorString(cudaError_t status);
