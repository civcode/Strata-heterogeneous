#pragma once
// CPU-only, versioned and fail-closed startup configuration for a separate
// CUDA worker process. No device/runtime header may be included here.
#include "strata_heterogeneous/host.hpp"
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::heterogeneous {

struct CudaWorkerConfig {
    std::string pack_dir;
    std::string profile_path;
    std::string index_sha256; // lowercase hex SHA-256 of the exact pack index
    uint64_t model_fingerprint = 0; // caller-attested identity, not an index hash
    uint32_t stage_index = 0;
    uint32_t device_ordinal = 0;
    uint32_t first_layer = 0;
    uint32_t end_layer = 0;
    uint32_t n_layers = 0;
    uint32_t hc = 0;
    uint32_t n_embd = 0;
    uint32_t n_vocab = 0;
    uint32_t max_cells = 0;
    uint32_t max_t = 0;
    uint32_t cpu_workers = 0;
    uint32_t max_cache_slots = 0;
    uint32_t io_timeout_ms = 0;
    uint64_t free_reserve_bytes = 0;
    uint64_t arena_budget_bytes = 0;
};

// Config is a strict ASCII key=value file with one required occurrence of
// each v1 key (blank lines/# comments allowed). Unknown/duplicate keys,
// invalid numbers and unsafe/relative paths are refused. Output unchanged
// on error, and parsing performs NO I/O beyond reading the config itself.
bool read_cuda_worker_config(const std::string& filename,
                             CudaWorkerConfig& out, std::string& error);

// Check immutable-looking external inputs BEFORE CUDA initialization: indexed
// bytes match the pinned digest, expert profile exists and has STRP/v1 magic,
// and the requested worker identity equals its own coordinator plan.
// A digest of index.txt alone is not proof of dense/experts.bin contents.
bool validate_cuda_worker_config(const CudaWorkerConfig& config,
                                 const Plan& plan, size_t stage_index,
                                 std::string& error);
// Can run without a Plan (for --check-config); also verifies paths/digest.
bool validate_cuda_worker_files(const CudaWorkerConfig& config,
                                std::string& error);

} // namespace strata::heterogeneous
