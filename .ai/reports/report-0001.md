# Report 0001 — compile-only integration

Status: compile-only checks passed; live validation remains separately gated.
Base: `20f5f7681c7811495d94989e07dd99a044727a05`. The final commit is the commit containing this report (a commit cannot contain its own hash); use `git rev-parse HEAD` after checkout.

## Results

- Standalone Linux host suite: GCC 13.3 Debug **27/27**, ROCm Clang 24 Debug **27/27**, GCC AddressSanitizer **27/27**, GCC UndefinedBehaviorSanitizer **27/27**. Builds used `--parallel 28`. These include CPU doubles and subprocess/preflight tests, not GPU inference.
- CUDA 12.0, sm_89: native CUDA-only worker and engine compiled and linked. HIP, ROCm Clang 24, gfx1201: native HIP-only worker, engine and complete adapter archive compiled and linked. HIP configuration required `-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1` to find `hipConfig.cmake`.
- Re-linked each native worker with its own adapter archive under `--whole-archive`; both links passed. The engine and other dependencies were linked normally, not whole-archived. No linked worker or probe was executed.
- Reviewed HIP startup identity and file validation before the runtime gate, bounded window admission, and verifier → native → resource teardown, alongside the CPU contract and process tests. These checks do not prove live-device fault handling or numerical correctness.

## Corrections

- Added HIP-named verifier binding doubles for the HIP native-stage and expert contract fixtures; previously they accidentally included the real HIP adapter header alongside incompatible CUDA fixture headers, preventing the host suite from compiling.
- Check HIP event and buffer release return values in both verifier adapters; fail-stop on release errors. This also fixes the ROCm compiler's `nodiscard` errors without suppressing them.
- Corrected the HIP stage-resource header's backend-isolation comment.

## Limits and environment

No GPU runtime initialization, GPU worker execution, model access, GPU diagnostics, device reset, inference or parity test was authorized or performed. Both live approval variables remained unset. No GitHub Actions workflow was run. The original stash and untracked `.env` and `tools/heterogeneous_pool_probe.cpp` were preserved.

The permitted staging directory was mounted read-only in this session; an initial configure there failed with `Read-only file system`. Build trees were therefore placed in ignored `build-handoff-0001-*` directories within the permitted worktree; bulk logs were not saved. GPU runtime teardown and inference remain unverified and require separate authorization.
