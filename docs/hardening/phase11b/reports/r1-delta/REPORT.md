# Phase 11B R1 integration — 2026-11-07

## Integration

Started in `/home/chris/workspace/Strata-dev` (clean) and identified the authoritative
worktree at `/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore`.
Branch remains `fix/phase11b-prefill-restore-lifetime`; HEAD remains
`0ef75af1730c9374bbf1cb66a361f410d699b805`.

Before integration, exported the actual dirty worktree with the existing evidence
script. Its SHA-256 exactly matched the R2-validated patch:
`8418546962721a09a0966d6135cbf8ceb972d6e7bbd25f0e3f58171432c8bcc6`.
Saved that export as `pre-integration.patch`; did not reapply any earlier patch.

Input: `/home/chris/tmp/strata-dev/STRATA_PHASE11B_R1_INCREMENTAL.patch`.
Original SHA-256: `b6b38eb017d4d5ce3ac8d610606136be6ad20a86da4d566e2fb94b9bc4910b07`.
The initial applicability check failed only on CMakeLists.txt. Its first hunk was
marked `@@ -1,10 +1,20 @@`, so Git treated it as file-start context. The exact
context exists at line 434. Changed that hint to `@@ -434,10 +434,20 @@` in an
isolated copy; the source delta was unchanged. The corrected copy passed
`git apply --check` and was applied once. Original input preserved as
`INPUT_ORIGINAL.patch`; the local `STRATA_PHASE11B_R1_INCREMENTAL.patch` contains
the corrected hint. No production source correction to R1 was necessary.

## Newly executed results

| Check | Result |
|---|---|
| Nine new CPU completion cases + existing 63 regressions | 72/72 passed |
| Same group, ASan/UBSan and leak detection | 72/72 passed |
| HIP gfx1201 actual peer_experts.cpp + production executable | Built successfully |
| CUDA 12.0 sm_89 actual peer_experts.cpp + production executable | Built successfully |
| Real PeerExperts native cases, R9700 #0 primary / #1 helper | 20/20 passed |
| Same cases, physical roles reversed | 20/20 passed |
| R2 optimization + remote-owner native regressions | 22/22 passed |
| Native CUDA short probes | 0 passed, 3 skipped (exit 77) |

All HIP workloads ran serially, with SDMA off. The harness checked the kernel
journal after each process and found no MES/TLB/page-fault/reset matches in its
recorded windows. Native fault wrappers do not hang or fault a GPU: they submit
valid work, or fence it successfully, then report a synthetic API failure.

HIP compilation reports ignored-return warnings in unchanged optional P2P probing
and its old On scope. That code was intentionally not rewritten. Build logs retain
the warnings; this is a successful build, not a warning-free qualification.

## Native production-class fixture

Added `tests/core/peer_experts_gpu_test.cpp` and 20 `peer_native_*` CTest entries.
The test links the actual production engine and calls public APIs; it does not copy
the launch/finish/close logic or access private members.

It creates model-independent zero-weight native Q2_0 blobs and pinned caller
x/out/source storage. Two initially resident experts and two refill candidates
exercise:

- Direct mapped output and staged output; exact routed-row output checks.
- Close with queued direct scatter, plus idempotent close and entry-device restoration.
- Accepted-work/report-error on the input, plan and staged result copies.
- Successful issue drains before returning false, and failed issue/finish drains
  exiting the child with status 1 before any peer or borrowed owner release.
- Teardown device sync and device selection failure, also fail-before-release.
- Normal refill publication; second-copy refill issue errors; event record errors;
  source exception after one earlier refill was accepted.
- Not-ready events remaining unpublished, and failed event query/wait taking the
  process-terminal path rather than publishing residency.
- Partial initialization with a helper allocation refusal after host owners exist.

Wrappers track peer allocations, borrowed pinned x/out/blob owners, outstanding
submission state and completion checks. Recoverable errors require a real drain
before the method returns. Terminal scenarios require injection + ERR markers,
child exit 1 and no owner-release marker. The parent forks/execs before HIP init.

PeerExperts currently requires helper ordinal >= 1. Both directions therefore use
logical primary 0/helper 1, reversing physical order through ROCR_VISIBLE_DEVICES:

- Forward: `GPU-e39306ee46993aad,GPU-2f182f01261ba30b`.
- Reverse: `GPU-2f182f01261ba30b,GPU-e39306ee46993aad`.

The first native bring-up falsely classified temporary ExpertCache admission-probe
allocations as retained teardown owners (`release phase=open ... teardown=0`).
Corrected only the fixture: ordinary admission-probe releases leave the retained
owner map without asserting teardown, while injected partial-init cleanup still
requires a successful fence before release. Retained failed/diagnostic logs; all
40 selected final R1 cases passed after this fixture correction. No native failure
was attributed to production code, and no new kernel fault was observed.

Updated the in-repository handoff at
`docs/hardening/PHASE11B_R1_CHATGPT_HANDOFF.md` with actual results and limits.
No prior general audit or Phase 11A preservation pass was repeated.

## Actual command summary

All shell calls had explicit outer timeouts. Raw outputs are under `evidence/`.

```sh
W=/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore
R=/home/chris/workspace/Strata-phase11b/r1-delta
B=/home/chris/workspace/Strata-builds
cd "$W"
# Corrected CMake hunk line hint in the isolated copy, then:
git apply --check "$R/STRATA_PHASE11B_R1_INCREMENTAL.patch"
git apply "$R/STRATA_PHASE11B_R1_INCREMENTAL.patch"
git diff --check
cmake --build "$B/phase11b-hip" --target strata -j 4
cmake --build "$B/phase11b-cuda" --target strata -j 4
cmake --build "$B/phase11b-hip" --target peer_experts_gpu_test remote_opt_gpu_test remote_experts_gpu_test -j 4

# Separately for D=phase11b-cpu and D=phase11b-sanitizers:
cmake --build "$B/$D" --target peer_experts_completion_test remote_opt_completion_test transfer_failure_test conversation_lifetime_cuda_test conversation_lifetime_hip_test prefill_completion_test pinned_ring_test conv_cache_test conversation_restore_orchestration_test -j 4
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir "$B/$D" -R '^(peer_experts_completion_|remote_opt_completion_|transfer_|lifetime_|prefill_completion_test$|pinned_ring_test$|conv_cache_test$|conversation_restore_orchestration_test$)' --output-on-failure

python3 "$R/run_gpu_tests.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-r1-forward
python3 "$R/run_gpu_tests.py" hip --uuid GPU-2f182f01261ba30b,GPU-e39306ee46993aad --label hip-r1-reverse
python3 "$R/r2_regression.py" hip --uuid GPU-e39306ee46993aad,GPU-2f182f01261ba30b --label hip-r2-regression
nvidia-smi -L
lspci -nn -s 04:00.0
python3 "$R/cuda_probe.py" cuda --label cuda-r1 --small
python3 "$R/finalize_evidence.py"
python3 "$R/package_incremental.py"
```

Native harnesses reuse the existing bounded serial driver, with the command
selection changed for R1. Each process has a 60-second timeout plus kill grace,
with a kernel check before continuing. JSON results record commands/environment.
CPU final logs are `cpu-final-tests.log` and `sanitizers-final-tests.log`.
The focused bring-up success run and earlier CPU run overlap final coverage and
are not additional unique tests.

## Artifacts

- `review.patch`: updated full uncommitted worktree patch against HEAD, including
  untracked source. SHA-256:
  `156d6fa43cae8e5fdd81674ee172083953ecf25b81d9c2f5036ae9def431ec1d`.
- `R1_VALIDATED_INCREMENTAL.patch`: R1 plus the native fixture and handoff updates,
  relative to the exact R2-validated dirty state. SHA-256:
  `012b2ae16330766723398ccb038d882294d44dda0aea9bd3bb4d7f22aedf326e`.
- Both patches passed applicability checks against their respective bases using
  a separate temporary index. The authoritative index remains empty. Do not reapply
  either output patch to the already-updated implementation worktree.
- MANIFEST.sha256 records patch hashes. Source identities, production binary
  hashes and native fixture binary hashes are in `evidence/`.

## Limits and next work

NVIDIA remains PCI-visible but NVML cannot obtain a device handle and runtime
probes report no CUDA-capable device. CUDA build success is not runtime proof.
Native R1 CUDA, SDMA-on R1, real-model multi-GPU, P2P, launch-exception injection,
free/get/restore-device error injection, physically failed GPU, long contexts,
performance comparisons and SYCL/Metal qualification were not run. No fresh TSan
or Compute Sanitizer qualification is claimed. Zero-weight rows check plumbing
and ownership, not general numerical accuracy.

R3 expert-cache startup and the earlier prefill local/borrowed cleanup review
remain outside this delta. Advisor discovery returned no available advisor.
No commit, push, GitHub Actions, driver/reset/system change, model download or
separate Strata-heterogeneous project inspection. Strata-dev remains clean;
Phase 11B remains uncommitted.
