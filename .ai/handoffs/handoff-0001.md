# Handoff 0001 — M2.0–M2.5 native CUDA/HIP integration and CPU validation

## Objective

Integrate and independently validate the complete *code-only* M2.0–M2.5
CUDA/HIP stage-worker increment on Oberon: make the standalone CPU host
test suite pass, compile and link both native workers independently, and
correct ordinary build/API/lifetime defects uncovered by local toolchains.

**This handoff does not authorize GPU runtime initialization, model loading,
live inference, resetting devices, rebooting the machine, or numerical parity.**
A compiled worker is not proof of working GPU inference.

## Context

- Working branch: `feature/heterogeneous-cuda-window-admission`
- Starting code commit: `27369212216260417beb33ff668334722a561413`
- Base milestone M1.18: `6b3b6be30a81d7d35d3b577e1f13b3c5c31aebfe`
- M2.0 stage-indexed worker `--config` routing / fork exec tests: `6534cb0f`.
- M2.1 HIP v2 GGUF/pack config, CPU-only fail-closed preflight: `baf4ebf7`.
- M2.2 separate HIP non-final verifier adapter: `4cd34ea5`.
- M2.3 separate HIP final/STFR verifier adapter: `a641eab3`.
- M2.4 HIP stage weight/session owner, NativeDense/PLE, Q2_0 file
  experts and CPU contract doubles: `f80107c2`.
- M2.5 opt-in, supervised native HIP executable composing these pieces:
  `27cc79ca`, corrected wrapper `n_vocab` field at `27369212`.
- Upstream tracking is **assessment-only**: latest observed maintainers'
  `main` `13eedf55`; no upstream commits have been integrated.
  Do not import upstream changes as a shortcut to solving an isolated build
  problem. The root upstream CMake refactor is significant.
- The 2026-11-13 ChatGPT container did not have a synced actual checkout
  or a native HIP toolchain. Focused HIP parser (50 assertions) and HIP
  config (17 assertions) ran with **local CPU host doubles**, not the actual
  repository. The M2.0–M2.5 full source and builds are **not yet verified**.
- GitHub code commits have already been published; this handoff owns
  **remaining integration and corrections**, not repetition of source
  generation.

## Tasks

1. Synchronize exactly the working branch and starting commit above;
   verify and preserve the existing user's stash and untracked probe.
   Do not change `main`, force-push, rebase milestone history, or run
   a new GitHub Actions workflow.
2. Run standalone experimental Linux CPU-only configure/build/CTest
   under GCC, Clang, ASan and UBSan as supported, with
   `cmake --build <dir> --parallel 28`. **27 CTest targets expected**:
   20 from M1.18 + 1 M2.0 HIP bootstrap + 1 M2.1 HIP config + 1 M2.2
   HIP non-final verifier + 1 M2.3 HIP final verifier + 3 M2.4 HIP
   resource/native/expert tests. The existing process CTest also probes
   real stage-indexed `posix_spawn` argv and CPU-only supervised HIP
   preflight refusal. Correct failures in code/tests/CMake as needed;
   do not relax the protocol, output-validity or ownership gates.
3. Independently build/link the native CUDA worker against its CUDA-only
   `strata_engine`, retaining M1.18 behavior and, where practical, the
   existing whole-archive adapter/engine link check.
4. Independently compile/link the **native HIP-only** worker, using
   `STRATA_ENABLE_HIP=ON`,
   `STRATA_BUILD_HETERO_HIP_WORKER=ON` and the appropriate
   gfx1201 compile target. Also verify the opt-in HIP adapter archive
   including non-final, final, weight owner, NativeDense/PLE and expert
   sources. If a native compile/link defect appears (CMake source shape,
   shim header, unresolved symbol, owner type), independently fix and
   retry; do not make either runtime depend on the other. Check the
   whole-archive HIP adapter link where feasible.
5. Audit, test and correct initial worker readiness and teardown:
   stage-specific config routing, HIP backend ordinal/layer identity,
   v2 config/file preflight before runtime, positive-prefix-only commit,
   nonfinite activation/head rejection, no provisional output published
   as successful inference, proper verifier -> native -> session/expert
   teardown. Ensure both live approval env vars remain unset for all
   test commands. Local CPU-double validation is allowed. Do not start
   either native GPU worker on actual devices, or access unapproved GGUF
   weights.
6. Batch ordinary fixes and tests into your own turn until the whole
   compile-only integration objective is met, then write a concise report.

## Constraints

- **Permitted worktree only:** `/home/chris/workspace/Strata-heterogeneous`.
- **Permitted staging/logs:** `/home/chris/tmp/strata-heterogeneous`.
- **Completely forbidden:** `/home/chris/workspace/Strata-dev` and
  `/home/chris/tmp/strata-dev`; do not inspect/search/read these trees
  or scan their parent directories.
- No model files without separate path-level permission. Do not initialize
  CUDA/HIP, probe device access, run live inference, reset GPU drivers,
  reboot, modify firmware, or start any GPU stress/diagnostic job.
- Do not set `STRATA_HETERO_CUDA_LIVE` or `STRATA_HETERO_HIP_LIVE`.
- Do not dispatch, run, or enable GitHub Actions, or add automatic triggers.
  Any new workflow, if actually required, must be manual-only
  `workflow_dispatch`.
- Preserve process separation (CUDA-only and HIP-only executables), the
  v4 and STFR protocol ABIs, existing mock behavior, bounded 16 MiB
  handoffs, stage-local state and explicit host-staged transfers.
- Do not claim GPU parity or successful inference from CPU tests/linking.
- Use local tools/compilers; `--parallel 28`; use `vi` or `vim` if
  interactive editing is necessary.
- You own this development turn exclusively until final report/push.

## Completion Criteria

- Actual checkout verifies all CPU CTest targets under supported builds,
  with any unsupported toolchain explicitly recorded (do not claim pass).
- CUDA worker and HIP worker build/link independently using their own
  backend engine; limitations of whole-archive verification documented.
- Any recoverable compile/link/test or lifecycle failures found have been
  fixed and rerun, without weakening safety/admission boundaries.
- No GPU runtime/model load/action, no GitHub Actions, no unrelated
  workspace changes.
- Necessary source corrections and matching `.ai/reports/report-0001.md`
  committed and pushed to the working branch. A genuine blocked or
  partial result is acceptable only when local constraints remain after
  reasonable debugging, with specific evidence.

## Reporting

Write `.ai/reports/report-0001.md` with meaningful results only:
native CUDA/HIP compile/link status, CPU CTest counts by compiler/
sanitizer, corrections made, residual blockers and the exact final
commit. Keep bulk logs outside the repository, under the permitted
staging tree. Then stop modifying the repository and return:

```text
Iteration: 0001
Status: Completed | Partial | Blocked
Branch: feature/heterogeneous-cuda-window-admission
Commit: <pushed-final-commit>
Summary: <one or two sentences>
Report: .ai/reports/report-0001.md
```

The Human User will relay this short message to ChatGPT, returning
exclusive write ownership. Live GPU validation remains separately gated.
