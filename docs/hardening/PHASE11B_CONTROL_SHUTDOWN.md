# Phase 11B — control-vector and penalty-history GPU shutdown (WIP)

## Scope and evidence

The committed pointer ledger at
[stage-resources](phase11b/reports/stage-resources/REPORT.md) attributes all
four surviving logical GPU1 allocations in a healthy two-card run:
control-vector g_dev[1].dir (491,520 B), .s (192 B), .on (4 B), and the
penalty-history buffer d_hist (131,072 B). The startup-refusal path has zero
surviving GPU1 allocations. These were public hipMalloc lifetime observations,
not proof of a VRAM leak after process destruction.

This follow-up adds owner-device release for these resources without changing
the serving hot path. The control-vector tables stay alive until the engine's
other automatic GPU owners have unwound. A shutdown guard releases all resident
device tables with a device-scoped completion fence and checked cudaFree calls.
Replacement of an existing control vector uses the same checked release.
Partial cvec upload failure also reclaims any initialized tables.

The penalty history has a DeviceOwnedBuffer registered before the GpuStage
vector; its custom deleter runs after the dependent stages and verifiers and
selects/fences the allocation device. All early returns use the same owner.

## Local checks (not native qualification)

The CPU-only fake-runtime suite tests empty, both-device, partial and
penalty-history release, plus get/select/sync/free terminal errors. GCC/Clang
Release and ASan/UBSan: 9 scenarios each (36 child cases, 4 Python test methods),
all passed locally. Intentional terminal cases disable leak detection; positive
cases remain sanitizer-checked. Native HIP/CUDA production compilation and
hardware execution are **pending Oberon**, not claimed by this commit.

## Oberon handoff

Use only the shared branch
fix/phase11b-prefill-restore-lifetime in the existing linked Phase 11B worktree.
Fetch the new commit without reapplying old patches. Do not modify the protected
checkout, either heterogeneous project, driver settings, or user configs.

1. Review the source diff and run the focused Python suite under
   tests/tools/phase11b_control_shutdown_test.py. Run the existing 89-case
   CPU/sanitizer group and 34+ Python harness tests.
2. Rebuild HIP gfx1201 and CUDA sm_89 production targets locally with **28
   parallel jobs**, with one backend build at a time.
3. Run existing native HIP 67-case group in both physical GPU orders and CUDA
   lifetime 8-case group; keep serial CTest and kernel-journal/NVRM monitoring.
4. Repeat the bounded two-card healthy A/B/A test in the first physical order,
   then inspect before reverse. Repeat the controlled late-stage stream-create
   refusal in that order, inspect, then reverse.
5. Repeat pointer-level HIP hipMalloc/hipFree and stream/event attribution on
   both devices. Verify that each control-vector table and d_hist has exactly
   one successful owner-device free and that none is released before its last
   graph user. Do not conclude leak-free cleanup from counts alone.
6. Check no new GPU fault, crash, timeout, wrong-device free, failed free,
   unexpected memory growth, or post-exit process is observed. On failure,
   stop and preserve exact evidence; do not weaken the test acceptance.
7. Commit the separate validation report and compact ledger evidence after
   verification, then push the same branch after checking GitHub Actions path
   filters. Never dispatch GitHub Actions.

## Outstanding after this change

Primary-device raw serve buffers and stream/event lifetimes, some process-wide
conversation copy pools, other startup failures, P2P-enabled operation,
two-NVIDIA layer split, long contexts, and whole-runtime memory accounting
are separate qualifications. Do not start RC1 or the PR until the required
validation and release-candidate decisions are complete.
