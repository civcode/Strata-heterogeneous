# Phase 11B later-stage startup refusal — Oberon

## Result

**Passed in both physical GPU orders**, one native invocation per order, with the first result inspected before enabling reverse. No production defect was demonstrated or production source changed. This qualifies only the supplied later-stage stream-creation refusal, not all startup unwind or asynchronous lifetime paths.

Started from clean `/home/chris/workspace/Strata-dev`; selected the existing linked `/home/chris/workspace/Strata-worktrees/phase11b-prefill-restore`, HEAD `0ef75af1730c9374bbf1cb66a361f410d699b805`.

Input remains at `/home/chris/tmp/strata-dev/STRATA_PHASE11B_LAYER_SPLIT_STARTUP_FAILURE_INCREMENTAL.patch`.
SHA-256: `f38cc2e11a5551510cf0692434d9ceae6d81599bd4110121e335832184e08736`.
`git apply --check` passed and this patch was applied once. No earlier patch was reapplied. The input accidentally includes two ignored CPython 3.13 bytecode files; after application only those two incoming artifacts were removed. They were not used by Python 3.12 and are excluded from the refreshed source patches. All seven source/doc additions are unchanged from the input.

## Before native execution

- Read and followed `docs/hardening/PHASE11B_LAYER_SPLIT_STARTUP_FAILURE.md`.
- CPU tests: 15/15 passed, including actual host-compiler interposer ABI/forwarding controls, wrong-device/no-opt-in controls, skip-one control and child-timeout ownership test. Python byte-compilation passed.
- Built the test-only library with the documented ROCm `hipcc -std=c++17 -O2 -fPIC -shared ... -ldl` command. Never linked into the production executable or installed system-wide.
- Dry run passed before native execution and created no native output directory.
- Reused the already-tested Phase 11B HIP production executable and existing read-only `Strata-phase10c/out/dual-final-off-config.json`. Qwen Q2_0, int8 KV, context 24576, automatic two-card split, same existing model/tokenizer/profile paths. No production C++ changed, so no production rebuild was needed.
- Existing asset paths checked; profile-save absent from effective argv. Preflight showed about 56 GiB available host memory and 32567 MiB free VRAM on each R9700. No competing model engine was observed.

## Controls and actual mapping

`monitored_startup.py` runs exactly one order per explicit invocation. It calls the supplied gate/verification and substitutes a monitored equivalent of `run_owned` so the actual engine process group is checked during its wait. The engine has its own session/group, 180-second deadline, and only that group is killed on a stop/timeout. Each shell invocation also had an explicit 220-second outer timeout.

A separate HIP runtime process queries ordinal PCI bus IDs and UUIDs under the same effective visibility environment, **without any LD_PRELOAD or fault settings**, immediately before each fault run. The fault engine receives only the new fault interposer; the prior mapping audit was not chained. Mapping assertions compare actual runtime identities with the physical inventory, rather than assuming mask order renumbers cards.

| Run | Logical 0 | Logical 1 (fault target) | Engine PID | Result |
|---|---|---|---:|---|
| forward | 03:00.0 / e39306ee46993aad | 07:00.0 / 2f182f01261ba30b | 125298 | pass |
| reverse | 07:00.0 / 2f182f01261ba30b | 03:00.0 / e39306ee46993aad | 126212 | pass |

Forward log, result, journal and mapping were read before reverse; the review record is `evidence/forward-reviewed.txt`. The launcher refuses reverse without a passed forward supervisor result and that explicit record. No skip search or retry was performed (`--skip 0` both times).

The established MES timeout/failure, INVALIDATE_TLBS, page-fault, GPU-reset and KFD/amdgpu fault regex was checked before/after mapping, during engine execution at 0.5-second polling intervals, and afterward. Journal reads succeeded: 9 checks forward, 8 reverse, **no matching stop conditions**. Runtime resume/initialization messages are retained in raw journals and were inspected; they are not a fault-match or a manual reset.

## Observed acceptance

Both raw logs end with exactly:

```
PHASE11B_FAULT_INJECTED stream-create device=1 skipped=0
strata generate: layer split, CUDA1: its streams failed
```

Each engine exited **exactly 1**, without a signal/crash or READY announcement. Owned process groups no longer existed after reaping. Native wait durations were about 2.02 s forward and 1.52 s reverse, not timeouts. Post-run process inspection found no lingering engine.

Source inspection locates the refusal in `generate.cpp`'s later `GpuStage` stream setup, after its weight loading for this config and before later stage session/expert-cache initialization. The marker/refusal sequence demonstrates that branch, not an unrelated earlier stream failure. This does not establish ExpertCache teardown coverage at a later startup point or inspect every resource release device.

Config, expert profile, executable and interposer hashes were identical before/after each run.

## Evidence and hashes

Root: `/home/chris/workspace/Strata-phase11b/layer-split-startup-failure/`.

- `forward/` and `reverse/`: focused raw `startup-failure.log` and gate `results.json`.
- `evidence/`: per-order supervisors, journal/mapping logs, input/post hashes, first-order review, CPU/build/dry-run logs, physical inventory and before/after memory.
- `monitored_startup.py`: exact external validation wrapper, not installed in production.
- `libphase11b_stream_create_fault.so`: copy of the tested temporary library. SHA-256 `18d6e41d3eb56221a8f75ed97863deb2a422d27f1f1e2013ea87d3bb1c89fc43`.
- Production executable SHA-256 `2d04887accc5c8be881af4882f4f9087755532ee0a4d1a46463715ce46bced3c`.
- `pre-integration.patch` exactly matches the prior layer-split full patch: `c396e5511f8fed299a153dd215ba8bc07ffa9e62f6b4c0fc1ec3ebafa3c3350e`.
- `review.patch`: refreshed full worktree, SHA-256 `e2c530de2fff71cc1e1afd57738c6fe9af7084964680a096153ee6672189366a`.
- `STARTUP_FAILURE_VALIDATED_INCREMENTAL.patch`: seven source/doc additions only, SHA-256 `efff8524705fa4763c6b359fd15d644708f8769f65132406c2a1d70576399bd5`.

Full-patch applicability against HEAD verified through a separate temporary index; authoritative index empty. Tracked and untracked whitespace checks passed. No post-test source adjustment. Advisor discovery found no available tool.

## Limits

This is one submission-time refusal, not an accepted-work error, unknown completion, delayed failure, hang or reset. It does not prove every startup failure branch, later stage cache/session failure, complete allocation-device release correctness, P2P, native CUDA or broader backend safety. Existing R4/R5/R6 functionality was left unchanged; the unrelated 89-case CPU group was not repeated.

No commits, pushes, Actions, driver changes, device resets, downloads, user config/profile writes or separate heterogeneous-project access/changes. Strata-dev remains clean; implementation stays uncommitted in the linked worktree. Do not reapply any output patch to that already-updated worktree.
