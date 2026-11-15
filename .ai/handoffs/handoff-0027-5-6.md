# Oberon handoff — 0027.5 and 0027.6 preparation

Read report-0027-5-6.md and experimental/heterogeneous/docs/NATIVE_HC_NUMERICAL_DIVERGENCE.md. Base hardware evidence is e2dbe960. The candidate is diagnostic only; preserve native arithmetic and the unchanged KL gate. Run autonomously on Oberon, report and commit actual results. Limit writes to approved checkout /home/chris/workspace/Strata-heterogeneous and scratch /home/chris/tmp/strata-heterogeneous/m7-0027-5/. Use prior approved assets read-only. Preserve tools/heterogeneous_pool_probe.cpp. No private/upstream writes, downloads/repack, forced push, fault injection/forced GPU kills, hardware/settings changes, GitHub Actions, or performance tuning. Use vanilla toolchain, existing vanilla venv and -j28.

## 1. Build and CPU checks

```sh
cmake --build build-hetero-smoke-host -j28
cmake --build build-hetero-smoke-cuda -j28
cmake --build build-hetero-smoke-hip -j28
cmake --build build-hetero-smoke-cuda --target strata-hc-replay -j28
ctest --test-dir build-hetero-smoke-host -j28 --output-on-failure
PYTHONPATH=tools .venv/bin/python -m unittest test_heterogeneous_prefill_stage_gate test_heterogeneous_hc_replay test_heterogeneous_hc_replay_capture test_heterogeneous_prefill_first_token test_heterogeneous_layer0_arithmetic test_heterogeneous_prefill_state test_heterogeneous_prefill_parity test_heterogeneous_mtp_resident test_heterogeneous_mtp_generation test_heterogeneous_mtp_provenance test_heterogeneous_mtp_capture_integration test_heterogeneous_mtp_qualification test_heterogeneous_mtp_three_round test_heterogeneous_mtp_acceptance test_heterogeneous_mtp_reuse test_heterogeneous_mtp_front test_heterogeneous_mtp_state test_heterogeneous_mtp_admission test_heterogeneous_compare_parity
PYTHONPATH=tools .venv/bin/python -m unittest serve.test_heterogeneous_resident
env -u PYTHONPATH .venv/bin/python -m pip check
```

Expected candidate counts if no further test changes: 98 CTests, 136 requested Python tests, 12 resident. Record actual counts. Check typed no-run CLI acceptance and refusal: stage-capture only with existing bounded A/B/A state capture, approved plan/context, group native HIP, chunk <= cut <= 16; reject reference without stage capture, missing state/parity/capture, wrong plan/context/chunk, incompatible first-token/HC/MTP/receiver/ordinary generation modes. Reference requires a completed authentic matching source capture; do not fabricate that marker. Fix only demonstrated diagnostic or adapter blockers, not arithmetic; report exact error and changes. Do not proceed past unresolved build or unexpected contract failure.

## 2. Bounded wire/reference matrix

Reuse exact previously approved cfg files and executable/CLI setup from 0027/0027.1–3. CUDA0 PCI04:00, HIP0 PCI03:00, HIP1 PCI07:00, layer split 12/18/18, context2048/max_t8/reserve536870912, full residency zero host misses. Keep all native settings and source bytes unchanged. Check assets using previous read-only approved digest evidence. Both shared and socket host transports; live bulk R was socket-framed even in shared mode. No external arm or overlap flag changes, no HC replay mode for this matrix.

Run these five fixed cases in each host mode, with actual teacher-forced IDs:

| Cut | Chunk | Complete sequence |
|---:|---:|---|
| 1 | 1 | 248045 846 |
| 2 | 1 | 248045 846 198 |
| 2 | 2 | 248045 846 198 |
| 6 | 3 | 248045 846 198 2523 513 32767 264 |
| 6 | 6 | 248045 846 198 2523 513 32767 264 |

Each case is two fresh processes (wire and independent file-fed stage reference), each running the existing serial-A/native/serial-B A/B/A with actual teacher forcing. Use distinct absolute prefixes WIRE and REF inside restricted scratch; capture stderr/stdout in chronological order outside capture prefixes. Keep raw files restricted and read-only once complete; no raw commits/uploads. Historical captures may be used for the matching uninstrumented baseline only if exact identity/settings are verified; otherwise obtain fresh legacy A/B/A baselines (without new flags). Compare overlapping state/logit/bridge results, excluding expected completion-marker changes, so the new observation does not silently change the native path.

Wire: same historical approved command, --generate 1 --prompt-chunk 1 --prefill-parity-at CUT --native-prefill-chunk CHUNK --prefill-state-capture --prefill-stage-capture --capture-prefix WIRE --run, with the sequence on stdin. No new flags in ordinary regressions. Both physical HIP stages run native chain and normal R bridge.

Before using a wire capture as immutable reference:
```sh
PYTHONPATH=tools .venv/bin/python -c 'import sys; from pathlib import Path; from heterogeneous_prefill_stage_gate import read_capture; r=read_capture(Path(sys.argv[1])); assert not r["marker"]["stage_reference"] and r["bridge"]["status"]=="PASS" and r["local_links_exact"] and r["serial_controls_exact"]' "$WIRE"
```
Check actual marker cut/chunk/tokens/transport/source identity against the next command and logs. Check full residency, zero host misses and successful EOF. Quality failure does not disqualify collection of independent interoperability evidence, provided all these integrity gates pass.

Reference: same case, exact same assets, native paths, owner devices and command except fresh --capture-prefix REF and --prefill-stage-reference "$WIRE". CUDA independently computes the same native prefix and does not forward live bulk R. HIP0 and HIP1 separately execute native Prefill with the matching immutable producer outputs from WIRE. Both ACKs precede the same teacher-forced continuation. No new transport protocol. Reference is not full original vanilla Prefill::run caller or independent math implementation.

Retain all raw per-stage in/out R, checkpoint manifests/fields, original bridge and CUDA adoption snapshot and STFR. Check stage snapshots cover correct GDN/QSA ordinals; PLE storage exists on all three but only stage0 owns the PLE computation. Nonfinite/unexpected state or replay mismatch requires localization and no admission; do not hide by changing tolerances. A stage mismatch could also reflect native scheduling, especially independent HIP stages, not necessarily cross-runtime error.

Pair analysis (single pair produces incomplete matrix, exit 1 is expected):
```sh
PYTHONPATH=tools .venv/bin/python tools/heterogeneous_prefill_stage_gate.py --pair "$WIRE" "$REF"
```
Run the same tool with ten --pair WIRE REF arguments for all five cases and both transports; save complete JSON outside the capture stems. Exit 0 only if the complete bounded matrix passes both axes; exit 1 for failure/incomplete, 2 for invalid data. Record exact pair and matrix interoperability and unchanged quality gates independently. Historical cut2 KL is expected to fail; record it and continue the bounded interoperability investigation if all integrity/interop checks pass. Do not relax 0.001. If new quality failure or unexplained mismatch appears, preserve first differing stage/chunk/state/row and stop unsafe continuation. No overall admission decision from stage replay.

## 3. MTP preparation, regressions and stopping boundary

The new CPU seed API is tested in the standalone contract. No GPU caller uses it. Do not enable bulk+MTP. Later integration, subject to independent qualification and review, must feed N-1 known prompt pairs using synchronous native draft priming on the terminal HIP owner while each on_chunk residual is valid; never borrow a pointer across reuse. Final actual prompt token goes through verifier for actual greedy anchor. Keep committed-row vs window counters distinct, target adoption only after all ACKs; use native arithmetic (do not force batched draft_kv or substitute precision). Compare same native prefill without MTP, fresh checkpoints and traced R/K/V/all 22 front fields and natural rejection/acceptance across transports, then separately qualify resident/cancel/recovery. This hardware work is not in the present candidate.

Run previously qualified serial MTP 0025 nonresident, 0026 resident A/B/A and cancellation/recovery, ordinary resident, and existing diagnostic/no-MTP controls on both transports using reports/handoffs of those milestones. No unrelated performance work. Verify previous outputs and regression gates. 0028 consolidation is pending hardware evidence. Do not general-admit; target checkpoint provenance remains separately unresolved.

Update .ai/reports/report-0027-5-6.md with actual builds/tests, exact commands/raw paths, all interop and quality results (including failures), first mismatch/localization and any corrective changes. Commit and push all relevant code and report changes; preserve unrelated untracked file. Do not commit raw data.
