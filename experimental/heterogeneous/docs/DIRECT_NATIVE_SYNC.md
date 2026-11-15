# Direct native CUDA → HIP ready-flag path: graph qualification candidate

Hardware update: the bounded model-free graph matrix and a terminal timeout passed on Oberon; see `.ai/reports/report-0037.md` for scope and remaining gaps. Real-model integration and inference performance remain unqualified.

Base inspected: `ff97e6ad694a373c0509478bea33e574a8598109` on `feature/heterogeneous-hip-only`. Handoff 0036 remains outstanding; this patch is prepared separately and does not change its branch or production defaults.

## Objective and scope

Reuse the unmodified native `handoff_publish`, `wait_handoff`, and `copy_from_mapped` kernels from vanilla Strata. Register the same size-sealed Linux memfd independently with CUDA and HIP; pass an inherited file descriptor, not a foreign runtime pointer or event. The HIP-to-HIP edge uses the existing coherent mapped host allocation pattern. Each GPU uses a separately resolved local alias. Both downstream native graphs are captured before any live wait; HIP1 and HIP0 graphs are launched before CUDA publishes the first activation. The probe uses the real R/bo/inj handoff shape of HC=4 and n_embd=2560, with 1..8 rows (51,216–409,728 bytes per edge). It never copies payload between runtimes via the CPU. The host does seed the original CUDA test input and read back both HIP outputs for verification.

This is a model-free graph and registration candidate, **not integrated inference**. It does not modify either native synchronization kernel or enable an optimized server path. Do not use the native graph probe's host orchestration timings as inference throughput. It adds no dependency or workflow.

## Build and first hardware run

Use the existing vanilla Strata build toolchain and the same venv setup. Build in the existing independent CUDA and HIP build caches; 28 parallel jobs. Do not run GitHub Actions.

```sh
cmake --build "$CUDA_BUILD" --target strata_heterogeneous_direct_graph_probe -j28
cmake --build "$HIP_BUILD" --target strata_heterogeneous_direct_graph_probe -j28
cmake --build "$HOST_BUILD" --target strata_heterogeneous_direct_handoff_layout_tests -j28
ctest --test-dir "$HOST_BUILD" -R direct_handoff_layout -j28 --output-on-failure
.venv/bin/python -m unittest -v tools.test_heterogeneous_direct_graph_probe
```

Resolve the actual existing probe executable paths and device ordinals by their PCI identities. Reserve exclusive use of the GPUs, use new private scratch and keep safety reserves unchanged. Never forcibly kill a potentially stalled native GPU worker. Run the shortest graph probe first:

```sh
umask 077
.venv/bin/python tools/heterogeneous_direct_graph_probe.py \
  --cuda-worker "$CUDA_DIRECT_PROBE" --hip-worker "$HIP_DIRECT_PROBE" \
  --cuda-device 0 --hip-devices 0 1 --rows 1 --rounds 16 --warmup 2 \
  --output "$NEW_PRIVATE_EVIDENCE/rows1.json"
```

Only after success, repeat in independent processes for 4 and 8 rows, 128 then 1024 rounds, with unique output filenames and warmups. Qualify full-byte correctness at both HIP GPUs, initial pending state, ready/error checks, fresh pattern across rounds, both native stream drains and explicit BYE/unregister/exit. A failure is not permission to skip verification, change device settings, or increase timeout to hide missing publication. The optional `--exercise-timeout` withholds CUDA publication for a terminal round (separate fresh process suggested). It must reject the window, drain, poison the session and exit cleanly; it deliberately does not reuse a timed-out slot. The vanilla wait kernel permits the rest of the graph to drain even after timeout; its output is discarded. Do not start live-model execution on failing probe evidence. If any process remains alive, report the PID and stop expanding the test.

## Real model integration boundary

The existing CUDA `CudaVerifierDriver`, HIP `HipFinalVerifierExecutor`, and `FinalPipelineSession` are the seams for the real inference edge. Extend the existing opt-in receiver-arm path only after the above graph runs pass. Use a dedicated registered mapping distinct from the ordinary CPU activation slot. On admitted windows, the grouped HIP worker must attach the mapped input and ready flag before native graph capture and prelaunch; CUDA must attach the same mapping's local payload/flag aliases as its native verifier output. Both workers must agree on exact identity, geometry, capacity, ownership, and mapping generation before any graph is armed. The existing receiver-arm ACK must precede producer launch. No CPU payload relay, CPU ready publication, CUDA→HIP runtime handle, or native kernel rewrite. Preserve the HIP-local native chain. Ensure producer/consumer completion evidence, timeout poisoning, graph destruction before unregister, and no slot reuse before all accepted-prefix commit ACKs. Keep the old transport as a separately selectable reference/fallback, not an automatic retry of a poisoned session.

The existing CPU transport validates activation finiteness before handing bytes to HIP. A direct GPU-consumed slot must retain an equivalent safety gate; the native ready word alone conveys publication, not finite-state validity. The current native wait can drain on timeout using stale bytes, so a timed-out result must never be published/committed. Do not remove the guards or authorize the full-model path solely on a synthetic probe. Establish native finite-state validation, then bounded teacher-forced full-logit/state comparisons, exact generated token comparisons, reset/reuse/cancel and partial-prefix state parity. Native asynchronous bulk prompt prefill requires separate chunk/event ownership and validation; this graph probe is the decode-window mechanism, not a new bulk-prefill protocol.

Use the user-selected experimental 15/18/15 placement and 12/18/18 reference; do not change defaults. Actual throughput improvements must be measured trace-off using matched prompts, model, sampling, context, KV precision and MTP state; compare existing group, direct candidate, and matched vanilla. No speculation until no-draft controls are reliable. Real graph capture, cross-vendor ordering, model integration, quality, hardware throughput and production qualification remain unverified in Chat AI's environment.

## Local validation of this patch

- Python: five tests passed, including real sealed memfd and separate **CPU fake** subprocesses, expected failure rejection, explicit shutdown and withheld-publication control. Syntax check passed.
- C++: layout/payload-pattern fixture passed under `-DNDEBUG -O2 -Wall -Wextra -Wpedantic -Werror`; both CUDA/HIP conditional translation-unit syntax checks passed against **synthetic runtime declarations**, not actual CUDA/ROCm headers or GPU binaries.
- Full checkout unavailable (GitHub DNS resolution failed in container); no native runtime build, GPU run or GitHub Actions. Treat all hardware statuses as NOT_RUN until Oberon evidence.


## Serial verified real-model candidate (after 0037)

The explicit native CLI flag `--direct-verified-shared` attaches **the same size-sealed shared transport object** (dedicated to this mode) to both workers before graph preparation, using independent runtime-local aliases and new mode-specific HELLO kinds. The native CUDA verifier publishes directly into this mapping, bypassing the separate D2H staging and CPU activation copy. The HIP entry consumes directly from the registered mapping via vanilla's native ready-word wait and mapped reads; HIP0→HIP1 uses the existing native overlap chain. No native synchronization kernel changed.

This is deliberately **serial at the CUDA→HIP admission boundary**, not fully prelaunched direct inference: the coordinator waits for the producer drain and its checksum-bearing activation reply; both original finite-activation checks and existing checksum/identity/accepted-prefix commit gates still execute before HIP consumes the data. The CPU scans bytes but does not relay the activation payload. The existing one-slot protocol owns reset/reuse after all commit ACKs. No foreign runtime pointers or events. Direct registration aliases outlive graph destruction. On failure the experimental process group requests EOF/drain and does not signal workers on timeout; report any unexited PID.

Use the known native CLI fixed-token or resident serial setup on the approved 15/18/15 or 12/18/18 mixed plan (HC=4, embedding 2560, max_t=8, full residency), with `--shared --group-native-hip --hip-split-overlap --direct-verified-shared`. The reference omits the final flag, preserving all other settings. Use full logits, no native bulk, external receiver arm, CUDA mapped egress, MTP, lookup, or unrelated diagnostic captures. Ordinary fixed-window full-logit captures remain available. This is a validation-only opt-in, not a new server mode or production default. Clean native build, real-model parity, failure lifecycle, and throughput are NOT_RUN in Chat AI's environment. Test the new C++ binding fixture and host final-pipeline protocol suite first. Then qualify actual fixed one-row/multirow partial-prefix windows and reset/greedy before measuring. Future vendor-boundary prelaunch still needs a genuinely equivalent pre-consumption GPU finite gate and independent lifecycle qualification.
