# Phase 11B — bounded real-model layer-split integration gate

This is the next additive implementation after the validated PeerExperts primary-device correction. It does not modify production inference, GPU drivers, or model assets.

## Integration

Agent starts from `/home/chris/workspace/Strata-dev` and identifies the existing linked Phase 11B worktree. Incoming patch is staged in `/home/chris/tmp/strata-dev/`. Apply **once** there after `git apply --check`; do not reapply any older R1/R2/R3/peer-primary patches.

This patch only adds `tools/phase11b_layer_split_gate.py`, its CPU tests, and this note. No work is permitted in either separate heterogeneous-project directory. Do not commit, push, run GitHub Actions, reset devices, or install drivers.

## Check the gate itself

```sh
python3 -m unittest discover -s tests/tools -p 'phase11b_layer_split_gate_test.py' -v
python3 -m py_compile tools/phase11b_layer_split_gate.py
```

## Bounded native HIP execution

Use the existing validated HIP production executable and an **existing** Qwen Q2_0 two-R9700 config in JSON with `backend: hip`, valid `tokenizer`, model paths and other production args. The gate builds tokenized 512-token A/B prompts using the exact tokenizer/template from that config, or accepts two existing comma-separated ID files. It is dry-run by default and refuses existing evidence directories.

Example (substitute the validated paths before execution):

```sh
cd /home/chris/workspace/Strata-dev
python3 tools/phase11b_layer_split_gate.py \
  --config /path/to/existing-two-card-config.json \
  --engine /path/to/validated-hip-strata \
  --output /home/chris/workspace/Strata-phase11b/layer-split-smoke \
  --uuid-a GPU-e39306ee46993aad \
  --uuid-b GPU-2f182f01261ba30b
```

The above dry run creates no outputs or starts no model. After checking availability, sufficient host/GPU RAM, executable/config paths, and the existing kernel-journal stop policy, repeat with `--run` under the agent's explicit timeout wrapper. Do not change user model/profile files. The tool launches two engines serially, one per UUID visibility-list order, sets `HSA_ENABLE_SDMA=0`, requires an actual two-stage startup log and a parked-image restore after A→B→A, and retains raw logs and `results.json`. `--run` is the specific opt-in for GPU execution.

It overrides *only child-process flags* to ensure a 2 GiB conversation cache, four checkpoint slots and 128-token checkpoints; it does not write the profile. The generated prompts are 512 tokens or slightly longer depending on tokenization. Inspect the *actual* selected model, context and memory fit before enabling the native run.

Run with the existing journal monitor/stop-on-MES/TLB/page-fault/reset policy. If an engine fails startup, no actual restore occurs, the GPU is unstable, or a journal fault appears, stop rather than escalating prompt length or running other GPU tests.

## Acceptance and remaining limits

- Both requested UUID mask orders: two-stage startup present, three normal `DONE` results, nonempty generated IDs and observed A-return `reused > 0` plus fresh `restored N tokens` log line. Verify actual ordinal-to-physical-device mapping from runtime evidence: the mask order alone may not renumber devices on every ROCm platform.
- No unconditional token-ID parity claim (Phase 8 established execution-path differences); preserve ids for later analysis.
- Record source/patch/executable/config hashes, GPU UUID order, kernel journal and output path. Keep subsequent changes in an incremental patch; do not duplicate the old 89-test CPU group unless code touched its boundary.
- P2P remains separately unqualified, as do CUDA execution, longer contexts, borrowed/converted single-stage real-model restore faults and other backends.

If the existing test config cannot boot this gate safely, report the precise blocker and preserve logs; don't alter production defaults to force it.
