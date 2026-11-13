# M2.6 — CPU-only canonical Q2_0 model/plan consistency gate

**Baseline:** Oberon M2.0–M2.5 compile/link result `d3a8aa61`.
No GPU runtime or model file mutation is part of this change.

Run `tools/heterogeneous_model_gate.py` for independent cross-stage
consistency checks. An additional native CPU-only config preflight is
optional (use --preflight /absolute/path if desired):

```bash
cd /home/chris/workspace/Strata-heterogeneous
python3 tools/heterogeneous_model_gate.py \
  --plan cuda:0:0-12,hip:0:12-30,hip:1:30-48 \
  --layers 48 --hc 4 --n-embd 2560 --fingerprint 2026111301 \
   --config "$PWD/build-q2-stage-configs/cuda-stage0.cfg" \
  --config "$PWD/build-q2-stage-configs/hip-stage1.cfg" \
  --config "$PWD/build-q2-stage-configs/hip-stage2.cfg"
```

Requires ordered **absolute** config paths. The additional CPU-only
--preflight executable is **optional** and its filename need not be
HIP-specific. It does not replace the core Python GGUF/pack/profile checks.
Stage ordinals are backend-local: CUDA 0 and HIP 0 are distinct.

## Checks

- Coordinator partition of [0,48) with no gaps/overlaps or reused
  (backend,ordinal), at least one stage of each backend.
- Each v2 config's exact stage index, ordinal, layer range, model
  fingerprint, compiled geometry and vocabulary. Uniform pack,
  STRP profile, original ordered GGUF shards and exact index SHA-256.
- Uses Strata's existing bounded GGUF-v3 header reader. The primary
  shard must be qwen4exp; every layer must carry exactly the
  Q2_0 gate/up/down routed expert tensors. Refuses missing, duplicate,
  noncanonical or IQ3_XXS routed expert types.
- Exactly one original shard contains the layer-1 PLE tensor
  `per_layer_token_embd.weight` with IQ4_NL type and
  [160,320001536] shape. The stage covering layer 1 must name
  that shard; all other stages must omit `ple_gguf_path`.
- Checks the canonical Q2_0 pack directory has index.txt, dense.bin,
  embd.bin and experts.bin, does not contain native_experts.txt, and
  has the expected 48x512x1,382,400 expert blob bytes. The index
  has 19-column unique rows spanning all 48 layers.
- Validates full canonical STRP/v1 ranking and inverse mapping for
  48x512 expert pairs (24,576). Checks the SHA-256 of index.txt
  against the pinned configuration. Does **not** modify source inputs.

## Tests and limitations

Run `python3 -m unittest discover -s tools
-p 'test_heterogeneous_model_gate.py' -v`. 13 CPU-only
contract tests cover happy path, missing/duplicate/reordered
stages, PLE mismatches, IQ expert rejection, incomplete expert
source, profile corruption and index tamper. Registered as one
extra CTest target when Python3 is installed (**28** CTests).
The local fixture uses synthetic GGUF-header objects and a
*sparse* experts.bin; full real-data validation must run on Oberon.

The index hash is **not** a trusted digest of entire model binaries;
GGUF header checks and canonical packed file sizes are **not**
cryptographic provenance or a numerical verifier. For a live run,
verify authorized artifacts and device ordinals separately.
Nothing sets the HIP/CUDA live environment gates; **no** GPU,
driver, recovery, model load, inference or numerical parity is
authorized or performed. No GitHub Actions.
