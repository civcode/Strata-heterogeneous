# Example run configs (Oberon)

`mixed.example.json` (CUDA [0,15) + HIP [15,33) + [33,48)) and `hiponly.example.json` (the report 0067 HIP-only
configuration). Report 0080 adds two HIP-only variants: `hiponly-native.example.json` (the same on the native Q2_0 pack,
`tools/iq_pack.py --base ${STRATA_DATA}/packs/q2_0 --out ${STRATA_DATA}/packs/q2_0-native`) and
`hiponly-dflash2.example.json` (native pack, `"drafting": "dflash2"`, no `--pipeline-windows`; the drafter pack in
`${STRATA_DATA}/dflash2/`). Vanilla refuses DFlash2 on the canonical pack.
`mixed-native.example.json` (report 0081) is the mixed mirror on that native pack; the front stage gets the host's `--pack`. Both run vision and the scaled control vector. Start either with `hetero/tools/serve_mixed.py`, which fills the placeholders, writes the filled config
and the server log into `RUN_DIR`, gives a mixed split one shared expert arena, and becomes the server:

```sh
export STRATA_DATA=/home/chris/workspace/Strata-data \
       STRATA_MODEL_DIR=/home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
       STRATA_PROFILE=/home/chris/workspace/Strata/data/expert-profile.bin \
       STRATA_VANILLA=/home/chris/workspace/Strata ROCM_LIB=/opt/rocm/core-10.1/lib
.venv/bin/python hetero/tools/serve_mixed.py --config hetero/configs/mixed.example.json
```

| Placeholder | Meaning |
| --- | --- |
| `STRATA_REPO` | this checkout (default) |
| `RUN_DIR` | where the filled config and the log go (default `runs/` in this checkout) |
| `STRATA_DATA` | the pack and the MTP drafter |
| `STRATA_MODEL_DIR` | the GGUF shards (and the vision projector) |
| `STRATA_PROFILE` | the expert profile |
| `STRATA_VANILLA` | the vanilla checkout, read only: the control vector file (HIP-only config; data, not a program) |
| `ROCM_LIB` | the ROCm libraries |

Both examples name this checkout's `build-0071-hip` (the mixed one also `build-0071-cuda`) and the vision builds (the
toolchains in `.ai/PROJECT.md`). After a successful build of each tree, write `git rev-parse HEAD` into its
`BUILD_COMMIT` file. Before starting, the launcher (`check_launch.py`) checks that each executable comes from a build of
this checkout, that its libraries come from that build or a system or toolkit path, and that the sources have not
changed since the stamp. A commit that touches only notes, reports, run logs, docs, `*.md` files, the scripts in
`hetero/tools/` or the configs here does not make a build stale. Vision helpers come from the host build or from a build of `tools/vision` in this checkout, stamped the
same way (with `STRATA_VISION_HIP=ON` for `"gpu": true`). Do not bypass the check.
Not supported in a mixed split yet: `--batch`, DFlash2. `--pipeline-windows` works since report 0077 (both
examples run it with `STRATA_PIPELINE_AUTO=0` and `STRATA_MTP_STAGE0=0`, as production does); `STRATA_SPLIT_TIMING=1`
times the serial and the pipelined windows and every link call (report 0078). Prompt chunks overlap across the link
since report 0078: the front stage reads chunk c+1 while the first R9700 reads chunk c (`STRATA_MIXED_PF_AHEAD=0`
turns it off, for A/B runs). `"layer_split": "auto"` works since report 0075: vanilla's search plans the host's cards over
layers `[front_layers, 48)` (on Oberon K=31, against the configs' fixed 33). `STRATA_SPLIT_TIMING=1` in the
environment logs a mixed decode window's parts per request (report 0075).

Drivers in `hetero/tools/`: `engine_drive.py` (the engine over the serve protocol, token prompts from `mkprompts.py`),
`server_checks.py` (the HTTP checks of report 0069), `front_client.cpp` (the front stage alone), `run_engine.py` (one
accounted engine run from a config), `serve_session.py` (one accounted server session from a config: the checks, the
cache/vision/cvec features, or the code/long/16K measurement), `cmp_tokens.py` (token-exact comparison of two runs), `tps_table.py` (prefill / decode tok/s and first-token time
over the samples of run_engine.py results), `session_table.py` (the same, with drafts, over `serve_session.py … measure` results), `concurrent_table.py` (aggregate and per-request tables over `serve_session.py … concurrent` results); `mkprompts.py` also writes a 32K prompt (`long32k`).
