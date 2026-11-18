# Example run configs (Oberon)

`mixed.example.json` (CUDA [0,15) + HIP [15,33) + [33,48)) and `hiponly.example.json` (the report 0067 HIP-only
configuration). Both run vision and the scaled control vector. Start either with `hetero/tools/serve_mixed.py`, which fills the placeholders, writes the filled config
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
changed since the stamp. A commit that touches only notes, reports, run logs, docs or `*.md` files does not make a
build stale. Vision helpers come from the host build or from a build of `tools/vision` in this checkout, stamped the
same way (with `STRATA_VISION_HIP=ON` for `"gpu": true`). Do not bypass the check.
Not supported in a mixed split yet: `--batch`, DFlash2,
`--pipeline-windows`, `--layer-split auto`.

Drivers in `hetero/tools/`: `engine_drive.py` (the engine over the serve protocol, token prompts from `mkprompts.py`),
`server_checks.py` (the HTTP checks of report 0069), `front_client.cpp` (the front stage alone).
