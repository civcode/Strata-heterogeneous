# Example run configs (Oberon)

`mixed.example.json` (CUDA [0,15) + HIP [15,33) + [33,48)) and `hiponly.example.json` (the report 0067 HIP-only
configuration). Start either with `hetero/tools/serve_mixed.py`, which fills the placeholders, writes the filled config
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
| `STRATA_VANILLA` | the vanilla checkout, read only: control vector and vision engines (HIP-only config) |
| `ROCM_LIB` | the ROCm libraries |

The build trees named in `exe` and `--front-exe` are this checkout's `build-0067-hip` and `build-0067-cuda` (the
toolchains in `.ai/PROJECT.md`). Not supported in a mixed split yet: vision, control vectors, `--batch`, DFlash2,
`--pipeline-windows`, `--layer-split auto`.

Drivers in `hetero/tools/`: `engine_drive.py` (the engine over the serve protocol, token prompts from `mkprompts.py`),
`server_checks.py` (the HTTP checks of report 0069), `front_client.cpp` (the front stage alone).
