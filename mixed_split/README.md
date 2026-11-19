# Mixed split: one model on an NVIDIA card and AMD cards together

**Experimental.** The first layers of the model run on an NVIDIA card (a CUDA engine, the "front stage"); the rest
run on two or more AMD cards (the HIP engine, with vanilla Strata's layer split). Both engines are compiled from this
checkout with the CMake option `STRATA_MIXED_SPLIT`. Linux only.

Tested on one PC: an RTX 4070 Ti SUPER (16 GB) and two Radeon AI PRO R9700 (32 GB each), with Qwen3.8-Flash-Next Q2_0
and 15 layers on the NVIDIA card. The release-candidate measurements will be added here (`.ai/RC.md`).

## Set it up

Run setup as for vanilla Strata, from this checkout:

```sh
./setup.sh --setup
```

On a PC with NVIDIA and AMD cards, the first question offers a third answer:

```
  1) NVIDIA: ...   (recommended)
  2) AMD: ...
  3) Both together: mixed split - the first layers on the NVIDIA card, the rest on the AMD cards
Which cards? [1]:
```

Answer `3`. Setup then asks how many layers go on the NVIDIA card (15 on a 16 GB card, scaled by its VRAM), and
goes on with vanilla's questions: the model, its size, the context, pictures, and so on. It compiles:
- the HIP engine with the option, into `engine/` (as vanilla does, plus `STRATA_MIXED_SPLIT=ON`);
- the CUDA front-stage engine, into `engine-front/` (it needs the CUDA toolkit; setup offers to install it as it does
  for an NVIDIA engine).

Without questions: `./setup.sh --setup --backend mixed_split [--front-layers 15] --yes ...`

After that, start it as vanilla: `./setup.sh` (or the `run-<model>.sh` script setup writes). The start line says
`mixed split: layers 0-14 on ... (CUDA), the rest on the AMD cards (HIP)`.

To go back to the AMD cards alone, run `./setup.sh --setup` again and answer `2` (or `--backend hip`): the engine is
compiled again without the option, and the model's config loses the front stage.

## What setup changes in the run config

The model's `strata-<model>.json` gets a `"mixed_split"` key (the front stage's engine, its layers, the NVIDIA card)
and the engine options `--front-exe` and `--front-layers`. Your own options in the config are kept, as vanilla keeps
them, except those the engine refuses beside the mixed split; setup leaves those out and says so:
`--split-skip-if-fits`, `--batch` / `--slots`, `--peer-device`, `--expert-cache-remote`. It also runs one request at
a time (`"parallel": 1`).

## Limits
- Two or more AMD cards; one AMD card is not supported yet.
- One request at a time; no DFlash2 drafter; the list above.
- Tested with Qwen3.8-Flash-Next only.
- Setup from this checkout only. The desktop app and `strata launch` do not know the mixed split.
- What differs from vanilla, and why: `.ai/DIVERGENCE.md`.
