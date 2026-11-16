# Report 0065

Status: **Completed**, with measurement limitations below. Vanilla accepted the pack, native and PLE GGUFs, expert profile, MTP draft and control vector, served all three requests, and shut down cooperatively.

## Configuration and startup

Copied the vanilla Q2_0 run config into the worktree (ignored `strata-q2_0.json`). Relative to the original, only these keys changed: `exe` to `/home/chris/workspace/Strata-mixed-native/build-hip/strata`, `log` to `/home/chris/workspace/Strata-mixed-native/strata-q2_0-0065.log`, and `cwd` to `/home/chris/workspace/Strata-mixed-native`. The cwd was redirected to prevent incidental writes in the vanilla checkout. Started the branch server with its checkout-local `.venv`, `HSA_ENABLE_SDMA=0`, port 8081, and `--open`, as in the vanilla launcher. The original vision executable and input files were used. No engine code changed.

Server: `[strata] layer split across GPUs [0, 1] (auto)`; `[strata] Drafting: MTP (the DFlash2 drafter isn't downloaded yet)`. This is the native two-stage layer-split engine, not the all-cards split engine. Selected engine log lines (HIP devices are labeled CUDA by the compatibility layer):

```text
strata: gfx12 (RDNA4): exact extra switches on by default (STRATA_GFX12_DEFAULTS=0 turns them off; a switch you set is kept): HEAD_MIX_MULTI
strata: gfx12 (RDNA4): 12 exact speed switches on by default (STRATA_GFX12_DEFAULTS=0 turns them off; a switch you set is kept): GDN_HEAD GDN_PP GDN_CONVL2 GDN_NOY CVEC_FUSE ATTN_LANECELL ATTN_MERGE_V2 TSUM EXPDEC SMALL_UNR GR_Q32_G GR_Q8_PFS
strata generate: gfx12: --prefill auto may take chunks up to 32768 (STRATA_GFX12_CHUNK=16 keeps 16384, =0 keeps 8192)
strata: gfx12 (RDNA4): the hyper-connection projections are read as Q8_0 (quantized at load, decode faster, answers differ slightly); STRATA_HC_Q8=0 keeps the bf16 read
strata: the prompt attention on matrix cores on (gfx1201) by default; STRATA_HIP_WMMA=0 turns it off
strata generate: layer split auto: K=24 - predicted 46.3 ms per decode window; the caches hold 24576 of 24576 profiled pairs (~100.0% of the routed mass), best of 46 placements
strata mtp: draft layer loaded, 1026 MiB of VRAM (experts 675, dense 111), files read in 0.11 s (6932 MiB/s)
strata: prompt kernels: every chunk runs 48 of 48 layers on the FP16 path.
strata serve: layer split: layers 0-23 (CUDA0), 24-47 (CUDA1), one hand-off per window
strata serve: --pipeline-windows 2: two verifiers per stage, 2 stages overlap across windows (decode too; the GDN snapshots beside the window)
```

The engine's exact expanded invocation:

```text
/home/chris/workspace/Strata-mixed-native/build-hip/strata --serve --pack /home/chris/workspace/Strata-data/packs/q2_0 --native /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf --ple-gguf /home/chris/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf --expert-profile /home/chris/workspace/Strata/data/expert-profile.bin --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp /home/chris/workspace/Strata-data/mtp/rt --max-context 196608 --kv int8 --vision --vram-reserve-mib 700 --control-vector-scaled /home/chris/workspace/Strata/data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf:1.0 --control-vector-layer-range 4 44 --cvec-mode project --cvec-dir per-layer --layer-split auto
```

## Requests

Non-streaming OpenAI chat, `temperature: 0`, `max_tokens: 200` for the first two identical requests and 50 for the third. All returned `finish_reason: length`. The long prompt was **5,397 tokens**, not the intended approximately 4,000; its reading rate is reported for that actual length. The third response exhausted its token limit while still thinking and had no answer. TTFT was **not reported** in the server response or logs and was not measured by this non-streaming client; prompt time is not presented as TTFT.

| Request | Prompt tokens (reused / read) | Completion | Prompt time | Prefill tok/s | Decode tok/s | Drafts accepted |
| --- | --- | --- | --- | --- | --- | --- |
| Cold | 69 (0 / 69) | 200 | 320 ms | 215.5 | 85.0 | 105 / 184 |
| Repeat | 69 (64 / 5) | 200 | 29 ms | 171.4 | 84.9 | 106 / 186 |
| Long | 5397 (0 / 5397) | 50 | 2364 ms | 2283.0 | 96.1 | 25 / 45 |

Values above are from engine log lines. The JSON response rounds the second prefill rate to 171.2 tok/s instead of the log's 171.4. The repeat reused 64 cached tokens, so its prefill rate is for only five newly read tokens. Suffix drafts were separately reported as 3/7, 6/10 and 2/5 for the respective requests.

## Memory and shutdown

`amd-smi metric -m` after ready, before the requests, reported GPU 0 used/free **26689 / 5935 MB**, GPU 1 **24360 / 8264 MB** (each total 32624 MB). The engine separately reported 6112 and 8470 MiB free at startup. Sent one SIGINT to the server; it logged `stopping` then `stopped`. Server, engine and vision process exited; no force was used. Afterwards AMD SMI reported **78 MB used and 32546 MB free on each card**: there is residual device/system usage, not zero global VRAM usage; no process from this run remained.
