# Report 0080

**Status: Completed, no-go.** Part A only. On this rig, vanilla's DFlash2 drafter decodes slower than the production
mirror's MTP on all three prompts, by more than the spread of the samples. Part B was not started. No engine source
changed.

## Summary

- **Vanilla does not run DFlash2 on the production pack.** The canonical Q2_0 pack (`packs/q2_0`, the one vanilla's
  setup builds on an AVX-512 CPU) is refused with "DFlash2 is off here: it needs a native pack"
  (`generate.cpp`, the `why` chain at the DFlash2 load). To get DFlash2 at all, I built the native Q2_0 pack vanilla's
  setup builds on other CPUs (`tools/iq_pack.py --base packs/q2_0`, hard-linked `dense.bin`, a few seconds of CPU).
- **Vanilla drops `--pipeline-windows` with DFlash2.** The launcher's `drafting.blocked_by` keeps DFlash2 off when the
  arguments hold `--pipeline-windows`. The engine refuses the pair: "DFlash2 drafts from the finished window's taps
  on every stage, so the next window's drafts cannot exist while this one still runs". So DFlash2 always runs serial
  windows, and with serial windows `STRATA_SPLIT_OVERLAP=1` is active.
- **Decode:** DFlash2 is 5.5%, 6.9% and 16.7% slower than the mirror on the code, long and 16K prompts. Against MTP
  with the same pack and serial windows it is −1.7%, −3.2% and +5.9%. That is the drafter on its own, but it is not
  the production setting.
- **Prefill is faster with DFlash2 (+15 to +29% against the mirror), but DFlash2 is not the cause.** MTP with the same
  native pack and serial windows prefills exactly as fast (−0.7%, +0.6%, 0.0%). The gain comes from the native pack and
  from losing `--pipeline-windows` (which turns `STRATA_SPLIT_OVERLAP` off).
- **Side finding for Chat AI (not a production change):** the native pack prefills 16.5% (long) and 13.0% (16K) faster
  than the canonical pack, at the same decode, with MTP and `--pipeline-windows 2`.

## Part A

### The drafter pack

- **Source:** `incoai/Qwen3.8-Flash-Next-DFlash2`, at the revision vanilla pins (`a22a1744`, Apache-2.0, 1.2 GB,
  `hf download`). No licence step. The SHA-256 of `config.json` and `model.safetensors` match `strata_control/drafting.py`
  `FILES`.
- **Packing:** `tools/dflash2_pack.py --q4_0`, then `--convert-selector`. The packs match vanilla's pins (`PACK_SHA256`
  `cda40d62…`, `CONVERTED_SHA256` `4a42b982…`). Then I ran `drafting_fetch.ensure()`, offline because the pack is
  already there. It wrote the `.ok` marker and the draft vocabulary, as vanilla's fetch does. The safetensors source
  is deleted, as vanilla deletes it.
- **Location:** `${STRATA_DATA}/dflash2/`, outside both checkouts. This is the stock pack. The tuned pack comes only
  from Strata's signed feed, and vanilla does not use it on gfx1201 or Q2_0 (`SET_DEFAULTS`).
- **Native Q2_0 pack:** `${STRATA_DATA}/packs/q2_0-native`, built with `tools/iq_pack.py --gguf … --base packs/q2_0`.
  It holds an index and `native_experts.txt`; the experts are read from the GGUF.

### How DFlash2 was started

As vanilla's launcher applies a DFlash2 choice made by name (`"drafting": "dflash2"` in the run config;
`serve/server.py` `engine_args` → `drafting.apply`):
- the flags `--draft dflash2 --draft-model …q4_0-selq4.pack --draft-fallback none --mtp-draft-vocab
  …/mtp/rt/draft_vocab.bin --spec 6 --draft-conf 0.35`;
- no `--mtp` and no `--spec-min-p`;
- `--pipeline-windows` removed, because otherwise the launcher falls back to MTP ("DFlash2 isn't available in this
  mode").

The engine reports: "DFlash2 drafter on …; windows up to 8 tokens; no MTP layer is loaded; layer split: the drafter is
on CUDA1 (last stage), the other stages' taps are gathered onto it". `layer split auto` picks K=24 in every arm (canonical and native).

Configs: `hetero/configs/hiponly-native.example.json` (mirror on the native pack) and
`hetero/configs/hiponly-dflash2.example.json`. The serial MTP arm (mirror on the native pack without
`--pipeline-windows`) is `.ai/runs/0080-hiponly/hiponly-native-serial.example.json`.

### Table

Through the server: `serve_session.py … measure`, untraced, one fresh server per sample. Order: rounds of
mirror → native → DFlash2, then three native-serial samples. Values are the median, with the samples in brackets.
Generated tokens: 300 / 200 / 128. Built with `hetero/tools/session_table.py`; the same table is in
`.ai/runs/0080-hiponly/table.md`.

| Prompt | Config | Prefill tok/s | Decode tok/s | Drafts accepted / drafted |
| --- | --- | --- | --- | --- |
| code (118) | **Mirror** (canonical, MTP, pw 2) | 326.6 (326.6 / 328.8 / 201.4) | **113.1** (113.1 / 118.7 / 107.8) | 155/237, 152/224, 149/240 |
| code (118) | Native, MTP, pw 2 | 289.1 (289.1 / 203.6 / 296.3) | 114.0 (114.0 / 114.3 / 113.2) | 147/204, 148/208, 144/205 |
| code (118) | Native, MTP, serial | 378.9 (375.4 / 378.9 / 380.4) | 108.8 (108.8 / 107.3 / 122.4) | 168/255, 170/278, 192/285 |
| code (118) | **Native, DFlash2** (serial) | 376.3 (376.3 / 378.8 / 372.5) | **106.9** (109.3 / 104.5 / 106.9) | 170/260, 163/255, 162/226 |
| long (5,000) | **Mirror** | 2,392.7 (2,405.9 / 2,392.7 / 2,381.0) | **114.8** (107.2 / 126.3 / 114.8) | 104/179, 106/143, 106/144 |
| long (5,000) | Native, MTP, pw 2 | 2,788.5 (2,806.5 / 2,788.5 / 2,764.4) | 114.0 (112.5 / 114.0 / 125.4) | 107/179, 104/154, 109/145 |
| long (5,000) | Native, MTP, serial | 3,032.9 (3,003.0 / 3,032.9 / 3,045.6) | 110.4 (110.4 / 113.7 / 105.9) | 119/178, 119/160, 115/178 |
| long (5,000) | **Native, DFlash2** | 3,052.5 (3,063.4 / 3,052.5 / 3,033.6) | **106.9** (106.9 / 104.2 / 107.6) | 116/180, 113/179, 116/175 |
| 16K (16,102) | **Mirror** | 3,510.5 (3,521.9 / 3,510.5 / 3,494.7) | **137.5** (124.6 / 137.5 / 139.3) | 77/102, 75/102, 78/99 |
| 16K (16,102) | Native, MTP, pw 2 | 3,968.2 (3,981.4 / 3,966.2 / 3,968.2) | 129.0 (127.2 / 133.8 / 129.0) | 78/106, 81/107, 77/108 |
| 16K (16,102) | Native, MTP, serial | 4,511.2 (4,469.8 / 4,518.8 / 4,511.2) | 108.2 (108.2 / 109.2 / 105.1) | 76/120, 76/116, 76/133 |
| 16K (16,102) | **Native, DFlash2** | 4,512.0 (4,547.2 / 4,509.5 / 4,512.0) | **114.6** (114.6 / 119.6 / 112.3) | 78/108, 83/118, 80/127 |

**DFlash2 against the mirror:**
- **Decode:**
  - code −5.5% (DFlash2 104.5 to 109.3 against 107.8 to 118.7, overlapping at the edges, median clearly lower);
  - long −6.9%;
  - 16K −16.7% (112.3 to 119.6 against 124.6 to 139.3, no overlap).
- **Prefill:** +15.2%, +27.6%, +28.5%. MTP gets the same from the native pack and serial windows (native-serial row).
  This is not the drafter.
- **Drafts:** DFlash2 accepts a few more drafts per request than the mirror (about 165 against 152 on code; 115 against
  105 on long; 80 against 77 at 16K) at a similar rate (63 to 70%). Serial windows are dearer than the mirror's
  pipelined ones, so the extra accepted drafts do not pay. The DFlash2 suffix drafts (prompt lookup) added 10, 6 and 7
  windows with nearly all drafts accepted.

**Pipelined windows with DFlash2:** vanilla does not keep `--pipeline-windows 2`; it is serial (see the summary for
both refusals). `STRATA_SPLIT_OVERLAP=1` is then active ("one hand-off per window (overlapped)"). Under
`--pipeline-windows 2` the server says "STRATA_SPLIT_OVERLAP is off beside --pipeline-windows 2".

**The gfx1201 "prompt stall"** that vanilla's `drafting.AUTO_MTP` names as the reason for not using DFlash2 by default
on R9700 (0.2.11) did not occur. That is 3 of 3 starts and 9 of 9 prompts, too few to say it is gone.

### Decision

**No-go.** The decision rule asks whether DFlash2 is faster than MTP by more than the spread. On decode it is slower on
all three prompts. Its prefill lead over the mirror is real, but MTP shows the same lead with the same pack and window
mode, so it is not an advantage of the drafter. Part B was not implemented.

### Findings for Chat AI (not acted on)

1. **Native Q2_0 pack:** prefill +16.5% (long) and +13.0% (16K) over the canonical production pack, decode within the
   spread (MTP, `--pipeline-windows 2`, the same K=24). On code the first prefill is noisy in both arms (one 201 / 203
   sample each). Changing the production pack is the Human User's call. The layer split stays K=24; the expert path changes (GGUF via mmap instead of `experts.bin`). With 60 GB of RAM, the
   two packs cannot both stay in the page cache: one mirror start after native runs did not get ready within the
   65 s start window (below).
2. **Serial against pipelined windows (native pack, MTP):** serial prefills 8.8% (long) and 13.7% (16K) faster because
   `STRATA_SPLIT_OVERLAP` runs in prompt chunks. It decodes 3 to 16% slower, so it is a prefill against decode trade.
   Report 0078's mixed numbers are not comparable (another split and pack).
3. **Mixed mode on the native pack** has not been tried. The front stage would load layers `[0, K)` from the GGUF
   natively.

## Part B

Not started (no-go). Nothing to report on tap ownership, checkpoints or parks.

## Files

- New: `hetero/configs/hiponly-native.example.json`, `hetero/configs/hiponly-dflash2.example.json`,
  `hetero/tools/session_table.py`. `hetero/configs/README.md`: the two configs and the tool.
- No vanilla file changed. `check_off_is_vanilla.py`: 6 of 6. `mixed_split_tests` was not run, because nothing under
  `hetero/` that it covers changed.

## Safety and budget

- **Boot ID:** `86259517-dc02-4bae-be5f-ccc530423ae4` before and after. There was no pause.
- **Kernel log:** no GPU fault line. Five new `amdgpu_amdkfd_restore_userptr_worker hogged CPU` workqueue notices at
  09:36 and 09:43, during native-pack starts; the same notice is in the kernel logs of runs 0071 to 0073 and in reports 0068 and 0069. Evidence
  in `.ai/runs/0080-safety/`.
- **Core dumps:** none.
- **GPU time:** 581.7 s (`.ai/runs/0080-hiponly/gpu-seconds.txt`). Every run was sequential and under 2 minutes (the
  longest was 65.5 s).
  - The 581.7 s include the first DFlash2 attempt on the canonical pack (4.0 s, refused at start). I removed its run
    directory before rerunning; its refusal line is in `.ai/runs/0080-hiponly/engine-lines.txt`.
  - They also include one mirror start that was not ready within serve_session's 65 s, because the native runs had
    pushed the canonical pack's `experts.bin` out of the page cache. The server was stopped by SIGINT (rc −2, no
    fault). Its result is in `failed/`. I read `experts.bin` into the page cache (no GPU) and repeated the sample.
- **Process handling:** no process was killed or signalled other than the servers' SIGINT.

## Evidence

- `.ai/runs/0080-hiponly/`: `m-{mtp,nat,nser,df2}-{0,1,2}-result.json`, the filled configs, `table.md`,
  `engine-lines.txt` (key engine lines; the server logs are git-ignored) and `failed/`.
- `.ai/runs/0080-safety/`: boot ID, kernel log and core dumps, before and after.

## Open

- Whether to move production to the native Q2_0 pack (finding 1), and whether to measure mixed mode on it.
- `--batch` go/no-go (next per `.ai/STATUS.md`).
- Untracked leftovers from earlier turns in this worktree (`front-0066-1.*` and older `prompts.json` copies) are left
  alone.
