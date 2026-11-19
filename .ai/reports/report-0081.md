# Report 0081

**Status: Completed, no-go.** Parts A and C done; part B not started. The CUDA card has room for two slot sessions at
131,072 tokens, but on HIP-only `parallel` 2 with `--batch-groups 1` decodes **40% less** in aggregate than `parallel`
1, far from the +20% the rule asks. No engine source changed.

## Summary

- **Throughput (HIP-only, production mirror, two clients, code then long, three rounds):** aggregate decode tok/s,
  median of three sessions:
  - `parallel` 1: **116.7**;
  - `parallel` 2, vanilla's default (two pipelined groups of one slot): 105.5 (−9.6%);
  - `parallel` 2, `--batch-groups 1`: **69.7 (−40.3%)**;
  - `parallel` 2, `--batch-groups 1 --batch-mtp`: 79.8 (−31.6%).

  Load wall time (end to end, prefill included): only the default groups are faster than `parallel` 1 (35.3 against
  38.9 s, aggregate output +10%).
- **Why:** with `--batch`, vanilla turns `--pipeline-windows` off ("not with --batch slots"). Without `--batch-mtp`
  the slots also draft nothing, so each slot gets one token per step. Two serial slots without drafts lose to one
  conversation with MTP and pipelined windows.
- **CUDA room:** one slot session for layers 0 to 14 is 0.98 GiB at 262,144 tokens (int8 KV) and 0.51 GiB at 131,072.
  The front stage leaves 1.45 GiB of the 16 GB card free. Vanilla's `--batch 2` carves two slot sessions beside the
  solo session: 1.96 GiB at 262,144 tokens does not fit, 1.02 GiB at 131,072 does. The room half of the rule passes;
  the throughput half fails.
- **Native pack in mixed mode (part C):** the CUDA front stage accepts it. Prefill +23.7% (code), +15.6% (long),
  +2.7% (16K). Decode medians are 5 to 8% lower, with overlapping sample ranges.

## Part A

### Throughput

Driver: new suite `serve_session.py … concurrent`. Two client threads start together. Each sends, streamed and
greedy, the code prompt (300 tokens) and then the long prompt (200 tokens), three rounds; every request has its own
prefix. Untraced, one fresh server per sample, three samples per config in rotation (p1, p2, p2g1, p2g1m). Built with
`hetero/tools/concurrent_table.py`; also in `.ai/runs/0081-hiponly/table.md`.

- **Aggregate decode tok/s:** generated tokens divided by the wall time during which at least one request was between
  its first and last token. With `parallel` 1 this is the decode rate alone.
- **Aggregate output tok/s:** tokens divided by the whole load's wall time.

| Config | Aggregate decode tok/s | Aggregate output tok/s | Load wall s |
| --- | --- | --- | --- |
| `parallel` 1 (the mirror) | **116.7** (116.7 / 124.3 / 115.5) | **77.2** (77.2 / 80.0 / 76.3) | **38.9** (38.9 / 37.5 / 39.3) |
| `parallel` 2, default groups (2 pipelined groups of 1) | **105.5** (106.2 / 105.3 / 105.5) | **84.9** (85.7 / 84.9 / 84.7) | **35.3** (35.0 / 35.3 / 35.4) |
| `parallel` 2, `--batch-groups 1` | **69.7** (70.5 / 69.7 / 69.5) | **69.1** (69.8 / 69.1 / 69.0) | **43.4** (43.0 / 43.4 / 43.5) |
| `parallel` 2, `--batch-groups 1 --batch-mtp` | **79.8** (86.0 / 79.8 / 79.2) | **79.0** (79.6 / 79.0 / 78.5) | **38.0** (37.7 / 38.0 / 38.2) |

Per request: medians over the 18 requests of each kind (3 sessions × 2 clients × 3 rounds). Client decode is measured
from the first to the last streamed token. First token is client-side and includes the wait for the other client.

| Config | Prompt | Decode tok/s per request (min to max) | Server decode tok/s | First token ms (min to max) |
| --- | --- | --- | --- | --- |
| `parallel` 1 | code | 116.0 (108.0 to 126.0) | 115.7 | 3,186 (385 to 3,890) |
| `parallel` 1 | long | 121.4 (114.4 to 130.7) | 120.8 | 5,224 (4,541 to 5,754) |
| `parallel` 2, default | code | 68.3 (65.2 to 71.3) | 69.0 | 564 (281 to 878) |
| `parallel` 2, default | long | 54.6 (40.5 to 68.6) | 55.5 | 2,995 (1,868 to 4,148) |
| `parallel` 2, `--batch-groups 1` | code | 43.5 (36.8 to 49.0) | 43.8 | 299 (293 to 777) |
| `parallel` 2, `--batch-groups 1` | long | 39.4 (33.1 to 50.2) | 39.9 | 1,883 (1,847 to 2,171) |
| `parallel` 2, `--batch-groups 1 --batch-mtp` | code | 56.9 (40.5 to 64.4) | 57.5 | 308 (300 to 753) |
| `parallel` 2, `--batch-groups 1 --batch-mtp` | long | 46.4 (35.0 to 65.4) | 47.1 | 1,896 (1,875 to 4,141) |

What vanilla does with `parallel` 2 on this split (from the server logs; key lines in
`.ai/runs/0081-hiponly/engine-lines.txt`):
- "--pipeline-windows 2 is off: not with --batch slots".
- The default: "--batch-groups auto: 2 groups of 1 slot", with no slot drafts.
- `--batch-groups 1` alone: one token per slot and step. The conversation cache says "slot N not parked before its
  overwrite (… no per-slot draft K/V without --batch-mtp)".
- `--batch-mtp`: "2 slot drafters on CUDA1 (313 MiB … each)".
- The slot sessions on the R9700s are 1.86 GiB each for 24 layers at 262,144 tokens.

`parallel` 2 shortens the time to first token, since nobody waits for the other conversation's decode. Aggregate
decode falls in every `parallel` 2 config.

### CUDA card room

From vanilla's own sizing (`session_bytes`, printed by `--batch` as "slot sessions on CUDA0 (X GiB each)"):
- Two start-only sizing runs, HIP-only with `layer_split` 15, so that stage 0 is exactly the front stage's range
  `[0,15)`: 0.98 GiB per slot session at 262,144 tokens and 0.51 GiB at 131,072. That is about
  0.04 + 0.47 × (context / 131,072) GiB. `session_bytes` depends only on the geometry, the layer range and the context,
  not on the backend.
- The mixed mirror's front stage: "expert cache 7680 slots, 9.89 GiB" (all of its pairs), "1591 MiB of VRAM free with
  everything loaded". `nvidia-smi` during the run: 14,457 of 16,376 MiB used, **1,487 MiB free**. The card idles at
  15 MiB.

| Extra on the front card | Size | Fits in 1.45 GiB free? |
| --- | --- | --- |
| One more session, 262,144 tokens | 0.98 GiB | yes (0.47 GiB left) |
| Vanilla's `--batch 2`: two slot sessions beside the solo one, 262,144 tokens | 1.96 GiB | **no**, unless the expert cache gives up about 0.5 GiB (about 400 of its 7,680 pairs) |
| Two slot sessions, 131,072 tokens (solo session still at 262,144) | 1.02 GiB | **yes** (0.43 GiB left) |
| Two slot sessions, everything at `--max-context` c | 3 × s(c) ≤ 1.45 + 0.98 | up to about 210,000 tokens (estimate, no margin) |

Not counted: `--batch` turns pipelined windows off, which frees the front stage's second verifier and its GDN
snapshots (2 × 37.4 MiB). The front stage's expert cache is sized after the sessions, so a tight fit would cost a few
resident experts rather than fail.

### Decision

- **+20% aggregate decode with `--batch-groups 1`:** **not met** (−40.3%; −31.6% with `--batch-mtp`). Even vanilla's
  pipelined default is −9.6% in decode, +10% end to end.
- **Two sessions at ≥ 131,072 tokens on the CUDA card:** met.

**No-go.** Part B was not started.

## Part C: the native pack in mixed mode

`hetero/configs/mixed-native.example.json`: the mixed mirror with `--pack ${STRATA_DATA}/packs/q2_0-native`. The front
stage gets the host's `--pack`. The CUDA build accepts it: "native pack: … experts (largest blob 1.38 MB), token
embedding Q3_K in mapped host memory". `layer split auto` picks K=31, as on the canonical pack.

The run went through the server (`serve_session.py … measure`), untraced, alternating mirror / native, four samples
each. Before each session, the arm's main file was read into the page cache (no GPU): `experts.bin` (34 GB) for the
mirror, GGUF shard 1 (37.6 GB) for native. 60 GB of RAM does not hold both. The first sample of each arm is reported
apart (`.ai/runs/0081-mixed/first/`) and left out of the medians: the mirror's first code prefill was 97 tok/s and
native's first long prefill 2,309 tok/s.

| Prompt | Config | Prefill tok/s | Decode tok/s | Drafts |
| --- | --- | --- | --- | --- |
| code (118) | Mixed mirror (canonical) | **209.4** (209.4 / 236.9 / 197.5) | **132.8** (132.8 / 150.1 / 126.9) | 158/222, 177/244, 146/209 |
| code (118) | Mixed, native pack | **259.0** (261.3 / 259.0 / 231.9) | **122.0** (129.1 / 120.0 / 122.0) | 160/234, 149/212, 146/200 |
| long (5,000) | Mixed mirror | **2,316.3** (2,316.3 / 2,318.8 / 2,311.9) | **131.6** (124.3 / 132.4 / 131.6) | 105/181, 107/152, 106/145 |
| long (5,000) | Mixed, native pack | **2,676.9** (2,676.9 / 2,679.1 / 2,665.0) | **124.9** (124.9 / 130.2 / 123.8) | 109/165, 110/156, 105/144 |
| 16K (16,102) | Mixed mirror | **3,897.2** (3,789.3 / 3,903.5 / 3,897.2) | **152.2** (137.5 / 157.8 / 152.2) | 77/108, 81/107, 81/108 |
| 16K (16,102) | Mixed, native pack | **4,002.3** (4,002.3 / 4,003.1 / 3,999.2) | **141.0** (141.0 / 143.1 / 139.0) | 78/106, 80/114, 74/94 |

- **Prefill:** +23.7% code, +15.6% long, +2.7% at 16K. On HIP-only (report 0080) it was +16.5% long and +13% at 16K.
  In mixed mode the 16K gain mostly disappears.
- **Decode:** −8.1%, −5.1%, −7.4% on the medians. Every pair of sample ranges overlaps, and the mirror's decode
  samples spread widely (126.9 to 150.1 on code). A small decode cost is possible but not established with three
  samples.
- The mixed native pack against HIP-only native (report 0080, same prompts): long prefill 2,677 against 2,789, 16K
  4,002 against 3,968; decode 124.9 against 114.0 (long), 141.0 against 129.0 (16K).

## Files

- `hetero/tools/serve_session.py`: suite `concurrent` (`--clients`, `--rounds`).
- New: `hetero/tools/concurrent_table.py`, `hetero/configs/mixed-native.example.json`. `hetero/configs/README.md`
  updated.
- The A/B arm configs are in `.ai/runs/0081-hiponly/` (`hiponly-p2`, `-p2g1`, `-p2g1m`, the two `-ls15` sizing configs).
- No vanilla file changed. `check_off_is_vanilla.py`: 6 of 6. I did not run `mixed_split_tests`: nothing it covers
  changed.

## Safety and budget

- **Boot ID:** `86259517-dc02-4bae-be5f-ccc530423ae4` before the first run and after the last (10:17). Then I hit a usage
  limit. At the resume the boot ID was `5242f948-6a93-452b-b79e-a43f2d1afa57`: the machine had restarted twice in the
  meantime. **The Human User confirmed the reboots were theirs and unrelated to this project.** Per `.ai/PROJECT.md` I
  ran nothing more on the GPU anyway (none was left to run). The rest of this turn was writing and committing only.
  - 11:52 in my boot: `strata` (named `strata-host`, general protection fault in libc, SIGSEGV) and `strata-vision-hip`
    (SIGABRT) dumped core. Both binaries are from **another install, `/home/chris/tmp/Strata/engine/`**, not from this
    checkout's builds, and the crash came 1.5 h after my last run. Two `node` SIGSEGVs (11:55, the agent harness) and one
    `tmux` SIGABRT (11:56) followed. Then a clean reboot at 11:59.
  - Boot `1bdba53a` (12:00 to 12:24) was not mine. At 12:20 the kernel logged **NVRM Xid 79, "GPU has fallen off the bus"**
    on the RTX 4070 Ti SUPER (PCI 04:00, under an `nvidia-smi` process), then `nvidia-modeset` errors until the reboot
    at 12:24. Only the passive `strata-gfxhub-recorder` service from `Strata-lab` is visible in that boot's log.
  - Evidence: `.ai/runs/0081-safety/` (`boot-after-pause.txt`, `kernel-boot-2-after-last-run.log`,
    `kernel-prevboot-after-snapshot.log`, `prevboot-tail.log`, `cores-after-pause.log`).
  - Noted only because the Xid 79 hit the front stage's card. It came from outside this project, during the Human User's
    own work.
- **Kernel log:** no GPU fault line. There are new `amdgpu_amdkfd_restore_userptr_worker` and `svm_range_restore_work`
  "hogged CPU" workqueue notices; both appear in the runs of 0071 to 0073 and in reports 0068, 0069 and 0075.
- **Core dumps:** none during the runs. The 11:52 to 11:56 dumps are described above.
- **GPU time:** 956.3 s (15.9 min), sequential, every run under 2 minutes (longest 69.1 s):
  - HIP-only: 640.3 s (`.ai/runs/0081-hiponly/gpu-seconds.txt`);
  - mixed: 316.0 s (`.ai/runs/0081-mixed/gpu-seconds.txt`).

  No run failed. My first `nvidia-smi` sampling call went wrong in the shell (the backgrounded chain took the `cd` with
  it). The server never started in that call, so it cost no GPU time.
- **Process handling:** no process was killed or signalled other than the servers' SIGINT.

## Evidence

- `.ai/runs/0081-hiponly/`: `c-{p1,p2,p2g1,p2g1m}-{0,1,2}-result.json` (every request's times and timings),
  `size-ls15*-result.json`, the filled configs, `table.md` and `engine-lines.txt`.
- `.ai/runs/0081-mixed/`: `m-{mir,nat}-{1,2,3}-result.json`, `first/` (samples 0), `size-mirror-result.json`,
  `nvsmi-mirror.txt` and `table.md`.
- `.ai/runs/0081-safety/`.

## Open

- `--batch` in mixed mode is not worth building for throughput on this rig. If faster first tokens for a second
  client matter more than aggregate speed, vanilla's pipelined default would be the variant to revisit. It needs the
  front stage inside the pipelined groups, about +2 turns per report 0075.
- The native pack: on HIP-only it wins on prefill at every length. In mixed mode it wins on the code and long prompts,
  hardly at 16K, and decode is not settled. Deciding a switch would need more decode samples.
