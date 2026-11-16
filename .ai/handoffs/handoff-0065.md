# Handoff 0065

## Tier
2

## Objective
Run the vanilla HIP-only baseline that handoffs 0063 and 0064 could not start, and find out whether vanilla 0.2.15
accepts this machine's existing model files.

## Context
- Read `.ai/PROJECT.md` first. Its rule about the vanilla checkout now says what to do with a copied run config.
- Report 0064: all three trees build (`build-cuda13` with CUDA 13.0), the probe passes. The copied config
  `strata-q2_0.json` in the worktree differs from the vanilla one only in `exe`, and still names a `log` inside the
  vanilla checkout and that checkout as `cwd`.
- Continue in `/home/chris/workspace/Strata-mixed-native`. No code has changed since report 0064.

## Tasks
1. In the copied config, redirect `log` into the worktree. Set `cwd` to the worktree as well if the server or
   engine would otherwise create or change any file under the vanilla checkout; reading model, pack, profile and
   control-vector files from their existing places is fine. Change nothing else.
2. Start the server on the two R9700 with this branch's HIP build, the way `run-q2_0.sh` starts vanilla (same port
   and environment). Wait until it is ready.
3. Send the same short greedy chat request twice (about 200 new tokens; the first is the cold one). Then send one
   request with a prompt of about 4000 tokens and about 50 new tokens, for a prompt-reading rate.
4. Shut down cleanly and confirm no engine process and no GPU memory is left.

## Constraints
- You may change only the copied run config, as task 1 says.
- If the engine refuses a file (pack, draft layer, profile, control vector) or fails to start, report the exact
  message and stop; do not convert, rebuild or download anything. That refusal is the result.
- Do not change vanilla engine, kernel, server or Terminal files.
- GPU processes are never killed by force.

## Hardware Test Budget
5 minutes.

## Completion Criteria
- Either the three requests completed and the server shut down cleanly, or the start failed and the exact refusal
  is recorded.
- `.ai/reports/report-0065.md` is committed and pushed to `feature/mixed-native`.

## Reporting
- The keys changed in the copied config.
- Which engine and drafting mode vanilla chose (layer split, all-cards split engine, MTP or DFlash2), and the
  startup lines that show the layer split boundaries and any architecture defaults applied.
- For each request: prompt tokens, completion tokens, time to first token, prefill tokens per second, decode
  tokens per second and drafts accepted, as the server reports them.
- Memory in use and free per card while loaded.
- The engine's own startup line that lists the arguments the server expanded the config to.
