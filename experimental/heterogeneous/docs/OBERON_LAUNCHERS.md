# Oberon: start Q2_0 heterogeneous Strata for HTTP chat

These Linux launchers follow the user's known-good shell + JSON setup and use the existing Strata HTTP service with the resident native coordinator. No changes to the productive Strata installation, no GPU build or dependency installation during launch.

After the active GPU benchmark has finished, run from this checkout:

```sh
./run-q2_0-heterogeneous.sh
# OR, after the first server and its workers have exited normally:
./run-q2_0-hip-only.sh
```

Both listen at http://127.0.0.1:8081, use model ID `qwen3.8-flash-next-q2_0`, and run in the foreground. Stop with a normal Ctrl+C and verify the native workers exited before switching topology. Do not run either while any other Strata engine or GPU benchmark is in progress. The shared lock serializes these two launchers; process detection is best-effort, not global GPU reservation. No processes are killed or stopped by a launcher.

For CPU-only validation:

```sh
./run-q2_0-heterogeneous.sh --prepare-only
./run-q2_0-hip-only.sh --prepare-only
```

The output path is a private vanilla-style run config (`exe`, `args`, `cwd`, `tokenizer`, `model_name`, `log`, `port`) with a `heterogeneous` topology extension. The launch uses the normal `python -m serve.server --engine strata --config ...` entry point and the same HTTP Service. The existing qualification JSON syntax remains readable for old benchmark and diagnostic tools. Native stderr is appended to the configured engine log. The output path is under `~/tmp/strata-heterogeneous/launchers/`. In normal mode the server runs its existing CPU source/config preflight before native worker startup. For a separate checkout use `STRATA_PYTHON=/absolute/path/to/existing/.venv/bin/python`; do not create a new venv. The scripts default to the current checkout's existing .venv. To change the local port, append e.g. `--port 18081`.

## Settings and local paths

Edit the root JSON launch profiles `strata-q2_0-heterogeneous.json` and `strata-q2_0-hip-only.json` (for example, with vi). They intentionally omit machine-specific executable and stage config paths. The launcher binds those paths from the real qualified server JSON from [report 0044](../../../.ai/reports/report-0044.md), discovered by recorded SHA-256 within `~/tmp/strata-heterogeneous/handoff-0044/`. It does not select arbitrary recent configs or invent binary/model paths. If private qualification artifacts have been relocated, pass `--reference-config /absolute/path/to/qualified-server.json` to either launcher. This override must still match the chosen topology and settings. The real native source/config preflight checks model binding, capacity, worker configuration and actual capability; the launcher does not attest that binary contents have not changed since 0044. Requalify changed builds before relying on the hardware results.

The Q2_0 pack and full-resident worker files are reused rather than repacking IQ3_XXS. The user's original configuration supplies the 256K, INT8 KV, ROCm core-10.1 and port-8081 reference. Qualified settings retain mixed 15/18/15, HIP-only 24/24, native prefill 128 and >=1 GiB per-owner reserve (in actual stage files), not the vanilla 700 MiB reserve. The shell retains HSA_ENABLE_SDMA=0, STRATA_KEEP_LEFTOVERS=1 and STRATA_NO_PRESET=1; trace variables are cleared for normal chat and STRATA_NO_HANDOVER=1 is set. The original unlimited core-dump setting is not applied. The shared option names `--max-context`, `--kv` (vanilla spelling `int8` or `fp16`), `--prefill` (explicit integer only), `--pack`, `--native`, `--ple-gguf` and `--expert-profile` are used in `args`. Model paths are checked against the bound entry stage, and unsupported engine flags are rejected rather than ignored. MTP, vision, non-greedy sampling, control vector, prompt/conversation cache, and experimental direct overlap are not qualified in this mode. The native protocol and its diagnostic/request-summary text are not yet identical to vanilla. The existing heterogeneous HTTP path rejects --open and is loopback only, so open a client manually.

The launch profiles request up to 2048 completion tokens (editable within 1..4096). This is an operational choice, not a previously qualified output length at 256K. Report 0044 qualified four output tokens, repeated long requests and native lifecycle; HTTP at 256K remains unqualified.

## Test from another terminal

```sh
curl http://127.0.0.1:8081/v1/models
curl http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3.8-flash-next-q2_0",
    "messages":[{"role":"user","content":"Explain how a binary search works."}],
    "temperature":0,
    "max_tokens":256,
    "stream":true
  }'
```

Use a bearer key if you configured one via the service's supported API-key settings. Start with a short text request and extend cautiously; production HTTP readiness and long-context HTTP operation have not yet been demonstrated. The launch helper's isolated CPU tests and three shell syntax checks passed; real Oberon preflight, actual launch, HTTP replies and shutdown require local validation. No GitHub Actions.
