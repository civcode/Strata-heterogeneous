# Report 0085

**Status: Partial. Tier escalation recommended.** Fresh builds and CPU seam tests are complete; the session driver, stub test and all GPU qualification remain undone. No engine or front-stage code was changed. This is not an RC qualification result.

## Builds and configuration

- Reset to `origin/feature/mixed-native` at `fcdc3ed67`. Release/Ninja builds from that revision succeeded with 28 jobs: `build-0085-cuda` (CUDA 13.0, sm_89), `build-0085-hip` (ROCm 10.1, gfx1201, both MMQ options on), `build-0085-vision-cpu`, and `build-0085-vision-hip` (gfx1201). All four have `BUILD_COMMIT` stamps at that revision. HIP configuration needed `CMAKE_PREFIX_PATH=/opt/rocm/core-10.1`.
- `mixed_split_tests` passed 1/1 in each of the CUDA and HIP build trees; these are CPU/mock seam tests, not GPU tests. Build and test logs are local under `.ai/runs/0085-build/`.
- The two production mirrors now point to these four builds. Their arguments were not changed; `--max-context` was already 262144. Both parse as JSON. Launch checks and runtime verification were not performed.

## Unfinished qualification

The requested driver, table script and stub test were not implemented. In particular, the correctness-probe design needs a reliable way to observe the **actual generated token IDs** through the 0.2.15 server without changing the engine or front stage: the engine protocol carries `T <id>`, but the server's OpenAI logprob entries in `serve/server.py` expose token text and bytes, not the ID. Re-tokenizing the final text is not a demonstrated equivalent, especially for control tokens and tool/reasoning parsing. A server-side capture or other approach within the handoff's permitted new-tool scope needs design and validation; no proof that such an approach is impossible is claimed. Tier escalation is recommended for this unresolved integration design and the remaining substantial driver work.

The server does have `/v1/messages/count_tokens` for rendered Anthropic requests, but equivalence to the requested OpenAI/pi form needs checking. Reused-token collection, request identity, stub bookkeeping and probe replay remain unverified. No partial driver is presented as functional.

- Task 3 rename/vision/exactness checks: **not run**.
- Scaled sessions and full sessions, including probe replays: **not run**. No comparison tables exist.
- RC criteria: stability **not measured**; correctness **not measured**; resources **not measured**; performance **not measured**. The RC2 soak also remains entirely open.
- GPU runtime: **0**. No GPU workload was started, so the before/after GPU safety snapshots were not taken; no claim is made about fault lines or coredumps. No Actions were run. The pre-existing untracked `tools/heterogeneous_pool_probe.cpp` was left untouched.
