# Report 0059 — resident fail-stop HTTP integration

## Result

**PASS for focused CPU-only adapter, HTTP and ordinary service regressions.** Tested source `0762cd905ba8373334a7eef5611e78e286db0a54`; no implementation or fixture fixes were needed. No native build or GPU work was performed, as requested. This does not qualify native GPU failure recovery, high-position correctness or production readiness. Unrelated untracked `tools/heterogeneous_pool_probe.cpp` remains untouched.

## Checks

Read `.ai/TEST_BUDGET.md`; used checkout `.venv` and existing tools:

```sh
env PYTHONPATH=tools .venv/bin/python -m unittest -v serve.test_heterogeneous_resident serve.test_heterogeneous_retirement serve.test_heterogeneous_http_disconnect tools.test_heterogeneous_mtp_stream_contract
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_server.EngineDeath serve.test_server.StatusHandover serve.test_server.RestartWindow serve.test_server.SharingTheGpu serve.test_server.OpenAIChunksTerminal serve.test_server.PrepareCancelled serve.test_server.LostStep serve.test_server.ClientHangUp serve.test_lifecycle serve.test_restart_waiters serve.test_fatal_gpu_restart
```

Focused suite **41/41**, ordinary service/lifecycle suite **73/73**. Log files are `/tmp/handoff-0059-focused.log` and `/tmp/handoff-0059-service.log`. Fatal-GPU-restart cases in the latter are simulated CPU fixtures, not hardware faults.

The actual loopback synthetic-transport HTTP disconnect path cancelled at its gated checkpoint, drained the ACK, and admitted a queued recovery only afterward. The other actual loopback path injected a malformed resident completion through the real adapter: the SSE response contained an error, explicit operator-restart message and `[DONE]`; the adapter was poisoned. The following non-stream request returned **503**, explicitly refused automatic restart, sent no additional native command, invoked no restart and released its reservation. A separate service test checks healthy MTP admission and verifies `before_load` and restart are not called after failure. Adapter matrix covers ordinary opt-in/non-opt-in, MTP including stale opt-in and closed state; ordinary opt-in additionally verifies an old live process group prevents restart and an absent group permits launch. Generic service restart, handover, status, load hooks and error-message regressions passed.

Searched `serve` and relevant `tools` Python fixtures for `object.__new__(ResidentEngine)` / equivalent: the single fixture initializes `_restart_clean`; no additional missing fixture was found. Existing default batch policy and configs were not changed. No GPU run, native fault injection, forced kills, dependency/asset/driver/clock changes or GitHub Actions.
