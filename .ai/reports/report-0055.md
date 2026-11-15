# Report 0055 — HTTP disconnect and ordered cancellation drain

## Result

**PASS for real loopback HTTP disconnect at a genuine native bulk checkpoint, ordered cancellation drain, queued recovery and cooperative retirement**, using short requests with 262144 configured cells. Tested source `bc3817ea4a57c57363b4f919acae1ec64943161e`; no production or fixture corrections were needed. No high-position correctness, native GPU fault recovery, transient peak or production claim. Preserved unrelated untracked `tools/heterogeneous_pool_probe.cpp`.

## CPU verification

Read `.ai/TEST_BUDGET.md`. Reused report-0054 matching native binaries: this handoff changes only Python HTTP handling, tests and docs, not native source. Existing checkout `.venv`:

```sh
env PYTHONPATH=tools .venv/bin/python -m unittest -v serve.test_heterogeneous_http_disconnect serve.test_heterogeneous_resident serve.test_heterogeneous_retirement tools.test_heterogeneous_mtp_stream_contract
env PYTHONPATH=tools .venv/bin/python -m unittest serve.test_server.ClientHangUp serve.test_server.StatusHandover serve.test_server.ConnectionClose serve.test_server.OpenAIChunksTerminal serve.test_server.MaxTokens serve.test_server.UnfinishedToolCall serve.test_responses serve.test_structured serve.test_mcp
```

First suite **37/37**; second **119 tests, OK (3 skipped)**. The first includes actual loopback synthetic transport with gated checkpoint and ACK, queued second HTTP request, and retained OpenAI source closure; adapter and retirement tests cover cancel, failure and group verification. The second covers existing OpenAI/Anthropic/Responses endpoint semantics, disconnect/FIFO, connection close, structured output and MCP. Source inspection confirms SSE handler `finally` closes encoders, OpenAI wrapper closes its retained source, structured wrapper closes its nested chunks, and Responses/Anthropic encoders close Service iterators. Service holds its request reservation while the adapter's cancel ACK drains. These tests do not prove every network timing or every client implementation.

## Real native HTTP sequence

Fresh process/listener and NVIDIA/AMD compute checks showed no active Strata listener or GPU compute process; AMD 0/1 each used 57 MB. Snapshot is not a reservation. Reused report-0054 private 262144-cell HIP 24/24 profile/config unchanged, Q2_0/Q8 `/home/chris/workspace/Strata-data/mtp/rt`, max_t 8, INT8 KV, full experts, >=1 GiB reserve, chunk 32, spec 4/min-p 0.5, workload and long-context enabled, `STRATA_*` cleared and `HSA_ENABLE_SDMA=0`. No existing service contacted. Private driver, raw records and logs: `/home/chris/tmp/strata-heterogeneous/handoff-0055/`.

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0055/http_native.py
```

The in-process private loopback driver wraps the real adapter receive/send only for observation and event gates; it returns the real native records unmodified and does not fabricate cancellation. Sequence:

1. Actual POST socket request 1 entered native prefill with 59 prompt tokens. Real coordinator PID 1713980 emitted `HETERO_PROGRESS 1 1 58 BATCH 0 0`; the adapter receive wrapper held this record before returning it to `generate()`.
2. The actual client TCP socket was shut down and closed. The HTTP watcher incremented `client_disconnect_cancels` to 1. Only then did the driver release the checkpoint; the adapter sent `cancel` and the real coordinator returned `finish_reason=cancelled`, `prompt_tokens=59`, `token_ids=[]`. The first client's HTTP status was not read before disconnect.
3. The real ACK was held before delivery to the adapter. A second actual HTTP request queued; the Service reservation and adapter lock remained held, and the native command log contained only request 1 and its `cancel`—no request 2 command. The ACK gate was then released.
4. Request 2 ran on the **same coordinator PID**, with 55 prompt tokens and output cap 8. HTTP **200 text/event-stream** ended with finish `length` and `[DONE]`, with no error. Native record had 8 final IDs and progress positions 54,55,57,62; 8 proposed/5 accepted. Its short response contained reasoning text rather than a final answer, consistent with the explicit length cap. Highest committed GPU position was **62**.

The driver stopped its own HTTP listener, invoked cooperative `ResidentEngine.close()`, checked exit code 0, joined reader and independently verified the original process group absent. The native log ended `HETERO_CLOSED`; final process and compute checks were empty. No mid-batch disconnect was attempted. Wall time from native launch through retirement **17.54 s**.

Both owners reported 12288/12288 target experts, resident INT8 KV and zero host expert routes/misses. HIP1 draft admission logged context bytes 420,482,832, free 10,110,369,792, estimate 3,465,350,928 and base reserve 1,073,741,824 bytes; first terminal prefill allocation left 7,876,902,912 bytes against the 1,610,612,736-byte augmented reserve. These are snapshots, not measured transient peaks. No actual long/high-capacity prompt, forced kill, injected GPU failure, dependency/asset/driver/clock changes, spill, reserve reduction or GitHub Actions. Broader HTTP failure timing and high-position behavior remain unverified.
