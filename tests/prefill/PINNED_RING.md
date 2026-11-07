# Pinned-ring regression tests

`pinned_ring_test` compiles the real ring and host-grouping gate against a CPU-only runtime.
Failed submissions can leave work queued; failed waits leave it pending. The fixture detects freeing
or resubmitting such storage. It covers rotation, growth, release, destruction, poisoned retries,
allocation failure, H2D/D2H and the grouping consumer/cleanup gate. No device fault is induced.

`pinned_ring_gpu_test` checks exact real-runtime transfers on visible discrete GPUs, including the
64 KiB boundary and repeated slot reuse. Run under both `HSA_ENABLE_SDMA=0` and `1`.

## Real prefill error propagation (Linux HIP, model required)

The build also produces `libpinned_ring_prefill_inject.so`. It is a test-only interposer, never linked
into or installed with the engine. Use an isolated model process and normally the corrected engine;
the safe sync-only baseline exception is described below.

Start the normal model with `--prefill 1280 --expert-cache 1024` and these **per-process** settings:

```sh
LD_PRELOAD=/absolute/build/path/libpinned_ring_prefill_inject.so \
STRATA_GROUP_COPY=1 STRATA_PF_FUSED=0 HSA_ENABLE_SDMA=1 <engine command>
```

Send a text prompt longer than 1280 tokens. With top-k 10, the routed-ID download is 51,200 bytes into
a ring allocation of 65,536 bytes. The interposer logs its match. It reports a synchronization error
only after the real stream synchronization succeeds. The request must fail with
`prefill: the routed ids' download failed`, without generating an answer. The ring reports retaining
65,536 bytes. No actual GPU fault is induced.

In a **fresh process**, add `PINNED_RING_INJECT_SUBMIT=1` to report an error after real enqueue instead.
The corrected ring must drain the submitted work, return failure and not publish the IDs. Do not use
that mode on an old engine whose failed submission skips the drain.

To exercise failed teardown as well, add `PINNED_RING_INJECT_STICKY=1` in a fresh sync-error process.
Subsequent waits also report errors only after real completion. The engine must log that it retains
the staging owners until process exit; it must not destroy `Prefill::Impl` after those failed fences.
The interposer also logs any attempt to free the selected ring slot after the reported sync error;
that line must be absent from the corrected engine's log. A sync-only baseline comparison is safe
because real work was completed first; never use submission-error mode on the original engine.

For the healthy control, use `PINNED_RING_OBSERVE_ONLY=1`. It logs the first matching transfer without
changing an API result. A missing match means no integration coverage, not a passing failure test.
The shape is specific to this fixture; other model top-k values/chunk sizes need a corresponding test.

The interposer validates the real caller's error propagation. Unknown outstanding work and all cleanup
ownership assertions are tested independently in the CPU model, not by destabilizing hardware.
