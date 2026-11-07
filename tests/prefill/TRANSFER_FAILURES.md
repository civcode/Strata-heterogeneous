# Expert-transfer failure checks

Build with `STRATA_BUILD_TESTS=ON`. The test controls are confined to test binaries and an explicitly loaded
Linux HIP interposer. They are not linked into or installed with the engine.

CPU-only checks (no GPU runtime):

```sh
ctest --test-dir BUILD -R '^transfer_' -E '^transfer_hip_' --output-on-failure
```

These compile the production `RemoteReadback`, `ExpertStager` and event-query gate against a delayed fake
runtime. Submission can queue work and still report an error. Assertions cover incomplete publication,
slot reuse, actual pinned/pageable frees, shutdown, clean drains and terminal allocation/submission bounds.
The Stager tests include multiple workers, batched claims and successive generations.

Linux HIP checks (only on an isolated, healthy machine):

```sh
HSA_ENABLE_SDMA=0 ctest --test-dir BUILD -R '^transfer_hip_|^remote_owner_' --output-on-failure -j1
```

`remote_owner_*` uses the real `RemoteExperts` and `RemotePrefillSource` with synthetic expert bytes on two
GPUs. It checks exact hit/miss data, clean allocation fallback, rejection, repeated failed calls, reopen refusal
and retention of both the cache arena and landing buffer. Reverse cases exchange primary/helper roles.
`transfer_hip_*` exercises the production landing/worker/query components. Synchronization-error wrappers
complete real valid work before returning synthetic errors. They do not induce hardware faults or prove that
a faulted device can recover. Fixture reclamation is separate from production cleanup.

`libtransfer_prefill_inject.so` can be explicitly preloaded into an isolated Q2_0 model process. Set
`STRATA_TRANSFER_INJECT=observe|remote-submit|remote-sync|stager-record|stager-wait`. It selects a 1,382,400-byte
expert transfer with the appropriate owned-host allocation flags, then logs `TRANSFER_INJECT selected`.
A missing selection is not coverage. Never preload it into a production service. Record/wait error modes
also report failed cleanup fences after real completion. No environment knob in the normal engine enables it.

Failure policy:

- Remote D2H writes land in a reusable helper-owned arena, never a borrowed destination. Only a positively
  completed, successful batch is copied to the caller. A clean landing-allocation refusal may use file fallback.
  A submitted-transfer failure rejects the batch and poisons the helper. Cache mutation/reopen is refused.
- Stager `issued` is a recorded-event watermark, not completion. Errors stop workers/issuers and reject prefill.
  Shutdown joins CPU workers; only a successful owning-stream fence permits freeing submitted host slots.
- A failed prefill can still reference borrowed cache/session VRAM. After the stage chain joins, callers must
  inspect `terminal_transfer_failure()`. The engine exits without stack unwinding on that terminal status,
  before cancellation/refill can reuse external storage. Library users cannot free raw loans merely because
  `run()` returned false. Independent Stager destruction protects its own sources, not an arbitrary destination.
- Q36 query errors take its existing fatal `CK` policy, before CPU routing reads. There is no in-process recovery.

Retention is bounded by the failed object's existing allocations, not by a process-wide quarantine cap. Do not
construct replacement owners indefinitely in a failed process. These checks do not complete the separate
PeerExperts direct-buffer, RemoteExpertOpt, generic ExpertCache startup, or CUDA async snapshot audits.

Phase 10D validation stopped on a real MES `INVALIDATE_TLBS` timeout during a broad SDMA-off run. Do not infer
hardware stability, final GPU validation, Q36 model coverage or a performance sign-off from the focused tests.
