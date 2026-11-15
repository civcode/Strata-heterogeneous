# Bounded pipeline and guarded direct handoff (M5–M7)

The existing 256K qualified path remains the default. See reports 0044 and
0045 and the binding test policy in .ai/TEST_BUDGET.md.

## Pipeline profiles (experimental)

The resident CLI accepts `--pipeline-mode baseline|fast`. The HTTP profile
can explicitly set `"pipeline_mode": "baseline"` or `"pipeline_mode": "fast"`
inside `heterogeneous`. Both require the new native coordinator and matching
workers. Omission retains prior environment-controlled behavior, so existing
launchers/profiles are not changed. An old coordinator refuses explicit mode.

Baseline explicitly disables compact results, fast fence polling and fanout.
Fast enables native compact results (prediction + nonfinite guard instead of
full vocabulary result), 50-us fence polling, fanout of commit/reset and, for
the two-domain mixed topology, early final-window metadata dispatch. The
downstream worker can begin metadata admission while CUDA executes. It still
waits for the normal matching finite/checksummed activation before doing GPU
work. All endpoints must acknowledge before publishing committed output.
These are candidate latency optimizations, not measured gains on the current
revision. Full diagnostic logit captures remain on their dedicated path.

The resident benchmark summary accepts `--axis pipeline` on matched explicit
baseline/fast A/B/B/A artifacts. KV and topology comparisons must hold the
pipeline selection fixed. Historical omitted modes are labeled legacy and
cannot enter the explicit pipeline comparison.

## Guarded GPU publication prototype

The new guarded publisher checks FP32 exponent bits while copying the original
R/bo/inj payload, accumulates each block's failure status with the completion
counter, and publishes 1 only for an entirely finite completed payload; 2
signals rejection. The bounded checked wait accepts exactly 1. Invalid,
withdrawn or missing publication latches a consumer-local failure bit. The
existing conditional copy kernel then substitutes zeros for ALL payload
segments rather than reading stale/nonfinite input. A rejected window must
be drained and permanently discarded. Zero quarantine is not usable model
state or permission to commit.

The model-free direct graph probe now supports `--guarded` and separate
`--exercise-timeout` / `--exercise-nonfinite` terminal cases, with bit-exact
zero readback on both HIP devices. Neither this primitive nor the probe is
enabled in live model execution. GPU compiler, publication/coherency, actual
cross-runtime prelaunch and safe native retirement must be qualified before
live integration. In particular, do not treat ready 2 or UINT32_MAX as a
successful publication, and never re-use a slot after terminal rejection.

## Short HTTP checks on an already running server

From the repo using the existing Strata venv:

```sh
python -m tools.heterogeneous_quick_smoke --output /tmp/strata-quick-smoke-unique.jsonl
```

Defaults: local port 8081, model qwen3.8-flash-next-q2_0, one pass, 90-second
admission budget, 20-second request timeout, <=64 output tokens per request.
No server launch, restart, real tool execution, repeated benchmark or long
context. Run only with permission to use that server. The wall budget stops
admitting new HTTP work; an in-flight operation may exceed it. Evidence
records incomplete/skipped cases, transport and answer checks separately.
A client cancellation is not a native cancellation ACK. Fixture quality
failures are observations, not proof of new numerical drift. A passing
smoke is not production admission or proof of a context-capacity bound.
