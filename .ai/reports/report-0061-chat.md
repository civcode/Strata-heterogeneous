# Report 0061 (Chat-side source increment) — PARTIAL

Branch starting revision `32d3ff45dd37f56d1d9f51651d9f8ba80599d6d6`.
Exact pinned vanilla commit `8db7ddc46cd2575514408c4ab7dd878f69df2f5f`
was reported available on Oberon in report 0060, but the GitHub tree request
returned 422 in the previous audit. This container's git ls-remote fails
DNS resolution of github.com; no full local repo or GPU environment. We
used exact GitHub branch file contents, the native ancestor
`146df3c468739147f520ac3816de806f71232571` and prior pinned-source
findings, not an inferred alternative algorithm.

## Implementation

- Native setup's two server argv builders extracted unchanged into one
  shared server_command; the mixed launcher calls the same function and
  adds the existing offline argument. Both checked-in profiles carry
  port=8081/parallel=1. Old --engine heterogeneous remains available
  for qualification; public --engine strata selects mixed runtime via
  explicit extension. Legacy profile fields NOT yet migrated into args.
- Native shared environment/library path/offline code, child working
  directory, log and native browser path reused. The mixed coordinator
  keeps backend-local GPU selection/defaults until native per-worker
  extraction can be validated. Normal log routing does not consume
  resident protocol; raw qualification trace mode remains explicit.
- Native model-loading/heartbeat narration shared; actual staged
  expert-cache/startup events and incremental prompt progress remain
  to be integrated. No fake vanilla output. No forced worker kill.
- Original shared HTTP service/front-end only; existing resident adapter
  remains with its feature restrictions. No claim of full HTTP parity.
- No inference algorithm, spec width, precision, performance flag,
  capacity, cache, worker identity or partition changed. In particular,
  pinned report 0060 states spec=4 legacy candidate count is not native
  --spec 4 total-window semantics. This remains blocking.

## Tests performed here

Local Python ran the exact new `strata_control/server_launch.py`
(Git blob edd7da76ba885f00225caeb1d4e4e7e9d6344258) and its exact test
module (blob 1bc107344eebc83da46fa5f27441891d714ec333): both tests PASS.
The supplied vanilla JSON parses; args and args_written have the same
46 entries. Supplied shell script passes sh -n.

Three additional tests PASS in an isolated local extraction of the
changed _native, ResidentEngine._diagnostic/_read_records and child_env
functions, with an actual Python subprocess, temporary log, explicit
cwd/env, queue, native OFFLINE_ENV constant and mocked homogeneous hooks.
This is NOT a full module or integration run. Extracted code compiles.
The two full-server entrypoint/refusal tests are written but NOT run
here; the complete repository Python suites, package import path, native
builds, HTTP/live requests, startup log sequence, GPU retirement and
actual effectiveness of accepted options remain unverified. The two
new tests check routing to preflight and explicit rejection of ordinary
args until shared parser integration; they intentionally do not claim
native option support.

All eight repository workflows are manual workflow_dispatch. None
modified, dispatched or activated. No GPU run, forced kill or reset.

## Next

Handoff 0061 and APP_COMPATIBILITY.md are mandatory before additional
performance qualification. Full pinned-source comparison and native
parser/option, engine/API, cache, vision, sampler, control-vector,
prefill and normal output parity remain open. The existing 256-token cap
and greedy restrictions remain. Complete CPU/full checkout tests first.
