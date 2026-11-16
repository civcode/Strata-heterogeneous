# Report 0062 — direct native application compatibility implementation

Status: partial source implementation; isolated CPU checks pass; full application parity is NOT established.

## Reference

Accessible upstream Niko1221/Strata main: c77572806996349b49995e4f98d41b5294d36a21.
The previously pinned 8db7ddc46cd2575514408c4ab7dd878f69df2f5f is not found by the GitHub API.
No pin was changed. Inspected actual upstream generate.cpp, launch/config and server code;
the local branch's native request-log formatter matches the accessible upstream grammar.
This environment cannot resolve github.com for git clone/ls-remote; the connector
can read and write source blobs and commits but does not provide a build checkout.

## Changes

- Reuse the ordinary launch builder from the previous increment, with native args,
  args_written, user_args precedence and exe/tokenizer/model-path binding.
  Public --prefill 32 (mixed) / 128 (HIP-only), --max-context, --kv, --kv-resident 0,
  --mtp, --spec and --spec-min-p replace corresponding private profile settings.
  Path flags are checked against the actual coordinator source, not treated as
  arbitrary new model selection. Unsupported options fail closed.
- Public --spec 4 means four total verifier rows (anchor and up to three draft
  candidates). --spec 5 means four draft candidates plus anchor. Legacy private
  spec=4 keeps its historical meaning. Canonical mode requests the existing
  native confidence-window selector and checks coordinator/terminal HIP-worker
  capabilities before startup. Existing all-owner commit and resource gates remain.
- Extract the actual native C++ request-summary formatter and call it from both
  vanilla generate.cpp and the heterogeneous coordinator. Committed counts and
  C++ timing are used; the coordinator time includes transport/control, and no
  nonexistent cache reuse or checkpoints are claimed. Share native log follower
  and HTTP Service progress fields.
- Share native pack tokenizer construction and end-token policy. Public mixed
  startup checks the tokenizer against the original GGUF. Reuse existing HTTP
  config behavior rather than adding a separate route/server.

## Tests actually run in this environment

- Exact Git blob matches for request_log.hpp (16e07a86aa55654934b89327477075fc9992ecda),
  long_context.hpp (07e01feac2e207a16d52d279219de5bb3f6942a6),
  mtp_request_policy.hpp (4b142eb120a6c97df317d5f264e5608b614d5fcc)
  and mtp_request_policy_tests.cpp (e4a9d76553209b8a457c5b93463ecc43a25d3dbd).
  g++ -std=c++17 -Wall -Wextra -Werror -O2: PASS 86274 request-policy assertions.
- Compiled isolated native request log header with g++ -std=c++17 -Wall -Wextra
  -Werror; exact normal and cancelled stderr fixtures matched with cmp.
- Exact mixed_runconfig.py blob 6aec8345bee88c5f3fbb7c8b3b117b990003cb41;
  four isolated Python unit tests passed using extracted native user_args
  groups/merge/stamp helpers. Not a full checkout test.
- Supplied vanilla shell syntax and JSON syntax passed.
- No complete build, GPU run, real tokenizer/HTTP integration test, or Actions run.

## Remaining differences

The mixed engine still has a resident protocol adapter and process supervisor,
private stage configuration and a constrained runtime. The full native C++
argument parser, option defaults and resource policy are not shared. Public
prefill auto, expert-cache auto, mmap-experts, 700 MiB reserve (current
long-context gate requires at least 1 GiB), prompt/conversation caching,
vision, control vectors, pipeline settings and other native features remain
unsupported or unqualified. Native MTP request/catch-up policy parity is not
proved by width alone. Completion cap and greedy-only constraints remain.
Raw native wire protocol and all startup events are not identical; bulk
progress is emitted only after an actual committed checkpoint. Diagnostic
records remain in the log. The shared HTTP/web application still delegates
to ResidentEngine rather than StrataEngine. Full parity requires native
integration and real builds/runtime tests, not just matching labels.
