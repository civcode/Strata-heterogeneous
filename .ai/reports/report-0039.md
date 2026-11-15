# Report 0039 — native INT8 KV default (initial implementation)

## Starting evidence and user decision
0038 returned at `b8e7aa939eb59c687fbb400c70ba7790ba1c76d9`: byte-identical committed FP16 full-logit STFR packets for 1/4/8-row direct/reference cases and partial-prefix continuation, bounded resident cancellation/reset; host CTest 101/101. Full native-state parity, broader negatives and direct throughput remain NOT_RUN. User explicitly approved Q8 KV as default subsequent to 0038.

## Chat AI changes
Implementation checkpoint `45b95196699716ad43c98626b429771fc5c4dc37`. Ordinary native/Python CLI and resident server now default to vanilla `int8` KV (Q8 signed codes with FP16 scales per 64). Explicit `f16` reference retained. Legacy specialized FP16 parity/MTP diagnostics continue their existing FP16 contract; explicit INT8 refused there until independently qualified. The coordinator passes explicit KV argv to both runtime workers before native session sizing, and verifies actual QSA format/residency per physical stage. New `HETERO_KV` log and selected-mode source description provide evidence for future benchmarks. No native kernel changes or independent KV implementation, no prefill or MTP/compact flag promotion, no new supported context cap. Native launch now uses existing graceful-only drain path for all selected modes; timeout reports surviving PID, not forced kill. This preserves the hardware safety constraint for qualification. RDNA4 INT8 WMMA is eligible under the native architecture-dependent dispatch, actual activation and throughput NOT_RUN.

Local isolated C++17 format-header test passed with -Wall -Wextra -Wpedantic -Werror: exact blob SHA `330e40229f550e20ef3f7751f95f1647b19b70a8` verified, default INT8, two round trips, seven unsupported names rejected. Added repository CPU bootstrap and Python CLI/server fixtures; those integrated tests and all native builds/hardware remain NOT_RUN here: full checkout failed GitHub DNS resolution. No GitHub Actions. Prior rates are historical FP16; do not infer Q8 gains. No 256K runtime qualification.

## Oberon qualification
PENDING. Append actual evidence per handoff 0039.
