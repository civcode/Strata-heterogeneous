# Report 0049 — HIP-only MTP, streaming and cancellation

## Status

**PASS for bounded HIP-only Q8 MTP streaming, mid-batch cancellation/recovery, mixed streaming regression, and focused local integration.** The HIP-only streamed 32-token output matched its ordinary reference; A/B/cancel/A exercised variable prompts, terminal stop, cancellation and reset on one resident process set. Mixed streamed output matched report 0048's buffered MTP output. No source defect was demonstrated; no source changes were required. This is not production or 256K admission, physical state-parity proof, or a robust performance claim. Optional private HTTP smoke was not run.

## Provenance and safety

Source `a1cd947e` on `feature/heterogeneous-hip-only`, existing Release build caches and checkout `.venv` Python 3.12.3; CUDA `/usr/bin/nvcc` sm_89, HIP `/opt/rocm/core-10.1/lib/llvm/bin/clang++` gfx1201. SHA-256 of rebuilt executables:

- host `build-256k-host/strata_heterogeneous_live_smoke`: `e16b7cf83dc4b01ae88328e477d847b5c5df4efa85266ac5caa074fdb299c491`
- CUDA `build-256k-cuda/experimental/heterogeneous/strata_heterogeneous_cuda_worker`: `50ed5184927d4d23bab315a58a4afb94e2619c1597f6e764d4074d33d57b0da5`
- HIP `build-256k-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker`: `68677d6a815a110764c7cebb07bb622dcbf5b5d7ec7a9b547019a73c45093a81`

Fresh process/listener checks and NVIDIA/AMD process and memory snapshots showed no local Strata listener or GPU compute process; this was a snapshot, not an exclusive reservation. Xorg used 15 MiB NVIDIA, each 32 GB AMD GPU used 57 MB. Rechecked GPU processes after builds; none detected. No existing service was contacted or interrupted. No GitHub Actions, dependency installs, asset/repack changes, long-context tests, clock/driver changes, spill, reserve lowering or forced kills. Unrelated untracked `tools/heterogeneous_pool_probe.cpp` was preserved. Logs, captures and private driver are under `/home/chris/tmp/strata-heterogeneous/handoff-0049/`, directory created with `umask 077`.

## Build and focused CPU tests

```sh
cmake --build build-256k-host --parallel 28 --target strata_heterogeneous_live_smoke strata_heterogeneous_final_pipeline_tests strata_heterogeneous_host_tests strata_heterogeneous_window_admission_tests strata_heterogeneous_hip_final_verifier_contract_tests strata_heterogeneous_hip_verifier_contract_tests strata_heterogeneous_mtp_prefix_cursor_tests strata_heterogeneous_mtp_request_policy_tests strata_heterogeneous_mtp_stream_batch_tests
cmake --build build-256k-cuda --parallel 28 --target strata_heterogeneous_cuda_worker strata_heterogeneous_live_smoke
cmake --build build-256k-hip --parallel 28 --target strata_heterogeneous_hip_worker strata_heterogeneous_live_smoke
ctest --test-dir build-256k-host --output-on-failure -R 'strata_heterogeneous_(final_pipeline|host|window_admission|hip_final_verifier_contract|hip_verifier_contract|mtp_prefix_cursor|mtp_request_policy|mtp_stream_batch)_tests'
env PYTHONPATH=tools .venv/bin/python -m unittest -v serve.test_heterogeneous_resident tools.test_heterogeneous_hip_only_cli tools.test_heterogeneous_mtp_stream_contract tools.test_heterogeneous_mtp_qualification tools.test_heterogeneous_mtp_generation
```

Builds passed in 2.04/21.68/11.87 s; compiler warnings retained in private logs. CTest 8/8 in 8.62 s, Python 44/44 in 0.25 s. Tests include legacy/fixed admission and ordinary/buffered protocol fixtures, batch cursor and malformed inputs, streaming cancellation and capped-final behavior.

## GPU execution and observations

Reused approved Q2_0 runtime `/home/chris/workspace/Strata-data/mtp/rt`, pack index `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`, fingerprint 2026111301; exact 2048-cell stage configs, max_t=8, INT8 KV, full expert maps and >=1 GiB reserve per owner. `STRATA_*` environment overrides were cleared and `HSA_ENABLE_SDMA=0`. HIP-only reference command (exact shell-escaped command in `hip-reference.command`):

```sh
.venv/bin/python strata.py heterogeneous-generate \
 --coordinator "$PWD/build-256k-host/strata_heterogeneous_live_smoke" \
 --hip-worker "$PWD/build-256k-hip/experimental/heterogeneous/strata_heterogeneous_hip_worker" \
 --plan hip:0:0-24,hip:1:24-48 --fingerprint 2026111301 \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/hip-2k/stage0.cfg \
 --config /home/chris/tmp/strata-heterogeneous/qualification-256k/hip-2k/stage1.cfg \
 --prompt-file /home/chris/tmp/strata-heterogeneous/handoff-0045/short.txt \
 --max-new-tokens 32 --prompt-chunk 8 --resident --max-context 2048 --shared --group-native-hip --kv int8 --output json --run
```

Private streaming driver (configuration and all raw progress/final records saved in `stream_check.py`, `hip-stream.json`, `mixed-stream.json`):

```sh
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0049/stream_check.py hip
env PYTHONPATH="$PWD" .venv/bin/python /home/chris/tmp/strata-heterogeneous/handoff-0049/stream_check.py mixed
```

The driver uses `prepare_server`/`ResidentEngine`, 2K configs, `--mtp` pinned runtime, spec 4, min-p 0.5, buffered native prefill disabled, streaming enabled and prompt chunk 8. HIP-only plan `hip:0:0-24,hip:1:24-48` used one grouped HIP process, no CUDA worker; mixed used CUDA0/HIP0/HIP1 15/18/15. Reference HIP overlap was off; MTP overlap was on, so timings are not comparable.

| HIP request | Prompt / budget | Result | Proposed / accepted | Committed end |
|---|---|---|---:|---:|
| 1 A | seven `23066` / 32 | length, 32 IDs `23066` | 25 / 24 | 38 |
| 2 B | `[248045,846,198,2523]` / 24 | stop `248046`, 13 IDs `[513,449,14791,17313,49110,440,21156,264,6150,4939,23014,23850,13]` | 18 / 5 | 17 |
| 3 A | seven `23066` / 32 | cancelled at a five-token batch after yielding its first token; cancelled final ACK contained no output IDs | not summarized on cancellation | 12 |
| 4 A | seven `23066` / 32 | length, same 32 IDs as request 1 and ordinary reference | 25 / 24 | 38 |

All four requests used the same resident process and logged draft reset for each. Request 3 progress was `HETERO_PROGRESS 3 2 12 BATCH 0 5 23066 23066 23066 23066 23066`; two total tokens had reached the client, after which cancellation was acknowledged and request 4 reset and completed normally. Noncancelled yielded tokens excluding stop were compared against the validated final records. Native and Python batch cursor checks enforce committed positions, ordered prefix, no duplicate anchor/unaccepted IDs, terminal status and final-record identity; capped-final withholding is additionally covered by CPU tests and implementation, rather than an independent packet-timestamp measurement. Real nonempty batches of up to five were observed; request 2 produced terminal batch flag 1. The HIP A output and reference each consisted entirely of 32 copies of 23066; equal output does not establish full state parity.

The separate mixed streaming request completed length with 32 IDs identical to report 0048 buffered MTP, proposed 33 / accepted 18, with real batches and final-record validation. No deliberate GPU failure injection was run.

HIP logs showed HIP0 and HIP1 12288/12288 experts resident, target KV INT8 resident on both, no host expert routes or misses, and 1 GiB reserve configured on both. HIP1 MTP admission logged 12,029,263,872 bytes free, 3,044,868,096-byte conservative estimate and 1,073,741,824-byte reserve; draft VRAM reported 1,019,533,980 bytes. Successful initialization/reset exercised the draft's Q8 dense code/scale and no-host/no-elastic checks. Source inspection confirms service passes the HIP1 embedding as a per-drafter override rather than installing it as the process-global target embedding; device ownership was HIP1. These observations do not establish transient peak headroom or independent physical-state correctness.

All three GPU process sets returned 0 and `HETERO_CLOSED`, with no surviving native workers or AMD/NVIDIA compute process afterward. Host wall times: HIP reference 11.76 s, HIP streamed A/B/cancel/A 19.11 s, mixed streamed regression 16.01 s; total **46.88 s** of admitted GPU process wall time. No HTTP request was made. Broader prompts, repeated cancellation modes, real-GPU fault retirement, network streaming and long-context MTP remain unverified.
