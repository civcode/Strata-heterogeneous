# Report 0018 — dynamic native MTP relay and draft-state qualification

**Status: live relay exercised; draft-state reuse FAIL, multi-round native relay disabled.** Starting commit `0c3e7059519daa96c24a1119dc5aa8e98fc72073`. CUDA0 [0,12) → HIP0 [12,30) → HIP1 [30,48), grouped HIP split-overlap, unchanged reserves and approved Q2_0/MTP assets. No GitHub Actions, native fault injection or forced GPU termination. No production generation or throughput claim.

## Build, tests and environment

Existing vanilla host, CUDA and HIP builds passed `cmake --build ... -j28`, as did the HIP rebuild after the fail-closed guard. Host `ctest --test-dir build-hetero-smoke-host --output-on-failure -j28`: **97/97**. `PYTHONPATH=tools .venv/bin/python -m unittest test_heterogeneous_mtp_admission test_heterogeneous_prefill_state test_heterogeneous_prefill_parity`: **25/25**. Initial test invocation without `PYTHONPATH=tools` failed imports; corrected invocation passed. Separate Python resident suite not run.

The apparent ROS packages were not installed in the venv: the inherited `PYTHONPATH=/opt/ros/jazzy/lib/python3.12/site-packages` exposed system ROS packages. With `env -u PYTHONPATH`, checkout `.venv/bin/python -m pip check` passes. No system ROS files or venv packages were removed; test commands explicitly replace or remove the ambient PYTHONPATH. An earlier pip check with inherited PYTHONPATH failed on ROS dependencies missing setuptools.

Runtime hashes match report 0017: `rt/dense.txt` `991456cf4eb266eb8a888c349e548d11bc640f5dbd9149439aa9786af0f732fd`; `dense.bin` `c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a`; `experts.bin` `09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f`; `draft_vocab.bin` `b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91`. Three configured stages use `/home/chris/workspace/Strata-data/packs/q2_0`; `index.txt` SHA-256 `c344c01f59f547ed8a59003857a82eb90499e5299154241dbca787acb54b5b7c`. No raw re-audit. Target conversion's exact checkpoint revision remains unverified.

## Live results before guard

Original full-logit keep2 fixed input, three windows, shared transport, reproduced one-shot draft `[14075,8818,364,264]` at cell 5, anchor 264. The committed prompt was `[248045,846,198,2523,513,32767]`. The actual live relay then ran three rounds with this prompt, independently in shared, socket and repeated shared modes. All three completed six windows with orderly shutdown, full expert residency and no host misses. These are actual candidate windows, not replayed hardcoded offers:

| position | anchor | actual draft offer | accepted draft tokens | authoritative next |
|---|---:|---|---:|---:|
| 6 | 264 | 14075, 8818, 364, 264 | 0 | 2724 |
| 7 | 2724 | 7189, 303, 264, 4954 | 0 | 579 |
| 8 | 579 | 1118, 9665, 314, 264 | 0 | 5713 |

All three runs produced identical offers, counts and next tokens. The coordinator reported each relay window committed before the next offer, and final completion markers were published after successful shutdown; no draft acceptance was observed in this case. No broader natural-acceptance search was done once draft-state qualification failed.

Independent fresh one-row teacher-forced same-plan run of the first eight committed tokens agreed exactly with the first eight accepted rows of the live relay: exact predictions/top-1, KL 0, max absolute logit difference 0. Same-plan full-evaluated-row comparisons across shared/socket and repeated shared: **27/27 rows**, exact predictions/top-1, KL 0, max difference 0. Comparisons used existing validated STFR parser and same-plan comparator with `allow_partial=True`, 0.001 KL limit, and accepted-row slicing only for the serial comparison. Raw driver and JSON results are in `/home/chris/tmp/strata-heterogeneous/m7-0018/`. An initial invocation of the ordinary full-commit comparator failed as expected on partial commits; the explicit partial-prefix path was then used, without changing gates.

## Fresh-prefix draft-state gate: FAIL

Fresh independent one-shot native probes were primed with exactly the accepted prefix, one row per window, same plan and next anchor. At cell 6, anchor 2724, fresh draft was `[579,23513,1830,364]`, whereas live reuse offered `[7189,303,264,4954]`. At cell 7, anchor 579, fresh draft was `[5713,11889,310,8290]`, versus reused `[1118,9665,314,264]`. Serial target accepted-row parity above rules out an observed difference in target logits at the compared positions; the cause of the native draft divergence is not yet established. No committed draft K/V per-cell comparison was available/run; neither logical causal overwrite nor physical suffix clearing is proved. Results cannot be interpreted as a successful rollback. Fresh candidates would have matched the next target token in these two positions, but they were not live-submitted and are not accepted-token evidence.

`experimental/heterogeneous/include/strata_heterogeneous/native_mtp_probe.hpp` now rejects `relay_rounds > 1` during native probe preparation, before allocating the draft probe; the host mock protocol remains testable and one-shot functionality remains available. The newly added native guard was compiled but not exercised in a post-change live failure run. The pre-guard live results above are diagnostic only; do not reuse the resulting draft state. Resident MTP and native reset remain unqualified and disabled; no cold reload is claimed as reset. Native bulk prefill remains disabled and the first CUDA GDN state disparity remains open. The 160 accepted output tok/s and 3840 prompt tok/s figures remain workload-dependent and unverified.

## Next

Investigate native `draft_first` versus fresh `prefill` state at committed cells using a narrow terminal-device diagnostic and exact same target residuals/positions, keeping vanilla math unchanged. Restore multi-round admission only after exact fresh-prefix candidate and committed K/V qualification, then rerun socket/shared, acceptance and serial gates. Raw files: `control*`, `relay-{shared,socket,shared-repeat}*`, `fresh-{7,8}*`, `compare-accepted.py`, `accepted-serial-parity.json`, and `relay-*-parity.json` in the scratch directory. The pre-existing untracked `tools/heterogeneous_pool_probe.cpp` remains untouched.
