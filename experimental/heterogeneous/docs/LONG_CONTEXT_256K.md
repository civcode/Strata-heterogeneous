# Q8 resident context expansion to 256K

Implementation candidate. Oberon capacity runs reached 256K on both topologies,
but lifecycle qualification remains incomplete; see [results](LONG_CONTEXT_256K_OBERON.md).
Prior numerical discrepancies are recorded separately; do not require their elimination before
testing architecture, ownership, memory or context capacity. Preserve all
existing parity criteria and report their status accurately.

## Mode and memory admission

- Ordinary generation remains unchanged (Q8 default, explicit F16 reference).
  Existing workload mode remains capped at 32768. Extended context is explicit
  via `--workload-mode --long-context --group-native-hip --native-prefill-chunk N`.
- Context is the total prompt plus completion limit, 2048 through 262144,
  aligned to 256, and must match every worker's configured capacity. The
  existing completion limit of 4096 is retained. The CLI uses a 4 MiB prompt
  text bound and the resident token command has a 3 MiB bound; existing
  activation frame bounds are unchanged.
- Initial support is the approved Q2_0 full-resident 48-layer pack, native INT8
  QSA KV, two grouped HIP owners or CUDA entry followed by the grouped HIP
  owners. Require all 512 routed experts per owned layer, max_t=8,
  uncapped owned arena and at least 1 GiB reserve on every GPU. No F16 fallback,
  partial expert residency, cache spilling or precision substitution.
- The coordinator checks exact capacity and worker capability. Workers verify
  native QSA pools and expert residency. Native prefill and verifier use the
  existing native execution paths. The entire long prefill phase has an
  extended 30-minute deadline; mixed transfer retains its per-piece deadline.
- Native allocation guards use the actual session, dense/PLE and expert
  allocation paths. After verifier initialization, reserve plus native
  `Prefill::bytes_needed_owned` is checked against actual free memory on each
  owner. Repeat checks surround prefill allocation and drain and the first
  three committed windows. Failure is explicit rather than silent downscaling.
- `HETERO_CAPACITY` records stage, phase, free and total bytes, reserve,
  additional estimated bytes and fit verdict. These are allocator-boundary
  observations, not an exhaustive measurement of transient peak. Native
  sizing and telemetry must be confirmed against actual device allocations;
  do not treat a planning estimate as proof of fit.

## Oberon qualification

Use the existing vanilla toolchain and virtual environment, with 28 parallel
build jobs (see [known-good configuration](../../../docs/KNOWN_GOOD_STRATA_CONFIG.md)).
Rebuild host, CUDA and HIP binaries from the same revision and run host
CTest and Python tests. No GitHub Actions.

Generate separate private configs from the approved existing mixed configs.
These must be in plan order with matching model identity and at least 1 GiB
reserve. The command below only writes configs and a manifest; it does not
launch GPUs. Set the source paths and a fresh absolute output directory first.

```bash
python -m tools.heterogeneous_long_context_config \
  --source "$CUDA_CFG" "$HIP0_CFG" "$HIP1_CFG" \
  --output-dir "$PRIVATE/hip-32k" --split 24 24 --context 32768
```

For mixed, use `--split 15 18 15` as a candidate, or a smaller CUDA allocation
such as 12/18/18 if CUDA headroom is insufficient. The manifest gives the
plan, model fingerprint and config hashes. Preserve reserves and full residency
when selecting another split. No split is guaranteed to fit.

Use the existing heterogeneous-generate CLI with the new options, matching
worker/config paths and plan from the manifest, `--resident --max-context N
--kv int8 --workload-mode --long-context --group-native-hip
--native-prefill-chunk 128`. Run CPU preflight without `--run` first;
only add `--run` for the hardware test. Use the same options for a generated
server config where applicable.

Test in stages:

1. Regress existing 2K ordinary Q8 and F16, plus explicit long-context at 2K.
2. Test HIP-only 24/24 at 32K, 64K, 128K and 256K; for each, verify startup
   and short generation before exercising a near-capacity prompt with measured
   token counts and remaining completion budget. Measure actual device memory
   throughout, in addition to the native phase observations.
3. Repeat for the mixed topology, adapting placement if reserve checks decline
   admission. Stop on OOM, unexpected residency, lifecycle errors or insufficient
   reserve; never lower the reserve to obtain a passing result.
4. Repeat independent requests, verify reset/commit positions, cancellation at
   committed checkpoints, stale worker rejection, exact capacity enforcement,
   and prompt-plus-completion overflow refusal.

Report separately: CPU tests, actual configuration/placement, native admission,
completed prompt and decode token counts, per-owner sampled and independently
monitored peak memory, headroom, lifecycle results and any limitations.
Allocation at 256K without completing a near-capacity request is not 256K
inference qualification. Numerical diagnostics and throughput optimization are
separate follow-up activities.
