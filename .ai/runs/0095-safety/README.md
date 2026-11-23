# Handoff 0095 tools (not engine code)

- `runner.py NAME [--rec RECORDING] [--limit N] [--env K=V] [--cfg KEY=JSON]`: 0094's runner (copy of the maxperf
  config, port 8081, kernel watcher, cooperative stop), with `--rec` for a trimmed recording.
- `wrap-nsys.sh`: an engine under nsys (CUDA trace), SDMA off; used for the CUDA-only run.
- `pltrace_stats.py PLTRACE`: intervals from `STRATA_PIPELINE_TRACE` (needs `STRATA_PIPELINE_DEBUG=1`).
- `plmodel.py PLTRACE [PARAMS.json]`: the trace-driven model of the pipelined loop; scenarios in `../0095-model/`.
- `before.txt`, `after.txt`: safety records. Trimmed recordings and raw traces stay in
  `strata-local-archive/profiles/0095/`.
