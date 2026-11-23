# Handoff 0094 tools (not engine code)

- `runner.py NAME [--limit N] [--env K=V] [--cfg KEY=JSON] [--exe-wrap W] [--front-wrap W] [--sampler] [--pyspy]`: one replay of the
  recording on a fresh server (copy of the maxperf config, port 8081), kernel watcher, cooperative stop.
- `timers.py SERVER_LOG REPLAY_DIR`: per-phase tables from STRATA_DECODE_TIMING / VERIFY_PROFILE / SPLIT_TIMING lines.
- `cpusampler/`: LD_PRELOAD on-CPU sampler (perf is not available: perf_event_paranoid 4) and its analysis.
- `wrap-rocprof.sh`, `wrap-nsys.sh`: profiler wrappers (`exe` / `--front-exe`), SDMA off.
- `rocprof_summary.py`, `rocprof_decode.py`, `rocprof_hip1_only.py`: kernel-trace summaries.
- `before.txt`, `after.txt`: safety records.
