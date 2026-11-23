#!/bin/sh
# Handoff 0094: the host engine under perf record (inherited: every thread it starts and the front stage it spawns).
# The first rule: SDMA off.
export HSA_ENABLE_SDMA=0
OUT=/home/chris/workspace/strata-local-archive/profiles/0094/perf-inherit
mkdir -p "$OUT"
exec perf record -F 499 -g -o "$OUT/perf.data" -- "$STRATA_0094_REAL_EXE" "$@"
