#!/bin/sh
# Handoff 0094: the host engine under rocprofv3 (kernel trace + stats).  The first rule: SDMA off.
export HSA_ENABLE_SDMA=0
OUT=/home/chris/workspace/strata-local-archive/profiles/0094/amd/rocprof
mkdir -p "$OUT"
exec /opt/rocm/core-10.1/bin/rocprofv3 --kernel-trace --stats -d "$OUT" -o run -f csv -- "$STRATA_0094_REAL_EXE" "$@"
