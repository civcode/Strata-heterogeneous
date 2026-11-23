#!/bin/sh
# Handoff 0095: an engine under Nsight Systems (CUDA trace, no CPU sampling).  The first rule: SDMA off.
export HSA_ENABLE_SDMA=0
OUT=${STRATA_0095_NSYS_OUT:-/home/chris/workspace/strata-local-archive/profiles/0095/cuda-only/engine}
exec /usr/local/bin/nsys profile -t cuda --sample=none --cpuctxsw=none --force-overwrite true -o "$OUT" "$STRATA_0095_REAL_EXE" "$@"
