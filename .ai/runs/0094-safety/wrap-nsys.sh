#!/bin/sh
# Handoff 0094: the front stage under Nsight Systems (CUDA + OS runtime trace, no CPU sampling).  The first rule: SDMA off.
export HSA_ENABLE_SDMA=0
OUT=/home/chris/workspace/strata-local-archive/profiles/0094/nvidia
mkdir -p "$OUT"
exec /usr/local/bin/nsys profile -t cuda,osrt --sample=none --cpuctxsw=none --force-overwrite true -o "$OUT/front" "$STRATA_0094_REAL_FRONT" "$@"
