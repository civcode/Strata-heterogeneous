#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PROFILE=$1
shift
export HSA_ENABLE_SDMA=0
export STRATA_KEEP_LEFTOVERS=1
export STRATA_NO_PRESET=1
export STRATA_NO_HANDOVER=1
export ROCM_PATH=/opt/rocm/core-10.1
export LD_LIBRARY_PATH="$ROCM_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Normal chat, not the original diagnostic trace run.
unset STRATA_TRANSFER_TRACE STRATA_HETERO_WINDOW_TRACE STRATA_HETERO_PERF_TRACE STRATA_VERIFY_TRACE
PY=${STRATA_PYTHON:-"$ROOT/.venv/bin/python"}
exec "$PY" -m tools.heterogeneous_oberon_launch --profile "$ROOT/$PROFILE" "$@"
