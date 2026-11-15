#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$ROOT/tools/run-oberon-strata.sh" strata-q2_0-hip-only.json "$@"
