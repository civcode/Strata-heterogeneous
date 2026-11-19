#!/bin/bash
# run_server_checks.sh CONFIG PORT RUN_DIR [serve_mixed.py options...]
# Start the server through serve_mixed.py, wait until it answers, run server_checks.py, stop the server with SIGTERM
# (its Ctrl+C path; a server started with & from a script ignores SIGINT) and wait for the engines to drain. Never kills.
# The placeholders come from the environment (mixed_split/configs/README.md).
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CFG=$1; PORT=$2; RUN=$3; shift 3
NAME=$(basename "$CFG" .json); NAME=${NAME%.example}
mkdir -p "$RUN"
"$REPO/.venv/bin/python" -I "$REPO/mixed_split/tools/serve_mixed.py" --config "$CFG" --port "$PORT" --run-dir "$RUN" "$@" \
    > "$RUN/$NAME.server.out" 2>&1 &
SP=$!
# Cooperatively stop the server before a single GPU run reaches two minutes.
( sleep 105; kill -TERM "$SP" 2>/dev/null || true ) &
WATCHDOG=$!
up=0
for i in $(seq 1 100); do
    sleep 1
    kill -0 $SP 2>/dev/null || break
    curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && { up=1; break; }
done
echo "server up=$up after ${i}s"
LOG=$("$REPO/.venv/bin/python" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['log'])" "$RUN/$NAME.json")
MODEL=$("$REPO/.venv/bin/python" -I -c "import json,sys; print(json.load(open(sys.argv[1]))['model_name'])" "$RUN/$NAME.json")
[ $up = 1 ] && timeout 100 "$REPO/.venv/bin/python" -I "$REPO/mixed_split/tools/server_checks.py" "$PORT" "$MODEL" "$LOG" "$RUN/$NAME.checks.json"
kill -TERM $SP 2>/dev/null
kill "$WATCHDOG" 2>/dev/null || true
wait "$WATCHDOG" 2>/dev/null || true
for i in $(seq 1 120); do sleep 1; kill -0 $SP 2>/dev/null || break; done
echo "server ended after ${i}s: $(kill -0 $SP 2>/dev/null && echo STILL RUNNING || echo gone)"
ps -eo pid,args | grep "[b]uild-0067-.*/strata" | cut -c1-120
