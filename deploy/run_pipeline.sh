#!/usr/bin/env bash
# Pull the latest code and run the whole pipeline, unless this commit already ran successfully.
# Usage: deploy/run_pipeline.sh [--force] [extra run_all.py args, e.g. --scale small]
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

FORCE=0
if [ "${1:-}" = "--force" ]; then FORCE=1; shift; fi

git pull --ff-only -q || .venv/bin/python -c "from src import notify; notify.send('git pull failed on the VPS')"
.venv/bin/pip install -q -r requirements.txt

COMMIT="$(git rev-parse --short HEAD)"
if [ "$FORCE" = 0 ] && [ -f logs/last_ok_commit ] && [ "$(cat logs/last_ok_commit)" = "$COMMIT" ]; then
  echo "$(date) commit $COMMIT already ran, nothing to do"
  exit 0
fi

ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then ARGS=(--scale full); fi

LOG="logs/run_$(date +%Y%m%d_%H%M%S)_$COMMIT.log"
.venv/bin/python -m pytest -q > "$LOG" 2>&1 || {
  .venv/bin/python -c "from src import notify; notify.send('tests FAILED on commit $COMMIT, not running')"
  exit 1
}
.venv/bin/python scripts/run_all.py "${ARGS[@]}" >> "$LOG" 2>&1
STATUS=$?
.venv/bin/python -c "from src import notify; notify.send_file('$LOG', 'full log of commit $COMMIT')"
if [ $STATUS -eq 0 ]; then echo "$COMMIT" > logs/last_ok_commit; fi
exit $STATUS
