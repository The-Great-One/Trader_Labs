#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$ROOT/reports/auto_iteration_runs"
PYTHON="${AUTOTRADER_PYTHON:-/home/ubuntu/Auto_Trader/venv/bin/python}"
LAB="$ROOT/scripts/auto_iteration_lab.py"
PID_FILE="$RUN_DIR/current.pid"
LOG_FILE_REF="$RUN_DIR/current.log"
EXIT_FILE="$RUN_DIR/current.exit"
STARTED_FILE="$RUN_DIR/current.started"
DELIVERED_FILE="$RUN_DIR/current.delivered"
RESULT_FILE="$ROOT/reports/auto_iteration_latest.json"

mkdir -p "$RUN_DIR"

running_pid() {
  if [[ -s "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      printf '%s' "$pid"
      return 0
    fi
  fi
  return 1
}

start_run() {
  local pid run_id log_file
  if pid="$(running_pid)"; then
    echo "🔬 Nightly strategy auto-iteration is already running (PID $pid)."
    return 0
  fi

  run_id="$(date '+%Y%m%d_%H%M%S')"
  log_file="$RUN_DIR/$run_id.log"
  rm -f "$EXIT_FILE" "$DELIVERED_FILE"
  printf '%s\n' "$log_file" > "$LOG_FILE_REF"
  date -Iseconds > "$STARTED_FILE"

  nohup bash -c '
    root="$1"
    python="$2"
    lab="$3"
    log_file="$4"
    exit_file="$5"
    cd "$root"
    set +e
    AUTOTRADER_ROOT=/home/ubuntu/Auto_Trader \
      AT_RESEARCH_MODE=1 \
      AT_LAB_PRECACHE=0 \
      AT_DISABLE_FILE_LOGGING=1 \
      "$python" "$lab" >"$log_file" 2>&1
    rc=$?
    tmp="${exit_file}.tmp.$$"
    printf "%s\n" "$rc" > "$tmp"
    mv "$tmp" "$exit_file"
    exit "$rc"
  ' _ "$ROOT" "$PYTHON" "$LAB" "$log_file" "$EXIT_FILE" </dev/null >/dev/null 2>&1 &

  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  echo "🔬 Nightly strategy auto-iteration started (PID $pid). Results will be delivered when complete."
}

status_run() {
  local pid rc log_file
  if pid="$(running_pid)"; then
    return 0
  fi
  if [[ ! -s "$EXIT_FILE" || -e "$DELIVERED_FILE" ]]; then
    return 0
  fi

  rc="$(cat "$EXIT_FILE")"
  log_file="$(cat "$LOG_FILE_REF" 2>/dev/null || true)"
  if [[ "$rc" != "0" ]]; then
    echo "⚠️ Nightly strategy auto-iteration failed on Oracle (exit $rc)."
    if [[ -n "$log_file" && -f "$log_file" ]]; then
      echo
      tail -20 "$log_file"
    fi
    touch "$DELIVERED_FILE"
    return 0
  fi

  if [[ ! -s "$RESULT_FILE" ]]; then
    echo "⚠️ Nightly strategy auto-iteration exited successfully but produced no result file."
    touch "$DELIVERED_FILE"
    return 0
  fi

  "$PYTHON" - "$RESULT_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
rows = data.get("results", [])[:5]
print("🧪 Nightly Strategy Auto-Iteration Complete")
print(
    f"{data.get('symbols_loaded', 0)} symbols · "
    f"{data.get('combinations_tested', 0)} combinations · "
    f"{data.get('date_range', 'unknown range')}"
)
if not rows:
    print("\nNo ranked strategy results were produced.")
else:
    print("\nTop strategies:")
    for index, row in enumerate(rows, 1):
        agg = row.get("agg", {})
        print(
            f"{index}. {row.get('family', 'unknown')} — "
            f"CAGR {float(agg.get('cagr_pct', 0)):+.1f}% · "
            f"DD {float(agg.get('max_drawdown_pct', 0)):.1f}% · "
            f"Sharpe {float(agg.get('sharpe_ratio', 0)):.2f}"
        )
PY
  touch "$DELIVERED_FILE"
}

stop_run() {
  local pid
  if pid="$(running_pid)"; then
    kill "$pid"
    echo "Stopped nightly strategy auto-iteration (PID $pid)."
  else
    echo "Nightly strategy auto-iteration is not running."
  fi
}

case "${1:-start}" in
  start) start_run ;;
  status) status_run ;;
  stop) stop_run ;;
  *) echo "usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
