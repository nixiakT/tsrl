#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC_PATH="${1:-configs/overnight_trading_search.json}"
OUTPUT_DIR="${2:-$ROOT_DIR/runs/overnight_$(date '+%Y%m%d_%H%M%S')}"
DEADLINE="${3:-$(date -d 'tomorrow 10:00' '+%Y-%m-%dT%H:%M:%S%:z')}"

if [[ "$SPEC_PATH" != /* ]]; then
  SPEC_PATH="$ROOT_DIR/$SPEC_PATH"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
  OUTPUT_DIR="$ROOT_DIR/$OUTPUT_DIR"
fi

mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/overnight.log"

setsid bash -lc "cd '$ROOT_DIR' && exec /usr/local/bin/uv run tsrl-train overnight-watchdog --spec '$SPEC_PATH' --output '$OUTPUT_DIR' --deadline '$DEADLINE'" \
  >"$LOG_PATH" 2>&1 < /dev/null &

PID="$!"
echo "pid: $PID"
echo "output: $OUTPUT_DIR"
echo "log: $LOG_PATH"
echo "deadline: $DEADLINE"
echo "stop_file: $OUTPUT_DIR/STOP"
