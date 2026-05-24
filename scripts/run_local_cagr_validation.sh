#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_FILE="intermediary_files/lab_status/current_cagr_validation.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
export AT_RESEARCH_MODE="${AT_RESEARCH_MODE:-1}"
export AT_LAB_MODE="${AT_LAB_MODE:-1}"

exec rtk python3 scripts/weekly_universe_cagr_check.py
