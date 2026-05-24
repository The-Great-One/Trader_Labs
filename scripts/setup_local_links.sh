#!/usr/bin/env bash
set -euo pipefail
LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTOTRADER_ROOT="${AUTOTRADER_ROOT:-$(cd "$LAB_ROOT/../Stocks" && pwd)}"
mkdir -p "$LAB_ROOT/reports" "$LAB_ROOT/log" "$LAB_ROOT/logs"
if [ ! -e "$LAB_ROOT/intermediary_files" ]; then
  ln -s "$AUTOTRADER_ROOT/intermediary_files" "$LAB_ROOT/intermediary_files"
fi
if [ ! -e "$LAB_ROOT/external" ] && [ -e "$AUTOTRADER_ROOT/external" ]; then
  ln -s "$AUTOTRADER_ROOT/external" "$LAB_ROOT/external"
fi
echo "Trader_Labs linked to Auto_Trader at $AUTOTRADER_ROOT"
