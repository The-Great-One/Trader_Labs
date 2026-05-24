#!/usr/bin/env bash
# Walk-forward validation for curated_combo_282 (best variant from batch 801)
# Buy: sr_breakout_enabled=1, sr_breakout_buffer_pct=0.005, volume_confirm_mult=0.85, adx_strong_min=18, ich_cloud_bull=0
# Sell: momentum_exit_rsi=38.0, equity_review_rsi=45.0
set -euo pipefail

cd "$(dirname "$0")/.."

export AT_RESEARCH_MODE=1
export AT_LAB_PRECACHE=0
export AT_BUY_SR_BREAKOUT_ENABLED=1
export AT_BUY_SR_BREAKOUT_BUFFER_PCT=0.005
export AT_BUY_VOLUME_CONFIRM_MULT=0.85
export AT_BUY_ADX_STRONG_MIN=18
export AT_BUY_ICH_CLOUD_BULL=0
export AT_SELL_MOMENTUM_EXIT_RSI=38
export AT_SELL_EQUITY_REVIEW_RSI=45

echo "=== Walk-forward validation for curated_combo_282 ==="
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

python3 scripts/weekly_universe_cagr_check.py 2>&1

echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"