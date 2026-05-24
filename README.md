# Trader_Labs

Research and validation workspace for Auto_Trader.

This repository contains lab-only strategy research, walk-forward validation,
Kronos experiments, option research, OOS/CAGR hunts, and Telegram/channel
learning analysis. The live Auto_Trader repo remains focused on runtime,
execution, paper shadowing, dashboards, and operational reports.

## Relationship to Auto_Trader

- Default sibling live repo: `../Stocks`
- Override with: `AUTOTRADER_ROOT=/path/to/Auto_Trader`
- Lab scripts import live rule/runtime modules from Auto_Trader for parity, but
  lab changes are developed here first.
- Improvements are promoted back to Auto_Trader only after validation.

## First-time local setup

```bash
./scripts/setup_local_links.sh
python -m venv venv
source venv/bin/activate
pip install -r ../Stocks/requirements.txt
pip install -r requirements-research.txt
pip install -r requirements-kronos.txt  # optional, for Kronos
```

## Promotion rule

Do not edit live trading rules directly from labs. Promotion requires:

1. Kite-cache-only backtest or paper evidence.
2. Walk-forward/OOS validation.
3. A short promotion note under `promotion_notes/`.
4. A small, reviewed patch/PR/cherry-pick into Auto_Trader.

See `PROMOTION.md`.

## Lab-only Kite data refresh

Trader_Labs can refresh Kite auth/history without using the live Auto_Trader
trading service session.

1. Create ignored credentials:

```bash
cp secrets/kite_lab_secrets.example.py secrets/kite_lab_secrets.py
# fill API_KEY, API_SECRET, USER_NAME, PASS, TOTP_KEY
```

2. Generate a lab-only token:

```bash
python scripts/kite_lab_token.py refresh --browser
python scripts/kite_lab_token.py check
```

3. Fetch historical OHLCV into the lab cache:

```bash
python scripts/kite_lab_fetch_history.py --universe NIFTY200 --interval day --years 5
# or preview first
python scripts/kite_lab_fetch_history.py --symbols RELIANCE,TCS,INFY --dry-run
```

Token and fetched data are written under `intermediary_files/`, which is ignored
and should remain separate from live trading service state.
