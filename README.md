# Trader_Labs

Trader_Labs is the focused research deployment for the nightly RSI-momentum auto-iteration lab. It does not place orders. Strategy promotion requires reviewed evidence and a separate Auto_Trader change.

## Live script closure

```text
scripts/auto_iteration_lab.py
└── scripts/rsi_224466_rotation_lab.py
```

`scripts/__init__.py` makes the directory importable. `scripts/run_auto_iteration_nightly.sh` supplies the `start`, `status`, and `stop` lifecycle for the nightly lab.

## Nightly auto-iteration

The Hermes nightly chain is:

```text
Hermes cron 7af4ce25de03
→ run_auto_iteration.sh
→ scripts/run_auto_iteration_nightly.sh start
→ scripts/auto_iteration_lab.py
```

The live baseline in `auto_iteration_lab.py` is RSI 22/44/66, momentum 63, `3W-FRI`, top 8, SMA100 regime, MACD, momentum-rank blend 0.3, and inverse-volatility weighting with a 10-day lookback. Keep it synchronized with the deployed Auto_Trader paper-shadow parameters.

Qualification gates and scoring weights are read from `config/auto_iteration_lab.json`. The lab writes:

- `reports/auto_iteration_latest.json`
- `reports/auto_iteration_history.jsonl`
- `reports/auto_iteration_runs/`

The history is append-only. Only candidates that pass the configured consistency gates can become the research champion; a research champion is not an automatic live promotion.

Lifecycle commands on the runtime server:

```bash
cd /home/ubuntu/Trader_Labs
scripts/run_auto_iteration_nightly.sh start
scripts/run_auto_iteration_nightly.sh status
scripts/run_auto_iteration_nightly.sh stop
```

Do not start the nightly lab during routine verification.

## Runtime and promotion boundary

- Runtime host: `ubuntu@144.24.112.62` (`arunnet`)
- Deployment root: `/home/ubuntu/Trader_Labs`
- Auto_Trader dependency: `/home/ubuntu/Auto_Trader` (runtime venv only; no source imports)
- Generated `reports/` and market-data caches remain gitignored.
- Do not edit live trading rules from this repository.
- Require consistency qualification, walk-forward/OOS review, execution/data-parity checks, and a separate reviewed Auto_Trader patch before promotion.
