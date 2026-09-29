# Trader_Labs Project Map

Trader_Labs has one intentionally small live closure: the nightly RSI-momentum auto-iteration lab.

## Roots

- Local: `/Users/sahilgoel/Desktop/Projects/trading/Trader_Labs`
- Runtime: `/home/ubuntu/Trader_Labs`
- Host: `ubuntu@144.24.112.62` (`arunnet`)
- Runtime Python: `/home/ubuntu/Auto_Trader/venv/bin/python`

## Nightly auto-iteration flow

```text
Hermes cron 7af4ce25de03
→ run_auto_iteration.sh
→ scripts/run_auto_iteration_nightly.sh start
→ scripts/auto_iteration_lab.py
→ scripts/rsi_224466_rotation_lab.py
```

- `config/auto_iteration_lab.json` owns qualification gates, selection weights, and walk-forward thresholds. Do not change it during deployment-only work.
- `scripts/run_auto_iteration_nightly.sh` uses Linux flock and the `run_lab_worker.py` process-group supervisor; it propagates one required run ID.
- `scripts/auto_iteration_lab.py` explores canonical parameter identities, atomically persists current-schema history, re-evaluates the incumbent, and permits only consistency+walk-forward-qualified candidates to be `champion_ready`.
- `scripts/portfolio_simulator.py` performs stateful D-close/D+1-open execution with explicit cash, drifting units, fail-closed marks, and actual-notional fees.
- `scripts/walk_forward.py` evaluates fixed candidates in isolated, non-overlapping retrospective folds.
- `scripts/rsi_224466_rotation_lab.py` supplies the RSI-rotation data loader and backtest primitives shared with the deployed Auto_Trader shadow.

Current baseline (must stay in parity with the deployed shadow): RSI 22/44/66, momentum 63, `3W-FRI`, top 8, SMA100 regime, MACD enabled, blend 0.3, inverse-volatility weighting, volatility lookback 10, 10 bps costs, and maximum three names per sector.

Outputs:

- `reports/auto_iteration_latest.json`
- `reports/auto_iteration_history.jsonl`
- `reports/auto_iteration_runs/`

## Kept files

- `scripts/__init__.py`
- `scripts/auto_iteration_lab.py`
- `scripts/rsi_224466_rotation_lab.py`
- `scripts/run_auto_iteration_nightly.sh`
- `config/auto_iteration_lab.json`

## Research and promotion rules

- Complete calendar years drive consistency qualification; partial years are reported but excluded from those gates.
- Qualification includes worst-year return, Sharpe, drawdown, rolling 12-month floor, outlier dependence, transaction costs, and minimum completed years.
- Walk-forward evidence must distinguish causal signal computation from candidate-selection leakage and current-universe survivorship bias.
- Full-history ranking is always `retrospective_only`; `champion_ready` requires every configured walk-forward gate and same-snapshot incumbent comparison.
- Beating a baseline in every fold does not pass a strict all-positive-fold gate if any candidate fold is negative.
- A research champion is never automatically promoted. Promotion requires reviewed walk-forward/OOS evidence, data and execution parity, and a separate minimal Auto_Trader patch.

## Operational checks

```bash
# Syntax only; does not run the lab
python3 -m py_compile scripts/__init__.py scripts/auto_iteration_lab.py \
  scripts/rsi_224466_rotation_lab.py

# Lifecycle status; do not use start during routine verification
scripts/run_auto_iteration_nightly.sh status
```

Generated reports, caches, secrets, `.serena/`, and Python bytecode remain ignored.
