#!/usr/bin/env python3
"""Sweep RSI+momentum universe filters and rebalance frequencies.

Purpose:
- compare validated 363-symbol setup with broader/looser universes
- compare monthly end (ME) vs weekly Friday (W-FRI) rebalances
- optionally run walk-forward on a small subset to judge weekly robustness

Writes:
- reports/rsi_momentum_universe_rebalance_sweep_latest.json
- timestamped copy under reports/
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rsi_momentum_report import find_hist_dir
from scripts.rsi_224466_rotation_lab import (
    load_prices as lab_load_prices,
    rebalance_dates as lab_rebalance_dates,
    rsi_dataframe as lab_rsi,
)

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)


@dataclass
class SweepConfig:
    name: str
    min_rows: int
    min_end_date: str
    rebalance: str


def load_prices(min_rows: int, min_end_date: str) -> tuple[pd.DataFrame, dict]:
    hist_dir = Path(find_hist_dir(""))
    prices_raw, ctx = lab_load_prices(
        hist_dir,
        min_rows=min_rows,
        min_end_date=min_end_date,
        symbols=set(),
        max_symbols=0,
    )
    return prices_raw.ffill(limit=3), ctx


def strategy_daily_returns(
    prices: pd.DataFrame,
    top_n: int = 10,
    cost_bps: float = 10.0,
    momentum_period: int = 21,
    rebalance: str = "ME",
) -> tuple[pd.Series, list[dict]]:
    rsi_score = (lab_rsi(prices, 22) + lab_rsi(prices, 44) + lab_rsi(prices, 66)) / 3.0
    mom_1m = prices.pct_change(momentum_period, fill_method=None)
    returns = prices.pct_change(fill_method=None).fillna(0)
    dates = lab_rebalance_dates(prices.index, rebalance)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    turnover_l = pd.Series(0.0, index=prices.index)
    prev = pd.Series(0.0, index=prices.columns)
    pick_log: list[dict] = []

    for i, d in enumerate(dates):
        pos = prices.index.get_loc(d)
        if pos + 1 >= len(prices.index):
            continue
        td = prices.index[pos + 1]
        ed = dates[i + 1] if i + 1 < len(dates) else prices.index[-1]
        target = pd.Series(0.0, index=prices.columns)
        rsi_at = rsi_score.loc[d].copy()
        mom_at = mom_1m.loc[d].copy()
        combined = rsi_at.where(mom_at > 0, 0)
        scored = combined.dropna().sort_values(ascending=False)
        picks = [
            s
            for s in scored.index
            if s in prices.columns and pd.notna(prices.loc[d, s]) and scored[s] > 0
        ][:top_n]
        if picks:
            target.loc[picks] = 1.0 / len(picks)
        turnover_l.loc[td] = abs(target - prev).sum()
        prev = target
        mask = (prices.index >= td) & (prices.index <= ed)
        weights.loc[mask, :] = target.values
        pick_log.append(
            {
                "signal_date": str(d.date()),
                "trade_date": str(td.date()),
                "pick_count": int(len(picks)),
                "picks": picks,
            }
        )

    gross = (weights * returns).sum(axis=1)
    net = gross - turnover_l * (cost_bps / 10000.0)
    active = weights.sum(axis=1) > 0
    return net.loc[active].copy(), pick_log


def metrics_from_returns(r: pd.Series) -> dict:
    if r.empty:
        raise ValueError("empty return series")
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())
    vol = float(r.std() * math.sqrt(252)) if len(r) > 1 else 0.0
    sharpe = float((r.mean() * 252) / vol) if vol > 0 else 0.0
    yearly = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {
        "days": int(len(r)),
        "start": str(r.index[0].date()),
        "end": str(r.index[-1].date()),
        "total_return_pct": round(float((eq.iloc[-1] - 1) * 100), 2),
        "cagr_pct": round(float(cagr * 100), 2),
        "xirr_pct": round(float(cagr * 100), 2),
        "max_drawdown_pct": round(float(dd * 100), 2),
        "vol_pct": round(float(vol * 100), 2),
        "sharpe_like": round(sharpe, 3),
        "worst_year_pct": round(float(yearly.min() * 100), 2),
        "positive_years": int((yearly > 0).sum()),
        "total_years": int(len(yearly)),
    }


def trade_stats(pick_log: list[dict]) -> dict:
    prev = set()
    total_buys = total_sells = total_changes = 0
    empty_rebalances = 0
    for row in pick_log:
        cur = set(row["picks"])
        if not cur:
            empty_rebalances += 1
        buys = cur - prev
        sells = prev - cur
        total_buys += len(buys)
        total_sells += len(sells)
        total_changes += len(buys) + len(sells)
        prev = cur
    return {
        "rebalance_count": int(len(pick_log)),
        "active_rebalances": int(sum(1 for r in pick_log if r["pick_count"] > 0)),
        "empty_rebalances": int(empty_rebalances),
        "initial_and_switch_buys": int(total_buys),
        "switch_sells": int(total_sells),
        "total_symbol_side_changes": int(total_changes),
        "avg_names_changed_per_rebalance": round(total_changes / len(pick_log), 2) if pick_log else 0.0,
        "final_pick_count": int(pick_log[-1]["pick_count"]) if pick_log else 0,
    }


def walkforward_summary(
    prices: pd.DataFrame,
    top_n: int = 10,
    cost_bps: float = 10.0,
    momentum_period: int = 21,
    rebalance: str = "ME",
    test_months: int = 6,
    min_train_years: int = 2,
) -> dict:
    rsi_score = (lab_rsi(prices, 22) + lab_rsi(prices, 44) + lab_rsi(prices, 66)) / 3.0
    mom_1m = prices.pct_change(momentum_period, fill_method=None)
    returns = prices.pct_change(fill_method=None).fillna(0)
    all_dates = prices.index

    start = all_dates[0]
    end_dt = all_dates[-1]
    train_end = all_dates[all_dates >= start + pd.DateOffset(years=min_train_years) - pd.Timedelta(days=1)][0]
    fold_cagrs = []
    fold_dds = []
    positive = 0
    worst = None
    fold_count = 0

    while True:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end_dt)
        if test_end <= test_start or test_end - test_start < pd.Timedelta(days=60):
            break

        test_mask = (all_dates >= test_start) & (all_dates <= test_end)
        if test_mask.sum() < 30:
            train_end = test_end
            continue
        test_dates = all_dates[test_mask]
        rb_dates = lab_rebalance_dates(test_dates, rebalance)
        if len(rb_dates) < 2:
            train_end = test_end
            continue

        fold_count += 1
        weights = pd.DataFrame(0.0, index=test_dates, columns=prices.columns)
        turnover_l = pd.Series(0.0, index=test_dates)
        prev = pd.Series(0.0, index=prices.columns)

        for i, d in enumerate(rb_dates):
            pos_idx = test_dates.get_loc(d)
            if pos_idx + 1 >= len(test_dates):
                continue
            td = test_dates[pos_idx + 1]
            ed = test_dates[test_dates.get_loc(rb_dates[i + 1])] if i + 1 < len(rb_dates) else test_dates[-1]
            target = pd.Series(0.0, index=prices.columns)
            rsi_at = rsi_score.loc[d].copy()
            mom_at = mom_1m.loc[d].copy()
            combined = rsi_at.where(mom_at > 0, 0)
            sc = combined.dropna().sort_values(ascending=False)
            picks = [s for s in sc.index if s in prices.columns and pd.notna(prices.loc[d, s]) and sc[s] > 0][:top_n]
            if picks:
                target.loc[picks] = 1.0 / len(picks)
            turnover_l.loc[td] = abs(target - prev).sum()
            prev = target
            mask = (test_dates >= td) & (test_dates <= ed)
            weights.loc[mask, :] = target.values

        gross = (weights * returns.loc[test_dates]).sum(axis=1).fillna(0)
        net = gross - turnover_l * (cost_bps / 10000.0)
        eq = (1 + net).cumprod()
        if eq.iloc[-1] > 0 and len(net) > 30:
            y = len(net) / 252
            c = eq.iloc[-1] ** (1 / y) - 1
            d = float((eq / eq.cummax() - 1).min())
        else:
            c = 0.0
            d = 0.0
        fold_cagrs.append(c * 100)
        fold_dds.append(d * 100)
        if c > 0:
            positive += 1
        worst = c * 100 if worst is None else min(worst, c * 100)
        train_end = test_end

    return {
        "fold_count": int(fold_count),
        "positive_folds": int(positive),
        "positive_fold_pct": round((positive / fold_count) * 100, 1) if fold_count else 0.0,
        "mean_test_cagr_pct": round(float(np.mean(fold_cagrs)), 2) if fold_cagrs else None,
        "worst_test_cagr_pct": round(float(min(fold_cagrs)), 2) if fold_cagrs else None,
        "mean_test_max_dd_pct": round(float(np.mean(fold_dds)), 2) if fold_dds else None,
    }


def run_config(cfg: SweepConfig, top_n: int = 10, cost_bps: float = 10.0, with_wf: bool = False) -> dict:
    prices, ctx = load_prices(cfg.min_rows, cfg.min_end_date)
    r, pick_log = strategy_daily_returns(prices, top_n=top_n, cost_bps=cost_bps, rebalance=cfg.rebalance)
    result = {
        "config": asdict(cfg),
        "symbols_loaded": int(ctx["symbols_loaded"]),
        "headline": metrics_from_returns(r),
        "trade_stats": trade_stats(pick_log),
    }
    if with_wf:
        result["walk_forward"] = walkforward_summary(prices, top_n=top_n, cost_bps=cost_bps, rebalance=cfg.rebalance)
    return result


def main() -> int:
    configs = [
        SweepConfig("validated_700_fresh_ME", 700, "2026-04-17", "ME"),
        SweepConfig("validated_700_fresh_WFRI", 700, "2026-04-17", "W-FRI"),
        SweepConfig("broader_500_fresh_ME", 500, "2026-04-17", "ME"),
        SweepConfig("broader_500_fresh_WFRI", 500, "2026-04-17", "W-FRI"),
        SweepConfig("broader_400_fresh_ME", 400, "2026-04-17", "ME"),
        SweepConfig("broader_400_fresh_WFRI", 400, "2026-04-17", "W-FRI"),
        SweepConfig("broader_250_loose_ME", 250, "", "ME"),
        SweepConfig("broader_250_loose_WFRI", 250, "", "W-FRI"),
    ]
    wf_names = {
        "validated_700_fresh_ME",
        "validated_700_fresh_WFRI",
        "broader_250_loose_ME",
        "broader_250_loose_WFRI",
    }

    results = []
    for cfg in configs:
        results.append(run_config(cfg, with_wf=cfg.name in wf_names))

    ranked_by_xirr = sorted(results, key=lambda x: x["headline"]["xirr_pct"], reverse=True)
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "top_n": 10,
        "cost_bps": 10.0,
        "results": results,
        "ranked_by_xirr": [
            {
                "name": r["config"]["name"],
                "symbols_loaded": r["symbols_loaded"],
                "rebalance": r["config"]["rebalance"],
                "xirr_pct": r["headline"]["xirr_pct"],
                "cagr_pct": r["headline"]["cagr_pct"],
                "max_drawdown_pct": r["headline"]["max_drawdown_pct"],
            }
            for r in ranked_by_xirr
        ],
    }

    latest = OUT_DIR / "rsi_momentum_universe_rebalance_sweep_latest.json"
    stamped = OUT_DIR / f"rsi_momentum_universe_rebalance_sweep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    latest.write_text(json.dumps(summary, indent=2))
    stamped.write_text(json.dumps(summary, indent=2))

    print(f"Saved: {latest}")
    print(f"Saved: {stamped}")
    for row in ranked_by_xirr[:5]:
        print(
            f"{row['config']['name']}: symbols={row['symbols_loaded']} "
            f"rebalance={row['config']['rebalance']} XIRR={row['headline']['xirr_pct']:.2f}% "
            f"DD={row['headline']['max_drawdown_pct']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
