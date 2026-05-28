#!/usr/bin/env python3
"""Compare official RSI+momentum lab logic with deployed paper-shadow logic.

This script quantifies parity gaps between:
- scripts/rsi_momentum_report.py style research validation logic
- scripts/rsi_momentum_paper_shadow.py deployed paper-shadow logic

Outputs:
- reports/rsi_momentum_parity_check_latest.json
- timestamped copy under reports/
"""

from __future__ import annotations

import json
import math
import sys
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
from scripts.rsi_momentum_paper_shadow import compute_rotation as paper_compute_rotation
from scripts.rsi_momentum_paper_shadow import load_hist as paper_load_hist

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)


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


def official_load_prices() -> tuple[pd.DataFrame, dict]:
    hist_dir = find_hist_dir("")
    prices_raw, ctx = lab_load_prices(
        hist_dir,
        min_rows=700,
        min_end_date="2026-04-17",
        symbols=set(),
        max_symbols=0,
    )
    return prices_raw.ffill(limit=3), ctx


def official_strategy_returns(prices: pd.DataFrame, top_n: int = 10, cost_bps: float = 10.0, momentum_period: int = 21) -> tuple[pd.Series, list[dict]]:
    rsi_score = (lab_rsi(prices, 22) + lab_rsi(prices, 44) + lab_rsi(prices, 66)) / 3.0
    mom_1m = prices.pct_change(momentum_period, fill_method=None)
    returns = prices.pct_change(fill_method=None).fillna(0)
    dates = lab_rebalance_dates(prices.index, "ME")
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
        picks = [s for s in scored.index if s in prices.columns and pd.notna(prices.loc[d, s]) and scored[s] > 0][:top_n]
        if picks:
            target.loc[picks] = 1.0 / len(picks)
        turnover_l.loc[td] = abs(target - prev).sum()
        prev = target
        mask = (prices.index >= td) & (prices.index <= ed)
        weights.loc[mask, :] = target.values
        pick_log.append({"signal_date": str(d.date()), "trade_date": str(td.date()), "pick_count": int(len(picks)), "picks": picks})

    gross = (weights * returns).sum(axis=1)
    net = gross - turnover_l * (cost_bps / 10000.0)
    active = weights.sum(axis=1) > 0
    return net.loc[active].copy(), pick_log


def safe_number(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def compare_pick_logs(official: list[dict], paper_result: dict) -> dict:
    official_last = official[-1] if official else {"signal_date": None, "picks": []}
    paper_latest = paper_result.get("latest_signal", {})
    off = set(official_last.get("picks", []))
    pap = set(paper_latest.get("picks", []))
    return {
        "official_signal_date": official_last.get("signal_date"),
        "paper_signal_date": paper_latest.get("date"),
        "official_pick_count": len(off),
        "paper_pick_count": len(pap),
        "overlap_count": len(off & pap),
        "official_only": sorted(off - pap),
        "paper_only": sorted(pap - off),
        "exact_match": sorted(off) == sorted(pap),
    }


def main() -> int:
    official_prices, official_ctx = official_load_prices()
    official_returns, official_picks = official_strategy_returns(official_prices)
    official_metrics = metrics_from_returns(official_returns)

    paper_prices = paper_load_hist(ROOT / "intermediary_files" / "Hist_Data")
    paper_result = paper_compute_rotation(paper_prices)
    paper_metrics = paper_result.get("backtest_metrics", {})

    metric_fields = ["cagr_pct", "xirr_pct", "total_return_pct", "max_drawdown_pct", "vol_pct", "sharpe"]
    metric_deltas = {}
    for field in metric_fields:
        left = safe_number(official_metrics.get(field if field != "sharpe" else "sharpe_like"))
        right = safe_number(paper_metrics.get(field if field != "sharpe" else "sharpe"))
        if left is None or right is None:
            metric_deltas[field] = None
        else:
            metric_deltas[field] = round(left - right, 4)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "official_research": {
            "data_context": official_ctx,
            "metrics": official_metrics,
        },
        "paper_shadow": {
            "symbols_loaded": int(len(paper_prices.columns)),
            "date_range": [str(paper_prices.index.min().date()), str(paper_prices.index.max().date())] if not paper_prices.empty else None,
            "latest_result": paper_result,
        },
        "latest_pick_parity": compare_pick_logs(official_picks, paper_result),
        "metric_deltas_official_minus_paper": metric_deltas,
        "structural_differences": [
            "official loader uses min_rows=700 and min_end_date=2026-04-17; paper-shadow uses MIN_ROWS env default 500 and no min_end_date freshness filter",
            "official logic uses rsi_dataframe from rsi_224466_rotation_lab; paper-shadow uses its own EWM-style RSI implementation",
            "official report loads 363 symbols in current dataset; paper-shadow currently loads a broader universe",
        ],
    }

    latest = OUT_DIR / "rsi_momentum_parity_check_latest.json"
    stamped = OUT_DIR / f"rsi_momentum_parity_check_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    latest.write_text(json.dumps(report, indent=2))
    stamped.write_text(json.dumps(report, indent=2))

    print(f"Saved: {latest}")
    print(f"Saved: {stamped}")
    print(
        f"Official XIRR={official_metrics['xirr_pct']:.2f}% | "
        f"Paper XIRR={float(paper_metrics.get('xirr_pct', 0.0)):.2f}% | "
        f"Pick overlap={report['latest_pick_parity']['overlap_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
