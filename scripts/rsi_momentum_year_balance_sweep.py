#!/usr/bin/env python3
"""Search RSI+momentum variants that improve weak calendar years.

The validated top10 monthly RSI+momentum strategy has a strong headline CAGR,
but 2022 is nearly flat. This script sweeps simple, explainable variants and
ranks for balanced yearly performance: better 2022 / worst-year return while
preserving strong CAGR and drawdown.

Writes:
- reports/rsi_momentum_year_balance_sweep_latest.json
- timestamped copy
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


@dataclass(frozen=True)
class Variant:
    name: str
    top_n: int
    rebalance: str
    momentum_period: int
    momentum_min: float
    rsi_mode: str  # raw, rank_pct
    vol_target: float  # 0 = no vol target
    max_weight_scale: float


def load_prices() -> tuple[pd.DataFrame, dict]:
    hist_dir = Path(find_hist_dir(""))
    prices_raw, ctx = lab_load_prices(
        hist_dir,
        min_rows=700,
        min_end_date="2026-04-17",
        symbols=set(),
        max_symbols=0,
    )
    return prices_raw.ffill(limit=3), ctx


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
        "yearly_returns_pct": {str(k): round(float(v * 100), 2) for k, v in yearly.items()},
    }


def run_variant(prices: pd.DataFrame, v: Variant, cost_bps: float = 10.0) -> tuple[pd.Series, list[dict]]:
    rsi_score = (lab_rsi(prices, 22) + lab_rsi(prices, 44) + lab_rsi(prices, 66)) / 3.0
    if v.rsi_mode == "rank_pct":
        rsi_score = rsi_score.rank(axis=1, pct=True) * 100.0
    elif v.rsi_mode != "raw":
        raise ValueError(f"unknown rsi_mode {v.rsi_mode}")

    mom = prices.pct_change(v.momentum_period, fill_method=None)
    returns = prices.pct_change(fill_method=None).fillna(0)
    dates = lab_rebalance_dates(prices.index, v.rebalance)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    turnover_l = pd.Series(0.0, index=prices.index)
    prev = pd.Series(0.0, index=prices.columns)
    pick_log: list[dict] = []

    # Universe-level realized volatility proxy for risk scaling.
    universe_ret = returns.mean(axis=1).fillna(0)
    realized_vol = universe_ret.rolling(63, min_periods=40).std() * math.sqrt(252)

    for i, d in enumerate(dates):
        pos = prices.index.get_loc(d)
        if pos + 1 >= len(prices.index):
            continue
        td = prices.index[pos + 1]
        ed = dates[i + 1] if i + 1 < len(dates) else prices.index[-1]
        target = pd.Series(0.0, index=prices.columns)
        rsi_at = rsi_score.loc[d].copy()
        mom_at = mom.loc[d].copy()
        combined = rsi_at.where(mom_at > v.momentum_min, 0)
        scored = combined.dropna().sort_values(ascending=False)
        picks = [s for s in scored.index if s in prices.columns and pd.notna(prices.loc[d, s]) and scored[s] > 0][: v.top_n]
        if picks:
            scale = 1.0
            if v.vol_target > 0 and pd.notna(realized_vol.loc[d]) and realized_vol.loc[d] > 0:
                scale = min(v.max_weight_scale, v.vol_target / float(realized_vol.loc[d]))
            target.loc[picks] = scale / len(picks)
        turnover_l.loc[td] = abs(target - prev).sum()
        prev = target
        mask = (prices.index >= td) & (prices.index <= ed)
        weights.loc[mask, :] = target.values
        pick_log.append({"signal_date": str(d.date()), "trade_date": str(td.date()), "pick_count": int(len(picks)), "gross_exposure": round(float(target.sum()), 3), "picks": picks})

    gross = (weights * returns).sum(axis=1)
    net = gross - turnover_l * (cost_bps / 10000.0)
    active = weights.sum(axis=1) > 0
    return net.loc[active].copy(), pick_log


def trade_stats(pick_log: list[dict]) -> dict:
    prev = set()
    buys = sells = changes = 0
    for row in pick_log:
        cur = set(row["picks"])
        buys += len(cur - prev)
        sells += len(prev - cur)
        changes += len(cur - prev) + len(prev - cur)
        prev = cur
    exposures = [row["gross_exposure"] for row in pick_log]
    return {
        "rebalance_count": len(pick_log),
        "buys": buys,
        "sells": sells,
        "total_side_changes": changes,
        "avg_side_changes_per_rebalance": round(changes / len(pick_log), 2) if pick_log else 0.0,
        "avg_gross_exposure": round(float(np.mean(exposures)), 3) if exposures else 0.0,
        "min_gross_exposure": round(float(np.min(exposures)), 3) if exposures else 0.0,
    }


def score_variant(m: dict) -> float:
    yearly = m["yearly_returns_pct"]
    y2022 = yearly.get("2022", -999.0)
    worst = m["worst_year_pct"]
    cagr = m["cagr_pct"]
    dd = abs(m["max_drawdown_pct"])
    # Balanced objective: reward 2022/worst-year heavily, preserve CAGR, penalize DD.
    return round((1.8 * y2022) + (1.2 * worst) + (0.45 * cagr) - (0.25 * dd), 4)


def main() -> int:
    prices, ctx = load_prices()
    variants: list[Variant] = []
    for top_n in [6, 8, 10, 12, 15]:
        for rebalance in ["ME", "W-FRI"]:
            for momentum_period in [21, 42, 63]:
                for momentum_min in [0.0, 0.03, 0.05, 0.10]:
                    variants.append(Variant(
                        name=f"top{top_n}_{rebalance}_mom{momentum_period}_min{momentum_min:g}_raw",
                        top_n=top_n,
                        rebalance=rebalance,
                        momentum_period=momentum_period,
                        momentum_min=momentum_min,
                        rsi_mode="raw",
                        vol_target=0.0,
                        max_weight_scale=1.0,
                    ))
    # Risk-scaled variants around the validated config.
    for vt in [0.12, 0.15, 0.18, 0.22]:
        variants.append(Variant(
            name=f"top10_ME_mom21_min0_voltarget{vt:g}",
            top_n=10,
            rebalance="ME",
            momentum_period=21,
            momentum_min=0.0,
            rsi_mode="raw",
            vol_target=vt,
            max_weight_scale=1.0,
        ))
    # Cross-sectional percentile ranking variants.
    for top_n in [8, 10, 12]:
        for momentum_min in [0.0, 0.03, 0.05]:
            variants.append(Variant(
                name=f"top{top_n}_ME_mom21_min{momentum_min:g}_rankpct",
                top_n=top_n,
                rebalance="ME",
                momentum_period=21,
                momentum_min=momentum_min,
                rsi_mode="rank_pct",
                vol_target=0.0,
                max_weight_scale=1.0,
            ))

    results = []
    for v in variants:
        try:
            r, pick_log = run_variant(prices, v)
            m = metrics_from_returns(r)
            results.append({
                "variant": asdict(v),
                "metrics": m,
                "trade_stats": trade_stats(pick_log),
                "balance_score": score_variant(m),
            })
        except Exception as exc:
            results.append({"variant": asdict(v), "error": repr(exc), "balance_score": -999999})

    baseline = next(r for r in results if r.get("variant", {}).get("name") == "top10_ME_mom21_min0_raw")
    # Candidate filters: must beat 2022 materially and keep reasonable headline.
    filtered = [
        r for r in results
        if "metrics" in r
        and r["metrics"]["yearly_returns_pct"].get("2022", -999) > baseline["metrics"]["yearly_returns_pct"].get("2022", -999) + 3
        and r["metrics"]["cagr_pct"] >= 30
        and r["metrics"]["max_drawdown_pct"] >= -30
    ]
    ranked_balanced = sorted(filtered, key=lambda r: r["balance_score"], reverse=True)
    ranked_all = sorted([r for r in results if "metrics" in r], key=lambda r: r["balance_score"], reverse=True)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "data_context": ctx,
        "baseline": baseline,
        "best_balanced_filtered": ranked_balanced[:25],
        "best_balanced_all": ranked_all[:25],
        "all_results_count": len(results),
    }
    latest = OUT_DIR / "rsi_momentum_year_balance_sweep_latest.json"
    stamped = OUT_DIR / f"rsi_momentum_year_balance_sweep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    latest.write_text(json.dumps(report, indent=2))
    stamped.write_text(json.dumps(report, indent=2))
    print(f"Saved: {latest}")
    print(f"Saved: {stamped}")
    print("BASELINE", baseline["variant"]["name"], baseline["metrics"]["cagr_pct"], baseline["metrics"]["yearly_returns_pct"])
    for r in ranked_balanced[:10]:
        m = r["metrics"]
        print(r["variant"]["name"], "score", r["balance_score"], "CAGR", m["cagr_pct"], "DD", m["max_drawdown_pct"], "2022", m["yearly_returns_pct"].get("2022"), "worst", m["worst_year_pct"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
