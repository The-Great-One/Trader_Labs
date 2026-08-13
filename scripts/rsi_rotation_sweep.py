#!/usr/bin/env python3
"""RSI Rotation Auto-Iteration Lab — sweeps variations on the winning RSI momentum strategy.

Tests:
  1. RSI period combos (7/14/21, 14/28/42, 10/20/30, single periods)
  2. Score blending (RSI + momentum, RSI + volume, RSI only)
  3. Exit rules (trailing stop, time stop, rebalance-only)
  4. Position sizing (equal, RSI-weighted, vol-weighted)
  5. Universe filters (min volume, momentum gate)
  6. Regime detection (SMA cross, volatility)
  7. Rebalance frequency (weekly, biweekly, monthly)
  8. Top-N sweep (5-30)

Output: reports/rsi_rotation_sweep_latest.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("AT_DISABLE_FILE_LOGGING", "1")
os.environ.setdefault("AT_RESEARCH_MODE", "1")
os.environ.setdefault("AT_LAB_PRECACHE", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rsi_224466_rotation_lab import (
    load_prices, rsi_dataframe, rebalance_dates, find_hist_dir,
    build_regime_mask, metrics, RotationResult,
)

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)
OUTPUT = OUT_DIR / "rsi_rotation_sweep_latest.json"


# ── Score builders ───────────────────────────────────────────────────────

def build_rsi_score(prices: pd.DataFrame, periods: list[int]) -> pd.DataFrame:
    """Average RSI across given periods."""
    scores = [rsi_dataframe(prices, p) for p in periods]
    return sum(scores) / len(scores)


def build_rsi_momentum_score(prices: pd.DataFrame, rsi_periods: list[int], mom_period: int, mom_weight: float) -> pd.DataFrame:
    """Blend RSI average with momentum. Only positive momentum gets the blend."""
    rsi_score = build_rsi_score(prices, rsi_periods)
    mom = prices.pct_change(mom_period, fill_method=None)
    # Zero out RSI score where momentum is negative
    blended = rsi_score.where(mom > 0, 0)
    return blended


def build_rsi_volume_score(prices: pd.DataFrame, rsi_periods: list[int], vol_lookback: int) -> pd.DataFrame:
    """RSI score multiplied by volume ratio (higher vol = higher conviction)."""
    rsi_score = build_rsi_score(prices, rsi_periods)
    # Volume proxy: recent range / average range
    vol_ratio = (prices.rolling(vol_lookback).max() - prices.rolling(vol_lookback).min()) / \
                (prices.rolling(vol_lookback * 3).max() - prices.rolling(vol_lookback * 3).min()).replace(0, 1)
    return rsi_score * vol_ratio.clip(0.5, 2.0)


# ── Exit rule variants ───────────────────────────────────────────────────

def run_rotation_with_exits(
    prices_raw: pd.DataFrame,
    score: pd.DataFrame,
    rebalance: str,
    top_n: int,
    cost_bps: float,
    regime: str,
    ffill_limit: int,
    exit_atr_mult: float = 0,  # 0 = no trailing stop
    exit_max_hold: int = 0,    # 0 = no time stop
    sizing: str = "equal",     # "equal", "rsi_weighted", "vol_inverse"
) -> tuple[RotationResult, dict]:
    """RSI rotation with optional trailing stop and time stop exits."""
    prices = prices_raw.ffill(limit=ffill_limit)
    returns = prices.pct_change(fill_method=None).fillna(0)
    dates = rebalance_dates(prices.index, rebalance)
    regime_mask = build_regime_mask(prices, regime).fillna(False)

    # For trailing stop, compute ATR
    atr = None
    if exit_atr_mult > 0:
        tr = pd.DataFrame({
            "h_l": prices.rolling(2).max() - prices.rolling(2).min(),
            "h_c": (prices.rolling(2).max() - prices.shift(1)).abs(),
            "l_c": (prices.rolling(2).min() - prices.shift(1)).abs(),
        }).max(axis=1)
        atr = tr.rolling(14).mean()

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    turnover = pd.Series(0.0, index=prices.index)
    previous = pd.Series(0.0, index=prices.columns)
    picks_log: list[dict] = []

    # Track individual position entries for trailing stop
    entry_prices: dict[str, float] = {}
    highest_since_entry: dict[str, float] = {}
    entry_dates: dict[str, pd.Timestamp] = {}

    for i, d in enumerate(dates):
        pos = prices.index.get_loc(d)
        if pos + 1 >= len(prices.index):
            continue
        trade_date = prices.index[pos + 1]
        end_date = dates[i + 1] if i + 1 < len(dates) else prices.index[-1]

        # Check for trailing stop exits before rebalance
        if exit_atr_mult > 0 and atr is not None and entry_prices:
            exited = []
            for sym in list(entry_prices.keys()):
                if sym not in prices.columns:
                    exited.append(sym)
                    continue
                current = prices.loc[d, sym]
                if pd.isna(current):
                    continue
                highest = highest_since_entry.get(sym, current)
                if current > highest:
                    highest_since_entry[sym] = current
                stop = highest - exit_atr_mult * atr.loc[d, sym] if sym in atr.columns else highest * 0.95
                if pd.notna(stop) and current < stop:
                    exited.append(sym)
            for sym in exited:
                del entry_prices[sym]
                del highest_since_entry[sym]
                if sym in entry_dates:
                    del entry_dates[sym]

        # Check for time stop exits
        if exit_max_hold > 0 and entry_dates:
            expired = []
            for sym, entry_d in entry_dates.items():
                bars_held = len(prices.loc[entry_d:d].index) - 1
                if bars_held >= exit_max_hold:
                    expired.append(sym)
            for sym in expired:
                del entry_prices[sym]
                del highest_since_entry[sym]
                del entry_dates[sym]

        target = pd.Series(0.0, index=prices.columns)
        if bool(regime_mask.loc[d]):
            sc = score.loc[d].dropna().sort_values(ascending=False)
            picks = [s for s in sc.index if pd.notna(prices.loc[d, s])][:top_n]

            # Remove picks that were stopped out
            if exit_atr_mult > 0 or exit_max_hold > 0:
                picks = [s for s in picks if s not in entry_prices or s in entry_prices]

            if picks:
                if sizing == "equal":
                    target.loc[picks] = 1.0 / len(picks)
                elif sizing == "rsi_weighted":
                    raw = sc.loc[picks].clip(lower=0)
                    target.loc[picks] = (raw / raw.sum()).values
                elif sizing == "vol_inverse":
                    # Lower volatility gets higher weight
                    vol = returns[picks].rolling(20).std().iloc[-1]
                    inv_vol = 1.0 / vol.replace(0, 1)
                    target.loc[picks] = (inv_vol / inv_vol.sum()).values

                # Track entries for trailing stop
                if exit_atr_mult > 0 or exit_max_hold > 0:
                    for sym in picks:
                        entry_prices[sym] = prices.loc[d, sym]
                        highest_since_entry[sym] = prices.loc[d, sym]
                        entry_dates[sym] = trade_date

                picks_log.append({
                    "signal_date": str(d.date()),
                    "trade_date": str(trade_date.date()),
                    "picks": picks,
                    "scores": {s: round(float(sc.loc[s]), 2) for s in picks[:20]},
                })

        turnover.loc[trade_date] = abs(target - previous).sum()
        previous = target
        weights.loc[(prices.index >= trade_date) & (prices.index <= end_date), :] = target.values

    gross = (weights * returns).sum(axis=1)
    net = gross - turnover * (cost_bps / 10000.0)
    name = f"rsi_{rebalance}_top{top_n}_{regime}_exit{exit_atr_mult}x{exit_max_hold}_{sizing}"
    result = metrics(name, net, weights, turnover, {
        "rebalance": rebalance, "top_n": top_n, "cost_bps": cost_bps,
        "regime": regime, "symbols_loaded": prices.shape[1],
    })
    diagnostics = {
        "latest_picks": picks_log[-1] if picks_log else {},
        "rebalance_count": len(picks_log),
    }
    return result, diagnostics


# ── Main sweep ───────────────────────────────────────────────────────────

def main() -> int:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
    hist_dir = find_hist_dir("")
    prices_raw, data_context = load_prices(
        hist_dir=hist_dir, min_rows=700, min_end_date="2026-04-17",
        symbols=set(), max_symbols=0,
    )
    print(f"Loaded {data_context['symbols_loaded']} symbols, {data_context['date_range']}")

    all_combos: list[dict[str, Any]] = []

    # ── 1. RSI period combos ──────────────────────────────────────────
    rsi_combos = [
        ([7, 14, 21], "rsi7_14_21"),
        ([14, 28, 42], "rsi14_28_42"),
        ([10, 20, 30], "rsi10_20_30"),
        ([22, 44, 66], "rsi22_44_66"),  # baseline
        ([14], "rsi14"),
        ([21], "rsi21"),
        ([7], "rsi7"),
    ]
    for periods, label in rsi_combos:
        for reb in ["ME", "W-FRI"]:
            for top_n in [8, 10, 15, 20]:
                for regime in ["none", "universe_sma200"]:
                    all_combos.append({
                        "type": "rsi_periods",
                        "label": f"{label}_{reb}_top{top_n}_{regime}",
                        "rsi_periods": periods,
                        "rebalance": reb,
                        "top_n": top_n,
                        "regime": regime,
                        "score_type": "rsi_only",
                    })

    # ── 2. RSI + Momentum blend ───────────────────────────────────────
    for mom_period in [21, 63]:
        for mom_weight in [0.3, 0.5]:
            for reb in ["ME"]:
                for top_n in [10, 15]:
                    for regime in ["none"]:
                        all_combos.append({
                            "type": "rsi_momentum",
                            "label": f"rsi224466_mom{mom_period}w{mom_weight}_{reb}_top{top_n}",
                            "rsi_periods": [22, 44, 66],
                            "rebalance": reb,
                            "top_n": top_n,
                            "regime": regime,
                            "score_type": "rsi_momentum",
                            "mom_period": mom_period,
                            "mom_weight": mom_weight,
                        })

    # ── 3. RSI + Volume blend ────────────────────────────────────────
    for vol_lb in [10, 20]:
        for reb in ["ME"]:
            for top_n in [10, 15]:
                for regime in ["none"]:
                    all_combos.append({
                        "type": "rsi_volume",
                        "label": f"rsi224466_vol{vol_lb}_{reb}_top{top_n}",
                        "rsi_periods": [22, 44, 66],
                        "rebalance": reb,
                        "top_n": top_n,
                        "regime": regime,
                        "score_type": "rsi_volume",
                        "vol_lookback": vol_lb,
                    })

    # ── 4. Exit rules ─────────────────────────────────────────────────
    for exit_atr in [0, 2.0, 3.0]:
        for exit_hold in [0, 40, 60]:
            if exit_atr == 0 and exit_hold == 0:
                continue  # skip baseline (already covered)
            for reb in ["ME"]:
                for top_n in [10]:
                    for regime in ["none"]:
                        all_combos.append({
                            "type": "exit_rules",
                            "label": f"rsi224466_{reb}_top{top_n}_exit{exit_atr}x{exit_hold}",
                            "rsi_periods": [22, 44, 66],
                            "rebalance": reb,
                            "top_n": top_n,
                            "regime": regime,
                            "score_type": "rsi_only",
                            "exit_atr_mult": exit_atr,
                            "exit_max_hold": exit_hold,
                        })

    # ── 5. Position sizing ────────────────────────────────────────────
    for sizing in ["equal", "rsi_weighted", "vol_inverse"]:
        if sizing == "equal":
            continue  # baseline
        for reb in ["ME"]:
            for top_n in [10, 15]:
                for regime in ["none"]:
                    all_combos.append({
                        "type": "sizing",
                        "label": f"rsi224466_{reb}_top{top_n}_{sizing}",
                        "rsi_periods": [22, 44, 66],
                        "rebalance": reb,
                        "top_n": top_n,
                        "regime": regime,
                        "score_type": "rsi_only",
                        "sizing": sizing,
                    })

    # ── 6. Regime detection ───────────────────────────────────────────
    for regime in ["universe_sma50", "universe_sma100", "universe_sma200"]:
        for reb in ["ME", "W-FRI"]:
            for top_n in [10, 15]:
                all_combos.append({
                    "type": "regime",
                    "label": f"rsi224466_{reb}_top{top_n}_{regime}",
                    "rsi_periods": [22, 44, 66],
                    "rebalance": reb,
                    "top_n": top_n,
                    "regime": regime,
                    "score_type": "rsi_only",
                })

    # ── 7. Rebalance frequency ────────────────────────────────────────
    for reb in ["W-MON", "W-FRI", "2W-FRI", "ME", "MS"]:
        for top_n in [10, 15, 20]:
            for regime in ["none"]:
                all_combos.append({
                    "type": "rebalance",
                    "label": f"rsi224466_{reb}_top{top_n}_{regime}",
                    "rsi_periods": [22, 44, 66],
                    "rebalance": reb,
                    "top_n": top_n,
                    "regime": regime,
                    "score_type": "rsi_only",
                })

    # ── 8. Top-N sweep ────────────────────────────────────────────────
    for top_n in [5, 8, 10, 12, 15, 20, 25, 30]:
        for reb in ["ME"]:
            for regime in ["none"]:
                all_combos.append({
                    "type": "top_n",
                    "label": f"rsi224466_{reb}_top{top_n}_{regime}",
                    "rsi_periods": [22, 44, 66],
                    "rebalance": reb,
                    "top_n": top_n,
                    "regime": regime,
                    "score_type": "rsi_only",
                })

    # Deduplicate
    seen = set()
    unique = []
    for c in all_combos:
        key = c["label"]
        if key not in seen:
            seen.add(key)
            unique.append(c)
    all_combos = unique

    print(f"Testing {len(all_combos)} unique parameter combinations...")

    # Pre-ffill prices for score computation (match lab behavior)
    prices_ffill = prices_raw.ffill(limit=3)

    results: list[dict[str, Any]] = []
    for idx, combo in enumerate(all_combos):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(all_combos)}] {combo['label'][:80]}...")

        # Build score
        rsi_periods = combo["rsi_periods"]
        score_type = combo.get("score_type", "rsi_only")

        if score_type == "rsi_only":
            score = build_rsi_score(prices_ffill, rsi_periods)
        elif score_type == "rsi_momentum":
            score = build_rsi_momentum_score(
                prices_ffill, rsi_periods,
                combo["mom_period"], combo["mom_weight"]
            )
        elif score_type == "rsi_volume":
            score = build_rsi_volume_score(
                prices_ffill, rsi_periods,
                combo["vol_lookback"]
            )
        else:
            score = build_rsi_score(prices_ffill, rsi_periods)

        try:
            res, diag = run_rotation_with_exits(
                prices_raw, score,
                rebalance=combo["rebalance"],
                top_n=combo["top_n"],
                cost_bps=10,
                regime=combo["regime"],
                ffill_limit=3,
                exit_atr_mult=combo.get("exit_atr_mult", 0),
                exit_max_hold=combo.get("exit_max_hold", 0),
                sizing=combo.get("sizing", "equal"),
            )
            results.append({
                "label": combo["label"],
                "type": combo["type"],
                "params": combo,
                "cagr_pct": res.cagr_pct,
                "xirr_pct": res.xirr_pct,
                "max_drawdown_pct": res.max_drawdown_pct,
                "sharpe_like": res.sharpe_like,
                "total_return_pct": res.total_return_pct,
                "selection_score": res.selection_score,
                "turnover_monthly": res.turnover_monthly_equiv,
                "avg_positions": res.avg_positions,
                "positive_years": res.positive_years,
                "total_years": res.total_years,
                "worst_year_pct": res.worst_year_pct,
            })
        except Exception as e:
            print(f"    ERROR: {combo['label']}: {e}", file=sys.stderr)

    # Sort by selection score
    results.sort(key=lambda x: x.get("selection_score", 0), reverse=True)

    # Print top 40
    print(f"\n{'─' * 120}")
    print(f"{'Rank':<5} {'Type':<16} {'Label':<50} {'CAGR%':>8} {'XIRR%':>8} {'DD%':>8} {'Sharpe':>7} {'Score':>8} {'+Yrs':>5}")
    print(f"{'─' * 120}")
    for i, r in enumerate(results[:40], 1):
        print(f"{i:<5} {r['type']:<16} {r['label'][:49]:<50} {r['cagr_pct']:>8.1f} {r['xirr_pct']:>8.1f} {r['max_drawdown_pct']:>8.1f} {r['sharpe_like']:>7.2f} {r['selection_score']:>8.1f} {r['positive_years']:>4}")

    # Type summary
    print(f"\n{'─' * 90}")
    print(f"{'Category':<20} {'Best CAGR%':>10} {'Best XIRR%':>10} {'Best DD%':>10} {'Combos':>8}")
    print(f"{'─' * 90}")
    for cat in ["rsi_periods", "rsi_momentum", "rsi_volume", "exit_rules", "sizing", "regime", "rebalance", "top_n"]:
        cat_results = [r for r in results if r["type"] == cat]
        if cat_results:
            best = cat_results[0]
            print(f"{cat:<20} {best['cagr_pct']:>10.1f} {best['xirr_pct']:>10.1f} {best['max_drawdown_pct']:>10.1f} {len(cat_results):>8}")

    # Write output
    output = {
        "generated_at": datetime.now().isoformat(),
        "date_range": data_context["date_range"],
        "symbols_loaded": data_context["symbols_loaded"],
        "combinations_tested": len(all_combos),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(results)} results to {OUTPUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
