#!/usr/bin/env python3
"""Auto-iteration strategy lab v4 — self-improving enhancement framework.

Each night the lab:
1. Reads the history of past results (auto_iteration_history.jsonl)
2. Identifies the current best-performing enhancement
3. Builds a parameter grid that:
   a. Always re-tests the baseline (live params) as a control
   b. Re-tests the current champion to confirm it still wins
   c. Explores NEW variations around the champion's parameters
   d. Tests one new enhancement idea it hasn't tried yet
4. Saves results to history
5. Outputs the latest report with champion + challengers

The lab evolves its search based on what actually worked in past runs,
not a fixed grid. Over time it converges toward better and better configs.

Output:
  reports/auto_iteration_latest.json  (latest run summary)
  reports/auto_iteration_history.jsonl (append-only history)
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

os.environ.setdefault("AT_DISABLE_FILE_LOGGING", "1")
os.environ.setdefault("AT_RESEARCH_MODE", "1")
os.environ.setdefault("AT_LAB_PRECACHE", "0")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.rsi_224466_rotation_lab import (  # noqa: E402
    load_prices as lab_load_prices,
    rebalance_dates as lab_rebalance_dates,
    rsi_dataframe as lab_rsi,
    build_regime_mask,
)

HIST_DIR = REPO / "intermediary_files" / "Hist_Data"
OUTPUT = REPO / "reports" / "auto_iteration_latest.json"
HISTORY = REPO / "reports" / "auto_iteration_history.jsonl"
CONFIG = REPO / "config" / "auto_iteration_lab.json"
SCORING_VERSION = "v5.1_calendar_consistency"
OUTPUT.parent.mkdir(exist_ok=True)

# Live production baseline (must match cron env)
BASELINE = {
    "rsi_periods": [22, 44, 66],
    "momentum_period": 63,
    "regime_mode": "sma100",
    "use_macd": True,
    "top_n": 8,
    "rebalance_freq": "3W-FRI",
    "cost_bps": 10.0,
    "max_per_sector": 3,
    "blend_weight": 0.3,
}


def _load_lab_config() -> dict[str, Any]:
    """Load the lab's explicit qualification gates and scoring weights."""
    try:
        payload = json.loads(CONFIG.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load lab configuration: {CONFIG}") from exc
    consistency = payload.get("consistency")
    if not isinstance(consistency, dict):
        raise RuntimeError(f"Missing consistency configuration in {CONFIG}")
    return payload


def _consistency_metrics(
    daily_returns: pd.Series,
    *,
    sharpe: float,
    max_drawdown_pct: float,
    avg_turnover: float,
    cost_bps: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Score repeatability using complete calendar years and rolling 12-month returns."""
    returns = daily_returns.sort_index().astype(float)
    date_index = pd.DatetimeIndex(returns.index)
    returns.index = date_index
    yearly_returns: dict[str, float] = {}
    yearly_counts: dict[int, int] = {}
    years = np.array([int(str(timestamp)[:4]) for timestamp in date_index], dtype=int)
    for year_number in np.unique(years):
        values = returns.iloc[years == year_number]
        year_number = int(year_number)
        yearly_counts[year_number] = len(values)
        yearly_returns[str(year_number)] = round(
            float((np.prod(1.0 + values) - 1.0) * 100.0), 2
        )

    min_days = int(config["min_trading_days_per_year"])
    complete_years = sorted(year for year, count in yearly_counts.items() if count >= min_days)
    partial_years = sorted(year for year, count in yearly_counts.items() if count < min_days)
    complete_returns = np.array(
        [yearly_returns[str(year)] for year in complete_years], dtype=float
    )

    median_year = float(np.median(complete_returns)) if len(complete_returns) else float("nan")
    worst_year = float(np.min(complete_returns)) if len(complete_returns) else float("nan")
    best_year = float(np.max(complete_returns)) if len(complete_returns) else float("nan")
    annual_mad = (
        float(np.median(np.abs(complete_returns - median_year)))
        if len(complete_returns)
        else float("nan")
    )
    positive_year_ratio = (
        float(np.mean(complete_returns > 0.0)) if len(complete_returns) else 0.0
    )
    outlier_ratio = (
        best_year / median_year
        if len(complete_returns) and median_year > 0.0
        else float("inf")
    )

    rolling_days = int(config["rolling_year_days"])
    rolling_returns = (1.0 + returns).rolling(rolling_days).apply(np.prod, raw=True) - 1.0
    rolling_min = (
        float(rolling_returns.dropna().min() * 100.0)
        if not rolling_returns.dropna().empty
        else float("nan")
    )

    failures: list[str] = []
    if len(complete_years) < int(config["min_complete_years"]):
        failures.append("insufficient_complete_years")
    if not len(complete_returns) or worst_year < float(config["min_year_return_pct"]):
        failures.append("weak_worst_year")
    if sharpe < float(config["min_sharpe_ratio"]):
        failures.append("low_sharpe")
    if cost_bps < float(config["min_cost_bps_for_champion"]):
        failures.append("optimistic_transaction_costs")
    if abs(max_drawdown_pct) > float(config["max_drawdown_abs_pct"]):
        failures.append("excess_drawdown")
    if outlier_ratio > float(config["max_year_to_median_ratio"]):
        failures.append("year_outlier_ratio")
    if not np.isfinite(rolling_min) or rolling_min < float(config["min_rolling_12m_return_pct"]):
        failures.append("weak_rolling_12m")

    weights = config["score_weights"]
    outlier_limit = float(config["max_year_to_median_ratio"])
    outlier_excess = (
        max(0.0, best_year - outlier_limit * median_year)
        if np.isfinite(best_year) and np.isfinite(median_year)
        else 0.0
    )
    score = (
        float(weights["worst_year"]) * (worst_year if np.isfinite(worst_year) else -100.0)
        + float(weights["median_year"]) * (median_year if np.isfinite(median_year) else -100.0)
        + float(weights["sharpe"]) * sharpe
        + float(weights["rolling_12m_min"]) * (rolling_min if np.isfinite(rolling_min) else -100.0)
        - float(weights["drawdown"]) * abs(max_drawdown_pct)
        - float(weights["annual_mad"]) * (annual_mad if np.isfinite(annual_mad) else 100.0)
        - float(weights["turnover"]) * avg_turnover
        - float(weights["outlier_excess"]) * outlier_excess
        - float(config["disqualification_penalty"]) * len(failures)
    )

    return {
        "scoring_version": SCORING_VERSION,
        "calendar_year_returns": yearly_returns,
        "complete_years": complete_years,
        "partial_years": partial_years,
        "positive_years": int(np.sum(complete_returns > 0.0)),
        "total_years": len(complete_years),
        "positive_year_ratio": round(positive_year_ratio, 3),
        "worst_year_return_pct": round(worst_year, 2) if np.isfinite(worst_year) else None,
        "median_year_return_pct": round(median_year, 2) if np.isfinite(median_year) else None,
        "best_year_return_pct": round(best_year, 2) if np.isfinite(best_year) else None,
        "annual_return_mad_pct": round(annual_mad, 2) if np.isfinite(annual_mad) else None,
        "year_to_median_ratio": round(outlier_ratio, 3) if np.isfinite(outlier_ratio) else None,
        "min_rolling_12m_return_pct": round(rolling_min, 2) if np.isfinite(rolling_min) else None,
        "qualified": not failures,
        "qualification_failures": failures,
        "selection_score": round(score, 2),
    }


# Enhancement ideas the lab cycles through (one new one per night)
ENHANCEMENT_IDEAS = [
    # RSI tuning
    {"rsi_periods": [10, 20, 30]},
    {"rsi_periods": [14, 28, 42]},
    {"rsi_periods": [20, 40, 60]},
    {"rsi_periods": [30, 60, 90]},
    {"rsi_periods": [7, 14, 21]},
    # Momentum tuning
    {"momentum_period": 21},
    {"momentum_period": 42},
    {"momentum_period": 84},
    # Blend tuning
    {"blend_weight": 0.0},
    {"blend_weight": 0.15},
    {"blend_weight": 0.25},
    {"blend_weight": 0.35},
    {"blend_weight": 0.4},
    {"blend_weight": 0.5},
    # Regime
    {"regime_mode": "sma200"},
    {"regime_mode": "none"},
    {"use_universe_regime": True, "universe_regime_mode": "universe_sma100"},
    {"use_universe_regime": True, "universe_regime_mode": "universe_sma200"},
    # Top-N
    {"top_n": 5},
    {"top_n": 6},
    {"top_n": 10},
    {"top_n": 12},
    # Rebalance
    {"rebalance_freq": "W-FRI"},
    {"rebalance_freq": "ME"},
    {"rebalance_freq": "2W-FRI"},
    # Cost sensitivity
    {"cost_bps": 0},
    {"cost_bps": 5},
    {"cost_bps": 20},
    # Sector
    {"max_per_sector": 0},
    {"max_per_sector": 2},
    {"max_per_sector": 5},
    # RSI thresholds
    {"rsi_min": 50},
    {"rsi_min": 55, "rsi_max": 80},
    {"rsi_min": 45, "rsi_max": 70},
    {"rsi_min": 40, "rsi_max": 75},
    # Volatility weighting
    {"vol_weight": True, "vol_lookback": 10},
    {"vol_weight": True, "vol_lookback": 20},
    {"vol_weight": True, "vol_lookback": 40},
    # RSI acceleration
    {"use_rsi_accel": True, "rsi_accel_weight": 0.10},
    {"use_rsi_accel": True, "rsi_accel_weight": 0.15},
    {"use_rsi_accel": True, "rsi_accel_weight": 0.25},
    # Turnover penalty
    {"turnover_penalty": 2.0},
    {"turnover_penalty": 5.0},
    {"turnover_penalty": 10.0},
    # Dynamic top-N
    {"dynamic_top_n": True},
    # Drawdown exit
    {"dd_exit_pct": -10},
    {"dd_exit_pct": -15},
    {"dd_exit_pct": -20},
    # No filters (ablation)
    {"use_macd": False},
    {"regime_mode": "none", "use_macd": False},
    # Combos
    {"rsi_min": 50, "blend_weight": 0.3},
    {"rsi_min": 50, "vol_weight": True, "vol_lookback": 20},
    {"blend_weight": 0.3, "vol_weight": True, "vol_lookback": 20},
    {"use_rsi_accel": True, "rsi_accel_weight": 0.15, "vol_weight": True, "vol_lookback": 20},
    {"rsi_min": 50, "rsi_max": 80, "blend_weight": 0.3, "use_rsi_accel": True, "rsi_accel_weight": 0.15},
    {"rsi_min": 50, "dd_exit_pct": -20, "blend_weight": 0.3},
    {"momentum_period": 84, "blend_weight": 0.3, "rsi_min": 50},
    {"rebalance_freq": "ME", "blend_weight": 0.3, "top_n": 10},
    {"rebalance_freq": "3W-FRI", "blend_weight": 0.4, "momentum_period": 84},
    {"rebalance_freq": "3W-FRI", "blend_weight": 0.3, "rsi_min": 50, "vol_weight": True, "vol_lookback": 20, "use_rsi_accel": True, "rsi_accel_weight": 0.15},
    {"rebalance_freq": "3W-FRI", "blend_weight": 0.35, "momentum_period": 84, "rsi_min": 50, "use_universe_regime": True, "universe_regime_mode": "universe_sma100"},
    {"rebalance_freq": "3W-FRI", "blend_weight": 0.3, "rsi_min": 45, "rsi_max": 75, "vol_weight": True, "vol_lookback": 20, "dd_exit_pct": -20},
    {"rebalance_freq": "3W-FRI", "blend_weight": 0.3, "rsi_min": 50, "use_rsi_accel": True, "rsi_accel_weight": 0.15, "turnover_penalty": 5.0},
    {"rebalance_freq": "3W-FRI", "blend_weight": 0.25, "momentum_period": 42, "rsi_min": 50, "use_rsi_accel": True, "rsi_accel_weight": 0.10},
]


def _load_instruments():
    p = REPO / "intermediary_files" / "Instruments.feather"
    return pd.read_feather(p) if p.exists() else pd.DataFrame()


def _filter_universe(prices, instruments, min_vol=50000, min_mcap=500.0):
    keep = []
    for col in prices.columns:
        sym = str(col).strip().upper()
        f = HIST_DIR / f"{sym}.feather"
        if min_vol > 0 and f.exists():
            try:
                df = pd.read_feather(f)
                vc = "volume" if "volume" in df.columns else ("Volume" if "Volume" in df.columns else None)
                if vc and df[vc].tail(20).mean() < min_vol:
                    continue
            except Exception:
                pass
        if min_mcap > 0 and not instruments.empty:
            m = instruments[instruments["Symbol"].str.upper() == sym]
            if not m.empty and "MarketCapCr" in m.columns:
                mc = m.iloc[0]["MarketCapCr"]
                if pd.notna(mc) and float(mc) < min_mcap:
                    continue
        keep.append(col)
    return prices[keep]


def _simulate(
    prices,
    params,
    instruments,
    consistency_config,
    *,
    evaluation_start=None,
    evaluation_end=None,
    include_daily_returns=False,
):
    if prices.empty or len(prices) < 200:
        return None

    pf = prices.ffill(limit=3)
    rsi_periods = tuple(params.get("rsi_periods", [22, 44, 66]))
    score = sum(lab_rsi(pf, p) for p in rsi_periods) / len(rsi_periods)

    mom_period = params.get("momentum_period", 63)
    mom = pf.pct_change(mom_period, fill_method=None)

    blend_w = params.get("blend_weight", 0.0)
    if blend_w > 0:
        mom_rank = mom.rank(axis=1, pct=True)
        score = (1 - blend_w) * score + blend_w * (mom_rank * 100)

    if params.get("use_rsi_accel", False):
        rsi_roc = score.diff(5)
        score = score + params.get("rsi_accel_weight", 0.15) * rsi_roc

    regime_mode = params.get("regime_mode", "sma100")
    if regime_mode == "none":
        regime = (pf > 0).astype(float)
    else:
        rw = int(regime_mode.replace("sma", ""))
        regime = (pf > pf.rolling(rw, min_periods=rw).mean()).astype(float)

    if params.get("use_universe_regime", False):
        uni_mask = build_regime_mask(pf, params.get("universe_regime_mode", "universe_sma100"))
    else:
        uni_mask = pd.Series(True, index=pf.index)

    use_macd = params.get("use_macd", True)
    if use_macd:
        ema_f = pf.ewm(span=12, min_periods=12).mean()
        ema_s = pf.ewm(span=26, min_periods=26).mean()
        macd_line = ema_f - ema_s
        macd_filter = (macd_line > macd_line.ewm(span=9, min_periods=9).mean()).astype(float)
    else:
        macd_filter = (pf > 0).astype(float)

    rebalance_freq = params.get("rebalance_freq", "3W-FRI")
    dates = lab_rebalance_dates(pf.index, rebalance_freq)
    if len(dates) < 3:
        return None
    actionable = [d for d in dates if pf.index.get_loc(d) + 1 < len(pf.index)]
    if len(actionable) < 3:
        return None

    returns = pf.pct_change(fill_method=None).fillna(0)
    cost_rate = params.get("cost_bps", 10.0) / 10000.0
    top_n = params.get("top_n", 8)
    max_sec = params.get("max_per_sector", 3)
    rsi_min = params.get("rsi_min", 0)
    rsi_max = params.get("rsi_max", 100)
    vol_weight = params.get("vol_weight", False)
    vol_lb = params.get("vol_lookback", 20)
    dynamic_n = params.get("dynamic_top_n", False)
    turnover_pen = params.get("turnover_penalty", 0)
    dd_exit = params.get("dd_exit_pct", 0)
    portfolio_returns = []
    portfolio_dates = []
    prev_picks = set()
    turnover_total = 0.0
    rebalance_count = 0
    entry_prices = {}

    for i, d in enumerate(actionable):
        ed = actionable[i + 1] if i + 1 < len(actionable) else pf.index[-1]

        in_market = uni_mask.loc[d] if d in uni_mask.index else True
        if not in_market:
            period_mask = (returns.index > d) & (returns.index <= ed)
            portfolio_returns.extend([0.0] * period_mask.sum())
            portfolio_dates.extend(returns.index[period_mask].tolist())
            prev_picks = set()
            continue

        rsi_at = score.loc[d].copy()
        mom_at = mom.loc[d].copy()
        combined = rsi_at.where(mom_at > 0, 0)
        if d in regime.index:
            combined = combined.where(regime.loc[d] > 0, 0)
        if use_macd and d in macd_filter.index:
            combined = combined.where(macd_filter.loc[d] > 0, 0)
        combined = combined.where(rsi_at >= rsi_min, 0)
        combined = combined.where(rsi_at <= rsi_max, 0)

        if turnover_pen > 0 and prev_picks:
            for s in prev_picks:
                if s in combined.index and combined[s] > 0:
                    combined[s] += turnover_pen

        scored = combined.dropna().sort_values(ascending=False)
        n_bullish = int((combined > 0).sum())
        current_n = max(3, min(top_n, n_bullish // 5)) if dynamic_n else top_n

        raw_picks = [s for s in scored.index if scored[s] > 0][:current_n * 2]

        if max_sec > 0 and not instruments.empty:
            sc = {}
            filtered = []
            for p in raw_picks:
                m = instruments[instruments["Symbol"].str.upper() == p]
                sec = str(m.iloc[0].get("Sector", "Unknown")) if not m.empty else "Unknown"
                if sc.get(sec, 0) < max_sec:
                    filtered.append(p)
                    sc[sec] = sc.get(sec, 0) + 1
                if len(filtered) >= current_n:
                    break
            picks = filtered[:current_n]
        else:
            picks = raw_picks[:current_n]

        if not picks:
            period_mask = (returns.index > d) & (returns.index <= ed)
            portfolio_returns.extend([0.0] * period_mask.sum())
            portfolio_dates.extend(returns.index[period_mask].tolist())
            prev_picks = set()
            continue

        rebalance_count += 1
        new_picks = set(picks)
        turnover_total += len(new_picks.symmetric_difference(prev_picks)) / 2

        if vol_weight:
            vols = {}
            for s in picks:
                col = pf[s].loc[:d].tail(vol_lb)
                vols[s] = float(col.pct_change().std()) if len(col) > 5 else 1.0
            iv = {s: 1.0 / (v + 1e-9) for s, v in vols.items()}
            ti = sum(iv.values())
            weights = {s: iv[s] / ti for s in picks}
        else:
            weights = {s: 1.0 / len(picks) for s in picks}

        period_mask = (returns.index > d) & (returns.index <= ed)
        period_days = returns.loc[period_mask]

        n_buy = len(new_picks - prev_picks) if i > 0 else len(picks)
        n_sell = len(prev_picks - new_picks) if i > 0 else 0
        tc = (n_buy + n_sell) * cost_rate / 2

        exited = set()
        daily_rets = []
        for idx_date in period_days.index:
            active = [s for s in picks if s not in exited]
            day_ret = sum(weights.get(s, 0) * (period_days.loc[idx_date, s] if s in period_days.columns else 0.0) for s in active)
            if dd_exit < 0:
                for s in list(active):
                    px = pf.loc[idx_date, s] if s in pf.columns else None
                    if px and not pd.isna(px) and s in entry_prices and entry_prices[s] > 0:
                        if (px / entry_prices[s] - 1) * 100 < dd_exit:
                            exited.add(s)
            daily_rets.append(day_ret - (tc / max(len(period_days), 1)))

        portfolio_returns.extend(daily_rets)
        portfolio_dates.extend(period_days.index.tolist())
        entry_prices = {s: float(pf.loc[d, s]) for s in picks if s in pf.columns}
        prev_picks = new_picks

    if not portfolio_returns:
        return None

    return_series = pd.Series(
        portfolio_returns,
        index=pd.DatetimeIndex(portfolio_dates),
        dtype=float,
    ).sort_index()
    if evaluation_start is not None:
        return_series = return_series.loc[return_series.index >= pd.Timestamp(evaluation_start)]
    if evaluation_end is not None:
        return_series = return_series.loc[return_series.index <= pd.Timestamp(evaluation_end)]
    if return_series.empty:
        return None

    rets = return_series.to_numpy()
    eq = np.cumprod(1 + rets)
    final = float(eq[-1])
    days = len(return_series)
    years = max(days / 252, 0.1)
    cagr = ((final) ** (1 / years) - 1) * 100 if final > 0 else -100.0
    peak = np.maximum.accumulate(eq)
    max_dd = float(np.max((peak - eq) / (peak + 1e-9) * 100))
    sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if len(rets) > 20 else 0.0

    avg_turn = turnover_total / max(rebalance_count, 1)
    consistency = _consistency_metrics(
        return_series,
        sharpe=sharpe,
        max_drawdown_pct=-max_dd,
        avg_turnover=avg_turn,
        cost_bps=float(params.get("cost_bps", 0.0)),
        config=consistency_config,
    )

    result = {
        "total_return_pct": round((final-1)*100, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(-max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "avg_turnover": round(avg_turn, 1),
        "rebalance_count": rebalance_count,
        "final_equity": round(final * 100000, 2),
        "universe_size": len(prices.columns),
        **consistency,
    }
    if include_daily_returns:
        result["_daily_returns"] = {
            str(date)[:10]: float(value) for date, value in return_series.items()
        }
    return result


def _read_history():
    """Read past results to find the champion."""
    if not HISTORY.exists():
        return []
    results = []
    for line in HISTORY.read_text().strip().split("\n"):
        try:
            results.append(json.loads(line))
        except Exception:
            continue
    return results


def _find_champion(history):
    """Return the best candidate that passes every consistency gate."""
    qualified = [
        entry
        for entry in history
        if entry.get("agg", {}).get("qualified") is True
        and entry.get("agg", {}).get("scoring_version") == SCORING_VERSION
    ]
    if not qualified:
        return None
    best = max(qualified, key=lambda x: x.get("agg", {}).get("selection_score", -999))
    return best


def _build_grid(history, champion):
    """Build tonight's parameter grid based on history and champion."""
    combos = []

    # 1. Always test baseline (control)
    base_params = {**BASELINE, "enhancement": "baseline"}
    combos.append(base_params)

    # 2. Re-test champion if exists
    if champion and champion.get("enhancement") != "baseline":
        champ_params = champion.get("params", {})
        champ_params = {k: v for k, v in champ_params.items() if k != "enhancement"}
        combos.append({**BASELINE, **champ_params, "enhancement": f"champion_{champion['enhancement']}"})

    # 3. Determine which enhancement ideas have been tested
    tested_keys = set()
    for h in history:
        p = h.get("params", {})
        # Create a key from non-baseline params
        diff = {k: v for k, v in p.items() if k not in BASELINE or BASELINE.get(k) != v}
        tested_keys.add(json.dumps(diff, sort_keys=True, default=str))

    # 4. Pick enhancement ideas that haven't been tested yet
    untested = []
    for idea in ENHANCEMENT_IDEAS:
        key = json.dumps(idea, sort_keys=True, default=str)
        if key not in tested_keys:
            untested.append(idea)

    # 5. Add up to 20 new ideas tonight
    random.seed(datetime.now().day)
    random.shuffle(untested)
    for idea in untested[:20]:
        label = "_".join(f"{k}={v}" for k, v in sorted(idea.items()))
        combos.append({**BASELINE, **idea, "enhancement": label})

    # 6. If we have a champion, mutate around it
    if champion and champion.get("enhancement") != "baseline":
        champ = champion.get("params", {})
        # Generate mutations of the champion's key params
        for _ in range(10):
            mutated = {**BASELINE, **champ}
            # Randomly tweak one parameter
            tweaks = [
                ("blend_weight", [0.0, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]),
                ("momentum_period", [21, 42, 63, 84]),
                ("top_n", [5, 6, 8, 10, 12]),
                ("rebalance_freq", ["W-FRI", "2W-FRI", "3W-FRI", "ME"]),
                ("rsi_min", [0, 45, 50, 55]),
                ("rsi_max", [70, 75, 80, 100]),
            ]
            k, vals = random.choice(tweaks)
            mutated[k] = random.choice(vals)
            mutated["enhancement"] = f"mutation_{k}={mutated[k]}"
            # Dedup
            key = json.dumps({k2: v2 for k2, v2 in mutated.items() if k2 != "enhancement"}, sort_keys=True, default=str)
            if key not in tested_keys:
                combos.append(mutated)
                tested_keys.add(key)

    # Deduplicate combos
    seen = set()
    unique = []
    for c in combos:
        key = json.dumps({k: v for k, v in c.items() if k != "enhancement"}, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def main():
    t0 = time.time()
    lab_config = _load_lab_config()
    consistency_config = lab_config["consistency"]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
    instruments = _load_instruments()

    prices_raw, _ = lab_load_prices(HIST_DIR, min_rows=700, min_end_date="2026-04-17",
                                     symbols=set(), max_symbols=0)
    if prices_raw.empty:
        print("ERROR: No price data")
        return 1

    prices = prices_raw.ffill(limit=3)
    print(f"Loaded {len(prices.columns)} symbols, {len(prices)} days")
    prices = _filter_universe(prices, instruments)
    print(f"Universe: {len(prices.columns)} symbols")

    # Read history and find champion
    history = _read_history()
    champion = _find_champion(history)
    if champion:
        print(f"Champion: {champion['enhancement']} (CAGR {champion['agg']['cagr_pct']:.1f}%, SelScore {champion['agg']['selection_score']:.1f})")
    else:
        print("No history — starting fresh exploration")

    # Build grid
    combos = _build_grid(history, champion)
    print(f"Testing {len(combos)} combinations ({len(history)} past results in history)...")

    results = []
    baseline_result = None
    for idx, params in enumerate(combos):
        enh = params.get("enhancement", "unknown")
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(combos)}] {enh} ({time.time()-t0:.0f}s)")

        agg = _simulate(prices, params, instruments, consistency_config)
        if agg is None:
            continue

        is_base = enh == "baseline"
        if is_base:
            baseline_result = {"enhancement": enh, "params": {k: v for k, v in params.items() if k != "instruments"}, "agg": agg, "is_baseline": True}

        results.append({
            "enhancement": enh,
            "params": {k: v for k, v in params.items() if k not in ("instruments",)},
            "agg": agg,
            "is_baseline": is_base,
            "run_date": datetime.now().strftime("%Y-%m-%d"),
        })

    # Qualified strategies always outrank unqualified high-CAGR outliers.
    results.sort(
        key=lambda x: (x["agg"]["qualified"], x["agg"]["selection_score"]),
        reverse=True,
    )

    # Append to history
    with open(HISTORY, "a") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Print results
    base_cagr = baseline_result["agg"]["cagr_pct"] if baseline_result else 0
    base_sel = baseline_result["agg"]["selection_score"] if baseline_result else 0

    print(f"\n{'='*155}")
    print(f"{'Rank':<5} {'CAGR%':>8} {'MaxDD%':>8} {'Sharpe':>7} {'WorstY':>8} {'MedY':>8} {'Roll12':>8} {'Qual':>5} {'Score':>9} {'Enhancement':<50}")
    print(f"{'='*155}")
    for i, r in enumerate(results[:20], 1):
        a = r["agg"]
        marker = " <- BASELINE" if r.get("is_baseline") else (" <- CHAMPION" if r["enhancement"].startswith("champion_") else "")
        print(f"{i:<5} {a['cagr_pct']:>8.1f} {a['max_drawdown_pct']:>8.1f} {a['sharpe_ratio']:>7.2f} "
              f"{a['worst_year_return_pct']:>8.1f} {a['median_year_return_pct']:>8.1f} "
              f"{a['min_rolling_12m_return_pct']:>8.1f} {str(a['qualified']):>5} "
              f"{a['selection_score']:>9.1f} {r['enhancement'][:49]}{marker}")
        if not a["qualified"]:
            print(f"      rejected: {', '.join(a['qualification_failures'])}")

    # New champion? Unqualified strategies can never be promoted.
    new_champ = results[0] if results and results[0]["agg"]["qualified"] else None
    if new_champ and new_champ["agg"]["selection_score"] > base_sel:
        print(f"\n*** NEW CONSISTENCY CHAMPION: {new_champ['enhancement']} ***")
        print(f"    Worst year {new_champ['agg']['worst_year_return_pct']:.1f}% · median year {new_champ['agg']['median_year_return_pct']:.1f}%")
        print(f"    Minimum rolling 12m {new_champ['agg']['min_rolling_12m_return_pct']:.1f}% · Sharpe {new_champ['agg']['sharpe_ratio']:.2f}")
    elif not new_champ:
        print("\n*** NO QUALIFIED CHAMPION — live baseline remains unchanged ***")

    # Write output
    output = {
        "generated_at": datetime.now().isoformat(),
        "version": "v5.1_consistency_optimised",
        "consistency_config": consistency_config,
        "date_range": f"{prices.index[0].date()} -> {prices.index[-1].date()}",
        "symbols_loaded": len(prices.columns),
        "combinations_tested": len(combos),
        "history_size": len(history),
        "baseline_cagr": base_cagr,
        "champion_before": champion["enhancement"] if champion else "none",
        "champion_after": new_champ["enhancement"] if new_champ else "none",
        "results": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote {len(results)} results to {OUTPUT} ({time.time()-t0:.0f}s)")
    print(f"History: {len(history)} -> {len(history) + len(results)} entries in {HISTORY}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
