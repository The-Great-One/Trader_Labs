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
from dataclasses import asdict, dataclass
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
    load_ohlc_prices as lab_load_ohlc_prices,
    rebalance_dates as lab_rebalance_dates,
    rsi_dataframe as lab_rsi,
    build_regime_mask,
)
from scripts.portfolio_simulator import (  # noqa: E402
    ExecutionDataError,
    PortfolioSimulator,
    SignalIntent,
)
from scripts.walk_forward import evaluate_walk_forward  # noqa: E402

HIST_DIR = REPO / "intermediary_files" / "Hist_Data"
OUTPUT = REPO / "reports" / "auto_iteration_latest.json"
HISTORY = REPO / "reports" / "auto_iteration_history.jsonl"
CONFIG = REPO / "config" / "auto_iteration_lab.json"
SCORING_VERSION = "v5.1_calendar_consistency"
OUTPUT.parent.mkdir(exist_ok=True)

# Live production baseline (must match cron env + deployed paper trader).
# 2026-08: champion config (vol_weight, vol_lookback 10) promoted to live on
# the RSI paper trader — the lab's baseline control must mirror deployment.
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
    "vol_weight": True,
    "vol_lookback": 10,
}


class ConfigError(ValueError):
    """Strict lab configuration error raised before any market-data access."""


@dataclass(frozen=True)
class ScoreWeightsConfig:
    worst_year: float
    median_year: float
    sharpe: float
    rolling_12m_min: float
    drawdown: float
    annual_mad: float
    turnover: float
    outlier_excess: float


@dataclass(frozen=True)
class ConsistencyConfig:
    min_complete_years: int
    min_trading_days_per_year: int
    min_year_return_pct: float
    min_sharpe_ratio: float
    min_cost_bps_for_champion: float
    max_drawdown_abs_pct: float
    max_year_to_median_ratio: float
    min_rolling_12m_return_pct: float
    rolling_year_days: int
    score_weights: ScoreWeightsConfig
    disqualification_penalty: float


@dataclass(frozen=True)
class WalkForwardConfig:
    min_train_years: int
    test_months: int
    step_months: int
    min_test_days: int
    min_folds: int
    min_worst_fold_return_pct: float
    min_stitched_sharpe: float
    max_stitched_drawdown_abs_pct: float
    min_baseline_outperformance_ratio: float
    max_fold_to_median_ratio: float


@dataclass(frozen=True)
class ExecutionConfig:
    min_execution_open_coverage: float
    min_held_close_coverage: float
    max_close_ffill_rows: int
    allowed_rebalance_frequencies: tuple[str, ...]


@dataclass(frozen=True)
class SchemaConfig:
    result_schema_version: str
    execution_model: str


@dataclass(frozen=True)
class LabConfig:
    consistency: ConsistencyConfig
    walk_forward: WalkForwardConfig
    execution: ExecutionConfig
    schema: SchemaConfig

    def __getitem__(self, section: str) -> dict[str, Any]:
        value = getattr(self, section)
        return asdict(value)


def _strict_keys(section: str, value: Any, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{section} must be an object")
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing:
        raise ConfigError(f"{section} missing keys: {sorted(missing)}")
    if unknown:
        raise ConfigError(f"{section} unknown keys: {sorted(unknown)}")
    return value


def _number(section: str, key: str, value: Any, *, integer: bool = False, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{section}.{key} must be numeric")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ConfigError(f"{section}.{key} must be finite")
    if integer and numeric != int(numeric):
        raise ConfigError(f"{section}.{key} must be an integer")
    if minimum is not None and numeric < minimum:
        raise ConfigError(f"{section}.{key} must be >= {minimum}")
    if maximum is not None and numeric > maximum:
        raise ConfigError(f"{section}.{key} must be <= {maximum}")
    return int(numeric) if integer else numeric


def validate_lab_config(payload: Any) -> LabConfig:
    top = _strict_keys("config", payload, {"consistency", "walk_forward", "execution", "schema"})
    consistency_keys = {
        "min_complete_years", "min_trading_days_per_year", "min_year_return_pct",
        "min_sharpe_ratio", "min_cost_bps_for_champion", "max_drawdown_abs_pct",
        "max_year_to_median_ratio", "min_rolling_12m_return_pct", "rolling_year_days",
        "score_weights", "disqualification_penalty",
    }
    c = _strict_keys("consistency", top["consistency"], consistency_keys)
    weight_keys = set(ScoreWeightsConfig.__dataclass_fields__)
    weights_raw = _strict_keys("consistency.score_weights", c["score_weights"], weight_keys)
    weights = ScoreWeightsConfig(**{
        key: _number("consistency.score_weights", key, value, minimum=0)
        for key, value in weights_raw.items()
    })
    consistency = ConsistencyConfig(
        min_complete_years=_number("consistency", "min_complete_years", c["min_complete_years"], integer=True, minimum=0),
        min_trading_days_per_year=_number("consistency", "min_trading_days_per_year", c["min_trading_days_per_year"], integer=True, minimum=1),
        min_year_return_pct=_number("consistency", "min_year_return_pct", c["min_year_return_pct"]),
        min_sharpe_ratio=_number("consistency", "min_sharpe_ratio", c["min_sharpe_ratio"]),
        min_cost_bps_for_champion=_number("consistency", "min_cost_bps_for_champion", c["min_cost_bps_for_champion"], minimum=0),
        max_drawdown_abs_pct=_number("consistency", "max_drawdown_abs_pct", c["max_drawdown_abs_pct"], minimum=0, maximum=100),
        max_year_to_median_ratio=_number("consistency", "max_year_to_median_ratio", c["max_year_to_median_ratio"], minimum=0),
        min_rolling_12m_return_pct=_number("consistency", "min_rolling_12m_return_pct", c["min_rolling_12m_return_pct"]),
        rolling_year_days=_number("consistency", "rolling_year_days", c["rolling_year_days"], integer=True, minimum=2),
        score_weights=weights,
        disqualification_penalty=_number("consistency", "disqualification_penalty", c["disqualification_penalty"], minimum=0),
    )
    wf_keys = set(WalkForwardConfig.__dataclass_fields__)
    w = _strict_keys("walk_forward", top["walk_forward"], wf_keys)
    walk_forward = WalkForwardConfig(
        min_train_years=_number("walk_forward", "min_train_years", w["min_train_years"], integer=True, minimum=1),
        test_months=_number("walk_forward", "test_months", w["test_months"], integer=True, minimum=1),
        step_months=_number("walk_forward", "step_months", w["step_months"], integer=True, minimum=1),
        min_test_days=_number("walk_forward", "min_test_days", w["min_test_days"], integer=True, minimum=1),
        min_folds=_number("walk_forward", "min_folds", w["min_folds"], integer=True, minimum=1),
        min_worst_fold_return_pct=_number("walk_forward", "min_worst_fold_return_pct", w["min_worst_fold_return_pct"]),
        min_stitched_sharpe=_number("walk_forward", "min_stitched_sharpe", w["min_stitched_sharpe"]),
        max_stitched_drawdown_abs_pct=_number("walk_forward", "max_stitched_drawdown_abs_pct", w["max_stitched_drawdown_abs_pct"], minimum=0, maximum=100),
        min_baseline_outperformance_ratio=_number("walk_forward", "min_baseline_outperformance_ratio", w["min_baseline_outperformance_ratio"], minimum=0, maximum=1),
        max_fold_to_median_ratio=_number("walk_forward", "max_fold_to_median_ratio", w["max_fold_to_median_ratio"], minimum=0),
    )
    if walk_forward.step_months < walk_forward.test_months:
        raise ConfigError("walk_forward.step_months must be >= test_months to prevent overlapping folds")
    e = _strict_keys("execution", top["execution"], set(ExecutionConfig.__dataclass_fields__))
    frequencies = e["allowed_rebalance_frequencies"]
    allowed = {"W-FRI", "2W-FRI", "3W-FRI", "ME"}
    if not isinstance(frequencies, list) or not frequencies or any(item not in allowed for item in frequencies):
        raise ConfigError("execution.allowed_rebalance_frequencies contains an invalid rebalance enum")
    execution = ExecutionConfig(
        min_execution_open_coverage=_number("execution", "min_execution_open_coverage", e["min_execution_open_coverage"], minimum=0, maximum=1),
        min_held_close_coverage=_number("execution", "min_held_close_coverage", e["min_held_close_coverage"], minimum=0, maximum=1),
        max_close_ffill_rows=_number("execution", "max_close_ffill_rows", e["max_close_ffill_rows"], integer=True, minimum=0),
        allowed_rebalance_frequencies=tuple(frequencies),
    )
    schema_raw = _strict_keys("schema", top["schema"], set(SchemaConfig.__dataclass_fields__))
    for key, value in schema_raw.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"schema.{key} must be a non-empty string")
    schema = SchemaConfig(**schema_raw)
    return LabConfig(consistency, walk_forward, execution, schema)


def _load_lab_config() -> LabConfig:
    """Load and strictly validate every governance setting before data access."""
    try:
        payload = json.loads(CONFIG.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Unable to load lab configuration: {CONFIG}") from exc
    return validate_lab_config(payload)


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


def _prepare_close_inputs(
    closes: pd.DataFrame, *, max_close_ffill_rows: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the sole close forward-fill and retain each mark's source date."""
    ordered = closes.sort_index().copy()
    sources = pd.DataFrame(index=ordered.index, columns=ordered.columns, dtype="datetime64[ns]")
    for symbol in ordered.columns:
        observed = ordered[symbol].notna()
        sources.loc[observed, symbol] = ordered.index[observed]
    return ordered.ffill(limit=max_close_ffill_rows), sources.ffill(limit=max_close_ffill_rows)


def _select_target_weights(
    date: pd.Timestamp,
    *,
    closes: pd.DataFrame,
    score: pd.DataFrame,
    momentum: pd.DataFrame,
    regime: pd.DataFrame,
    macd_filter: pd.DataFrame,
    universe_mask: pd.Series,
    params: dict[str, Any],
    instruments: pd.DataFrame,
    previous: set[str],
) -> dict[str, float]:
    if not bool(universe_mask.get(date, True)):
        return {}
    rsi_at = score.loc[date].copy()
    combined = rsi_at.where(momentum.loc[date] > 0, 0)
    combined = combined.where(regime.loc[date] > 0, 0)
    combined = combined.where(macd_filter.loc[date] > 0, 0)
    combined = combined.where(rsi_at >= float(params.get("rsi_min", 0)), 0)
    combined = combined.where(rsi_at <= float(params.get("rsi_max", 100)), 0)
    turnover_penalty = float(params.get("turnover_penalty", 0))
    if turnover_penalty:
        for symbol in previous:
            if symbol in combined and combined[symbol] > 0:
                combined[symbol] += turnover_penalty
    ranked = combined.dropna().sort_values(ascending=False)
    top_n = int(params.get("top_n", 8))
    if params.get("dynamic_top_n", False):
        top_n = max(3, min(top_n, int((combined > 0).sum()) // 5))
    candidates = [symbol for symbol in ranked.index if ranked[symbol] > 0]
    max_sector = int(params.get("max_per_sector", 3))
    picks: list[str] = []
    sector_counts: dict[str, int] = {}
    for symbol in candidates:
        sector = "Unknown"
        if max_sector > 0 and not instruments.empty and "Symbol" in instruments:
            match = instruments[instruments["Symbol"].astype(str).str.upper() == str(symbol).upper()]
            if not match.empty:
                sector = str(match.iloc[0].get("Sector", "Unknown"))
            if sector_counts.get(sector, 0) >= max_sector:
                continue
        picks.append(symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(picks) == top_n:
            break
    if not picks:
        return {}
    if not params.get("vol_weight", False):
        return {symbol: 1.0 / len(picks) for symbol in picks}
    lookback = int(params.get("vol_lookback", 20))
    inverse: dict[str, float] = {}
    for symbol in picks:
        sample = closes.loc[:date, symbol].tail(lookback)
        volatility = float(sample.pct_change(fill_method=None).std())
        if not np.isfinite(volatility) or volatility <= 0:
            volatility = 1e-9
        inverse[symbol] = 1.0 / volatility
    total = sum(inverse.values())
    return {symbol: inverse[symbol] / total for symbol in picks}


def build_signal_intents(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    params: dict[str, Any],
    instruments: pd.DataFrame,
) -> list[SignalIntent]:
    """Construct causal D-close target intents; opens are used only for the calendar."""
    del opens  # execution prices are deliberately not visible to selection
    rsi_periods = tuple(params.get("rsi_periods", [22, 44, 66]))
    score = sum(lab_rsi(closes, period) for period in rsi_periods) / len(rsi_periods)
    momentum = closes.pct_change(int(params.get("momentum_period", 63)), fill_method=None)
    blend = float(params.get("blend_weight", 0.0))
    if blend > 0:
        score = (1 - blend) * score + blend * momentum.rank(axis=1, pct=True) * 100
    if params.get("use_rsi_accel", False):
        score = score + float(params.get("rsi_accel_weight", 0.15)) * score.diff(5)
    mode = params.get("regime_mode", "sma100")
    if mode == "none":
        regime = (closes > 0).astype(float)
    else:
        window = int(str(mode).replace("sma", ""))
        regime = (closes > closes.rolling(window, min_periods=window).mean()).astype(float)
    if params.get("use_universe_regime", False):
        universe_mask = build_regime_mask(closes, params.get("universe_regime_mode", "universe_sma100"))
    else:
        universe_mask = pd.Series(True, index=closes.index)
    if params.get("use_macd", True):
        fast = closes.ewm(span=12, min_periods=12).mean()
        slow = closes.ewm(span=26, min_periods=26).mean()
        line = fast - slow
        macd_filter = (line > line.ewm(span=9, min_periods=9).mean()).astype(float)
    else:
        macd_filter = (closes > 0).astype(float)

    rebalance = set(lab_rebalance_dates(closes.index, params.get("rebalance_freq", "3W-FRI")))
    intents: list[SignalIntent] = []
    target: dict[str, float] = {}
    entry_closes: dict[str, float] = {}
    dd_exit = float(params.get("dd_exit_pct", 0))
    sequence = 0
    for date in closes.index:
        new_target: dict[str, float] | None = None
        if date in rebalance:
            new_target = _select_target_weights(
                date, closes=closes, score=score, momentum=momentum, regime=regime,
                macd_filter=macd_filter, universe_mask=universe_mask, params=params,
                instruments=instruments, previous=set(target),
            )
        elif dd_exit < 0 and target:
            exited = {
                symbol for symbol in target
                if symbol in entry_closes and np.isfinite(closes.loc[date, symbol])
                and (float(closes.loc[date, symbol]) / entry_closes[symbol] - 1.0) * 100 < dd_exit
            }
            if exited:
                new_target = {symbol: weight for symbol, weight in target.items() if symbol not in exited}
        if new_target is None or new_target == target:
            continue
        sequence += 1
        old = set(target)
        target = new_target
        for symbol in set(target) - old:
            entry_closes[symbol] = float(closes.loc[date, symbol])
        for symbol in old - set(target):
            entry_closes.pop(symbol, None)
        intents.append(SignalIntent(f"signal-{sequence}-{date.date()}", date, dict(target)))
    return intents


def _execution_failure(exc: ExecutionDataError) -> dict[str, Any]:
    return {
        "reason": exc.reason,
        "signal_date": str(exc.signal_date.date()) if exc.signal_date is not None else None,
        "execution_date": str(exc.execution_date.date()) if exc.execution_date is not None else None,
        "valuation_date": str(exc.valuation_date.date()) if exc.valuation_date is not None else None,
        "missing_symbols": list(exc.missing_symbols),
        "coverage": exc.coverage,
        "source_mark_age": exc.source_mark_age,
    }


def _simulate(
    prices,
    params,
    instruments,
    consistency_config,
    *,
    evaluation_start=None,
    evaluation_end=None,
    include_daily_returns=False,
    execution_config=None,
):
    if isinstance(prices, dict):
        opens = prices["open"].sort_index()
        raw_closes = prices["close"].reindex(index=opens.index, columns=opens.columns)
    else:
        # Compatibility for callers that have not yet migrated to explicit OHLC.
        raw_closes = prices.sort_index()
        opens = raw_closes.copy()
    if raw_closes.empty or len(raw_closes) < 200:
        return None
    execution_config = execution_config or {}
    max_ffill = int(execution_config.get("max_close_ffill_rows", 3))
    closes, source_dates = _prepare_close_inputs(raw_closes, max_close_ffill_rows=max_ffill)
    intents = build_signal_intents(opens, closes, params, instruments)
    simulator = PortfolioSimulator(
        opens, closes, initial_cash=100_000.0, cost_bps=float(params.get("cost_bps", 10.0)),
        min_execution_open_coverage=float(execution_config.get("min_execution_open_coverage", 0.0)),
        min_held_close_coverage=float(execution_config.get("min_held_close_coverage", 1.0)),
        max_close_ffill_rows=max_ffill, close_source_dates=source_dates,
    )
    try:
        simulation = simulator.run(intents)
    except ExecutionDataError as exc:
        failure = _execution_failure(exc)
        return {
            "qualified": False, "qualification_failures": [exc.reason],
            "execution_failure": failure, "scoring_version": SCORING_VERSION,
            "selection_score": -float(consistency_config.get("disqualification_penalty", 100.0)),
            "accounting": {"traded_notional": 0.0, "fees": 0.0, "cash_final": 100_000.0,
                           "avg_one_way_turnover": 0.0, "fills": 0},
        }
    returns = simulation.returns
    if evaluation_start is not None:
        returns = returns.loc[returns.index >= pd.Timestamp(evaluation_start)]
    if evaluation_end is not None:
        returns = returns.loc[returns.index <= pd.Timestamp(evaluation_end)]
    if returns.empty:
        return None
    equity = (1.0 + returns).cumprod()
    final = float(equity.iloc[-1])
    years = max(len(returns) / 252.0, 0.1)
    cagr = (final ** (1.0 / years) - 1.0) * 100 if final > 0 else -100.0
    drawdown = equity / equity.cummax() - 1.0
    max_dd = float(drawdown.min() * 100.0)
    values = returns.to_numpy()
    sharpe = float(values.mean() / (values.std() + 1e-9) * np.sqrt(252)) if len(values) > 20 else 0.0
    events = simulation.execution_events
    avg_turnover = float(np.mean([event.one_way_turnover for event in events])) if events else 0.0
    consistency = _consistency_metrics(
        returns, sharpe=sharpe, max_drawdown_pct=max_dd, avg_turnover=avg_turnover,
        cost_bps=float(params.get("cost_bps", 0.0)), config=consistency_config,
    )
    result = {
        "total_return_pct": round((final - 1.0) * 100, 2), "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2), "sharpe_ratio": round(sharpe, 3),
        "avg_turnover": round(avg_turnover, 4), "rebalance_count": len(events),
        "final_equity": round(float(simulation.nav.iloc[-1]), 2), "universe_size": len(closes.columns),
        "accounting": {
            "traded_notional": simulation.final_state.traded_notional,
            "fees": simulation.final_state.cumulative_fees, "cash_final": simulation.final_state.cash,
            "avg_one_way_turnover": avg_turnover,
            "fills": sum(len(event.fills) for event in events),
        },
        **consistency,
    }
    if include_daily_returns:
        result["_daily_returns"] = {str(date)[:10]: float(value) for date, value in returns.items()}
        result["_execution_events"] = [
            {
                "signal_id": event.signal_id, "signal_date": str(event.signal_date)[:10],
                "execution_date": str(event.execution_date)[:10],
                "pre_trade_nav": event.pre_trade_nav,
                "traded_notional": event.gross_traded_notional, "fees": event.fees,
                "one_way_turnover": event.one_way_turnover,
                "fills": [fill.__dict__ for fill in event.fills],
            } for event in events
        ]
    return result


def _apply_walk_forward_readiness(
    result: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    """Attach WF evidence; consistency alone can never imply readiness."""
    result["walk_forward"] = evidence
    ready = bool(result.get("agg", {}).get("qualified")) and bool(evidence.get("qualified"))
    result["champion_ready"] = ready
    result["evidence_label"] = "champion_ready" if ready else "retrospective_only"
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
    """Return only a schema-current candidate with enforced WF readiness."""
    qualified = [
        entry
        for entry in history
        if entry.get("agg", {}).get("qualified") is True
        and entry.get("agg", {}).get("scoring_version") == SCORING_VERSION
        and entry.get("champion_ready") is True
        and entry.get("walk_forward", {}).get("qualified") is True
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
    walk_forward_config = lab_config.walk_forward
    execution_config = lab_config["execution"]
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
    instruments = _load_instruments()

    ohlc_raw, _ = lab_load_ohlc_prices(HIST_DIR, min_rows=700, min_end_date="2026-07-01",
                                       symbols=set(), max_symbols=0)
    closes_raw = ohlc_raw["close"]
    if closes_raw.empty:
        print("ERROR: No price data")
        return 1

    closes = _filter_universe(closes_raw, instruments)
    opens = ohlc_raw["open"].reindex(index=closes.index, columns=closes.columns)
    prices = {"open": opens, "close": closes}
    print(f"Loaded {len(closes.columns)} symbols, {len(closes)} days")
    print(f"Universe: {len(closes.columns)} symbols")

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

        if params.get("rebalance_freq") not in execution_config["allowed_rebalance_frequencies"]:
            raise ConfigError(
                f"strategy rebalance_freq {params.get('rebalance_freq')!r} is not allowed"
            )
        agg = _simulate(
            prices,
            params,
            instruments,
            consistency_config,
            execution_config=execution_config,
        )
        if agg is None:
            continue

        is_base = enh == "baseline"
        if is_base:
            baseline_result = {"enhancement": enh, "params": {k: v for k, v in params.items() if k != "instruments"}, "agg": agg, "is_baseline": True}

        result = {
            "enhancement": enh,
            "params": {k: v for k, v in params.items() if k not in ("instruments",)},
            "agg": agg,
            "is_baseline": is_base,
            "run_date": datetime.now().strftime("%Y-%m-%d"),
        }
        evidence = evaluate_walk_forward(
            prices,
            candidate_params=params,
            baseline_params=BASELINE,
            instruments=instruments,
            config=walk_forward_config,
            execution_config=lab_config.execution,
            intent_builder=build_signal_intents,
        )
        results.append(_apply_walk_forward_readiness(result, evidence))

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

    # A research champion requires consistency plus every walk-forward gate.
    new_champ = next((result for result in results if result.get("champion_ready") is True), None)
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
        "walk_forward_config": asdict(walk_forward_config),
        "walk_forward_evidence_note": "Retrospective fixed-candidate stability only; candidate discovery used full history.",
        "date_range": f"{closes.index[0].date()} -> {closes.index[-1].date()}",
        "symbols_loaded": len(closes.columns),
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
