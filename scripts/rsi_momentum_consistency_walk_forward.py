#!/usr/bin/env python3
"""Retrospective expanding-window walk-forward validation for RSI Momentum.

Signals for each test fold are computed with data truncated at that fold's end.
The fixed research champion is compared with the live baseline over non-overlapping
out-of-sample windows. Because the champion was discovered using the full history,
this is a retrospective stability test, not a genuinely unseen final holdout.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.auto_iteration_lab import (  # noqa: E402
    BASELINE,
    HIST_DIR,
    _filter_universe,
    _load_instruments,
    _load_lab_config,
    _simulate,
    lab_load_prices,
)

OUTPUT = REPO / "reports" / "rsi_momentum_walk_forward_latest.json"

CHAMPION = {
    **BASELINE,
    "momentum_period": 42,
    "rsi_min": 50,
    "vol_weight": True,
    "vol_lookback": 20,
    "use_rsi_accel": True,
    "rsi_accel_weight": 0.15,
}


@dataclass(frozen=True)
class FoldWindow:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_days: int


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    test_days: int
    candidate_return_pct: float
    candidate_cagr_pct: float
    candidate_max_drawdown_pct: float
    candidate_sharpe: float
    baseline_return_pct: float
    baseline_cagr_pct: float
    baseline_max_drawdown_pct: float
    baseline_sharpe: float
    candidate_outperformed: bool


def build_expanding_folds(
    dates: pd.DatetimeIndex,
    *,
    min_train_years: int,
    test_months: int,
    step_months: int,
    min_test_days: int,
) -> list[FoldWindow]:
    """Build non-overlapping expanding-train, fixed-test windows."""
    if dates.empty:
        return []
    if step_months < test_months:
        raise ValueError("step_months must be >= test_months to prevent overlapping test folds")

    ordered = pd.DatetimeIndex(dates).sort_values().unique()
    start = pd.Timestamp(ordered[0])
    end = pd.Timestamp(ordered[-1])
    test_cursor = start + pd.DateOffset(years=min_train_years)
    folds: list[FoldWindow] = []

    while test_cursor <= end:
        train_dates = ordered[ordered < test_cursor]
        intended_end = test_cursor + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        test_dates = ordered[(ordered >= test_cursor) & (ordered <= min(intended_end, end))]
        if len(train_dates) == 0:
            break
        if len(test_dates) >= min_test_days:
            folds.append(
                FoldWindow(
                    fold=len(folds) + 1,
                    train_start=pd.Timestamp(train_dates[0]),
                    train_end=pd.Timestamp(train_dates[-1]),
                    test_start=pd.Timestamp(test_dates[0]),
                    test_end=pd.Timestamp(test_dates[-1]),
                    test_days=len(test_dates),
                )
            )
        test_cursor = test_cursor + pd.DateOffset(months=step_months)

    return folds


def _series_from_result(result: dict[str, Any]) -> pd.Series:
    daily = result.pop("_daily_returns")
    return pd.Series(
        {pd.Timestamp(date): float(value) for date, value in daily.items()},
        dtype=float,
    ).sort_index()


def _stitched_metrics(returns: pd.Series) -> dict[str, Any]:
    values = returns.sort_index().astype(float)
    if values.empty:
        raise ValueError("cannot score empty stitched returns")
    equity = (1.0 + values).cumprod()
    years = len(values) / 252.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    volatility = float(values.std() * math.sqrt(252))
    sharpe = float(values.mean() * 252.0 / volatility) if volatility > 0 else 0.0
    drawdown = float((equity / equity.cummax() - 1.0).min() * 100.0)
    calendar = values.groupby(values.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    return {
        "days": len(values),
        "start": str(values.index[0].date()),
        "end": str(values.index[-1].date()),
        "total_return_pct": round(float((equity.iloc[-1] - 1.0) * 100.0), 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "max_drawdown_pct": round(drawdown, 2),
        "sharpe_ratio": round(sharpe, 3),
        "calendar_year_returns": {
            str(int(year)): round(float(value * 100.0), 2)
            for year, value in calendar.items()
        },
    }


def summarize_walk_forward(
    candidate_fold_returns: list[float],
    baseline_fold_returns: list[float],
    *,
    stitched_sharpe: float,
    stitched_drawdown_pct: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply explicit consistency gates to non-overlapping OOS fold returns."""
    candidate = np.asarray(candidate_fold_returns, dtype=float)
    baseline = np.asarray(baseline_fold_returns, dtype=float)
    if len(candidate) != len(baseline):
        raise ValueError("candidate and baseline fold counts differ")

    median = float(np.median(candidate)) if len(candidate) else float("nan")
    worst = float(np.min(candidate)) if len(candidate) else float("nan")
    best = float(np.max(candidate)) if len(candidate) else float("nan")
    outlier_ratio = best / median if len(candidate) and median > 0 else float("inf")
    outperformance_ratio = (
        float(np.mean(candidate > baseline)) if len(candidate) else 0.0
    )

    failures: list[str] = []
    if len(candidate) < int(config["min_folds"]):
        failures.append("insufficient_folds")
    if not len(candidate) or worst <= float(config["min_worst_fold_return_pct"]):
        failures.append("non_positive_worst_fold")
    if stitched_sharpe < float(config["min_stitched_sharpe"]):
        failures.append("low_stitched_sharpe")
    if abs(stitched_drawdown_pct) > float(config["max_stitched_drawdown_abs_pct"]):
        failures.append("excess_stitched_drawdown")
    if outperformance_ratio < float(config["min_baseline_outperformance_ratio"]):
        failures.append("insufficient_baseline_outperformance")
    if outlier_ratio > float(config["max_fold_to_median_ratio"]):
        failures.append("fold_outlier_ratio")

    return {
        "fold_count": len(candidate),
        "positive_folds": int(np.sum(candidate > 0.0)),
        "positive_fold_ratio": round(float(np.mean(candidate > 0.0)), 3) if len(candidate) else 0.0,
        "worst_fold_return_pct": round(worst, 2) if np.isfinite(worst) else None,
        "median_fold_return_pct": round(median, 2) if np.isfinite(median) else None,
        "best_fold_return_pct": round(best, 2) if np.isfinite(best) else None,
        "fold_return_std_pct": round(float(np.std(candidate)), 2) if len(candidate) else None,
        "fold_to_median_ratio": round(outlier_ratio, 3) if np.isfinite(outlier_ratio) else None,
        "baseline_outperformance_ratio": round(outperformance_ratio, 3),
        "passed": not failures,
        "failures": failures,
    }


def _run_candidate(
    prices: pd.DataFrame,
    instruments: pd.DataFrame,
    params: dict[str, Any],
    consistency_config: dict[str, Any],
    fold: FoldWindow,
) -> tuple[dict[str, Any], pd.Series]:
    truncated = prices.loc[: fold.test_end]
    result = _simulate(
        truncated,
        params,
        instruments,
        consistency_config,
        evaluation_start=fold.test_start,
        evaluation_end=fold.test_end,
        include_daily_returns=True,
    )
    if result is None:
        raise RuntimeError(f"simulation produced no returns for fold {fold.fold}")
    return result, _series_from_result(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward validate RSI Momentum consistency champion")
    parser.add_argument("--hist-dir", default=str(HIST_DIR))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()

    config = _load_lab_config()
    consistency_config = config["consistency"]
    walk_config = config["walk_forward"]
    hist_dir = Path(args.hist_dir)

    prices_raw, _ = lab_load_prices(
        hist_dir,
        min_rows=700,
        min_end_date="2026-04-17",
        symbols=set(),
        max_symbols=0,
    )
    if prices_raw.empty:
        raise RuntimeError(f"no usable price data in {hist_dir}")
    instruments = _load_instruments()
    prices = _filter_universe(prices_raw.ffill(limit=3), instruments)
    folds = build_expanding_folds(
        prices.index,
        min_train_years=int(walk_config["min_train_years"]),
        test_months=int(walk_config["test_months"]),
        step_months=int(walk_config["step_months"]),
        min_test_days=int(walk_config["min_test_days"]),
    )
    if not folds:
        raise RuntimeError("walk-forward configuration produced no folds")

    fold_results: list[FoldResult] = []
    candidate_oos: list[pd.Series] = []
    baseline_oos: list[pd.Series] = []

    print(f"Loaded {len(prices.columns)} symbols, {len(prices)} days")
    print(f"Running {len(folds)} expanding-window folds...")
    for fold in folds:
        candidate_result, candidate_returns = _run_candidate(
            prices, instruments, CHAMPION, consistency_config, fold
        )
        baseline_result, baseline_returns = _run_candidate(
            prices, instruments, BASELINE, consistency_config, fold
        )
        candidate_oos.append(candidate_returns)
        baseline_oos.append(baseline_returns)
        row = FoldResult(
            fold=fold.fold,
            train_start=str(fold.train_start.date()),
            train_end=str(fold.train_end.date()),
            test_start=str(fold.test_start.date()),
            test_end=str(fold.test_end.date()),
            test_days=fold.test_days,
            candidate_return_pct=float(candidate_result["total_return_pct"]),
            candidate_cagr_pct=float(candidate_result["cagr_pct"]),
            candidate_max_drawdown_pct=float(candidate_result["max_drawdown_pct"]),
            candidate_sharpe=float(candidate_result["sharpe_ratio"]),
            baseline_return_pct=float(baseline_result["total_return_pct"]),
            baseline_cagr_pct=float(baseline_result["cagr_pct"]),
            baseline_max_drawdown_pct=float(baseline_result["max_drawdown_pct"]),
            baseline_sharpe=float(baseline_result["sharpe_ratio"]),
            candidate_outperformed=float(candidate_result["total_return_pct"])
            > float(baseline_result["total_return_pct"]),
        )
        fold_results.append(row)
        print(
            f"Fold {fold.fold}: {row.test_start}→{row.test_end} · "
            f"candidate {row.candidate_return_pct:+.2f}% · "
            f"baseline {row.baseline_return_pct:+.2f}% · "
            f"DD {row.candidate_max_drawdown_pct:.2f}% · Sharpe {row.candidate_sharpe:.2f}"
        )

    candidate_stitched = pd.concat(candidate_oos).sort_index()
    baseline_stitched = pd.concat(baseline_oos).sort_index()
    if candidate_stitched.index.has_duplicates or baseline_stitched.index.has_duplicates:
        raise RuntimeError("walk-forward test folds overlap")

    candidate_metrics = _stitched_metrics(candidate_stitched)
    baseline_metrics = _stitched_metrics(baseline_stitched)
    summary = summarize_walk_forward(
        [row.candidate_return_pct for row in fold_results],
        [row.baseline_return_pct for row in fold_results],
        stitched_sharpe=float(candidate_metrics["sharpe_ratio"]),
        stitched_drawdown_pct=float(candidate_metrics["max_drawdown_pct"]),
        config=walk_config,
    )

    payload = {
        "generated_at": datetime.now().isoformat(),
        "validator_version": "v1_retrospective_expanding_walk_forward",
        "methodology": {
            "type": "expanding_train_fixed_non_overlapping_test",
            "signal_data_cutoff": "each fold is simulated with prices truncated at test_end",
            "selection_caveat": (
                "The fixed champion was discovered using the full historical sample. "
                "This validates causal signal stability but is not a genuinely unseen final holdout."
            ),
            "universe_caveat": (
                "Historical point-in-time index membership and delisted-security history are unavailable. "
                "All folds therefore use the lab's current eligible-symbol universe and retain survivorship bias."
            ),
            "config": walk_config,
        },
        "candidate": {"name": "rsi_momentum_42_consistency_champion", "params": CHAMPION},
        "baseline": {"name": "live_rsi_momentum_baseline", "params": BASELINE},
        "data": {
            "symbols": len(prices.columns),
            "days": len(prices),
            "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()),
        },
        "folds": [asdict(row) for row in fold_results],
        "candidate_stitched_oos": candidate_metrics,
        "baseline_stitched_oos": baseline_metrics,
        "summary": summary,
        "verdict": "passed_retrospective_walk_forward" if summary["passed"] else "failed_walk_forward",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nCandidate stitched OOS:", json.dumps(candidate_metrics, indent=2))
    print("Baseline stitched OOS:", json.dumps(baseline_metrics, indent=2))
    print("Summary:", json.dumps(summary, indent=2))
    print(f"Verdict: {payload['verdict']}")
    print(f"Saved: {output}")
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
