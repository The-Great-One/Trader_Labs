"""Retrospective fixed-candidate walk-forward stability evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.portfolio_simulator import ExecutionDataError, PortfolioSimulator, SignalIntent


class WalkForwardConfigError(ValueError):
    """Raised when the requested fold geometry cannot produce valid evidence."""


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_sessions: pd.DatetimeIndex


IntentBuilder = Callable[
    [pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame], list[SignalIntent]
]


def _config_value(config: Any, name: str) -> Any:
    return getattr(config, name) if hasattr(config, name) else config[name]


def build_folds(sessions: pd.DatetimeIndex, config: Any) -> list[WalkForwardFold]:
    """Build expanding-train, half-open, non-overlapping calendar test folds."""
    ordered = pd.DatetimeIndex(sessions).sort_values().unique()
    if ordered.empty:
        raise WalkForwardConfigError("sessions must not be empty")
    test_months = int(_config_value(config, "test_months"))
    step_months = int(_config_value(config, "step_months"))
    if step_months < test_months:
        raise WalkForwardConfigError("step_months must be >= test_months")

    train_start = pd.Timestamp(ordered[0])
    test_start = train_start + pd.DateOffset(years=int(_config_value(config, "min_train_years")))
    folds: list[WalkForwardFold] = []
    while test_start <= ordered[-1]:
        test_end = test_start + pd.DateOffset(months=test_months)
        training = ordered[ordered < test_start]
        observed = ordered[(ordered >= test_start) & (ordered < test_end)]
        if len(training) and len(observed) >= int(_config_value(config, "min_test_days")):
            folds.append(
                WalkForwardFold(
                    train_start=train_start,
                    train_end=pd.Timestamp(training[-1]),
                    test_start=pd.Timestamp(test_start),
                    test_end=pd.Timestamp(test_end),
                    test_sessions=pd.DatetimeIndex(observed),
                )
            )
        test_start = test_start + pd.DateOffset(months=step_months)

    minimum = int(_config_value(config, "min_folds"))
    if len(folds) < minimum:
        raise WalkForwardConfigError(
            f"minimum folds not met: required {minimum}, observed {len(folds)}"
        )
    return folds


def _target_at(intents: list[SignalIntent], date: pd.Timestamp) -> dict[str, float]:
    eligible = [intent for intent in intents if pd.Timestamp(intent.signal_date) <= date]
    if not eligible:
        return {}
    return dict(max(eligible, key=lambda intent: pd.Timestamp(intent.signal_date)).target_weights)


def _fold_signals(
    intents: list[SignalIntent], fold: WalkForwardFold, label: str
) -> list[SignalIntent]:
    seed = SignalIntent(
        signal_id=f"wf-{label}-seed-{fold.test_start.date()}",
        signal_date=fold.train_end,
        target_weights=_target_at(intents, fold.train_end),
        label="last_training_close_intent",
    )
    within_test = [
        SignalIntent(
            signal_id=f"wf-{label}-{intent.signal_id}",
            signal_date=pd.Timestamp(intent.signal_date),
            target_weights=dict(intent.target_weights),
            label=intent.label,
        )
        for intent in intents
        if fold.test_start <= pd.Timestamp(intent.signal_date) < fold.test_end
    ]
    return [seed, *within_test]


def _fold_returns(nav: pd.Series, initial_cash: float) -> pd.Series:
    returns = nav.pct_change(fill_method=None)
    if not returns.empty:
        returns.iloc[0] = float(nav.iloc[0]) / initial_cash - 1.0
    return returns.fillna(0.0)


def _run_candidate_fold(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    params: dict[str, Any],
    instruments: pd.DataFrame,
    fold: WalkForwardFold,
    execution_config: Any,
    intent_builder: IntentBuilder,
    label: str,
) -> tuple[pd.Series, Any]:
    # Indicators and intents may see warm-up history, but never rows at/after test_end.
    visible_sessions = opens.index[opens.index < fold.test_end]
    visible_opens = opens.loc[visible_sessions]
    visible_closes = closes.loc[visible_sessions]
    intents = intent_builder(visible_opens, visible_closes, params, instruments)
    signals = _fold_signals(intents, fold, label)

    test_opens = opens.loc[fold.test_sessions]
    test_closes = closes.loc[fold.test_sessions]
    max_ffill = int(_config_value(execution_config, "max_close_ffill_rows"))
    source_dates = pd.DataFrame(index=test_closes.index, columns=test_closes.columns, dtype="datetime64[ns]")
    for symbol in test_closes:
        observed = test_closes[symbol].notna()
        source_dates.loc[observed, symbol] = test_closes.index[observed]
    marked_closes = test_closes.ffill(limit=max_ffill) if max_ffill > 0 else test_closes.copy()
    source_dates = source_dates.ffill(limit=max_ffill) if max_ffill > 0 else source_dates
    simulator = PortfolioSimulator(
        test_opens,
        marked_closes,
        initial_cash=100_000.0,
        cost_bps=float(params.get("cost_bps", 10.0)),
        min_execution_open_coverage=float(
            _config_value(execution_config, "min_execution_open_coverage")
        ),
        min_held_close_coverage=float(
            _config_value(execution_config, "min_held_close_coverage")
        ),
        max_close_ffill_rows=max_ffill,
        close_source_dates=source_dates,
    )
    result = simulator.run(signals)
    return _fold_returns(result.nav, simulator.initial_cash), result


def _ratio_or_none(values: list[float]) -> float | None:
    positive = np.asarray([value for value in values if np.isfinite(value) and value > 0.0])
    if positive.size == 0:
        return None
    median = float(np.median(positive))
    if not np.isfinite(median) or median <= 0:
        return None
    return float(np.max(positive) / median)


def evaluate_walk_forward(
    prices: dict[str, pd.DataFrame],
    *,
    candidate_params: dict[str, Any],
    baseline_params: dict[str, Any],
    instruments: pd.DataFrame,
    config: Any,
    execution_config: Any,
    intent_builder: IntentBuilder,
) -> dict[str, Any]:
    """Evaluate fixed candidates fold-by-fold and enforce every configured gate."""
    opens = prices["open"].sort_index()
    closes = prices["close"].reindex(index=opens.index, columns=opens.columns)
    folds = build_folds(opens.index, config)
    fold_rows: list[dict[str, Any]] = []
    stitched: list[pd.Series] = []

    for number, fold in enumerate(folds, 1):
        try:
            candidate_returns, candidate = _run_candidate_fold(
                opens, closes, candidate_params, instruments, fold, execution_config,
                intent_builder, f"candidate-{number}",
            )
            baseline_returns, _ = _run_candidate_fold(
                opens, closes, baseline_params, instruments, fold, execution_config,
                intent_builder, f"baseline-{number}",
            )
        except ExecutionDataError as exc:
            raise WalkForwardConfigError(
                f"fold {number} execution failed: {exc.reason}"
            ) from exc
        candidate_return = float((1.0 + candidate_returns).prod() - 1.0)
        baseline_return = float((1.0 + baseline_returns).prod() - 1.0)
        stitched.append(candidate_returns)
        first_event = candidate.execution_events[0] if candidate.execution_events else None
        fold_rows.append(
            {
                "fold": number,
                "train_start": str(fold.train_start.date()),
                "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end_exclusive": str(fold.test_end.date()),
                "test_first_session": str(fold.test_sessions[0].date()),
                "test_last_session": str(fold.test_sessions[-1].date()),
                "test_days": len(fold.test_sessions),
                "initial_cash": 100_000.0,
                "starting_units": {},
                "ending_units": dict(candidate.final_state.units),
                "first_fill_date": str(first_event.execution_date.date()) if first_event else None,
                "final_mark_date": str(candidate.valuations[-1].date.date()),
                "candidate_return_pct": candidate_return * 100.0,
                "baseline_return_pct": baseline_return * 100.0,
                "baseline_outperformance_pct": (candidate_return - baseline_return) * 100.0,
            }
        )

    stitched_returns = pd.concat(stitched).sort_index()
    fold_returns = [row["candidate_return_pct"] for row in fold_rows]
    worst = float(min(fold_returns))
    values = stitched_returns.to_numpy(dtype=float)
    sharpe = float(values.mean() / (values.std() + 1e-9) * np.sqrt(252)) if len(values) > 20 else 0.0
    equity = (1.0 + stitched_returns).cumprod()
    drawdown = float((equity / equity.cummax() - 1.0).min() * 100.0)
    outperformance_ratio = float(
        np.mean([row["baseline_outperformance_pct"] >= 0.0 for row in fold_rows])
    )
    fold_ratio = _ratio_or_none(fold_returns)

    failures: list[str] = []
    if len(folds) < int(_config_value(config, "min_folds")):
        failures.append("insufficient_folds")
    if worst < float(_config_value(config, "min_worst_fold_return_pct")):
        failures.append("weak_worst_fold")
    if sharpe < float(_config_value(config, "min_stitched_sharpe")):
        failures.append("low_stitched_sharpe")
    if abs(drawdown) > float(_config_value(config, "max_stitched_drawdown_abs_pct")):
        failures.append("excess_stitched_drawdown")
    if outperformance_ratio < float(
        _config_value(config, "min_baseline_outperformance_ratio")
    ):
        failures.append("weak_baseline_outperformance")
    if fold_ratio is None:
        failures.append("invalid_fold_to_median_ratio")
    elif fold_ratio > float(_config_value(config, "max_fold_to_median_ratio")):
        failures.append("excess_fold_to_median_ratio")

    return {
        "qualified": not failures,
        "failures": failures,
        "retrospective": True,
        "retrospective_reason": "candidate discovery used full history; folds test fixed-candidate stability only",
        "fold_count": len(folds),
        "folds": fold_rows,
        "strict_all_positive": worst >= 0.0,
        "worst_fold_return_pct": worst,
        "stitched_sharpe": sharpe,
        "stitched_max_drawdown_pct": drawdown,
        "baseline_outperformance_ratio": outperformance_ratio,
        "fold_to_median_ratio": fold_ratio,
        "stitched_days": len(stitched_returns),
        "config": asdict(config) if hasattr(config, "__dataclass_fields__") else dict(config),
    }
