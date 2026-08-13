from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.auto_iteration_lab import (
    _evaluate_walk_forward_safely,
    _prepare_close_inputs,
    _simulate,
    build_signal_intents,
)


def _ohlc(periods: int = 280):
    dates = pd.bdate_range("2022-01-03", periods=periods)
    step = np.arange(periods, dtype=float)
    closes = pd.DataFrame(
        {
            "AAA": 100.0 + step * 0.50,
            "BBB": 90.0 + step * 0.30,
            "CCC": 80.0 + step * 0.10,
        },
        index=dates,
    )
    opens = closes.shift(1) * 1.01
    opens.iloc[0] = closes.iloc[0]
    return dates, opens, closes


def _params(**overrides):
    params = {
        "rsi_periods": [10, 20, 30],
        "momentum_period": 21,
        "regime_mode": "none",
        "use_macd": False,
        "top_n": 2,
        "rebalance_freq": "ME",
        "cost_bps": 10.0,
        "max_per_sector": 0,
        "vol_weight": True,
        "vol_lookback": 10,
    }
    params.update(overrides)
    return params


def _wf_config():
    return {
        "test_months": 3,
        "step_months": 3,
        "min_train_years": 0,
        "min_folds": 2,
        "min_training_days": 60,
        "min_test_days": 10,
        "min_worst_fold_return_pct": -100.0,
        "min_stitched_sharpe": -100.0,
        "max_stitched_drawdown_abs_pct": 100.0,
        "min_baseline_outperformance_ratio": 0.0,
        "require_all_positive_folds": False,
        "max_fold_to_median_ratio": 1000.0,
    }


def _execution_config():
    return {
        "min_execution_open_coverage": 0.5,
        "min_held_close_coverage": 0.5,
        "max_close_ffill_rows": 3,
    }


def test_walk_forward_execution_failure_disqualifies_candidate_without_raising():
    """A missing required open inside walk-forward must fail the candidate
    closed (visible reason), never abort the nightly run."""
    dates, opens, closes = _ohlc()
    wf = _wf_config()

    # Force a missing required open on an execution date: the first fold's
    # first execution session for a selected symbol.
    from scripts.walk_forward import build_folds

    folds = build_folds(opens.index, wf)
    first_exec = folds[0].test_sessions[0]
    opens.loc[first_exec, "AAA"] = np.nan

    evidence, failure = _evaluate_walk_forward_safely(
        prices={"open": opens, "close": closes},
        candidate_params=_params(),
        baseline_params=_params(),
        instruments=pd.DataFrame(),
        walk_forward_config=wf,
        execution_config=_execution_config(),
        intent_builder=build_signal_intents,
    )

    # The candidate must NOT qualify...
    assert evidence is not None
    assert evidence.get("qualified") is False
    # ...and the reason must be visible for the report.
    assert failure is not None
    assert failure.get("reason") == "invalid_required_open"
    assert "AAA" in failure.get("missing_symbols", [])
    assert failure.get("fold") == 1
    assert failure.get("execution_date") is not None


def test_walk_forward_success_has_no_failure():
    dates, opens, closes = _ohlc()
    evidence, failure = _evaluate_walk_forward_safely(
        prices={"open": opens, "close": closes},
        candidate_params=_params(),
        baseline_params=_params(),
        instruments=pd.DataFrame(),
        walk_forward_config=_wf_config(),
        execution_config=_execution_config(),
        intent_builder=build_signal_intents,
    )
    assert failure is None
    assert evidence is not None
    assert "fold_rows" in evidence or "fold_count" in evidence
    assert "qualified" in evidence
