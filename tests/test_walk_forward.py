from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from scripts.auto_iteration_lab import _load_lab_config
from scripts.portfolio_simulator import SignalIntent
from scripts.walk_forward import (
    WalkForwardConfigError,
    build_folds,
    evaluate_walk_forward,
)


def _wf_config(**overrides):
    config = _load_lab_config().walk_forward
    return replace(config, **overrides)


def test_build_folds_uses_expanding_train_and_half_open_non_overlapping_tests():
    sessions = pd.bdate_range("2020-01-02", "2023-12-29")
    folds = build_folds(
        sessions,
        _wf_config(min_train_years=1, test_months=6, step_months=6, min_test_days=40, min_folds=3),
    )

    first, second = folds[0], folds[1]
    assert first.train_start == sessions[0]
    assert first.train_end == sessions[sessions < first.test_start][-1]
    assert first.test_sessions.min() >= first.test_start
    assert first.test_sessions.max() < first.test_end
    assert second.train_start == first.train_start
    assert second.train_end == sessions[sessions < second.test_start][-1]
    assert second.test_start == first.test_end
    assert set(first.test_sessions).isdisjoint(second.test_sessions)


def test_build_folds_enforces_training_test_and_fold_minimums():
    sessions = pd.bdate_range("2022-01-03", "2023-02-28")
    with pytest.raises(WalkForwardConfigError, match="minimum folds"):
        build_folds(
            sessions,
            _wf_config(min_train_years=1, test_months=6, step_months=6, min_test_days=40, min_folds=2),
        )
    with pytest.raises(WalkForwardConfigError, match="step_months"):
        build_folds(
            sessions,
            _wf_config(test_months=6, step_months=3, min_folds=1),
        )


def _constant_target_builder(opens, closes, params, instruments):
    del opens, instruments
    symbol = params["symbol"]
    return [SignalIntent(f"{symbol}-{closes.index[0].date()}", closes.index[0], {symbol: 1.0})]


def test_fold_starts_cash_only_fills_first_test_open_carries_only_inside_fold_and_marks_final_close():
    sessions = pd.bdate_range("2020-01-02", "2022-12-30")
    opens = pd.DataFrame({"UP": 100.0, "FLAT": 100.0}, index=sessions)
    closes = opens.copy()
    # Each test interval rises after its first open; a fresh fold must buy at that open.
    closes["UP"] = np.linspace(100.0, 160.0, len(sessions))
    observed_max_dates = []

    def builder(o, c, params, instruments):
        observed_max_dates.append(c.index.max())
        target = _constant_target_builder(o, c, params, instruments)[0].target_weights
        # The evaluator must turn the last training close target into the fold's seed intent.
        return [SignalIntent("seed", c.index[0], target)]

    evidence = evaluate_walk_forward(
        {"open": opens, "close": closes},
        candidate_params={"symbol": "UP", "cost_bps": 0.0},
        baseline_params={"symbol": "FLAT", "cost_bps": 0.0},
        instruments=pd.DataFrame(),
        config=_wf_config(min_train_years=1, test_months=6, step_months=6, min_test_days=40, min_folds=3),
        execution_config={"max_close_ffill_rows": 0, "min_execution_open_coverage": 1.0, "min_held_close_coverage": 1.0},
        intent_builder=builder,
    )

    assert evidence["fold_count"] >= 3
    for fold in evidence["folds"]:
        assert fold["initial_cash"] == 100000.0
        assert fold["first_fill_date"] == fold["test_first_session"]
        assert fold["final_mark_date"] == fold["test_last_session"]
        assert fold["starting_units"] == {}
        assert fold["ending_units"] == {"UP": pytest.approx(fold["ending_units"]["UP"])}
    # Candidate and baseline are rebuilt per fold and only see data through that fold's test end.
    assert len(observed_max_dates) == evidence["fold_count"] * 2
    assert max(observed_max_dates) < sessions.max() or evidence["folds"][-1]["test_last_session"] == str(sessions.max().date())
    assert evidence["stitched_days"] == sum(fold["test_days"] for fold in evidence["folds"])


def test_verdict_consumes_all_gates_and_handles_positive_fold_ratio_edges():
    sessions = pd.bdate_range("2020-01-02", "2023-12-29")
    opens = pd.DataFrame({"GOOD": 100.0, "BASE": 100.0}, index=sessions)
    closes = opens.copy()
    # Positive first folds but a final collapse: relative baseline wins cannot rescue strict positivity.
    closes["GOOD"] = 100.0 + np.arange(len(sessions)) * 0.08
    closes.loc[closes.index >= "2023-07-03", "GOOD"] = np.linspace(
        closes.loc[closes.index >= "2023-07-03", "GOOD"].iloc[0], 50.0,
        len(closes.loc[closes.index >= "2023-07-03"]),
    )
    evidence = evaluate_walk_forward(
        {"open": opens, "close": closes},
        candidate_params={"symbol": "GOOD", "cost_bps": 0.0},
        baseline_params={"symbol": "BASE", "cost_bps": 0.0},
        instruments=pd.DataFrame(),
        config=_wf_config(
            min_train_years=1, test_months=6, step_months=6, min_test_days=40, min_folds=5,
            min_worst_fold_return_pct=0.0, min_stitched_sharpe=-99.0,
            max_stitched_drawdown_abs_pct=100.0, min_baseline_outperformance_ratio=0.0,
            max_fold_to_median_ratio=99.0,
        ),
        execution_config={"max_close_ffill_rows": 0, "min_execution_open_coverage": 1.0, "min_held_close_coverage": 1.0},
        intent_builder=_constant_target_builder,
    )

    assert evidence["baseline_outperformance_ratio"] == pytest.approx(
        sum(f["baseline_outperformance_pct"] >= 0 for f in evidence["folds"]) / evidence["fold_count"]
    )
    assert evidence["worst_fold_return_pct"] < 0
    assert evidence["qualified"] is False
    assert "weak_worst_fold" in evidence["failures"]
    assert evidence["strict_all_positive"] is False
    assert evidence["fold_to_median_ratio"] > 0

    flat = evaluate_walk_forward(
        {"open": opens, "close": opens},
        candidate_params={"symbol": "GOOD", "cost_bps": 0.0},
        baseline_params={"symbol": "BASE", "cost_bps": 0.0},
        instruments=pd.DataFrame(),
        config=_wf_config(
            min_train_years=1, test_months=6, step_months=6, min_test_days=40, min_folds=5,
            min_worst_fold_return_pct=-1.0, min_stitched_sharpe=-1.0,
            max_stitched_drawdown_abs_pct=100.0, min_baseline_outperformance_ratio=0.0,
            max_fold_to_median_ratio=99.0,
        ),
        execution_config={"max_close_ffill_rows": 0, "min_execution_open_coverage": 1.0, "min_held_close_coverage": 1.0},
        intent_builder=_constant_target_builder,
    )
    assert flat["fold_to_median_ratio"] is None
    assert "invalid_fold_to_median_ratio" in flat["failures"]
