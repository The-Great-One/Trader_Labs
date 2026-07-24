from __future__ import annotations

import pandas as pd

from scripts.rsi_momentum_consistency_walk_forward import (
    build_expanding_folds,
    summarize_walk_forward,
)


def test_expanding_folds_are_ordered_non_overlapping_and_use_only_prior_training_data() -> None:
    dates = pd.bdate_range("2020-01-01", "2025-12-31")
    folds = build_expanding_folds(
        dates,
        min_train_years=3,
        test_months=6,
        step_months=6,
        min_test_days=40,
    )
    assert len(folds) >= 5
    for index, fold in enumerate(folds):
        assert fold.train_end < fold.test_start <= fold.test_end
        if index:
            assert folds[index - 1].test_end < fold.test_start


def test_walk_forward_summary_requires_consistency_not_one_outlier_fold() -> None:
    config = {
        "min_folds": 4,
        "min_worst_fold_return_pct": 0.0,
        "min_stitched_sharpe": 1.0,
        "max_stitched_drawdown_abs_pct": 35.0,
        "min_baseline_outperformance_ratio": 0.5,
        "max_fold_to_median_ratio": 3.0,
    }
    candidate_folds = [10.0, 12.0, 11.0, 150.0]
    baseline_folds = [8.0, 9.0, 8.0, 9.0]
    summary = summarize_walk_forward(
        candidate_folds,
        baseline_folds,
        stitched_sharpe=1.8,
        stitched_drawdown_pct=-20.0,
        config=config,
    )
    assert summary["passed"] is False
    assert "fold_outlier_ratio" in summary["failures"]


def test_walk_forward_summary_passes_uniform_positive_folds() -> None:
    config = {
        "min_folds": 4,
        "min_worst_fold_return_pct": 0.0,
        "min_stitched_sharpe": 1.0,
        "max_stitched_drawdown_abs_pct": 35.0,
        "min_baseline_outperformance_ratio": 0.5,
        "max_fold_to_median_ratio": 3.0,
    }
    summary = summarize_walk_forward(
        [10.0, 12.0, 11.0, 13.0],
        [8.0, 9.0, 10.0, 9.0],
        stitched_sharpe=1.8,
        stitched_drawdown_pct=-20.0,
        config=config,
    )
    assert summary["passed"] is True
    assert summary["failures"] == []