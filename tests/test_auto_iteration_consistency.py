from __future__ import annotations

import pandas as pd

from scripts.auto_iteration_lab import (
    SCORING_VERSION,
    _apply_walk_forward_readiness,
    _consistency_metrics,
    _find_champion,
    _load_lab_config,
    _simulate,
)


def _daily_years(yearly_targets: dict[int, float], days: int = 252) -> pd.Series:
    chunks = []
    for year, target_pct in yearly_targets.items():
        daily = (1.0 + target_pct / 100.0) ** (1.0 / days) - 1.0
        idx = pd.bdate_range(f"{year}-01-01", periods=days)
        chunks.append(pd.Series(daily, index=idx))
    return pd.concat(chunks).sort_index()


def test_smooth_returns_outrank_one_year_outlier() -> None:
    cfg = _load_lab_config()["consistency"]
    smooth = _consistency_metrics(
        _daily_years({2021: 25, 2022: 28, 2023: 24, 2024: 30}),
        sharpe=1.8,
        max_drawdown_pct=-18,
        avg_turnover=3,
        cost_bps=10,
        config=cfg,
    )
    outlier = _consistency_metrics(
        _daily_years({2021: 12, 2022: 11, 2023: 300, 2024: 10}),
        sharpe=0.7,
        max_drawdown_pct=-28,
        avg_turnover=4,
        cost_bps=10,
        config=cfg,
    )
    assert smooth["qualified"] is True
    assert outlier["qualified"] is False
    assert "year_outlier_ratio" in outlier["qualification_failures"]
    assert smooth["selection_score"] > outlier["selection_score"]


def test_partial_year_is_reported_but_not_used_for_qualification() -> None:
    cfg = _load_lab_config()["consistency"]
    complete = _daily_years({2022: 20, 2023: 22, 2024: 18})
    partial_idx = pd.bdate_range("2025-01-01", periods=40)
    partial = pd.Series(-0.002, index=partial_idx)
    metrics = _consistency_metrics(
        pd.concat([complete, partial]),
        sharpe=1.5,
        max_drawdown_pct=-20,
        avg_turnover=2,
        cost_bps=10,
        config=cfg,
    )
    assert metrics["complete_years"] == [2022, 2023, 2024]
    assert metrics["partial_years"] == [2025]
    assert "2025" in metrics["calendar_year_returns"]
    assert metrics["worst_year_return_pct"] > 0


def test_champion_requires_walk_forward_readiness() -> None:
    history = [
        {"enhancement": "outlier", "agg": {"qualified": False, "selection_score": 999, "scoring_version": SCORING_VERSION}},
        {"enhancement": "consistency_only", "agg": {"qualified": True, "selection_score": 50, "scoring_version": SCORING_VERSION}},
        {
            "enhancement": "ready",
            "agg": {"qualified": True, "selection_score": 20, "scoring_version": SCORING_VERSION},
            "champion_ready": True,
            "walk_forward": {"qualified": True},
        },
    ]
    assert _find_champion(history)["enhancement"] == "ready"
    assert _find_champion(history[:2]) is None


def test_old_scoring_schema_cannot_supply_champion() -> None:
    stale = {"enhancement": "stale", "agg": {"qualified": True, "selection_score": 999}}
    assert _find_champion([stale]) is None


def test_optimistic_transaction_cost_run_cannot_be_champion() -> None:
    cfg = _load_lab_config()["consistency"]
    metrics = _consistency_metrics(
        _daily_years({2021: 25, 2022: 28, 2023: 24, 2024: 30}),
        sharpe=1.8,
        max_drawdown_pct=-18,
        avg_turnover=3,
        cost_bps=0,
        config=cfg,
    )
    assert metrics["qualified"] is False
    assert "optimistic_transaction_costs" in metrics["qualification_failures"]


def test_simulation_can_return_a_leakage_safe_evaluation_slice() -> None:
    dates = pd.bdate_range("2022-01-03", periods=520)
    trend = pd.Series(range(len(dates)), index=dates, dtype=float)
    prices = pd.DataFrame(
        {
            "AAA": 100 + trend * 0.20,
            "BBB": 100 + trend * 0.15,
            "CCC": 100 + trend * 0.10,
        },
        index=dates,
    )
    cfg = _load_lab_config()["consistency"]
    start, end = dates[400], dates[480]
    result = _simulate(
        prices,
        {
            "rsi_periods": [10, 20, 30],
            "momentum_period": 21,
            "regime_mode": "none",
            "use_macd": False,
            "top_n": 2,
            "rebalance_freq": "3W-FRI",
            "cost_bps": 10.0,
            "max_per_sector": 0,
        },
        pd.DataFrame(),
        cfg,
        evaluation_start=start,
        evaluation_end=end,
        include_daily_returns=True,
    )
    assert result is not None
    daily = result.pop("_daily_returns")
    observed = pd.DatetimeIndex(daily)
    assert observed.min() >= start
    assert observed.max() <= end
    assert len(observed) == len(dates[(dates >= start) & (dates <= end)])


def test_full_history_winner_remains_retrospective_when_walk_forward_fails() -> None:
    result = {
        "enhancement": "full_history_winner",
        "agg": {"qualified": True, "selection_score": 99.0},
    }
    updated = _apply_walk_forward_readiness(
        result,
        {
            "qualified": False,
            "failures": ["weak_worst_fold"],
            "fold_count": 6,
        },
    )

    assert updated["evidence_label"] == "retrospective_only"
    assert updated["champion_ready"] is False
    assert updated["walk_forward"]["failures"] == ["weak_worst_fold"]


def test_only_consistency_and_walk_forward_qualified_result_is_champion_ready() -> None:
    result = {"enhancement": "stable", "agg": {"qualified": True, "selection_score": 20.0}}
    updated = _apply_walk_forward_readiness(
        result,
        {"qualified": True, "failures": [], "fold_count": 6},
    )

    assert updated["evidence_label"] == "champion_ready"
    assert updated["champion_ready"] is True
