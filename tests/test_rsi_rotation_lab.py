from __future__ import annotations

import math

import pandas as pd

from scripts.rsi_224466_rotation_lab import metrics


def test_metrics_xirr_includes_cash_days_after_first_trade():
    dates = pd.bdate_range("2024-01-01", periods=5)
    returns = pd.Series([0.10, 0.0, 0.0, 0.0, 0.0], index=dates)
    weights = pd.DataFrame({"AAA": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=dates)
    turnover = pd.Series([1.0, 1.0, 0.0, 0.0, 0.0], index=dates)

    result = metrics(
        "cash_gap_case",
        returns,
        weights,
        turnover,
        {"rebalance": "ME", "top_n": 1, "cost_bps": 0, "regime": "test", "symbols_loaded": 1},
    )

    active_only_years = 1 / 252
    full_calendar_years = 5 / 252
    expected_active_cagr = ((1.10 ** (1 / active_only_years)) - 1) * 100
    expected_xirr = ((1.10 ** (1 / full_calendar_years)) - 1) * 100

    assert result.cagr_pct == round(expected_active_cagr, 2)
    assert result.xirr_pct == round(expected_xirr, 2)
    assert result.xirr_pct < result.cagr_pct
