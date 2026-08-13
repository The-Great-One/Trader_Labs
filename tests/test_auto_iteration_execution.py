from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.auto_iteration_lab import (
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


def _consistency_config():
    return {
        "min_complete_years": 0,
        "min_trading_days_per_year": 1,
        "min_year_return_pct": -100.0,
        "min_sharpe_ratio": -100.0,
        "min_cost_bps_for_champion": 0.0,
        "max_drawdown_abs_pct": 100.0,
        "max_year_to_median_ratio": 1000.0,
        "min_rolling_12m_return_pct": -100.0,
        "rolling_year_days": 20,
        "score_weights": {
            "worst_year": 0.0,
            "median_year": 0.0,
            "sharpe": 0.0,
            "rolling_12m_min": 0.0,
            "drawdown": 0.0,
            "annual_mad": 0.0,
            "turnover": 0.0,
            "outlier_excess": 0.0,
        },
        "disqualification_penalty": 0.0,
    }


def test_close_boundary_forward_fills_once_and_preserves_source_dates():
    dates, _, closes = _ohlc()
    closes.loc[dates[100], "AAA"] = np.nan

    prepared, sources = _prepare_close_inputs(closes, max_close_ffill_rows=3)

    assert prepared.loc[dates[100], "AAA"] == closes.loc[dates[99], "AAA"]
    assert sources.loc[dates[100], "AAA"] == dates[99]
    assert closes.loc[dates[100], "AAA"] is np.nan or pd.isna(closes.loc[dates[100], "AAA"])


def test_intents_use_d_close_and_inverse_volatility_through_d_only():
    dates, opens, closes = _ohlc()
    intents = build_signal_intents(opens, closes, _params(), pd.DataFrame())
    assert intents
    first = intents[0]
    assert first.signal_date in closes.index
    execution_date = opens.index[opens.index.get_loc(first.signal_date) + 1]
    assert execution_date > first.signal_date

    changed_future = closes.copy()
    changed_future.loc[execution_date:, :] *= [100.0, 0.01, 50.0]
    changed = build_signal_intents(opens, changed_future, _params(), pd.DataFrame())
    same_date = next(intent for intent in changed if intent.signal_date == first.signal_date)
    assert same_date.target_weights == pytest.approx(first.target_weights)
    assert sum(first.target_weights.values()) == pytest.approx(1.0)
    assert len(set(round(v, 8) for v in first.target_weights.values())) > 1


def test_simulation_reports_stateful_execution_accounting_and_next_open_fills():
    dates, opens, closes = _ohlc()
    result = _simulate(
        {"open": opens, "close": closes},
        _params(),
        pd.DataFrame(),
        _consistency_config(),
        include_daily_returns=True,
    )

    assert result is not None
    assert result["qualified"] is True
    assert result["accounting"]["fills"] > 0
    assert result["accounting"]["traded_notional"] > 0
    assert result["accounting"]["fees"] > 0
    assert result["accounting"]["cash_final"] >= -1e-8
    assert result["accounting"]["avg_one_way_turnover"] > 0
    first = result["_execution_events"][0]
    assert pd.Timestamp(first["execution_date"]) > pd.Timestamp(first["signal_date"])
    assert first["fills"][0]["price"] == opens.loc[pd.Timestamp(first["execution_date"]), first["fills"][0]["symbol"]]
    assert dates[0].strftime("%Y-%m-%d") in result["_daily_returns"]


def test_drawdown_exit_after_fill_creates_real_sell_and_fee():
    dates, opens, closes = _ohlc()
    # Force AAA to be selected, filled, and then breach a close-based drawdown.
    crash_date = dates[240]
    closes.loc[crash_date:, "AAA"] *= 0.40
    opens.loc[crash_date + pd.offsets.BDay(1):, "AAA"] = closes.loc[crash_date, "AAA"] * 0.99
    result = _simulate(
        {"open": opens, "close": closes},
        _params(top_n=1, dd_exit_pct=-20),
        pd.DataFrame(),
        _consistency_config(),
        include_daily_returns=True,
    )
    assert result is not None
    sells = [fill for event in result["_execution_events"] for fill in event["fills"] if fill["side"] == "SELL"]
    assert sells
    assert any(fill["fee"] > 0 for fill in sells)


def test_missing_required_open_disqualifies_candidate_with_visible_reason():
    _, opens, closes = _ohlc()
    intents = build_signal_intents(opens, closes, _params(top_n=3), pd.DataFrame())
    execution = opens.index[opens.index.get_loc(intents[0].signal_date) + 1]
    symbol = next(iter(intents[0].target_weights))
    opens.loc[execution, symbol] = np.nan

    result = _simulate(
        {"open": opens, "close": closes},
        _params(top_n=3),
        pd.DataFrame(),
        _consistency_config(),
    )

    assert result is not None
    assert result["qualified"] is False
    assert "invalid_required_open" in result["qualification_failures"]
    assert result["execution_failure"]["missing_symbols"] == [symbol]
