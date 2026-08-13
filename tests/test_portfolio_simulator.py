from __future__ import annotations

import pandas as pd
import pytest

from scripts.portfolio_simulator import ExecutionDataError, PortfolioSimulator, PortfolioState, SignalIntent


def _frames():
    dates = pd.bdate_range("2024-01-02", periods=4)
    opens = pd.DataFrame({"AAA": [10.0, 20.0, 22.0, 24.0], "BBB": [20.0, 40.0, 44.0, 48.0]}, index=dates)
    closes = pd.DataFrame({"AAA": [12.0, 21.0, 23.0, 25.0], "BBB": [21.0, 42.0, 50.0, 50.0]}, index=dates)
    return dates, opens, closes


def test_signal_fills_at_next_open_without_earning_pre_fill_overnight_return():
    dates, opens, closes = _frames()
    sim = PortfolioSimulator(opens, closes, initial_cash=1000.0)

    result = sim.run([SignalIntent("s1", dates[0], {"AAA": 1.0})])

    event = result.execution_events[0]
    assert event.signal_date == dates[0]
    assert event.execution_date == dates[1]
    assert event.fills[0].price == 20.0
    assert result.nav.loc[dates[0]] == 1000.0
    assert result.nav.loc[dates[1]] == 1050.0


def test_old_basket_receives_overnight_move_before_sale_to_new_basket():
    dates, opens, closes = _frames()
    sim = PortfolioSimulator(opens, closes, initial_cash=1000.0)

    result = sim.run(
        [
            SignalIntent("s1", dates[0], {"AAA": 1.0}),
            SignalIntent("s2", dates[1], {"BBB": 1.0}),
        ]
    )

    second = result.execution_events[1]
    assert second.pre_trade_nav == pytest.approx(1100.0)
    assert [(f.symbol, f.side, f.price) for f in second.fills] == [
        ("AAA", "SELL", 22.0),
        ("BBB", "BUY", 44.0),
    ]


def test_units_stay_fixed_between_fills_and_weights_drift():
    dates, opens, closes = _frames()
    sim = PortfolioSimulator(opens, closes, initial_cash=1000.0)

    result = sim.run([SignalIntent("s1", dates[0], {"AAA": 0.5, "BBB": 0.5})])

    first_units = result.valuations[1].units
    later_units = result.valuations[2].units
    assert later_units == first_units
    assert result.valuations[1].weights != result.valuations[2].weights


def test_cash_fees_turnover_and_label_independence_use_actual_notional():
    dates, opens, closes = _frames()
    signals = [SignalIntent("s1", dates[0], {"AAA": 0.4}, label="top5")]
    top5 = PortfolioSimulator(opens, closes, initial_cash=1000.0, cost_bps=10).run(signals)
    top10 = PortfolioSimulator(opens, closes, initial_cash=1000.0, cost_bps=10).run(
        [SignalIntent("s2", dates[0], {"AAA": 0.4}, label="top10")]
    )

    event = top5.execution_events[0]
    assert event.gross_traded_notional == pytest.approx(400.0)
    assert event.fees == pytest.approx(0.4)
    assert event.one_way_turnover == pytest.approx(0.4)
    assert top5.final_state.cash == pytest.approx(599.6)
    assert top10.execution_events[0].fees == pytest.approx(event.fees)


def test_signal_at_sample_end_is_pending_and_non_actionable():
    dates, opens, closes = _frames()
    sim = PortfolioSimulator(opens, closes, initial_cash=1000.0)

    result = sim.run([SignalIntent("last", dates[-1], {"AAA": 1.0})])

    assert result.execution_events == []
    assert [s.signal_id for s in result.pending_signals] == ["last"]


@pytest.mark.parametrize("bad_open", [float("nan"), 0.0, -1.0])
def test_missing_target_open_fails_atomically_with_diagnostics(bad_open):
    dates, opens, closes = _frames()
    opens.loc[dates[1], "AAA"] = bad_open
    state = PortfolioState(cash=1000.0)
    before = repr(state)
    sim = PortfolioSimulator(opens, closes, initial_cash=1000.0)

    with pytest.raises(ExecutionDataError) as caught:
        sim.execute_signal(state, SignalIntent("s1", dates[0], {"AAA": 1.0}), dates[1])

    assert repr(state) == before
    assert caught.value.signal_date == dates[0]
    assert caught.value.execution_date == dates[1]
    assert caught.value.missing_symbols == ("AAA",)
    assert caught.value.coverage == pytest.approx(0.5)
    assert caught.value.reason == "invalid_required_open"


def test_missing_held_sell_open_fails_atomically():
    dates, opens, closes = _frames()
    state = PortfolioState(cash=0.0, units={"AAA": 50.0}, executed_signal_ids={"old"})
    opens.loc[dates[1], "AAA"] = float("nan")
    before = repr(state)
    sim = PortfolioSimulator(opens, closes, initial_cash=1000.0)

    with pytest.raises(ExecutionDataError, match="invalid_required_open") as caught:
        sim.execute_signal(state, SignalIntent("s2", dates[0], {"BBB": 1.0}), dates[1])

    assert repr(state) == before
    assert caught.value.missing_symbols == ("AAA",)


def test_execution_coverage_threshold_fails_before_mutation():
    dates, opens, closes = _frames()
    opens.loc[dates[1], "BBB"] = float("nan")
    state = PortfolioState(cash=1000.0)
    before = repr(state)
    sim = PortfolioSimulator(opens, closes, initial_cash=1000.0, min_execution_open_coverage=0.75)

    with pytest.raises(ExecutionDataError, match="insufficient_execution_open_coverage") as caught:
        sim.execute_signal(state, SignalIntent("s1", dates[0], {"AAA": 1.0}), dates[1])

    assert repr(state) == before
    assert caught.value.coverage == pytest.approx(0.5)


def test_duplicate_signal_id_fails_atomically():
    dates, opens, closes = _frames()
    state = PortfolioState(cash=1000.0, executed_signal_ids={"s1"})
    before = repr(state)
    sim = PortfolioSimulator(opens, closes, initial_cash=1000.0)

    with pytest.raises(ExecutionDataError, match="duplicate_signal_id"):
        sim.execute_signal(state, SignalIntent("s1", dates[0], {"AAA": 1.0}), dates[1])

    assert repr(state) == before


def test_missing_held_close_mark_fails_atomically_with_source_age():
    dates, opens, closes = _frames()
    closes.loc[dates[1], "AAA"] = float("nan")
    sources = pd.DataFrame({"AAA": dates, "BBB": dates}, index=dates)
    sources.loc[dates[1], "AAA"] = dates[0]
    state = PortfolioState(cash=0.0, units={"AAA": 50.0})
    before = repr(state)
    sim = PortfolioSimulator(opens, closes, close_source_dates=sources, initial_cash=1000.0, max_close_ffill_rows=0)

    with pytest.raises(ExecutionDataError, match="invalid_held_close") as caught:
        sim.mark_state(state, dates[1])

    assert repr(state) == before
    assert caught.value.valuation_date == dates[1]
    assert caught.value.missing_symbols == ("AAA",)
    assert caught.value.source_mark_age == {"AAA": 1}


def test_stale_held_close_and_held_coverage_fail_closed():
    dates, opens, closes = _frames()
    closes.loc[dates[2], "AAA"] = closes.loc[dates[1], "AAA"]
    sources = pd.DataFrame({"AAA": dates, "BBB": dates}, index=dates)
    sources.loc[dates[2], "AAA"] = dates[0]
    state = PortfolioState(cash=0.0, units={"AAA": 10.0, "BBB": 10.0})
    sim = PortfolioSimulator(
        opens,
        closes,
        close_source_dates=sources,
        initial_cash=1000.0,
        max_close_ffill_rows=1,
        min_held_close_coverage=1.0,
    )

    with pytest.raises(ExecutionDataError, match="insufficient_held_close_coverage") as caught:
        sim.mark_state(state, dates[2])

    assert caught.value.coverage == pytest.approx(0.5)
    assert caught.value.source_mark_age["AAA"] == 2
