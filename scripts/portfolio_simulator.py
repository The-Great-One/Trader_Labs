"""Deterministic stateful D-close/D+1-open research portfolio simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd


class ExecutionDataError(ValueError):
    """Typed fail-closed market-data diagnostic."""

    def __init__(
        self,
        reason: str,
        *,
        signal_date: pd.Timestamp | None = None,
        execution_date: pd.Timestamp | None = None,
        valuation_date: pd.Timestamp | None = None,
        missing_symbols: tuple[str, ...] = (),
        coverage: float = 0.0,
        source_mark_age: dict[str, int] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.signal_date = signal_date
        self.execution_date = execution_date
        self.valuation_date = valuation_date
        self.missing_symbols = missing_symbols
        self.coverage = coverage
        self.source_mark_age = source_mark_age or {}


@dataclass(frozen=True)
class SignalIntent:
    signal_id: str
    signal_date: pd.Timestamp
    target_weights: Mapping[str, float]
    label: str = ""


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str
    quantity: float
    price: float
    notional: float
    fee: float


@dataclass(frozen=True)
class ExecutionEvent:
    signal_id: str
    signal_date: pd.Timestamp
    execution_date: pd.Timestamp
    pre_trade_nav: float
    gross_traded_notional: float
    one_way_turnover: float
    fees: float
    cash_after: float
    fills: tuple[Fill, ...]


@dataclass(frozen=True)
class ValuationEvent:
    date: pd.Timestamp
    nav: float
    cash: float
    units: dict[str, float]
    weights: dict[str, float]
    marks: dict[str, float]


@dataclass
class PortfolioState:
    cash: float
    units: dict[str, float] = field(default_factory=dict)
    last_marks: dict[str, float] = field(default_factory=dict)
    cumulative_fees: float = 0.0
    traded_notional: float = 0.0
    executed_signal_ids: set[str] = field(default_factory=set)
    execution_events: list[ExecutionEvent] = field(default_factory=list)
    valuation_events: list[ValuationEvent] = field(default_factory=list)


@dataclass(frozen=True)
class SimulationResult:
    final_state: PortfolioState
    execution_events: list[ExecutionEvent]
    valuations: list[ValuationEvent]
    nav: pd.Series
    returns: pd.Series
    pending_signals: list[SignalIntent]


class PortfolioSimulator:
    def __init__(
        self,
        opens: pd.DataFrame,
        closes: pd.DataFrame,
        *,
        initial_cash: float,
        cost_bps: float = 0.0,
        min_execution_open_coverage: float = 0.0,
        min_held_close_coverage: float = 0.0,
        max_close_ffill_rows: int = 0,
        close_source_dates: pd.DataFrame | None = None,
    ) -> None:
        self.opens = opens.sort_index()
        self.closes = closes.reindex(index=self.opens.index, columns=self.opens.columns)
        self.initial_cash = float(initial_cash)
        self.cost_rate = float(cost_bps) / 10_000.0
        self.min_execution_open_coverage = float(min_execution_open_coverage)
        self.min_held_close_coverage = float(min_held_close_coverage)
        self.max_close_ffill_rows = int(max_close_ffill_rows)
        self.close_source_dates = close_source_dates.reindex_like(self.closes) if close_source_dates is not None else None

    def run(self, signals: list[SignalIntent]) -> SimulationResult:
        state = PortfolioState(cash=self.initial_cash)
        by_execution: dict[pd.Timestamp, SignalIntent] = {}
        pending: list[SignalIntent] = []
        for signal in sorted(signals, key=lambda item: item.signal_date):
            later = self.opens.index[self.opens.index > pd.Timestamp(signal.signal_date)]
            if len(later) == 0:
                pending.append(signal)
            else:
                by_execution[later[0]] = signal

        nav_values: dict[pd.Timestamp, float] = {}
        for date in self.opens.index:
            signal = by_execution.get(date)
            if signal is not None:
                self.execute_signal(state, signal, date)
            valuation = self.mark_state(state, date)
            state.valuation_events.append(valuation)
            nav_values[date] = valuation.nav

        nav = pd.Series(nav_values, dtype=float)
        returns = nav.pct_change(fill_method=None).fillna(0.0)
        return SimulationResult(
            final_state=state,
            execution_events=state.execution_events,
            valuations=state.valuation_events,
            nav=nav,
            returns=returns,
            pending_signals=pending,
        )

    def execute_signal(self, state: PortfolioState, signal: SignalIntent, date: pd.Timestamp) -> None:
        if signal.signal_id in state.executed_signal_ids:
            raise ExecutionDataError(
                "duplicate_signal_id", signal_date=signal.signal_date, execution_date=date
            )
        prices = self.opens.loc[date]
        finite_positive = prices.map(lambda value: bool(np.isfinite(value) and value > 0))
        coverage = float(finite_positive.mean()) if len(finite_positive) else 0.0
        if coverage < self.min_execution_open_coverage:
            raise ExecutionDataError(
                "insufficient_execution_open_coverage",
                signal_date=signal.signal_date,
                execution_date=date,
                missing_symbols=tuple(sorted(finite_positive.index[~finite_positive])),
                coverage=coverage,
            )
        required = set(state.units) | set(signal.target_weights)
        invalid_required = tuple(sorted(symbol for symbol in required if symbol not in prices or not finite_positive[symbol]))
        if invalid_required:
            raise ExecutionDataError(
                "invalid_required_open",
                signal_date=signal.signal_date,
                execution_date=date,
                missing_symbols=invalid_required,
                coverage=coverage,
            )
        pre_trade_nav = state.cash + sum(quantity * float(prices[symbol]) for symbol, quantity in state.units.items())
        target_notional = {symbol: pre_trade_nav * float(weight) for symbol, weight in signal.target_weights.items()}
        symbols = sorted(set(state.units) | set(target_notional))
        deltas = {
            symbol: target_notional.get(symbol, 0.0) / float(prices[symbol]) - state.units.get(symbol, 0.0)
            for symbol in symbols
        }
        fills: list[Fill] = []
        for side in ("SELL", "BUY"):
            for symbol in symbols:
                delta = deltas[symbol]
                if (side == "SELL" and delta >= 0) or (side == "BUY" and delta <= 0):
                    continue
                price = float(prices[symbol])
                quantity = abs(delta)
                notional = quantity * price
                fee = notional * self.cost_rate
                if side == "SELL":
                    state.cash += notional - fee
                else:
                    affordable = max(state.cash / (price * (1 + self.cost_rate)), 0.0)
                    quantity = min(quantity, affordable)
                    notional = quantity * price
                    fee = notional * self.cost_rate
                    state.cash -= notional + fee
                signed = -quantity if side == "SELL" else quantity
                state.units[symbol] = state.units.get(symbol, 0.0) + signed
                if abs(state.units[symbol]) < 1e-12:
                    state.units.pop(symbol, None)
                fills.append(Fill(symbol, side, quantity, price, notional, fee))

        gross = sum(fill.notional for fill in fills)
        fees = sum(fill.fee for fill in fills)
        event = ExecutionEvent(
            signal_id=signal.signal_id,
            signal_date=pd.Timestamp(signal.signal_date),
            execution_date=date,
            pre_trade_nav=pre_trade_nav,
            gross_traded_notional=gross,
            one_way_turnover=gross / pre_trade_nav if pre_trade_nav else 0.0,
            fees=fees,
            cash_after=state.cash,
            fills=tuple(fills),
        )
        state.cumulative_fees += fees
        state.traded_notional += gross
        state.executed_signal_ids.add(signal.signal_id)
        state.execution_events.append(event)

    def mark_state(self, state: PortfolioState, date: pd.Timestamp) -> ValuationEvent:
        row_position = self.closes.index.get_loc(date)
        marks: dict[str, float] = {}
        invalid: list[str] = []
        ages: dict[str, int] = {}
        for symbol in state.units:
            value = self.closes.loc[date, symbol]
            source_date = date if self.close_source_dates is None else self.close_source_dates.loc[date, symbol]
            if pd.isna(source_date):
                age = self.max_close_ffill_rows + 1
            else:
                source_position = self.closes.index.get_indexer([pd.Timestamp(source_date)])[0]
                age = row_position - source_position if source_position >= 0 else self.max_close_ffill_rows + 1
            ages[symbol] = int(age)
            if not np.isfinite(value) or value <= 0 or age < 0 or age > self.max_close_ffill_rows:
                invalid.append(symbol)
            else:
                marks[symbol] = float(value)
        coverage = (len(state.units) - len(invalid)) / len(state.units) if state.units else 1.0
        if invalid or coverage < self.min_held_close_coverage:
            raise ExecutionDataError(
                "insufficient_held_close_coverage"
                if coverage < self.min_held_close_coverage
                else "invalid_held_close",
                valuation_date=date,
                missing_symbols=tuple(sorted(invalid)),
                coverage=coverage,
                source_mark_age={symbol: ages[symbol] for symbol in invalid},
            )
        nav = state.cash + sum(quantity * marks[symbol] for symbol, quantity in state.units.items())
        weights = {symbol: quantity * marks[symbol] / nav for symbol, quantity in state.units.items()} if nav else {}
        state.last_marks = marks
        return ValuationEvent(date, nav, state.cash, dict(state.units), weights, marks)
