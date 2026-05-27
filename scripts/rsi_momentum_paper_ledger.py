#!/usr/bin/env python3
"""RSI + Momentum Paper Ledger — daily portfolio simulation.

Reads the latest paper shadow picks and simulates an equal-weight
portfolio. Rebalances on month-end signal dates, marks to market daily.
Tracks full P&L history, drawdown, and risk metrics.

State file: reports/paper_ledger_rsi_momentum_state.json
Output: reports/paper_ledger_rsi_momentum_latest.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports"
HIST_DIR = ROOT / "intermediary_files" / "Hist_Data"
OUT_DIR.mkdir(exist_ok=True)

# Config
INITIAL_CAPITAL = float(os.getenv("RSI_LEDGER_CAPITAL", "1000000"))  # ₹10L
COST_BPS = float(os.getenv("RSI_LEDGER_COST_BPS", "10"))
PAPER_SHADOW_FILE = OUT_DIR / "paper_shadow_rsi_momentum_latest.json"
STATE_FILE = OUT_DIR / "paper_ledger_rsi_momentum_state.json"
OUTPUT_FILE = OUT_DIR / "paper_ledger_rsi_momentum_latest.json"


# ── Data loading ──────────────────────────────────────────────

def load_prices(hist_dir: Path, min_rows: int = 350) -> pd.DataFrame:
    """Load OHLCV close prices from feather files."""
    if not hist_dir.is_dir():
        return pd.DataFrame()
    loaded = {}
    for fpath in sorted(hist_dir.glob("*.feather")):
        symbol = fpath.stem
        try:
            df = pd.read_feather(fpath)
        except Exception:
            continue
        if any(kw in symbol for kw in ["FUT", "OPT", "-I", "-II"]):
            continue
        date_col = next((c for c in ["date", "Date", "datetime"] if c in df.columns), None)
        close_col = next((c for c in ["close", "Close", "CLOSE"] if c in df.columns), None)
        if date_col is None or close_col is None:
            continue
        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        s = df.set_index(date_col)[close_col].dropna().sort_index()
        if len(s) >= min_rows:
            loaded[symbol] = s
    return pd.DataFrame(loaded).sort_index()


# ── State management ──────────────────────────────────────────

@dataclass
class PortfolioState:
    """Persistent state for the paper ledger."""
    cash: float = INITIAL_CAPITAL
    positions: dict[str, float] = field(default_factory=dict)  # symbol → shares
    cost_basis: dict[str, float] = field(default_factory=dict)  # symbol → avg buy price
    total_invested: float = 0.0
    last_rebalance_date: str = ""
    daily_values: list[dict] = field(default_factory=list)  # [{date, value, return}]
    trade_log: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PortfolioState":
        return cls(
            cash=d.get("cash", INITIAL_CAPITAL),
            positions=d.get("positions", {}),
            cost_basis=d.get("cost_basis", {}),
            total_invested=d.get("total_invested", 0.0),
            last_rebalance_date=d.get("last_rebalance_date", ""),
            daily_values=d.get("daily_values", []),
            trade_log=d.get("trade_log", []),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


def load_state() -> PortfolioState:
    if STATE_FILE.exists():
        try:
            return PortfolioState.from_dict(json.loads(STATE_FILE.read_text()))
        except Exception:
            pass
    state = PortfolioState(created_at=datetime.now().isoformat())
    return state


def save_state(state: PortfolioState) -> None:
    state.updated_at = datetime.now().isoformat()
    STATE_FILE.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


# ── Core simulation ──────────────────────────────────────────

def get_latest_signal() -> Optional[dict]:
    """Read the latest paper shadow signal."""
    if not PAPER_SHADOW_FILE.exists():
        return None
    try:
        data = json.loads(PAPER_SHADOW_FILE.read_text())
        return data.get("latest_signal")
    except Exception:
        return None


def portfolio_value(state: PortfolioState, prices: dict[str, float]) -> float:
    """Calculate current portfolio value (cash + positions MTM)."""
    position_value = 0.0
    for symbol, shares in state.positions.items():
        if symbol in prices and prices[symbol] > 0:
            position_value += shares * prices[symbol]
    return state.cash + position_value


def execute_rebalance(
    state: PortfolioState,
    picks: list[str],
    prices_series: pd.Series,
    date: str,
    cost_bps: float = COST_BPS,
) -> PortfolioState:
    """Sell everything, buy new picks equal-weight."""
    cost_rate = cost_bps / 10000.0

    # 1. Sell existing positions
    sold_value = 0.0
    for symbol, shares in list(state.positions.items()):
        if symbol in prices_series and prices_series[symbol] > 0:
            px = float(prices_series[symbol])
            gross = shares * px
            cost = gross * cost_rate
            net = gross - cost
            state.cash += net
            sold_value += net
            state.trade_log.append({
                "date": date,
                "action": "SELL",
                "symbol": symbol,
                "shares": round(shares, 2),
                "price": round(px, 2),
                "gross": round(gross, 2),
                "cost": round(cost, 2),
                "net": round(net, 2),
            })
    state.positions.clear()
    state.cost_basis.clear()

    # 2. Buy new picks equal-weight
    available = [s for s in picks if s in prices_series and pd.notna(prices_series[s]) and prices_series[s] > 0]
    if not available:
        state.last_rebalance_date = date
        return state

    per_symbol_capital = state.cash / len(available)
    for symbol in available:
        px = float(prices_series[symbol])
        gross_allocation = per_symbol_capital
        cost = gross_allocation * cost_rate
        net_allocation = gross_allocation - cost
        shares = net_allocation / px
        state.positions[symbol] = shares
        state.cost_basis[symbol] = px
        state.cash -= gross_allocation
        state.trade_log.append({
            "date": date,
            "action": "BUY",
            "symbol": symbol,
            "shares": round(shares, 2),
            "price": round(px, 2),
            "gross": round(gross_allocation, 2),
            "cost": round(cost, 2),
            "net": round(net_allocation, 2),
        })

    state.last_rebalance_date = date
    return state


def compute_metrics(daily_values: list[dict]) -> dict:
    """Compute performance metrics from daily value history."""
    if len(daily_values) < 20:
        return {"error": "insufficient history"}

    df = pd.DataFrame(daily_values)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    initial = df["value"].iloc[0]
    final = df["value"].iloc[-1]
    total_return = (final / initial) - 1

    # Daily returns
    df["returns"] = df["value"].pct_change().fillna(0)
    years = len(df) / 252
    cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0

    # Drawdown
    peak = df["value"].cummax()
    drawdown = df["value"] / peak - 1
    max_dd = float(drawdown.min())

    # Vol + Sharpe
    daily_r = df["returns"].iloc[1:]  # exclude first day
    vol = float(daily_r.std() * math.sqrt(252)) if len(daily_r) > 1 else 0.0
    sharpe = float((daily_r.mean() * 252) / vol) if vol > 0 else 0.0

    # Monthly returns
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    monthly = df.groupby("month")["returns"].apply(lambda x: (1 + x).prod() - 1)
    positive_months = int((monthly > 0).sum())

    return {
        "days_tracked": len(df),
        "years": round(years, 2),
        "initial_capital": round(initial, 2),
        "current_value": round(final, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "vol_pct": round(vol * 100, 1),
        "sharpe": round(sharpe, 3),
        "positive_months": positive_months,
        "total_months": int(len(monthly)),
        "monthly_returns": {str(k): round(float(v) * 100, 2) for k, v in monthly.tail(12).items()},
    }


def should_rebalance(state: PortfolioState, signal: dict, today: str) -> bool:
    """Check if today is the first trading day after a new signal."""
    signal_date = signal.get("date", "")
    if not signal_date:
        return False
    # Rebalance if signal is newer than last rebalance
    if not state.last_rebalance_date:
        return True
    return signal_date > state.last_rebalance_date


def log_daily(state: PortfolioState, value: float, date: str) -> None:
    """Record daily portfolio value."""
    if state.daily_values:
        prev_value = state.daily_values[-1]["value"]
        daily_return = (value / prev_value) - 1 if prev_value > 0 else 0.0
    else:
        daily_return = 0.0
    state.daily_values.append({
        "date": date,
        "value": round(value, 2),
        "return_pct": round(daily_return * 100, 4),
        "positions": len(state.positions),
        "cash": round(state.cash, 2),
    })
    # Keep last 2 years
    if len(state.daily_values) > 504:
        state.daily_values = state.daily_values[-504:]


# ── Main ────────────────────────────────────────────────────

def main() -> int:
    prices_df = load_prices(HIST_DIR)
    if prices_df.empty:
        print("ERROR: no price data")
        return 1

    signal = get_latest_signal()
    if signal is None:
        print("WARN: no paper shadow signal found — skipping")
        return 0

    signal_date = signal.get("date", "")
    picks = signal.get("picks", [])

    if not picks:
        print("WARN: no picks in signal")
        return 0

    # Today = latest available date in price data (end-of-day)
    # In cron: this is today's EOD data
    today = str(prices_df.index[-1].date())
    today_prices = prices_df.iloc[-1]  # latest row prices

    # Signal date prices — for executing buys/sells at correct entry prices
    signal_dt = pd.Timestamp(signal_date)
    if signal_dt in prices_df.index:
        signal_prices = prices_df.loc[signal_dt]
    else:
        # Find the nearest trading day at or after signal date
        idx = prices_df.index.searchsorted(signal_dt)
        if idx < len(prices_df):
            signal_prices = prices_df.iloc[idx]
        else:
            signal_prices = prices_df.iloc[-1]

    # Load state
    state = load_state()

    # Check if rebalance needed
    if should_rebalance(state, signal, signal_date):
        print(f"REBALANCE: signal {signal_date} is newer than last rebalance {state.last_rebalance_date}")
        state = execute_rebalance(state, picks, signal_prices, signal_date)
    elif not state.positions:
        # First run — initialize with current signal
        print(f"INIT: first run, buying {len(picks)} picks from signal {signal_date}")
        state = execute_rebalance(state, picks, signal_prices, signal_date)

    # MTM current positions — use last available price per position symbol
    prices_dict = {}
    for sym in state.positions:
        if sym in prices_df.columns:
            col = prices_df[sym].ffill()
            last_valid = col.last_valid_index()
            if last_valid is not None and col.loc[last_valid] > 0:
                prices_dict[sym] = float(col.loc[last_valid])
    current_value = portfolio_value(state, prices_dict)

    # Log daily value
    log_daily(state, current_value, today)

    # Compute metrics
    metrics = compute_metrics(state.daily_values)

    # Save state
    save_state(state)

    # Build output
    positions_detail = {}
    for sym, shares in state.positions.items():
        if sym in prices_dict:
            px = prices_dict[sym]
            mv = shares * px
            cost = state.cost_basis.get(sym, 0)
            positions_detail[sym] = {
                "shares": round(shares, 2),
                "avg_price": round(cost, 2),
                "current_price": round(px, 2),
                "market_value": round(mv, 2),
                "pnl_pct": round((px / cost - 1) * 100, 2) if cost > 0 else 0.0,
            }

    output = {
        "generated_at": datetime.now().isoformat(),
        "strategy": "rsi_momentum_rotation_paper_ledger",
        "signal": {
            "date": signal_date,
            "picks": picks,
        },
        "portfolio": {
            "cash": round(state.cash, 2),
            "position_value": round(current_value - state.cash, 2),
            "total_value": round(current_value, 2),
            "positions_count": len(state.positions),
            "positions": positions_detail,
            "last_rebalance": state.last_rebalance_date,
            "created_at": state.created_at,
        },
        "metrics": metrics,
        "latest_trades": state.trade_log[-20:],
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # Print summary
    print(f"\n=== RSI Momentum Paper Ledger ===")
    print(f"Date: {today} | Signal: {signal_date} | Picks: {len(picks)}")
    print(f"Portfolio:  ₹{current_value:,.2f}  (Cash: ₹{state.cash:,.2f}, Positions: {len(state.positions)})")
    if "total_return_pct" in metrics:
        print(f"Return:     {metrics['total_return_pct']:+.2f}%  CAGR: {metrics.get('cagr_pct', 0):+.2f}%")
        print(f"MaxDD:      {metrics['max_drawdown_pct']:+.2f}%  Sharpe: {metrics.get('sharpe', 0):.3f}")
    print(f"Saved: {OUTPUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
