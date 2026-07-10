#!/usr/bin/env python3
"""Auto-iteration strategy lab — runs on Oracle during non-market hours.

Tests multiple strategy families with parameter sweeps, saves results
incrementally. Designed to be cron-scheduled (e.g., every night at 10 PM IST).

Strategy families (sourced from Reddit r/algotrading):
  1. Trend Following — MA cross, MACD, RSI momentum
  2. Mean Reversion — RSI extremes, Bollinger Bands
  3. Breakout + ATR — Donchian/price channel with trailing stop
  4. Momentum Rotation — tactical allocation between symbols
  5. Volatility Breakout — range expansion
  6. Multi-Strategy — regime detection + strategy switching

Output: reports/auto_iteration_latest.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("AT_DISABLE_FILE_LOGGING", "1")
os.environ.setdefault("AT_RESEARCH_MODE", "1")
os.environ.setdefault("AT_LAB_PRECACHE", "0")

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import explore_strategies as ex

HIST_DIR = REPO / "intermediary_files" / "Hist_Data"
OUTPUT = REPO / "reports" / "auto_iteration_latest.json"
OUTPUT.parent.mkdir(exist_ok=True)

# ── Indicator computation ────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators needed across all strategy families."""
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # SMAs
    for p in [10, 20, 50, 100, 200]:
        df[f"SMA{p}"] = c.rolling(p).mean()

    # EMAs
    for p in [9, 12, 21, 26, 50]:
        df[f"EMA{p}"] = c.ewm(span=p, adjust=False).mean()

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # RSI for multiple periods
    for rsi_p in [2, 7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / rsi_p, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / rsi_p, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        df[f"RSI{rsi_p}"] = 100.0 - (100.0 / (1.0 + rs))

    # ATR
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()
    df["ATR20"] = tr.rolling(20).mean()

    # Bollinger Bands
    for bb_p in [20]:
        sma = c.rolling(bb_p).mean()
        std = c.rolling(bb_p).std()
        df[f"BB_upper_{bb_p}"] = sma + 2 * std
        df[f"BB_lower_{bb_p}"] = sma - 2 * std
        df[f"BB_mid_{bb_p}"] = sma
        df[f"BB_width_{bb_p}"] = (df[f"BB_upper_{bb_p}"] - df[f"BB_lower_{bb_p}"]) / sma

    # Volume ratio
    df["Vol_SMA20"] = v.rolling(20).mean()
    df["Vol_Ratio"] = v / df["Vol_SMA20"].replace(0, 1)

    # Returns
    df["Ret_1d"] = c.pct_change()
    df["Ret_5d"] = c.pct_change(5)
    df["Ret_20d"] = c.pct_change(20)

    # High-low range
    df["Range_pct"] = (h - l) / c.shift(1)

    # Donchian channels
    for dc_p in [10, 20]:
        df[f"DC_upper_{dc_p}"] = h.rolling(dc_p).max().shift(1)
        df[f"DC_lower_{dc_p}"] = l.rolling(dc_p).min().shift(1)

    return df


# ── Strategy simulators ──────────────────────────────────────────────────

def _simulate(
    df_raw: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any] | None:
    """Route to the correct strategy simulator based on family."""
    family = params.get("family", "trend")
    if family == "trend":
        return _sim_trend_following(df_raw, params, start, end)
    elif family == "mean_reversion":
        return _sim_mean_reversion(df_raw, params, start, end)
    elif family == "breakout":
        return _sim_breakout(df_raw, params, start, end)
    elif family == "momentum_rotation":
        return _sim_momentum_rotation(df_raw, params, start, end)
    elif family == "vol_breakout":
        return _sim_vol_breakout(df_raw, params, start, end)
    else:
        return None


def _get_date_mask(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    """Get valid index range for simulation."""
    dates = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    mask = (dates >= start) & (dates <= end)
    idxs = np.flatnonzero(mask.to_numpy())
    if len(idxs) < 30:
        return None, None
    warmup = 220
    start_idx = max(warmup, int(idxs[0]))
    end_idx = int(idxs[-1])
    if end_idx <= start_idx + 20:
        return None, None
    return start_idx, end_idx


def _compute_metrics(
    equity_curve: list[float],
    trades: int,
    wins: int,
    holding_bars: list[int],
    start_idx: int,
    end_idx: int,
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute standard metrics from equity curve."""
    final_equity = equity_curve[-1] if equity_curve else 100000.0
    total_return = (final_equity - 100000.0) / 100000.0 * 100

    days = (df.iloc[end_idx]["Date"] - df.iloc[start_idx]["Date"]).days
    years = max(days / 365.25, 0.1)
    cagr = ((final_equity / 100000.0) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

    eq_a = np.array(equity_curve)
    sharpe = 0.0
    if len(eq_a) > 20 and eq_a[:-1].mean() > 0:
        daily_ret = np.diff(eq_a) / (eq_a[:-1] + 1e-9)
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252)

    peak = np.maximum.accumulate(eq_a)
    dd = (peak - eq_a) / peak * 100
    max_dd = float(np.max(dd))

    return {
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(-max_dd, 2),
        "trades": trades,
        "wins": wins,
        "win_rate_pct": round(wins / max(trades, 1) * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "avg_holding_bars": round(float(np.mean(holding_bars)), 1) if holding_bars else 0,
        "final_equity": round(final_equity, 2),
    }


# ── Strategy 1: Trend Following ──────────────────────────────────────────

def _sim_trend_following(
    df_raw: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any] | None:
    """MA cross + RSI momentum trend following."""
    df = _compute_indicators(df_raw)
    start_idx, end_idx = _get_date_mask(df, start, end)
    if start_idx is None:
        return None

    ma_fast = int(params.get("ma_fast", 20))
    ma_slow = int(params.get("ma_slow", 50))
    rsi_period = int(params.get("rsi_period", 14))
    rsi_entry = float(params.get("rsi_entry", 50))  # enter when RSI > this
    atr_mult = float(params.get("atr_mult", 2.0))
    use_macd = bool(params.get("use_macd", False))

    fast_col = f"SMA{ma_fast}" if ma_fast in [10, 20, 50, 100, 200] else f"EMA{ma_fast}"
    slow_col = f"SMA{ma_slow}" if ma_slow in [10, 20, 50, 100, 200] else f"EMA{ma_slow}"
    rsi_col = f"RSI{rsi_period}"

    if fast_col not in df.columns:
        df[fast_col] = df["Close"].ewm(span=ma_fast, adjust=False).mean()
    if slow_col not in df.columns:
        df[slow_col] = df["Close"].ewm(span=ma_slow, adjust=False).mean()

    close = df["Close"].to_numpy(dtype=float)
    fast_ma = df[fast_col].to_numpy(dtype=float)
    slow_ma = df[slow_col].to_numpy(dtype=float)
    rsi = df[rsi_col].to_numpy(dtype=float) if rsi_col in df.columns else np.full(len(close), 50.0)
    atr = df["ATR14"].to_numpy(dtype=float)
    macd_hist = df["MACD_Hist"].to_numpy(dtype=float) if use_macd else None

    capital = 100000.0
    qty = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    trades = 0
    wins = 0
    equity_curve: list[float] = []
    holding_bars: list[int] = []
    entry_idx: int | None = None

    for i in range(start_idx, end_idx + 1):
        price = close[i]
        if np.isnan(price) or price <= 0:
            if qty > 0:
                capital += qty * max(close[max(0, i - 1)], 1.0)
                qty = 0
            equity_curve.append(capital)
            continue

        if qty == 0:
            # Entry: fast MA > slow MA AND RSI > threshold
            f = fast_ma[i]
            s = slow_ma[i]
            r = rsi[i]
            if np.isnan(f) or np.isnan(s):
                equity_curve.append(capital)
                continue

            ma_ok = f > s
            rsi_ok = r > rsi_entry
            macd_ok = True
            if use_macd and macd_hist is not None:
                macd_ok = macd_hist[i] > 0

            if ma_ok and rsi_ok and macd_ok:
                invest = capital * 0.95
                buy_qty = int(invest // price)
                if buy_qty > 0:
                    qty = buy_qty
                    capital -= qty * price
                    entry_price = price
                    entry_idx = i
                    highest_since_entry = price
                    trades += 1
        else:
            bars_held = i - entry_idx if entry_idx is not None else 0
            if price > highest_since_entry:
                highest_since_entry = price

            # Exit: fast MA crosses below slow MA OR trailing stop
            exit_signal = False
            f = fast_ma[i]
            s = slow_ma[i]

            if not np.isnan(f) and not np.isnan(s) and f < s and bars_held > 3:
                exit_signal = True

            trail_stop = highest_since_entry - atr_mult * atr[i]
            if not np.isnan(trail_stop) and price < trail_stop and bars_held > 3:
                exit_signal = True

            if exit_signal:
                capital += qty * price
                pnl = (price - entry_price) * qty
                if price > entry_price:
                    wins += 1
                holding_bars.append(bars_held)
                qty = 0
                entry_price = 0.0
                entry_idx = None
                highest_since_entry = 0.0

        equity_curve.append(capital + qty * price)

    if qty > 0:
        price = float(close[end_idx])
        capital += qty * price
        if price > entry_price:
            wins += 1
        if entry_idx is not None:
            holding_bars.append(end_idx - entry_idx)

    return _compute_metrics(equity_curve, trades, wins, holding_bars, start_idx, end_idx, df)


# ── Strategy 2: Mean Reversion ───────────────────────────────────────────

def _sim_mean_reversion(
    df_raw: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any] | None:
    """RSI oversold/overbought + Bollinger Band mean reversion."""
    df = _compute_indicators(df_raw)
    start_idx, end_idx = _get_date_mask(df, start, end)
    if start_idx is None:
        return None

    rsi_period = int(params.get("rsi_period", 14))
    rsi_oversold = float(params.get("rsi_oversold", 30))
    rsi_overbought = float(params.get("rsi_overbought", 70))
    use_bb = bool(params.get("use_bb", True))
    max_hold = int(params.get("max_hold", 20))

    rsi_col = f"RSI{rsi_period}"
    close = df["Close"].to_numpy(dtype=float)
    rsi = df[rsi_col].to_numpy(dtype=float) if rsi_col in df.columns else np.full(len(close), 50.0)
    bb_lower = df["BB_lower_20"].to_numpy(dtype=float) if use_bb else None
    bb_upper = df["BB_upper_20"].to_numpy(dtype=float) if use_bb else None

    capital = 100000.0
    qty = 0
    entry_price = 0.0
    direction = 0  # 1=long, -1=short
    trades = 0
    wins = 0
    equity_curve: list[float] = []
    holding_bars: list[int] = []
    entry_idx: int | None = None

    for i in range(start_idx, end_idx + 1):
        price = close[i]
        if np.isnan(price) or price <= 0:
            if qty > 0:
                capital += qty * max(close[max(0, i - 1)], 1.0)
                qty = 0
            equity_curve.append(capital)
            continue

        if qty == 0:
            r = rsi[i]
            if np.isnan(r):
                equity_curve.append(capital)
                continue

            # Long entry: RSI oversold + price below BB lower
            long_signal = r < rsi_oversold
            if use_bb and bb_lower is not None:
                long_signal = long_signal and price < bb_lower[i]

            # Short entry: RSI overbought + price above BB upper
            short_signal = r > rsi_overbought
            if use_bb and bb_upper is not None:
                short_signal = short_signal and price > bb_upper[i]

            if long_signal:
                invest = capital * 0.95
                buy_qty = int(invest // price)
                if buy_qty > 0:
                    qty = buy_qty
                    capital -= qty * price
                    entry_price = price
                    direction = 1
                    entry_idx = i
                    trades += 1
            elif short_signal:
                # Simulate short: sell borrowed shares
                invest = capital * 0.95
                sell_qty = int(invest // price)
                if sell_qty > 0:
                    qty = sell_qty
                    capital += qty * price  # receive cash
                    entry_price = price
                    direction = -1
                    entry_idx = i
                    trades += 1
        else:
            bars_held = i - entry_idx if entry_idx is not None else 0

            # Exit: RSI crosses back to neutral OR time stop
            r = rsi[i]
            exit_signal = False

            if direction == 1:
                # Exit long when RSI > 50 (mean reversion complete)
                if r > 50 and bars_held > 1:
                    exit_signal = True
            elif direction == -1:
                # Exit short when RSI < 50
                if r < 50 and bars_held > 1:
                    exit_signal = True

            if max_hold > 0 and bars_held >= max_hold:
                exit_signal = True

            if exit_signal:
                if direction == 1:
                    capital += qty * price
                    pnl = (price - entry_price) * qty
                else:
                    capital -= qty * price
                    pnl = (entry_price - price) * qty

                if pnl > 0:
                    wins += 1
                holding_bars.append(bars_held)
                qty = 0
                direction = 0
                entry_price = 0.0
                entry_idx = None

        if direction == 1:
            equity_curve.append(capital + qty * price)
        elif direction == -1:
            equity_curve.append(capital - qty * price)
        else:
            equity_curve.append(capital)

    if qty > 0:
        price = float(close[end_idx])
        if direction == 1:
            capital += qty * price
        else:
            capital -= qty * price
        if entry_idx is not None:
            holding_bars.append(end_idx - entry_idx)

    return _compute_metrics(equity_curve, trades, wins, holding_bars, start_idx, end_idx, df)


# ── Strategy 3: Breakout + ATR ───────────────────────────────────────────

def _sim_breakout(
    df_raw: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any] | None:
    """Donchian/price channel breakout with ATR trailing stop."""
    df = _compute_indicators(df_raw)
    start_idx, end_idx = _get_date_mask(df, start, end)
    if start_idx is None:
        return None

    dc_period = int(params.get("dc_period", 20))
    atr_trail = float(params.get("atr_trail", 2.5))
    vol_mult = float(params.get("vol_mult", 1.0))
    sma_period = int(params.get("sma_period", 50))

    dc_col = f"DC_upper_{dc_period}"
    sma_col = f"SMA{sma_period}" if sma_period in [10, 20, 50, 100, 200] else f"EMA{sma_period}"

    close = df["Close"].to_numpy(dtype=float)
    dc_upper = df[dc_col].to_numpy(dtype=float) if dc_col in df.columns else np.full(len(close), np.nan)
    trend = df[sma_col].to_numpy(dtype=float) if sma_col in df.columns else np.full(len(close), np.nan)
    atr = df["ATR14"].to_numpy(dtype=float)
    vol_ratio = df["Vol_Ratio"].to_numpy(dtype=float)

    capital = 100000.0
    qty = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    trades = 0
    wins = 0
    equity_curve: list[float] = []
    holding_bars: list[int] = []
    entry_idx: int | None = None

    for i in range(start_idx, end_idx + 1):
        price = close[i]
        if np.isnan(price) or price <= 0:
            if qty > 0:
                capital += qty * max(close[max(0, i - 1)], 1.0)
                qty = 0
            equity_curve.append(capital)
            continue

        if qty == 0:
            upper = dc_upper[i]
            if np.isnan(upper) or upper <= 0:
                equity_curve.append(capital)
                continue

            breakout = price > upper
            vol_ok = not np.isnan(vol_ratio[i]) and vol_ratio[i] >= vol_mult
            trend_ok = not np.isnan(trend[i]) and price > trend[i]

            if breakout and vol_ok and trend_ok:
                invest = capital * 0.95
                buy_qty = int(invest // price)
                if buy_qty > 0:
                    qty = buy_qty
                    capital -= qty * price
                    entry_price = price
                    entry_idx = i
                    highest_since_entry = price
                    trades += 1
        else:
            bars_held = i - entry_idx if entry_idx is not None else 0
            if price > highest_since_entry:
                highest_since_entry = price

            trail_stop = highest_since_entry - atr_trail * atr[i]
            exit_signal = not np.isnan(trail_stop) and price < trail_stop and bars_held > 3

            if exit_signal:
                capital += qty * price
                pnl = (price - entry_price) * qty
                if price > entry_price:
                    wins += 1
                holding_bars.append(bars_held)
                qty = 0
                entry_price = 0.0
                entry_idx = None
                highest_since_entry = 0.0

        equity_curve.append(capital + qty * price)

    if qty > 0:
        price = float(close[end_idx])
        capital += qty * price
        if price > entry_price:
            wins += 1
        if entry_idx is not None:
            holding_bars.append(end_idx - entry_idx)

    return _compute_metrics(equity_curve, trades, wins, holding_bars, start_idx, end_idx, df)


# ── Strategy 4: Momentum Rotation ────────────────────────────────────────

def _sim_momentum_rotation(
    df_raw: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any] | None:
    """Buy top N symbols by recent momentum, rotate monthly."""
    df = _compute_indicators(df_raw)
    start_idx, end_idx = _get_date_mask(df, start, end)
    if start_idx is None:
        return None

    lookback = int(params.get("lookback", 20))
    rebalance_freq = int(params.get("rebalance_freq", 20))  # bars between rebalances
    top_n = int(params.get("top_n", 5))  # not used per-symbol, handled at portfolio level

    close = df["Close"].to_numpy(dtype=float)
    ret_col = f"Ret_{lookback}d"

    capital = 100000.0
    qty = 0
    entry_price = 0.0
    trades = 0
    wins = 0
    equity_curve: list[float] = []
    holding_bars: list[int] = []
    entry_idx: int | None = None
    last_rebalance = start_idx

    for i in range(start_idx, end_idx + 1):
        price = close[i]
        if np.isnan(price) or price <= 0:
            if qty > 0:
                capital += qty * max(close[max(0, i - 1)], 1.0)
                qty = 0
            equity_curve.append(capital)
            continue

        bars_since = i - last_rebalance

        if qty == 0:
            # Enter on rebalance: buy if momentum is positive
            mom = df[ret_col].iloc[i] if ret_col in df.columns else 0
            if bars_since >= rebalance_freq and not np.isnan(mom) and mom > 0:
                invest = capital * 0.95
                buy_qty = int(invest // price)
                if buy_qty > 0:
                    qty = buy_qty
                    capital -= qty * price
                    entry_price = price
                    entry_idx = i
                    trades += 1
                    last_rebalance = i
        else:
            bars_held = i - entry_idx if entry_idx is not None else 0

            # Exit on next rebalance
            if bars_since >= rebalance_freq:
                capital += qty * price
                pnl = (price - entry_price) * qty
                if price > entry_price:
                    wins += 1
                holding_bars.append(bars_held)
                qty = 0
                entry_price = 0.0
                entry_idx = None
                last_rebalance = i

        equity_curve.append(capital + qty * price)

    if qty > 0:
        price = float(close[end_idx])
        capital += qty * price
        if price > entry_price:
            wins += 1
        if entry_idx is not None:
            holding_bars.append(end_idx - entry_idx)

    return _compute_metrics(equity_curve, trades, wins, holding_bars, start_idx, end_idx, df)


# ── Strategy 5: Volatility Breakout ──────────────────────────────────────

def _sim_vol_breakout(
    df_raw: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any] | None:
    """Range expansion breakout: enter when today's range > N * avg range."""
    df = _compute_indicators(df_raw)
    start_idx, end_idx = _get_date_mask(df, start, end)
    if start_idx is None:
        return None

    range_mult = float(params.get("range_mult", 1.5))
    atr_trail = float(params.get("atr_trail", 2.0))
    max_hold = int(params.get("max_hold", 20))

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    range_pct = df["Range_pct"].to_numpy(dtype=float)
    atr = df["ATR14"].to_numpy(dtype=float)
    avg_range = df["ATR14"].to_numpy(dtype=float) / close  # approximate avg range %

    capital = 100000.0
    qty = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    trades = 0
    wins = 0
    equity_curve: list[float] = []
    holding_bars: list[int] = []
    entry_idx: int | None = None

    for i in range(start_idx, end_idx + 1):
        price = close[i]
        if np.isnan(price) or price <= 0:
            if qty > 0:
                capital += qty * max(close[max(0, i - 1)], 1.0)
                qty = 0
            equity_curve.append(capital)
            continue

        if qty == 0:
            rp = range_pct[i]
            ar = avg_range[i]
            if np.isnan(rp) or np.isnan(ar) or ar <= 0:
                equity_curve.append(capital)
                continue

            # Entry: range expansion AND price closes near the high (bullish)
            range_ok = rp > range_mult * ar
            bullish = price > (high[i] + low[i]) / 2

            if range_ok and bullish:
                invest = capital * 0.95
                buy_qty = int(invest // price)
                if buy_qty > 0:
                    qty = buy_qty
                    capital -= qty * price
                    entry_price = price
                    entry_idx = i
                    highest_since_entry = price
                    trades += 1
        else:
            bars_held = i - entry_idx if entry_idx is not None else 0
            if price > highest_since_entry:
                highest_since_entry = price

            trail_stop = highest_since_entry - atr_trail * atr[i]
            exit_signal = not np.isnan(trail_stop) and price < trail_stop and bars_held > 3

            if max_hold > 0 and bars_held >= max_hold:
                exit_signal = True

            if exit_signal:
                capital += qty * price
                pnl = (price - entry_price) * qty
                if price > entry_price:
                    wins += 1
                holding_bars.append(bars_held)
                qty = 0
                entry_price = 0.0
                entry_idx = None
                highest_since_entry = 0.0

        equity_curve.append(capital + qty * price)

    if qty > 0:
        price = float(close[end_idx])
        capital += qty * price
        if price > entry_price:
            wins += 1
        if entry_idx is not None:
            holding_bars.append(end_idx - entry_idx)

    return _compute_metrics(equity_curve, trades, wins, holding_bars, start_idx, end_idx, df)


# ── Aggregation ──────────────────────────────────────────────────────────

def _agg(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cagr_pct": 0, "max_drawdown_pct": 0, "trades": 0, "win_rate_pct": 0, "sharpe_ratio": 0, "active_symbols": 0, "profitable_symbols": 0, "symbols": 0, "total_return_pct": 0}
    syms = len(rows)
    total_start = 100000.0 * syms
    total_final = sum(float(r["final_equity"]) for r in rows.values())
    ret = (total_final / total_start - 1) * 100.0
    trades = sum(int(r["trades"]) for r in rows.values())
    wins = sum(int(r.get("wins", 0)) for r in rows.values())
    active = sum(1 for r in rows.values() if int(r["trades"]) > 0)
    prof = sum(1 for r in rows.values() if float(r["total_return_pct"]) > 0)
    years = 5.0
    cagr = ((total_final / total_start) ** (1.0 / years) - 1.0) * 100.0 if total_final > 0 else -100.0
    return {
        "symbols": syms,
        "active_symbols": active,
        "profitable_symbols": prof,
        "trades": trades,
        "win_rate_pct": round((wins / max(1, trades)) * 100.0, 2),
        "total_return_pct": round(float(ret), 2),
        "cagr_pct": round(float(cagr), 2),
        "max_drawdown_pct": round(float(np.mean([float(r["max_drawdown_pct"]) for r in rows.values()])), 2),
        "sharpe_ratio": round(float(np.mean([float(r["sharpe_ratio"]) for r in rows.values()])), 2),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
    data_map = ex._load_data()
    if not data_map:
        print("ERROR: No data loaded")
        return 1
    print(f"Loaded {len(data_map)} symbols")

    # Determine date range
    starts, ends = [], []
    for df in data_map.values():
        d = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        starts.append(d.min())
        ends.append(d.max())
    date_start = min(starts)
    date_end = max(ends)
    print(f"Date range: {date_start.date()} → {date_end.date()}")

    # ── Parameter grids ──────────────────────────────────────────────
    all_combos: list[dict[str, Any]] = []

    # 1. Trend Following
    for ma_fast, ma_slow in [(10, 20), (20, 50), (50, 200)]:
        for rsi_p in [14]:
            for rsi_entry in [40, 50, 60]:
                for atr_m in [2.0, 2.5]:
                    all_combos.append({"family": "trend", "ma_fast": ma_fast, "ma_slow": ma_slow, "rsi_period": rsi_p, "rsi_entry": rsi_entry, "atr_mult": atr_m, "use_macd": False})
    # Trend + MACD
    for ma_fast, ma_slow in [(20, 50)]:
        for rsi_entry in [50]:
            for atr_m in [2.0]:
                all_combos.append({"family": "trend", "ma_fast": ma_fast, "ma_slow": ma_slow, "rsi_period": 14, "rsi_entry": rsi_entry, "atr_mult": atr_m, "use_macd": True})

    # 2. Mean Reversion
    for rsi_p in [7, 14]:
        for rsi_os, rsi_ob in [(30, 70), (20, 80), (25, 75)]:
            for use_bb in [True, False]:
                for max_hold in [10, 20]:
                    all_combos.append({"family": "mean_reversion", "rsi_period": rsi_p, "rsi_oversold": rsi_os, "rsi_overbought": rsi_ob, "use_bb": use_bb, "max_hold": max_hold})

    # 3. Breakout + ATR
    for dc_p in [10, 20]:
        for atr_t in [2.0, 2.5]:
            for sma_p in [50, 100]:
                all_combos.append({"family": "breakout", "dc_period": dc_p, "atr_trail": atr_t, "vol_mult": 1.0, "sma_period": sma_p})

    # 4. Momentum Rotation
    for lookback in [10, 20, 60]:
        for rebalance in [10, 20]:
            all_combos.append({"family": "momentum_rotation", "lookback": lookback, "rebalance_freq": rebalance, "top_n": 5})

    # 5. Volatility Breakout
    for range_m in [1.5, 2.0, 2.5]:
        for atr_t in [2.0, 2.5]:
            for max_h in [10, 20]:
                all_combos.append({"family": "vol_breakout", "range_mult": range_m, "atr_trail": atr_t, "max_hold": max_h})

    print(f"Testing {len(all_combos)} parameter combinations across 5 strategy families...")

    results: list[dict[str, Any]] = []
    for idx, params in enumerate(all_combos):
        family = params["family"]
        label = f"{family}_" + "_".join(f"{k}={v}" for k, v in params.items() if k != "family")
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx + 1}/{len(all_combos)}] {label[:80]}...")

        per_sym: dict[str, dict[str, Any]] = {}
        for sym, df_raw in data_map.items():
            r = _simulate(df_raw, params, date_start, date_end)
            if r is not None:
                per_sym[sym] = r

        agg = _agg(per_sym)
        results.append({"label": label, "family": family, "params": params, "agg": agg})

    # Sort by CAGR
    results.sort(key=lambda x: x["agg"]["cagr_pct"], reverse=True)

    # Print top 30
    print(f"\n{'─' * 110}")
    print(f"{'Rank':<5} {'Family':<18} {'Label':<45} {'CAGR%':>8} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7} {'Sharpe':>7}")
    print(f"{'─' * 110}")
    for i, r in enumerate(results[:30], 1):
        a = r["agg"]
        print(f"{i:<5} {r['family']:<18} {r['label'][:44]:<45} {a['cagr_pct']:>8.1f} {a['max_drawdown_pct']:>8.1f} {a['trades']:>7} {a['win_rate_pct']:>7.1f} {a['sharpe_ratio']:>7.2f}")

    # Family summary
    print(f"\n{'─' * 80}")
    print(f"{'Family':<20} {'Best CAGR%':>10} {'Best DD%':>10} {'Combos':>8}")
    print(f"{'─' * 80}")
    for family in ["trend", "mean_reversion", "breakout", "momentum_rotation", "vol_breakout"]:
        fam_results = [r for r in results if r["family"] == family]
        if fam_results:
            best = fam_results[0]
            print(f"{family:<20} {best['agg']['cagr_pct']:>10.1f} {best['agg']['max_drawdown_pct']:>10.1f} {len(fam_results):>8}")

    # Write output
    output = {
        "generated_at": datetime.now().isoformat(),
        "date_range": f"{date_start.date()} → {date_end.date()}",
        "symbols_loaded": len(data_map),
        "combinations_tested": len(all_combos),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {len(results)} results to {OUTPUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
