#!/usr/bin/env python3
"""Donchian + Volume + SMA breakout strategy sweep.

Tests structural combinations of:
- Donchian breakout period (N-day high)
- Volume confirmation multiplier
- SMA trend filter period
- ATR trailing stop multiplier
- Max hold bars (time stop)

Output: reports/donchian_volume_sma_sweep_latest.json
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import explore_strategies as ex  # noqa: E402

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
OUTPUT = REPORTS / "donchian_volume_sma_sweep_latest.json"
HIST_DIR = ROOT / "intermediary_files" / "Hist_Data"


# ── Donchian indicator computation (per-symbol) ─────────────────────────

def _add_donchian(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Add Donchian channel columns to a precomputed indicator DataFrame."""
    df = df.copy()
    h = df["High"]
    l = df["Low"]
    # Shift so today's breakout doesn't look at today's high (no lookahead)
    df[f"DC_upper_{period}"] = h.rolling(period).max().shift(1)
    df[f"DC_lower_{period}"] = l.rolling(period).min().shift(1)
    df[f"DC_mid_{period}"] = (df[f"DC_upper_{period}"] + df[f"DC_lower_{period}"]) / 2
    df[f"DC_pct_{period}"] = (df["Close"] - df[f"DC_lower_{period}"]) / (
        df[f"DC_upper_{period}"] - df[f"DC_lower_{period}"]
    ).replace(0, 1)
    return df


# ── Lightweight indicator computation (no SuperTrend, no ADX, fast) ──

def _compute_light(df: pd.DataFrame) -> pd.DataFrame:
    """Compute only indicators needed for Donchian+Volume+SMA+RSI strategy."""
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    # EMAs (used as SMA fallback for non-standard periods)
    for p in [20, 50, 100, 150, 200]:
        df[f"EMA{p}"] = c.ewm(span=p, adjust=False).mean()

    # SMAs
    for p in [20, 50, 100, 200]:
        df[f"SMA{p}"] = c.rolling(p).mean()

    # ATR
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # Volume ratio
    df["Vol_SMA20"] = v.rolling(20).mean()
    df["Vol_Ratio"] = v / df["Vol_SMA20"].replace(0, 1)

    # RSI for periods 7, 14, 21
    for rsi_p in [7, 14, 21]:
        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / rsi_p, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / rsi_p, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        df[f"RSI{rsi_p}"] = 100.0 - (100.0 / (1.0 + rs))

    return df


# ── Per-symbol simulator ────────────────────────────────────────────────

def _simulate_donchian(
    df_raw: pd.DataFrame,
    params: dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any] | None:
    """Simulate one symbol with the Donchian+Volume+SMA strategy."""

    # Compute all indicators
    df = _compute_light(df_raw)
    lookback = int(params.get("donchian_period", 20))
    df = _add_donchian(df, period=lookback)

    dates = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    mask = (dates >= start) & (dates <= end)
    idxs = np.flatnonzero(mask.to_numpy())
    if len(idxs) < 30:
        return None

    warmup = max(220, lookback + 20)
    start_idx = max(warmup, int(idxs[0]))
    end_idx = int(idxs[-1])
    if end_idx <= start_idx + 20:
        return None

    # Parameters
    vol_mult = float(params.get("vol_mult", 1.0))
    sma_period = int(params.get("sma_period", 50))
    atr_trail = float(params.get("atr_trail", 2.0))
    max_hold = int(params.get("max_hold", 0))  # 0 = no time stop
    rsi_period = int(params.get("rsi_period", 14))
    rsi_max = float(params.get("rsi_max", 999))  # 999 = no RSI filter
    dc_upper_col = f"DC_upper_{lookback}"
    sma_col = f"SMA{sma_period}" if sma_period in [20, 50] else f"EMA{sma_period}"

    # Ensure SMA/EMA column exists
    if sma_col not in df.columns:
        if sma_period not in [20, 50]:
            df[sma_col] = df["Close"].ewm(span=sma_period, adjust=False).mean()
        else:
            df[sma_col] = df["Close"].rolling(sma_period).mean()

    # Pre-fetch arrays for speed
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    dc_upper = df[dc_upper_col].to_numpy(dtype=float)
    vol_ratio = df["Vol_Ratio"].to_numpy(dtype=float)
    trend = df[sma_col].to_numpy(dtype=float)
    atr = df["ATR"].to_numpy(dtype=float)
    rsi_col = f"RSI{rsi_period}"
    if rsi_max < 999 and rsi_col in df.columns:
        rsi_arr = df[rsi_col].to_numpy(dtype=float)
    else:
        rsi_arr = None  # type: ignore[assignment]

    capital = 100000.0
    qty = 0
    entry_price = 0.0
    entry_idx: int | None = None
    highest_since_entry = 0.0  # for trailing stop
    trades = 0
    wins = 0
    total_pnl = 0.0
    equity_curve: list[float] = []
    peak_equity = capital
    max_dd = 0.0
    holding_bars: list[int] = []

    for i in range(start_idx, end_idx + 1):
        price = close[i]
        if np.isnan(price) or price <= 0:
            if qty > 0:
                capital += qty * max(close[max(0, i - 1)], 1.0)
                qty = 0
            equity_curve.append(capital)
            continue

        if qty == 0:
            # Entry: close breaks above Donchian upper band
            upper = dc_upper[i]
            if np.isnan(upper) or upper <= 0:
                equity_curve.append(capital)
                continue

            breakout = close[i] > upper

            # Volume confirmation
            vr = vol_ratio[i]
            vol_ok = not np.isnan(vr) and vr >= vol_mult if vol_mult > 0 else True

            # SMA trend filter
            sma_val = trend[i]
            trend_ok = not np.isnan(sma_val) and close[i] > sma_val

            # RSI filter (entry only if RSI < rsi_max; e.g., avoid overbought breakouts)
            rsi_ok = True
            if rsi_arr is not None:
                rsi_val = rsi_arr[i]
                if not np.isnan(rsi_val):
                    rsi_ok = rsi_val < rsi_max

            if breakout and vol_ok and trend_ok and rsi_ok:
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

            # Exit conditions
            exit_signal = False

            # 1. Trailing stop: price < highest_since_entry - atr_trail * ATR
            trail_stop = highest_since_entry - atr_trail * atr[i]
            if not np.isnan(trail_stop) and price < trail_stop and bars_held > 3:
                exit_signal = True

            # 2. Time stop
            if max_hold > 0 and bars_held >= max_hold:
                exit_signal = True

            if exit_signal:
                capital += qty * price
                pnl = (price - entry_price) * qty
                total_pnl += pnl
                if price > entry_price:
                    wins += 1
                holding_bars.append(bars_held)
                qty = 0
                entry_price = 0.0
                entry_idx = None
                highest_since_entry = 0.0

        current_equity = capital + qty * price
        equity_curve.append(current_equity)
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd = (peak_equity - current_equity) / peak_equity * 100 if peak_equity > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Close any open position
    if qty > 0:
        price = float(close[end_idx])
        capital += qty * price
        pnl = (price - entry_price) * qty
        total_pnl += pnl
        if price > entry_price:
            wins += 1
        if entry_idx is not None:
            holding_bars.append(end_idx - entry_idx)

    final_equity = capital
    total_return = (final_equity - 100000.0) / 100000.0 * 100

    days = (df.iloc[end_idx]["Date"] - df.iloc[start_idx]["Date"]).days
    years = max(days / 365.25, 0.1)
    cagr = ((final_equity / 100000.0) ** (1.0 / years) - 1.0) * 100.0 if final_equity > 0 else -100.0

    eq_a = np.array(equity_curve)
    sharpe = 0.0
    if len(eq_a) > 20 and eq_a[:-1].mean() > 0:
        daily_ret = np.diff(eq_a) / (eq_a[:-1] + 1e-9)
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-9) * np.sqrt(252)

    return {
        "symbol": df_raw.name if hasattr(df_raw, "name") else "",
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(-max_dd, 2),
        "trades": trades,
        "wins": wins,
        "win_rate_pct": round(wins / max(trades, 1) * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "avg_holding_bars": round(float(np.mean(holding_bars)), 1) if holding_bars else 0,
        "final_equity": round(final_equity, 2),
        "skip": False,
    }


# ── Aggregate cross-symbol metrics ───────────────────────────────────────

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
    years = 5.0  # approximate for full Hist_Data range
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


# ── Main sweep ───────────────────────────────────────────────────────────

def main() -> int:
    print("Loading data...")
    data_map = ex._load_data()
    if not data_map:
        print("ERROR: No data loaded from Hist_Data")
        return 1
    print(f"Loaded {len(data_map)} symbols")

    # Precompute base indicators once (lightweight — no SuperTrend)
    print("Precomputing indicators (light)...")
    precomputed: dict[str, pd.DataFrame] = {}
    for sym, df in data_map.items():
        precomputed[sym] = _compute_light(df)

    # Determine date range (handle mixed tz-aware/tz-naive datetimes)
    starts, ends = [], []
    for df in data_map.values():
        d = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        starts.append(d.min())
        ends.append(d.max())
    date_start = min(starts)
    date_end = max(ends)
    print(f"Date range: {date_start.date()} → {date_end.date()}")

    # Parameter grid — lean focused sweep
    grid = []
    for dc_period in [10, 20]:
        for vol_mult in [1.0]:
            for sma_p in [50, 100, 200]:
                for atr_trail in [2.0, 2.5]:
                    grid.append({
                        "donchian_period": dc_period,
                        "vol_mult": vol_mult,
                        "sma_period": sma_p,
                        "atr_trail": atr_trail,
                        "max_hold": 0,
                        "rsi_period": 14,
                        "rsi_max": 999,  # no RSI filter
                    })

    # Also test volume thresholds and time stops on best base
    for vol_mult in [0.8, 1.2]:
        grid.append({"donchian_period": 20, "vol_mult": vol_mult, "sma_period": 100, "atr_trail": 2.0, "max_hold": 0, "rsi_period": 14, "rsi_max": 999})
    for max_hold in [40, 60]:
        grid.append({"donchian_period": 20, "vol_mult": 1.0, "sma_period": 100, "atr_trail": 2.0, "max_hold": max_hold, "rsi_period": 14, "rsi_max": 999})

    # RSI sweep: test RSI thresholds on the best base params (dc10, sma50, atr2.5)
    base_best = {"donchian_period": 10, "vol_mult": 1.0, "sma_period": 50, "atr_trail": 2.5, "max_hold": 0}
    for rsi_p in [7, 14, 21]:
        for rsi_max in [30, 40, 50, 60, 70]:
            p = dict(base_best)
            p["rsi_period"] = rsi_p
            p["rsi_max"] = rsi_max
            grid.append(p)

    # Also test RSI on second-best base (dc10, sma100, atr2.5)
    base_2nd = {"donchian_period": 10, "vol_mult": 1.0, "sma_period": 100, "atr_trail": 2.5, "max_hold": 0}
    for rsi_p in [14]:
        for rsi_max in [30, 40, 50, 60, 70]:
            p = dict(base_2nd)
            p["rsi_period"] = rsi_p
            p["rsi_max"] = rsi_max
            grid.append(p)
    all_combos = grid
    print(f"Testing {len(all_combos)} parameter combinations...")

    results: list[dict[str, Any]] = []
    for idx, params in enumerate(all_combos):
        label = f"dc{params['donchian_period']}_vol{params['vol_mult']}_sma{params['sma_period']}_atr{params['atr_trail']}_h{params['max_hold']}_rsi{params.get('rsi_period',14)}x{params.get('rsi_max',999)}"
        if (idx + 1) % 50 == 0:
            print(f"  [{idx + 1}/{len(all_combos)}] {label}...")

        per_sym: dict[str, dict[str, Any]] = {}
        for sym, df_raw in data_map.items():
            r = _simulate_donchian(df_raw, params, date_start, date_end)
            if r is not None and not r.get("skip"):
                per_sym[sym] = r

        agg = _agg(per_sym)
        results.append({
            "label": label,
            "params": params,
            "agg": agg,
        })

    # Sort by CAGR
    results.sort(key=lambda x: x["agg"]["cagr_pct"], reverse=True)

    # Print top 20
    print(f"\n{'─' * 100}")
    print(f"{'Rank':<5} {'Label':<40} {'CAGR%':>8} {'MaxDD%':>8} {'Trades':>7} {'Win%':>7} {'Sharpe':>7} {'Active':>6}")
    print(f"{'─' * 100}")
    for i, r in enumerate(results[:36], 1):
        a = r["agg"]
        print(f"{i:<5} {r['label']:<40} {a['cagr_pct']:>8.1f} {a['max_drawdown_pct']:>8.1f} {a['trades']:>7} {a['win_rate_pct']:>7.1f} {a['sharpe_ratio']:>7.2f} {a['active_symbols']:>6}")

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
