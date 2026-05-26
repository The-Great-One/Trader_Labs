#!/usr/bin/env python3
"""Quick RSI + momentum blend IS headliner test"""
import sys, math
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, '/Users/sahilgoel/Desktop/Trader_Labs')
from scripts.rsi_224466_rotation_lab import (
    load_prices, rsi_dataframe, rebalance_dates, find_hist_dir
)

hist_dir = find_hist_dir("")
prices_raw, _ = load_prices(hist_dir=hist_dir, min_rows=700, min_end_date="2026-04-17", symbols=set(), max_symbols=0)
prices = prices_raw.ffill(limit=3)
returns = prices.pct_change(fill_method=None).fillna(0)

rsi_score = (rsi_dataframe(prices, 22) + rsi_dataframe(prices, 44) + rsi_dataframe(prices, 66)) / 3.0
mom_1m = prices.pct_change(21, fill_method=None)

print("=== RSI + MOMENTUM BLEND (IS only) ===\n")

for top_n in [6, 8, 10]:
    for reb in ["ME"]:
        dates = rebalance_dates(prices.index, reb)
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        turnover = pd.Series(0.0, index=prices.index)
        previous = pd.Series(0.0, index=prices.columns)
        
        for i, d in enumerate(dates):
            pos = prices.index.get_loc(d)
            if pos + 1 >= len(prices.index):
                continue
            trade_date = prices.index[pos + 1]
            end_date = dates[i + 1] if i + 1 < len(dates) else prices.index[-1]
            target = pd.Series(0.0, index=prices.columns)
            rsi_at_d = rsi_score.loc[d].copy()
            mom_at_d = mom_1m.loc[d].copy()
            combined = rsi_at_d.where(mom_at_d > 0, 0)
            sc = combined.dropna().sort_values(ascending=False)
            picks = [s for s in sc.index if s in prices.columns and pd.notna(prices.loc[d, s]) and sc[s] > 0][:top_n]
            if picks:
                target.loc[picks] = 1.0 / len(picks)
            turnover.loc[trade_date] = abs(target - previous).sum()
            previous = target
            mask = (prices.index >= trade_date) & (prices.index <= end_date)
            weights.loc[mask, :] = target.values
        
        gross = (weights * returns).sum(axis=1)
        net = gross - turnover * (10.0 / 10000.0)
        active = weights.sum(axis=1) > 0
        if not active.any():
            continue
        r = net.loc[active]
        eq = (1 + r).cumprod()
        years = len(r) / 252
        cagr = eq.iloc[-1] ** (1/years) - 1 if years > 0 else 0
        dd = float((eq / eq.cummax() - 1).min())
        elapsed = net.loc[r.index[0]:]
        elapsed_eq = (1 + elapsed).cumprod()
        elapsed_y = len(elapsed) / 252
        xirr = elapsed_eq.iloc[-1] ** (1/elapsed_y) - 1 if elapsed_y > 0 else 0
        vol = r.std() * math.sqrt(252)
        sharpe = (r.mean() * 252) / vol if vol > 0 else 0
        print(f"  rsi+momentum_ME_top{top_n:<2d}  CAGR={cagr*100:6.2f}% XIRR={xirr*100:6.2f}% DD={dd*100:6.2f}% Sharpe={sharpe:.3f}")
