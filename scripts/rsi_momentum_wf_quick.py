#!/usr/bin/env python3
"""Quick RSI + Momentum blend WF validator"""
import sys, math, json
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, '/Users/sahilgoel/Desktop/Trader_Labs')
from scripts.rsi_224466_rotation_lab import (
    load_prices, rsi_dataframe, rebalance_dates, find_hist_dir
)

@dataclass
class Fold:
    fold: int
    test_start: str
    test_end: str
    cagr: float
    return_pct: float
    dd: float
    sharpe: float
    positive: bool

hist_dir = find_hist_dir("")
prices_raw, _ = load_prices(hist_dir=hist_dir, min_rows=700, min_end_date="2026-04-17", symbols=set(), max_symbols=0)
prices = prices_raw.ffill(limit=3)
returns = prices.pct_change(fill_method=None).fillna(0)
score = (rsi_dataframe(prices, 22) + rsi_dataframe(prices, 44) + rsi_dataframe(prices, 66)) / 3.0
mom_1m = prices.pct_change(21, fill_method=None)

all_dates = prices.index
start = all_dates[0]
end = all_dates[-1]

for top_n in [6, 8, 10]:
    print(f"\n=== ME_top{top_n} BLEND (momentum>0) ===")
    folds = []
    train_end = all_dates[all_dates >= start + pd.DateOffset(years=2) - pd.Timedelta(days=1)][0]
    
    for fi in range(7):
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=6) - pd.Timedelta(days=1), end)
        if test_end <= test_start or test_end - test_start < pd.Timedelta(days=60):
            break

        test_mask = (all_dates >= test_start) & (all_dates <= test_end)
        if test_mask.sum() < 30:
            train_end = test_end
            continue

        test_dates = all_dates[test_mask]
        rb_dates = rebalance_dates(test_dates, "ME")
        if len(rb_dates) < 2:
            train_end = test_end
            continue

        weights = pd.DataFrame(0.0, index=test_dates, columns=prices.columns)
        turnover = pd.Series(0.0, index=test_dates)
        prev = pd.Series(0.0, index=prices.columns)

        for i, d in enumerate(rb_dates):
            pos_idx = test_dates.get_loc(d)
            if pos_idx + 1 >= len(test_dates):
                continue
            td = test_dates[pos_idx + 1]
            ed = test_dates[test_dates.get_loc(rb_dates[i+1])] if i+1 < len(rb_dates) else test_dates[-1]
            target = pd.Series(0.0, index=prices.columns)
            rsi_at_d = score.loc[d].copy()
            mom_at_d = mom_1m.loc[d].copy()
            combined = rsi_at_d.where(mom_at_d > 0, 0)
            sc = combined.dropna().sort_values(ascending=False)
            picks = [s for s in sc.index if s in prices.columns and pd.notna(prices.loc[d, s]) and sc[s] > 0][:top_n]
            if picks:
                target.loc[picks] = 1.0 / len(picks)
            turnover.loc[td] = abs(target - prev).sum()
            prev = target
            mask = (test_dates >= td) & (test_dates <= ed)
            weights.loc[mask, :] = target.values

        gross = (weights * returns.loc[test_dates]).sum(axis=1).fillna(0)
        net = gross - turnover * (10.0 / 10000.0)
        eq = (1 + net).cumprod()
        
        if eq.iloc[-1] > 0 and len(net) > 30:
            years = len(net) / 252
            c = eq.iloc[-1] ** (1/years) - 1
            d = float((eq / eq.cummax() - 1).min())
            ret = float(eq.iloc[-1] - 1)
            vol = net.std() * math.sqrt(252)
            sh = float((net.mean() * 252) / vol) if vol > 0 else 0
        else:
            c = d = ret = sh = 0.0

        folds.append(Fold(
            fold=fi+1,
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            cagr=c*100,
            return_pct=ret*100,
            dd=d*100,
            sharpe=sh,
            positive=c > 0,
        ))
        
        train_end = test_end

    if folds:
        pos = sum(1 for f in folds if f.positive)
        print(f"  Folds: {pos}/{len(folds)} positive")
        for f in folds:
            s = "✅" if f.positive else "❌"
            print(f"    {s} {f.test_start}→{f.test_end} CAGR={f.cagr:+.1f}% Return={f.return_pct:+.1f}% DD={f.dd:+.1f}% Sharpe={f.sharpe:.3f}")
        print(f"  MEAN: CAGR={np.mean([ff.cagr for ff in folds]):+.1f}% DD={np.mean([ff.dd for ff in folds]):+.1f}%")
