#!/usr/bin/env python3
"""Quick RSI + Momentum blend WF validator.

Usage:
  python scripts/rsi_momentum_wf_quick.py                           # default: sweep top 6/8/10, mom21>0
  python scripts/rsi_momentum_wf_quick.py --top-n 8 --mom 63 --min 0  # single variant
  python scripts/rsi_momentum_wf_quick.py --variants top8_mom63,t10_m21  # named variants
"""
import sys, math, json, argparse
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


def run_wf(prices, returns, score, mom, top_n, mom_min, cost_bps=10.0):
    all_dates = prices.index
    start = all_dates[0]
    end = all_dates[-1]
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
            mom_at_d = mom.loc[d].copy()
            combined = rsi_at_d.where(mom_at_d > mom_min, 0)
            sc = combined.dropna().sort_values(ascending=False)
            picks = [s for s in sc.index if s in prices.columns and pd.notna(prices.loc[d, s]) and sc[s] > 0][:top_n]
            if picks:
                target.loc[picks] = 1.0 / len(picks)
            turnover.loc[td] = abs(target - prev).sum()
            prev = target
            mask = (test_dates >= td) & (test_dates <= ed)
            weights.loc[mask, :] = target.values

        gross = (weights * returns.loc[test_dates]).sum(axis=1).fillna(0)
        net = gross - turnover * (cost_bps / 10000.0)
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
    return folds


def parse_variants(args_str):
    """Parse comma-separated variant specs like 'top8_mom63,t10_m21_m5'.
    Format: t<N>_m<period>[_m<min_pct>]"""
    out = []
    for s in args_str.split(","):
        parts = s.strip().split("_")
        top_n = int(parts[0].lstrip("t"))
        mom_period = 21
        mom_min = 0.0
        for p in parts[1:]:
            if p.startswith("m") and not p.startswith("mom"):
                try:
                    mom_period = int(p[1:])
                except ValueError:
                    pass
            elif p.startswith("mom"):
                mom_period = int(p[3:])
        # Check for min threshold (e.g., m5 = 5% = 0.05)
        for p in parts:
            if p.startswith("min"):
                mom_min = float(p[3:]) / 100.0
        out.append((f"top{top_n}_mom{mom_period}_min{mom_min:g}", top_n, mom_period, mom_min))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--mom", type=int, default=21, help="Momentum period in days")
    parser.add_argument("--min", type=float, default=0.0, help="Momentum minimum threshold (0.05 = 5%%)")
    parser.add_argument("--variants", type=str, help="Comma-separated variants, e.g. 't8_m63,t10_m21_min5'")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    hist_dir = find_hist_dir("")
    prices_raw, _ = load_prices(hist_dir=hist_dir, min_rows=700, min_end_date="2026-04-17", symbols=set(), max_symbols=0)
    prices = prices_raw.ffill(limit=3)
    returns = prices.pct_change(fill_method=None).fillna(0)
    score = (rsi_dataframe(prices, 22) + rsi_dataframe(prices, 44) + rsi_dataframe(prices, 66)) / 3.0

    if args.variants:
        variants = parse_variants(args.variants)
    elif args.top_n:
        mom_period = args.mom
        mom = prices.pct_change(mom_period, fill_method=None)
        variants = [(f"top{args.top_n}_mom{mom_period}_min{args.min:g}", args.top_n, mom_period, args.min)]
    else:
        # Default sweep
        mom_1m = prices.pct_change(21, fill_method=None)
        variants = [
            (f"top6_mom21_min0", 6, 21, 0.0),
            (f"top8_mom21_min0", 8, 21, 0.0),
            (f"top10_mom21_min0", 10, 21, 0.0),
        ]
        # Run the default sweep inline (uses mom_1m)
        for name, top_n, _, mom_min in variants:
            folds = run_wf(prices, returns, score, mom_1m, top_n, mom_min, args.cost_bps)
            _print_results(name, top_n, 21, mom_min, folds)
        return

    for name, top_n, mom_period, mom_min in variants:
        mom = prices.pct_change(mom_period, fill_method=None)
        folds = run_wf(prices, returns, score, mom, top_n, mom_min, args.cost_bps)
        _print_results(name, top_n, mom_period, mom_min, folds)


def _print_results(name, top_n, mom_period, mom_min, folds):
    print(f"\n=== {name} (top_n={top_n}, mom={mom_period}d, min>{mom_min*100:.0f}%) ===")
    if not folds:
        print("  No valid folds")
        return
    pos = sum(1 for f in folds if f.positive)
    print(f"  Folds: {pos}/{len(folds)} positive")
    for f in folds:
        s = "✅" if f.positive else "❌"
        print(f"    {s} {f.test_start}→{f.test_end} CAGR={f.cagr:+.1f}% Return={f.return_pct:+.1f}% DD={f.dd:+.1f}% Sharpe={f.sharpe:.3f}")
    print(f"  MEAN: CAGR={np.mean([ff.cagr for ff in folds]):+.1f}% DD={np.mean([ff.dd for ff in folds]):+.1f}%")


if __name__ == "__main__":
    main()
