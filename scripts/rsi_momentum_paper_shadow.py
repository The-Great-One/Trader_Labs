#!/usr/bin/env python3
"""RSI + Momentum Rotation Paper Shadow.

Monthly rotation paper trader: ranks stocks by RSI(22,44,66) average,
filters to positive 1-month momentum, holds top-N equal-weight.
Publishes paper decision to paper_shadow_rsi_momentum_latest.json.
No real orders placed.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports"
HIST_DIR = ROOT / "intermediary_files" / "Hist_Data"
OUT_DIR.mkdir(exist_ok=True)

# Default params — override via env
TOP_N = int(os.getenv("RSI_MOM_TOP_N", "10"))
COST_BPS = float(os.getenv("RSI_MOM_COST_BPS", "10"))
MOMENTUM_PERIOD = int(os.getenv("RSI_MOM_MOMENTUM_PERIOD", "21"))
MIN_ROWS = int(os.getenv("RSI_MOM_MIN_ROWS", "500"))


def load_hist(hist_dir: Path) -> pd.DataFrame:
    """Load close prices from feather files."""
    if not hist_dir.is_dir():
        return pd.DataFrame()
    loaded = {}
    for fpath in sorted(hist_dir.glob("*.feather")):
        symbol = fpath.stem
        try:
            df = pd.read_feather(fpath)
        except Exception:
            continue
        # Skip derivatives
        if any(kw in symbol for kw in ["FUT", "OPT", "-I", "-II"]):
            continue

        date_col = next((c for c in ["date", "Date", "datetime"] if c in df.columns), None)
        close_col = next((c for c in ["close", "Close", "CLOSE"] if c in df.columns), None)
        if date_col is None or close_col is None:
            continue

        df[date_col] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
        s = df.set_index(date_col)[close_col].dropna().sort_index()
        if len(s) >= MIN_ROWS:
            loaded[symbol] = s

    return pd.DataFrame(loaded).sort_index()


def rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def month_end_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last trading day of each month."""
    df = pd.DataFrame({"d": index, "m": index.to_period("M")})
    return [g.iloc[-1]["d"] for _, g in df.groupby("m")]


def compute_rotation(prices: pd.DataFrame, top_n: int = TOP_N) -> dict:
    """Compute latest monthly rotation picks and publish paper decision."""
    if prices.empty or len(prices.columns) < top_n:
        return {"error": "insufficient symbols", "symbols_loaded": len(prices.columns)}

    prices_ffill = prices.ffill(limit=3)
    mom_1m = prices_ffill.pct_change(MOMENTUM_PERIOD)

    # RSI composite score
    rsi22 = prices_ffill.apply(lambda c: rsi(c, 22))
    rsi44 = prices_ffill.apply(lambda c: rsi(c, 44))
    rsi66 = prices_ffill.apply(lambda c: rsi(c, 66))
    score = (rsi22 + rsi44 + rsi66) / 3.0

    # Monthly rebalance dates
    dates = month_end_dates(prices_ffill.index)
    if len(dates) < 1:
        return {"error": "no rebalance dates"}

    # Latest signal
    latest_date = dates[-1]
    rsi_scores = score.loc[latest_date].dropna()
    mom_scores = mom_1m.loc[latest_date]

    # Filter: positive 1-month momentum, then rank by RSI score
    valid = rsi_scores[mom_scores > 0]
    ranked = valid.sort_values(ascending=False)

    picks = list(ranked.head(top_n).index)
    pick_scores = {s: round(float(ranked[s]), 2) for s in picks}

    # Historical backtest simulation for CAGR estimate
    weights = pd.DataFrame(0.0, index=prices_ffill.index, columns=prices_ffill.columns)
    turnover = pd.Series(0.0, index=prices_ffill.index)
    previous = pd.Series(0.0, index=prices_ffill.columns)
    returns = prices_ffill.pct_change(fill_method=None).fillna(0)

    for i, d in enumerate(dates):
        pos = prices_ffill.index.get_loc(d)
        if pos + 1 >= len(prices_ffill.index):
            continue
        trade_date = prices_ffill.index[pos + 1]
        end_date = dates[i + 1] if i + 1 < len(dates) else prices_ffill.index[-1]
        target = pd.Series(0.0, index=prices_ffill.columns)
        rsi_at = score.loc[d].dropna()
        mom_at = mom_1m.loc[d]
        combined = rsi_at[mom_at > 0]
        sc = combined.sort_values(ascending=False)
        sel = [s for s in sc.index if s in prices_ffill.columns and pd.notna(prices_ffill.loc[d, s]) and sc[s] > 0][:top_n]
        if sel:
            target.loc[sel] = 1.0 / len(sel)
        turnover.loc[trade_date] = abs(target - previous).sum()
        previous = target
        mask = (prices_ffill.index >= trade_date) & (prices_ffill.index <= end_date)
        weights.loc[mask, :] = target.values

    gross = (weights * returns).sum(axis=1)
    net = gross - turnover * (COST_BPS / 10000.0)
    active = weights.sum(axis=1) > 0
    if not active.any():
        return {"error": "no active periods in backtest"}

    r = net.loc[active]
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    xirr = cagr  # single initial allocation, no cashflows
    dd = float((eq / eq.cummax() - 1).min())
    vol = r.std() * (252 ** 0.5)
    sharpe = (r.mean() * 252) / vol if vol > 0 else 0.0
    yearly = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)

    # Last 12 months performance
    last_12m = net.last("365D") if len(net) > 252 else net
    eq_12m = (1 + last_12m).cumprod()
    ret_12m = eq_12m.iloc[-1] - 1 if len(eq_12m) > 0 else 0.0

    return {
        "generated_at": datetime.now().isoformat(),
        "strategy": "rsi_momentum_rotation",
        "params": {
            "top_n": top_n,
            "momentum_period": MOMENTUM_PERIOD,
            "cost_bps": COST_BPS,
        },
        "latest_signal": {
            "date": str(latest_date.date()),
            "picks": picks,
            "scores": pick_scores,
            "symbols_screened": len(valid),
        },
        "backtest_metrics": {
            "symbols_loaded": len(prices_ffill.columns),
            "date_range": [str(r.index[0].date()), str(r.index[-1].date())],
            "days": int(len(r)),
            "years": round(years, 2),
            "cagr_pct": round(float(cagr * 100), 2),
            "xirr_pct": round(float(xirr * 100), 2),
            "total_return_pct": round(float((eq.iloc[-1] - 1) * 100), 1),
            "max_drawdown_pct": round(float(dd * 100), 2),
            "vol_pct": round(float(vol * 100), 1),
            "sharpe": round(float(sharpe), 3),
            "positive_years": int((yearly > 0).sum()),
            "total_years": int(len(yearly)),
            "return_12m_pct": round(float(ret_12m * 100), 1),
        },
    }


def main():
    import os

    hist_dir = Path(os.getenv("RSI_MOM_HIST_DIR", str(HIST_DIR)))
    if not hist_dir.is_dir():
        print(f"ERROR: Hist_Data dir not found at {hist_dir}")
        return 1

    print(f"Loading {hist_dir}...")
    prices = load_hist(hist_dir)
    print(f"Loaded {len(prices.columns)} symbols, {len(prices)} days")

    result = compute_rotation(prices, top_n=TOP_N)
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return 1

    # Write output
    output_path = OUT_DIR / "paper_shadow_rsi_momentum_latest.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print summary
    picks = result["latest_signal"]["picks"]
    scores = result["latest_signal"]["scores"]
    bm = result["backtest_metrics"]

    print(f"\n=== RSI + Momentum Rotation Paper Shadow ===")
    print(f"Signal date: {result['latest_signal']['date']}")
    print(f"Top {TOP_N} picks:")
    for s in picks:
        print(f"  {s:<15s} RSI score: {scores.get(s, 'N/A')}")
    print(f"\nBacktest: {bm['cagr_pct']:.2f}% CAGR, {bm['max_drawdown_pct']:.1f}% MaxDD, "
          f"Sharpe {bm['sharpe']:.3f}, {bm['positive_years']}/{bm['total_years']} pos years")
    print(f"12-month return: {bm['return_12m_pct']:+.1f}%")
    print(f"\nSaved: {output_path}")
    return 0


if __name__ == "__main__":
    import os
    raise SystemExit(main())
