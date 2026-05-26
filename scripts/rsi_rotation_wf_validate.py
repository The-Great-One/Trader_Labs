#!/usr/bin/env python3
"""Walk-forward validation for RSI 22/44/66 rotation strategy.

This script extends the RSI rotation lab with proper expanding-window
walk-forward validation: train on previous years, test on the next
6-month period, then roll forward. It validates the 33.95% CAGR candidate
(rsi224466_W-FRI_top10_none) for out-of-sample robustness.

Key differences from the RSI lab:
- Uses expanding-window WFs, not just in-sample ranking
- Reports fold-by-fold OOS metrics
- Verdict based on WF stability, not just headline CAGR
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)

# --- Import regime mask from the RSI lab ---
from scripts.rsi_224466_rotation_lab import (
    build_regime_mask,
    load_prices,
    find_hist_dir as _find_hist_dir,
    rsi_dataframe as lab_rsi,
    rebalance_dates as lab_rebalance_dates,
)

def find_hist_dir(override: str = "") -> Path:
    return _find_hist_dir(override)


def load_prices(hist_dir: Path, min_rows: int = 700, min_end_date: str = "2026-04-17") -> tuple[pd.DataFrame, dict]:
    """Load OHLCV feather files, return price DataFrame and context dict."""
    loaded = {}
    skipped = {"derivative": 0, "not_requested": 0, "too_short": 0, "stale": 0, "read_error": 0}
    min_end = pd.Timestamp(min_end_date)

    for fpath in sorted(hist_dir.glob("*.feather")):
        symbol = fpath.stem
        try:
            df = pd.read_feather(fpath)
        except Exception:
            skipped["read_error"] += 1
            continue

        # Skip derivatives/options/futures
        if any(kw in str(fpath.name) for kw in ["FUT", "OPT", "NIFTY", "BANKNIFTY", "FINNIFTY"]):
            skipped["derivative"] += 1
            continue

        # Parse date column
        date_col = None
        for col in ["date", "Date", "datetime", "Datetime", "timestamp"]:
            if col in df.columns:
                date_col = col
                break
        if date_col is None:
            skipped["read_error"] += 1
            continue

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # Find close column
        close_col = None
        for col in ["close", "Close", "CLOSE"]:
            if col in df.columns:
                close_col = col
                break
        if close_col is None:
            skipped["read_error"] += 1
            continue

        close = df[close_col].dropna()
        if len(close) < min_rows:
            skipped["too_short"] += 1
            continue
        if close.index[-1] < min_end:
            skipped["stale"] += 1
            continue

        loaded[symbol] = close

    prices = pd.DataFrame(loaded).ffill(limit=3)
    data_context = {
        "hist_dir": str(hist_dir),
        "symbols_loaded": len(loaded),
        "skipped": skipped,
        "date_range": [str(prices.index[0].date()), str(prices.index[-1].date())],
        "min_rows": min_rows,
        "min_end_date": min_end_date,
        "loaded_symbols_sample": list(loaded.keys())[:20],
    }
    return prices, data_context


def rsi_dataframe(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """Vectorized RSI computation."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Generate rebalance dates aligned to W-FRI or ME."""
    if freq == "ME":
        return index[index.to_series().dt.is_month_end]
    if freq == "W-FRI":
        fridays = index[index.dayofweek == 4]
        # Take last Friday of each week group
        return fridays[fridays.to_series().diff().dt.days > 3]
    raise ValueError(f"Unknown rebalance freq: {freq}")


# --- Walk-forward engine ---

@dataclass
class WFFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_cagr_pct: float
    test_cagr_pct: float
    test_return_pct: float
    test_max_drawdown_pct: float
    test_sharpe: float
    test_trades: int
    test_positive: bool = False


@dataclass
class WFResult:
    candidate_name: str
    rebalance: str
    top_n: int
    cost_bps: float
    folds: list[WFFold] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        if not self.folds:
            return
        test_returns = [float(f.test_cagr_pct) for f in self.folds]
        test_return_values = [float(f.test_return_pct) for f in self.folds]
        test_drawdowns = [float(f.test_max_drawdown_pct) for f in self.folds]
        test_positives = sum(1 for f in self.folds if f.test_positive)
        self.summary = {
            "fold_count": len(self.folds),
            "positive_folds": test_positives,
            "positive_fold_pct": round(test_positives / len(self.folds) * 100, 1),
            "mean_test_cagr_pct": round(float(np.mean(test_returns)), 2),
            "median_test_cagr_pct": round(float(np.median(test_returns)), 2),
            "std_test_cagr_pct": round(float(np.std(test_returns)), 2),
            "best_test_cagr_pct": round(float(np.max(test_returns)), 2),
            "worst_test_cagr_pct": round(float(np.min(test_returns)), 2),
            "mean_test_return_pct": round(float(np.mean(test_return_values)), 2),
            "mean_test_max_dd_pct": round(float(np.mean(test_drawdowns)), 2),
        }


def run_wf_rotation(
    prices_raw: pd.DataFrame,
    rebalance: str,
    top_n: int,
    cost_bps: float,
    ffold_limit: int,
    regime: str = "none",
    train_years: int = 4,
    test_months: int = 6,
    step_months: int = 6,
    min_train_years: int = 2,
) -> WFResult:
    """Expanding-window walk-forward for RSI rotation."""
    prices = prices_raw.ffill(limit=ffold_limit)
    score = (rsi_dataframe(prices, 22) + rsi_dataframe(prices, 44) + rsi_dataframe(prices, 66)) / 3.0
    returns = prices.pct_change(fill_method=None).fillna(0)

    result = WFResult(
        candidate_name=f"rsi224466_{rebalance}_top{top_n}_{regime}",
        rebalance=rebalance,
        top_n=top_n,
        cost_bps=cost_bps,
    )

    all_dates = prices.index
    start = all_dates[0]
    end = all_dates[-1]
    fold = 0

    # Build full-period regime mask (same as RSI lab does)
    regime_mask = build_regime_mask(prices, regime).fillna(False)

    train_end = start + pd.DateOffset(years=min_train_years) - pd.Timedelta(days=1)
    while True:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end)

        if test_end <= test_start or test_end - test_start < pd.Timedelta(days=60):
            break

        # Filter to train window
        train_dates = all_dates[(all_dates >= start) & (all_dates <= train_end)]
        test_dates = all_dates[(all_dates >= test_start) & (all_dates <= test_end)]

        if len(test_dates) < 30:
            train_end = test_end
            continue

        fold += 1

        # --- Simulate in test window ---
        test_rb_dates = rebalance_dates(test_dates, rebalance)
        if len(test_rb_dates) == 0:
            train_end = test_end
            continue

        weights = pd.DataFrame(0.0, index=test_dates, columns=prices.columns)
        turnover = pd.Series(0.0, index=test_dates)
        previous = pd.Series(0.0, index=prices.columns)

        for i, d in enumerate(test_rb_dates):
            pos = test_dates.get_loc(d)
            if pos + 1 >= len(test_dates):
                continue
            trade_date = test_dates[pos + 1]
            end_idx = test_dates.get_loc(test_rb_dates[i + 1]) if i + 1 < len(test_rb_dates) else len(test_dates) - 1
            period_end = test_dates[end_idx]

            target = pd.Series(0.0, index=prices.columns)
            if bool(regime_mask.loc[d]):
                sc = score.loc[d].dropna().sort_values(ascending=False)
                picks = [s for s in sc.index if s in prices.columns and pd.notna(prices.loc[d, s])][:top_n]
                if picks:
                    target.loc[picks] = 1.0 / len(picks)

            turnover.loc[trade_date] = abs(target - previous).sum()
            previous = target
            mask = (test_dates >= trade_date) & (test_dates <= period_end)
            weights.loc[mask, :] = target.values

        gross = (weights * returns.loc[test_dates]).sum(axis=1).fillna(0)
        net = gross - turnover * (cost_bps / 10000.0)

        # Metrics
        eq = (1 + net).cumprod()
        if eq.iloc[-1] > 0 and len(net) > 30:
            years_test = len(net) / 252
            test_cagr = eq.iloc[-1] ** (1 / years_test) - 1 if years_test > 0 else 0
            test_dd = float((eq / eq.cummax() - 1).min())
            test_return = float(eq.iloc[-1] - 1)
            test_vol = net.std() * math.sqrt(252)
            test_sharpe = float((net.mean() * 252) / test_vol) if test_vol > 0 else 0
            test_trades = len(test_rb_dates)
            test_pos = test_cagr > 0
        else:
            test_cagr = 0.0
            test_dd = 0.0
            test_return = 0.0
            test_sharpe = 0.0
            test_trades = 0
            test_pos = False

        # Train metrics (IS)
        train_window = all_dates[(all_dates >= start) & (all_dates <= train_end)]
        train_eq = (1 + returns.loc[train_window].mean(axis=1).fillna(0)).cumprod()
        if train_eq.iloc[-1] > 0 and len(train_window) > 252:
            train_y = len(train_window) / 252
            train_cagr = train_eq.iloc[-1] ** (1 / train_y) - 1
        else:
            train_cagr = 0.0

        result.folds.append(WFFold(
            fold=fold,
            train_start=str(start.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            train_cagr_pct=round(float(train_cagr * 100), 2),
            test_cagr_pct=round(float(test_cagr * 100), 2),
            test_return_pct=round(float(test_return * 100), 2),
            test_max_drawdown_pct=round(float(test_dd * 100), 2),
            test_sharpe=round(float(test_sharpe), 3),
            test_trades=int(test_trades),
            test_positive=bool(test_pos),
        ))

        train_end = test_end

    result.finalize()
    return result


# --- Main ---

def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward validate RSI rotation candidate")
    parser.add_argument("--hist-dir", default="")
    parser.add_argument("--rebalance", default="W-FRI")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--regime", default="none")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--ffill-limit", type=int, default=3)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=6)
    parser.add_argument("--min-train-years", type=int, default=2)
    args = parser.parse_args()

    hist_dir = find_hist_dir(args.hist_dir)
    print(f"Loading data from {hist_dir}...")
    prices, data_context = load_prices(hist_dir)
    print(f"Loaded {prices.shape[1]} symbols, {prices.shape[0]} days ({prices.index[0].date()} to {prices.index[-1].date()})")

    wf = run_wf_rotation(
        prices_raw=prices,
        rebalance=args.rebalance,
        top_n=args.top_n,
        cost_bps=args.cost_bps,
        regime=args.regime,
        ffold_limit=args.ffill_limit,
        train_years=args.train_years,
        test_months=args.test_months,
        step_months=args.step_months,
        min_train_years=args.min_train_years,
    )

    print(f"\n=== WALK-FORWARD RESULTS: {wf.candidate_name} ===")
    for f in wf.folds:
        status = "✅" if f.test_positive else "❌"
        print(f"  Fold {f.fold}: {status} Test={f.test_start}→{f.test_end}  "
              f"CAGR={f.test_cagr_pct:+.2f}%  Return={f.test_return_pct:+.2f}%  "
              f"DD={f.test_max_drawdown_pct:+.2f}%  Sharpe={f.test_sharpe:.3f}")

    print(f"\n=== SUMMARY ===")
    for k, v in wf.summary.items():
        print(f"  {k}: {v}")

    verdict = "research_candidate"
    if wf.summary.get("positive_fold_pct", 0) >= 60 and wf.summary.get("mean_test_cagr_pct", 0) > 0:
        verdict = "promotable" if wf.summary.get("worst_test_cagr_pct", -100) > -10 else "needs_guardrails"
    elif wf.summary.get("mean_test_cagr_pct", 0) <= 0:
        verdict = "rejected"

    payload = {
        "generated_at": datetime.now().isoformat(),
        "validator_type": "rsi_rotation_walk_forward",
        "candidate": wf.candidate_name,
        "params": {
            "rebalance": wf.rebalance,
            "top_n": wf.top_n,
            "cost_bps": wf.cost_bps,
            "train_years": args.train_years,
            "test_months": args.test_months,
            "step_months": args.step_months,
        },
        "data_context": data_context,
        "folds": [asdict(f) for f in wf.folds],
        "summary": wf.summary,
        "verdict": verdict,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"rsi_wf_validate_{ts}.json"
    latest_path = OUT_DIR / "rsi_wf_validate_latest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nVerdict: {verdict}")
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
