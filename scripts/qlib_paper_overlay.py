#!/usr/bin/env python3
"""Research-only Qlib-style paper overlay for latest Kite-cache signals.

Trains the lightweight Qlib-style ranking bridge on historical labelled rows,
scores the latest available feature date, and writes a paper-only recommendation
report. It places no orders and does not alter Auto_Trader live rules.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qlib_alpha_lab as qlab

OUT_DIR = ROOT / "reports"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Latest Qlib-style paper overlay, no live trading")
    parser.add_argument("--hist-dir", default=os.getenv("AT_QLIB_LAB_HIST_DIR", ""))
    parser.add_argument("--symbols", default=os.getenv("AT_QLIB_LAB_SYMBOLS", ""))
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("AT_QLIB_LAB_MAX_SYMBOLS", "0") or "0"))
    parser.add_argument("--min-rows", type=int, default=int(os.getenv("AT_QLIB_LAB_MIN_ROWS", "700") or "700"))
    parser.add_argument("--min-end-date", default=os.getenv("AT_QLIB_LAB_MIN_END_DATE", "2026-04-17"))
    parser.add_argument("--model", default=os.getenv("AT_QLIB_LAB_MODEL", "sklearn_hgb"), choices=["lightgbm", "sklearn_hgb", "random_forest"])
    parser.add_argument("--horizon-days", type=int, default=int(os.getenv("AT_QLIB_LAB_HORIZON_DAYS", "21") or "21"))
    parser.add_argument("--top-n", type=int, default=int(os.getenv("AT_QLIB_PAPER_TOP_N", "10") or "10"))
    parser.add_argument("--embargo-days", type=int, default=int(os.getenv("AT_QLIB_LAB_EMBARGO_DAYS", "21") or "21"))
    parser.add_argument("--random-state", type=int, default=int(os.getenv("AT_QLIB_LAB_RANDOM_STATE", "42") or "42"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hist_dir = qlab.find_hist_dir(args.hist_dir)
    data, data_context = qlab.load_ohlcv(
        hist_dir=hist_dir,
        min_rows=args.min_rows,
        min_end_date=args.min_end_date,
        symbols=qlab.parse_symbols(args.symbols),
        max_symbols=args.max_symbols,
    )
    panel, _prices, feature_cols = qlab.build_feature_frame(data, args.horizon_days)
    labelled_dates = pd.DatetimeIndex(sorted(pd.unique(panel["date"])))
    embargo = max(args.horizon_days, args.embargo_days)
    train_cutoff_idx = max(0, len(labelled_dates) - embargo - 1)
    train_cutoff = labelled_dates[train_cutoff_idx]

    train = panel[panel["date"] <= train_cutoff].copy()

    # Latest scoring must not require a future-return label, otherwise the
    # report is silently delayed by horizon_days. Rebuild the feature panel and
    # filter only on features/current close.
    latest_panel = pd.concat(
        [qlab.add_symbol_features(symbol, df, args.horizon_days) for symbol, df in data.items()],
        ignore_index=True,
    ).sort_values(["date", "symbol"])
    base_feature_cols = [c for c in feature_cols if not c.startswith("xrank_")]
    for col in base_feature_cols:
        latest_panel[f"xrank_{col}"] = latest_panel.groupby("date")[col].rank(pct=True)
    latest_panel = latest_panel.dropna(subset=feature_cols + ["close"]).reset_index(drop=True)
    counts = latest_panel.groupby("date")["symbol"].nunique().sort_index()
    min_latest_rows = max(args.top_n * 3, 50)
    viable_dates = counts[counts >= min_latest_rows]
    if viable_dates.empty:
        latest_feature_date = counts.index[-1]
        latest = latest_panel[latest_panel["date"] == latest_feature_date].copy()
        data_quality_status = "BLOCKED_TOO_FEW_LATEST_SYMBOLS"
    else:
        latest_feature_date = viable_dates.index[-1]
        latest = latest_panel[latest_panel["date"] == latest_feature_date].copy()
        data_quality_status = "OK"

    # Extra guard against cache/path corruption: if the same close dominates
    # latest rows, the ranking is not safe to act on.
    if not latest.empty:
        top_close_share = float(latest["close"].round(2).value_counts(normalize=True).iloc[0])
        if top_close_share > 0.20:
            data_quality_status = "BLOCKED_DUPLICATE_PRICE_CLUSTER"
    else:
        top_close_share = 0.0

    if len(train) < 1000 or len(latest) < args.top_n:
        raise SystemExit(f"insufficient train/latest rows: train={len(train)} latest={len(latest)}")

    model = qlab.make_model(args.model, args.random_state)
    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train["label_fwd_return"].clip(-0.5, 0.5)
    X_latest = latest[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    model.fit(X_train, y_train)
    latest["score"] = model.predict(X_latest)
    latest = latest.sort_values("score", ascending=False)

    picks = []
    for _, row in latest.head(args.top_n).iterrows():
        picks.append({
            "symbol": str(row["symbol"]),
            "score": round(float(row["score"]), 6),
            "close": round(float(row["close"]), 2),
            "rsi_22": round(float(row.get("rsi_22", 0.0)), 2),
            "ret_21_pct": round(float(row.get("ret_21", 0.0)) * 100, 2),
            "sma_50_ratio_pct": round(float(row.get("sma_50_ratio", 0.0)) * 100, 2),
            "atr_14_pct": round(float(row.get("atr_14_pct", 0.0)) * 100, 2),
        })

    payload = {
        "generated_at": now_iso(),
        "paper_mode": True,
        "lab_type": "qlib_style_latest_overlay",
        "decision": "OBSERVE_ONLY" if data_quality_status == "OK" else "BLOCKED_DATA_QUALITY",
        "production_action": "NO_LIVE_TRADES",
        "model": args.model,
        "top_n": args.top_n,
        "horizon_days": args.horizon_days,
        "latest_feature_date": str(latest_feature_date.date()),
        "train_cutoff": str(train_cutoff.date()),
        "train_rows": int(len(train)),
        "latest_rows": int(len(latest)),
        "data_quality_status": data_quality_status,
        "latest_top_close_share": round(top_close_share, 4),
        "data_context": data_context,
        "picks": picks,
        "promotion_note": "Paper overlay only. Require repeated fresh OOS/paper evidence and live-parity RS7/RS2 comparison before any production use.",
    }
    latest_path = OUT_DIR / "qlib_paper_overlay_latest.json"
    ts_path = OUT_DIR / f"qlib_paper_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ts_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("generated_at", "decision", "model", "latest_feature_date", "train_rows", "latest_rows", "picks")}, indent=2))
    print(f"Saved: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
