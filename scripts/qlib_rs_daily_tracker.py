#!/usr/bin/env python3
"""Daily observe-only tracker comparing Qlib overlay picks with RS7/RS2.

For each Qlib paper-overlay pick, evaluate current RS7 entry diagnostics and
RS2 sell diagnostics using Kite cache data only. Persist daily snapshots plus a
history file so later reports can measure whether Qlib picks outperform RS7/RS2
alignment before any sizing/live use.
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

os.environ.setdefault("AT_RESEARCH_MODE", "1")
os.environ.setdefault("AT_DISABLE_FILE_LOGGING", "1")

ROOT = Path(__file__).resolve().parents[1]
# RULE_SET_2 mutates its Holdings.json state on SELL/stop updates. Force an
# isolated lab state dir before importing Auto_Trader modules so this tracker
# cannot touch live/runtime intermediary_files.
os.environ.setdefault("AT_STATE_DIR", str(ROOT / "reports" / "qlib_rs_tracker_state"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import qlib_paper_overlay

AUTOTRADER_ROOT = Path(os.getenv("AUTOTRADER_ROOT", str(ROOT.parent / "Stocks"))).expanduser()
# Put the live Auto_Trader repo before Trader_Labs so `Auto_Trader.*` resolves
# to the full runtime package, not Trader_Labs' lightweight research stubs.
if str(AUTOTRADER_ROOT) in sys.path:
    sys.path.remove(str(AUTOTRADER_ROOT))
sys.path.insert(0, str(AUTOTRADER_ROOT))

from Auto_Trader import RULE_SET_2, RULE_SET_7
from Auto_Trader import utils as at_utils

OUT_DIR = ROOT / "reports"
HISTORY_PATH = OUT_DIR / "qlib_rs_daily_tracker_history.jsonl"
LATEST_PATH = OUT_DIR / "qlib_rs_daily_tracker_latest.json"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def normalise_hist_df(raw: pd.DataFrame) -> pd.DataFrame:
    cmap = {str(c).lower(): c for c in raw.columns}
    required = ["date", "open", "high", "low", "close"]
    if not all(c in cmap for c in required):
        raise ValueError("missing OHLC columns")
    out = pd.DataFrame({
        "Date": pd.to_datetime(raw[cmap["date"]], errors="coerce"),
        "Open": pd.to_numeric(raw[cmap["open"]], errors="coerce"),
        "High": pd.to_numeric(raw[cmap["high"]], errors="coerce"),
        "Low": pd.to_numeric(raw[cmap["low"]], errors="coerce"),
        "Close": pd.to_numeric(raw[cmap["close"]], errors="coerce"),
        "Volume": pd.to_numeric(raw[cmap.get("volume", cmap["close"])], errors="coerce").fillna(0),
    }).dropna(subset=["Date", "Open", "High", "Low", "Close"])
    return out.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def load_symbol_frame(symbol: str, hist_dir: Path) -> pd.DataFrame:
    fp = hist_dir / f"{symbol}.feather"
    if not fp.exists():
        raise FileNotFoundError(str(fp))
    return at_utils.Indicators(normalise_hist_df(pd.read_feather(fp))).ffill().dropna(subset=["Close"]).reset_index(drop=True)


def latest_future_returns(df: pd.DataFrame, horizons: list[int]) -> dict:
    close = float(df.iloc[-1]["Close"])
    out = {"close": close, "latest_date": str(pd.to_datetime(df.iloc[-1]["Date"]).date())}
    for h in horizons:
        if len(df) > h:
            past = float(df.iloc[-1 - h]["Close"])
            out[f"ret_{h}d_pct"] = round(((close / past) - 1.0) * 100.0, 2) if past > 0 else None
        else:
            out[f"ret_{h}d_pct"] = None
    return out


def evaluate_symbol(symbol: str, pick: dict, hist_dir: Path) -> dict:
    base = {"symbol": symbol, "qlib_pick": pick, "status": "ok"}
    try:
        df = load_symbol_frame(symbol, hist_dir)
        row = df.iloc[-1].to_dict()
        row.setdefault("instrument_token", abs(hash(symbol)) % 2_000_000_000)
        empty_holdings = pd.DataFrame(columns=["instrument_token", "tradingsymbol", "average_price", "quantity", "t1_quantity", "bars_in_trade"])
        rs7_decision, rs7_details = RULE_SET_7.evaluate_signal(df, row, empty_holdings)
        held = pd.DataFrame([{
            "instrument_token": int(row["instrument_token"]),
            "tradingsymbol": symbol,
            "average_price": float(row.get("Close", 0.0) or 0.0),
            "quantity": 1,
            "t1_quantity": 0,
            "bars_in_trade": 1,
        }])
        try:
            rs2_decision = RULE_SET_2.buy_or_sell(df, row, held)
        except Exception as exc:
            rs2_decision = "ERROR"
            base["rs2_error"] = str(exc)[:200]
        readiness = float(rs7_details.get("readiness_score_pct", 0.0) or 0.0)
        gap = rs7_details.get("score_gap_to_buy")
        hard_blocks = rs7_details.get("hard_blocks") or []
        base.update({
            "latest": latest_future_returns(df, [1, 5, 10, 21]),
            "rs7_entry_decision": str(rs7_decision).upper(),
            "rs7_readiness_score_pct": round(readiness, 2),
            "rs7_score_gap_to_buy": gap,
            "rs7_hard_blocks": hard_blocks[:12],
            "rs7_hard_block_count": int(rs7_details.get("hard_block_count", len(hard_blocks)) or 0),
            "rs7_nearest_mode": rs7_details.get("nearest_mode"),
            "rs7_nearest_mode_missing": (rs7_details.get("nearest_mode_missing") or [])[:12],
            "rs2_exit_decision_if_held": str(rs2_decision).upper(),
            "alignment": "AGREE_BUY" if str(rs7_decision).upper() == "BUY" else "QLIB_ONLY",
        })
    except Exception as exc:
        base.update({"status": "error", "error": str(exc)[:300]})
    return base


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    agree = [r for r in ok if r.get("alignment") == "AGREE_BUY"]
    exit_flags = [r for r in ok if r.get("rs2_exit_decision_if_held") == "SELL"]
    return {
        "rows": len(rows),
        "ok_rows": len(ok),
        "rs7_agree_buy_count": len(agree),
        "rs7_agree_buy_symbols": [r["symbol"] for r in agree],
        "rs2_exit_if_held_count": len(exit_flags),
        "rs2_exit_if_held_symbols": [r["symbol"] for r in exit_flags],
        "avg_rs7_readiness_score_pct": round(float(np.mean([r.get("rs7_readiness_score_pct", 0.0) for r in ok])), 2) if ok else 0.0,
        "avg_1d_ret_pct": round(float(np.nanmean([r.get("latest", {}).get("ret_1d_pct", np.nan) for r in ok])), 2) if ok else None,
        "avg_5d_ret_pct": round(float(np.nanmean([r.get("latest", {}).get("ret_5d_pct", np.nan) for r in ok])), 2) if ok else None,
        "avg_21d_ret_pct": round(float(np.nanmean([r.get("latest", {}).get("ret_21d_pct", np.nan) for r in ok])), 2) if ok else None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Track Qlib overlay picks against RS7/RS2, observe-only")
    p.add_argument("--skip-overlay-refresh", action="store_true", help="Use existing qlib_paper_overlay_latest.json")
    p.add_argument("--top-n", type=int, default=int(os.getenv("AT_QLIB_TRACKER_TOP_N", "10") or "10"))
    p.add_argument("--min-rows", type=int, default=int(os.getenv("AT_QLIB_LAB_MIN_ROWS", "700") or "700"))
    p.add_argument("--min-end-date", default=os.getenv("AT_QLIB_LAB_MIN_END_DATE", "2026-04-17"))
    p.add_argument("--model", default=os.getenv("AT_QLIB_LAB_MODEL", "sklearn_hgb"))
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_overlay_refresh:
        old_argv = sys.argv[:]
        try:
            sys.argv = ["qlib_paper_overlay.py", "--top-n", str(args.top_n), "--min-rows", str(args.min_rows), "--min-end-date", args.min_end_date, "--model", args.model]
            rc = qlib_paper_overlay.main()
            if rc:
                raise SystemExit(rc)
        finally:
            sys.argv = old_argv
    overlay_path = OUT_DIR / "qlib_paper_overlay_latest.json"
    overlay = json.loads(overlay_path.read_text())
    hist_dir = Path(overlay.get("data_context", {}).get("hist_dir") or (ROOT / "intermediary_files" / "Hist_Data"))
    rows = [evaluate_symbol(str(p["symbol"]).upper(), p, hist_dir) for p in overlay.get("picks", [])[: args.top_n]]
    payload = {
        "generated_at": now_iso(),
        "paper_mode": True,
        "decision": "OBSERVE_ONLY",
        "production_action": "NO_LIVE_TRADES",
        "tracker_type": "qlib_vs_rs7_rs2_daily",
        "overlay_generated_at": overlay.get("generated_at"),
        "overlay_feature_date": overlay.get("latest_feature_date"),
        "overlay_data_quality_status": overlay.get("data_quality_status"),
        "summary": summarize(rows),
        "rows": rows,
        "promotion_note": "Track daily. Do not use for sizing/live unless repeated tracker history shows Qlib picks add value and RS7/RS2 alignment controls drawdown.",
    }
    LATEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    print(json.dumps({"generated_at": payload["generated_at"], "summary": payload["summary"]}, indent=2))
    print(f"Saved: {LATEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
