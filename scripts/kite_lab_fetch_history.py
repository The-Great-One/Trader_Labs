#!/usr/bin/env python3
"""Fetch Kite historical OHLCV into Trader_Labs cache without live service state.

This script is serial/rate-limit-aware by design. It uses the lab-only token
created by scripts/kite_lab_token.py and writes to Trader_Labs/intermediary_files
(or another explicit --hist-dir), not the live trading service session.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from dateutil.relativedelta import relativedelta
from kiteconnect import KiteConnect
from kiteconnect.exceptions import NetworkException, TokenException

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from kite_lab_token import DEFAULT_TOKEN_PATH, load_credentials, read_token  # noqa: E402

DEFAULT_HIST_DIR = ROOT / "intermediary_files" / "Hist_Data"
DEFAULT_INSTRUMENTS_CACHE = ROOT / "intermediary_files" / "kite_lab_instruments_cache.json"
DEFAULT_FETCH_MARKS = ROOT / "intermediary_files" / "kite_lab_fetched_history.json"

INTERVAL_LIMITS = {
    "day": 2000,
    "60minute": 400,
    "30minute": 200,
    "15minute": 200,
    "10minute": 120,
    "5minute": 100,
    "3minute": 90,
    "minute": 60,
}

NSE_INDEX_URLS = {
    "NIFTY50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY100": "https://archives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY200": "https://archives.nseindia.com/content/indices/ind_nifty200list.csv",
    "NIFTY500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "NIFTY MIDCAP 150": "https://archives.nseindia.com/content/indices/ind_niftymidcap150list.csv",
    "NIFTY SMALLCAP 250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}


def parse_symbols(raw: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in raw.replace("\n", ",").split(","):
        symbol = item.strip().upper()
        if symbol and symbol not in seen:
            seen.add(symbol)
            out.append(symbol)
    return out


def load_symbols_from_file(path: str) -> list[str]:
    p = Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"symbols file does not exist: {p}")
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        for col in ["Symbol", "SYMBOL", "tradingsymbol", "symbol"]:
            if col in df.columns:
                return parse_symbols(",".join(df[col].dropna().astype(str).tolist()))
        raise SystemExit(f"CSV {p} has no Symbol/SYMBOL column")
    return parse_symbols(p.read_text(encoding="utf-8"))


def fetch_nse_index_symbols(name: str) -> list[str]:
    import requests
    from io import StringIO

    key = name.strip().upper()
    if key not in NSE_INDEX_URLS:
        raise SystemExit(f"unknown universe {name!r}; choices: {', '.join(NSE_INDEX_URLS)}")
    response = requests.get(NSE_INDEX_URLS[key], headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    return parse_symbols(",".join(df[col].dropna().astype(str).tolist()))


def build_symbol_list(args: argparse.Namespace) -> list[str]:
    symbols: list[str] = []
    if args.symbols:
        symbols.extend(parse_symbols(args.symbols))
    if args.symbols_file:
        symbols.extend(load_symbols_from_file(args.symbols_file))
    if args.universe:
        symbols.extend(fetch_nse_index_symbols(args.universe))
    if not symbols:
        raise SystemExit("Provide --symbols, --symbols-file, or --universe")
    seen: set[str] = set()
    unique = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique[: args.limit] if args.limit else unique


def make_kite(args: argparse.Namespace) -> KiteConnect:
    creds = load_credentials(args.secrets_file)
    token_payload = read_token(Path(args.token_path).expanduser())
    token = token_payload.get("access_token")
    if not token:
        raise SystemExit(f"No lab access token found at {args.token_path}; run scripts/kite_lab_token.py refresh first")
    kite = KiteConnect(api_key=creds.api_key)
    kite.set_access_token(token)
    return kite


def load_instruments(kite: KiteConnect, cache_path: Path, refresh: bool) -> dict[str, int]:
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        token_map = payload.get("token_map", payload)
        return {str(k).upper(): int(v) for k, v in token_map.items()}
    instruments = kite.instruments("NSE")
    token_map = {str(row["tradingsymbol"]).upper(): int(row["instrument_token"]) for row in instruments if row.get("tradingsymbol") and row.get("instrument_token")}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"generated_at": datetime.now().astimezone().isoformat(), "exchange": "NSE", "token_map": token_map}, indent=2),
        encoding="utf-8",
    )
    return token_map


def interval_delta(interval: str) -> timedelta:
    if interval == "day":
        return timedelta(days=1)
    if interval == "minute":
        return timedelta(minutes=1)
    if interval.endswith("minute"):
        return timedelta(minutes=int(interval.replace("minute", "")))
    return timedelta(days=1)


def chunk_range(start_dt: datetime | date, end_dt: datetime | date, max_days: int) -> Iterable[tuple[datetime | date, datetime | date]]:
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + relativedelta(days=max_days), end_dt)
        yield cursor, chunk_end
        cursor = chunk_end


def read_existing(path: Path, interval: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_feather(path)
    if "Date" not in df.columns:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    if interval == "day":
        df["Date"] = df["Date"].dt.date
    return df.reset_index(drop=True)


def compute_start(existing: pd.DataFrame | None, interval: str, years: int, intraday_days: int) -> datetime | date:
    if existing is not None and len(existing):
        last = pd.to_datetime(existing["Date"], errors="coerce").max()
        if pd.notna(last):
            if interval == "day":
                return last.date() + timedelta(days=1)
            return last.to_pydatetime() + interval_delta(interval)
    if interval == "day":
        return date.today() - relativedelta(years=years)
    return datetime.now() - timedelta(days=intraday_days)


def normalize_frame(rows: list[dict[str, Any]], interval: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.rename(columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if interval == "day":
        df["Date"] = df["Date"].dt.date
    return df.sort_values("Date").drop_duplicates("Date", keep="last").reset_index(drop=True)


def is_market_hours_now() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return (now.hour, now.minute) >= (9, 0) and (now.hour, now.minute) <= (15, 45)


def fetch_symbol(kite: KiteConnect, symbol: str, token: int, args: argparse.Namespace) -> dict[str, Any]:
    hist_dir = Path(args.hist_dir).expanduser()
    hist_dir.mkdir(parents=True, exist_ok=True)
    out_path = hist_dir / f"{symbol}.feather"
    existing = read_existing(out_path, args.interval)
    start = compute_start(existing, args.interval, args.years, args.intraday_days)
    end = datetime.now() if args.interval != "day" else date.today()
    if start >= end:
        return {"symbol": symbol, "status": "up_to_date", "rows": int(len(existing) if existing is not None else 0), "path": str(out_path)}

    all_rows: list[dict[str, Any]] = []
    for sdt, edt in chunk_range(start, end, INTERVAL_LIMITS[args.interval]):
        last_error = ""
        for attempt in range(1, args.retries + 1):
            try:
                rows = kite.historical_data(token, from_date=sdt, to_date=edt, interval=args.interval, oi=False)
                all_rows.extend(rows)
                last_error = ""
                break
            except (NetworkException, TokenException) as exc:
                last_error = str(exc)
                time.sleep(args.retry_sleep * attempt)
            except Exception as exc:  # noqa: BLE001
                return {"symbol": symbol, "status": "error", "error": str(exc)[:300]}
        if last_error:
            return {"symbol": symbol, "status": "error", "error": last_error[:300]}
        time.sleep(args.pause)

    new_df = normalize_frame(all_rows, args.interval)
    if new_df.empty:
        return {"symbol": symbol, "status": "no_new_data", "rows": int(len(existing) if existing is not None else 0), "path": str(out_path)}
    if args.interval == "day" and args.drop_today_during_market and is_market_hours_now():
        today = date.today()
        new_df = new_df[new_df["Date"] != today]
    merged = new_df if existing is None or existing.empty else pd.concat([existing, new_df], ignore_index=True)
    merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
    merged = merged.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    if args.interval == "day":
        merged["Date"] = merged["Date"].dt.date
    merged.reset_index(drop=True).to_feather(out_path)
    return {"symbol": symbol, "status": "fetched", "rows": int(len(merged)), "new_rows": int(len(new_df)), "path": str(out_path)}


def load_marks(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def save_marks(path: Path, marks: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marks, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Kite historical OHLCV into Trader_Labs cache")
    parser.add_argument("--secrets-file", default=os.getenv("KITE_LAB_SECRETS_FILE", ""))
    parser.add_argument("--token-path", default=os.getenv("KITE_LAB_TOKEN_PATH", str(DEFAULT_TOKEN_PATH)))
    parser.add_argument("--hist-dir", default=os.getenv("KITE_LAB_HIST_DIR", str(DEFAULT_HIST_DIR)))
    parser.add_argument("--instruments-cache", default=str(DEFAULT_INSTRUMENTS_CACHE))
    parser.add_argument("--fetch-marks", default=str(DEFAULT_FETCH_MARKS))
    parser.add_argument("--refresh-instruments", action="store_true")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--symbols-file", default="")
    parser.add_argument("--universe", default="", help="NIFTY50/NIFTY100/NIFTY200/NIFTY500/etc")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--interval", default=os.getenv("KITE_LAB_INTERVAL", "day"), choices=sorted(INTERVAL_LIMITS))
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--intraday-days", type=int, default=60)
    parser.add_argument("--pause", type=float, default=0.35)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--skip-today-if-fetched", action="store_true", help="Skip symbols already fetched today for this interval")
    parser.add_argument("--drop-today-during-market", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    symbols = build_symbol_list(args)
    cache_path = Path(args.instruments_cache).expanduser()
    if args.dry_run and cache_path.exists() and not args.refresh_instruments:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        raw_map = payload.get("token_map", payload)
        token_map = {str(k).upper(): int(v) for k, v in raw_map.items()}
    elif args.dry_run and not cache_path.exists() and not args.refresh_instruments:
        token_map = {s: 0 for s in symbols}
    else:
        kite = make_kite(args)
        token_map = load_instruments(kite, cache_path, args.refresh_instruments)
    marks_path = Path(args.fetch_marks).expanduser()
    marks = load_marks(marks_path)
    today_key = date.today().isoformat()
    interval_marks = marks.setdefault(args.interval, {})

    planned = [s for s in symbols if s in token_map]
    missing = [s for s in symbols if s not in token_map]
    if args.skip_today_if_fetched:
        planned = [s for s in planned if interval_marks.get(s) != today_key]
    if args.dry_run:
        print(json.dumps({"status": "dry_run", "planned": planned, "missing_tokens": missing, "hist_dir": args.hist_dir}, indent=2))
        return 0

    results = []
    for idx, symbol in enumerate(planned, start=1):
        print(f"[{idx}/{len(planned)}] {symbol}", flush=True)
        result = fetch_symbol(kite, symbol, token_map[symbol], args)
        results.append(result)
        if result.get("status") in {"fetched", "up_to_date", "no_new_data"}:
            interval_marks[symbol] = today_key
            save_marks(marks_path, marks)

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "interval": args.interval,
        "hist_dir": args.hist_dir,
        "requested": len(symbols),
        "planned": len(planned),
        "missing_tokens": missing,
        "status_counts": pd.Series([r.get("status") for r in results]).value_counts().to_dict() if results else {},
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 1 if any(r.get("status") == "error" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
