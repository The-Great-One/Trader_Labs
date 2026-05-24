#!/usr/bin/env python3
"""Export Kite feather OHLCV history into Kronos finetune CSV datasets.

Kronos finetune_csv expects one CSV with columns:
  timestamps, open, high, low, close, volume, amount

For multi-symbol NSE/Kite training we concatenate per-symbol daily bars and add
`symbol` plus `split` metadata columns. The Kronos CSV loader ignores extra
columns in current examples/configs, while these fields let us audit date-based
train/validation/test splits and avoid random leakage.

Default source is the sibling Auto_Trader cache:
  ../Stocks/intermediary_files/Hist_Data/*.feather
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

LAB_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTOTRADER_ROOT = LAB_ROOT.parent / "Stocks"


@dataclass
class SymbolSummary:
    symbol: str
    rows: int
    train_rows: int
    val_rows: int
    test_rows: int
    start: str | None
    end: str | None
    skipped_reason: str | None = None


def _normalize_ohlcv(path: Path) -> pd.DataFrame:
    raw = pd.read_feather(path)
    cmap = {str(c).lower(): c for c in raw.columns}
    required = ["date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in cmap]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = pd.DataFrame(
        {
            "timestamps": pd.to_datetime(raw[cmap["date"]], errors="coerce"),
            "open": pd.to_numeric(raw[cmap["open"]], errors="coerce"),
            "high": pd.to_numeric(raw[cmap["high"]], errors="coerce"),
            "low": pd.to_numeric(raw[cmap["low"]], errors="coerce"),
            "close": pd.to_numeric(raw[cmap["close"]], errors="coerce"),
            "volume": pd.to_numeric(raw[cmap.get("volume", "Volume")], errors="coerce") if "volume" in cmap else 0,
        }
    )
    out = out.dropna(subset=["timestamps", "open", "high", "low", "close"])
    out = out.sort_values("timestamps").drop_duplicates(subset=["timestamps"], keep="last")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).clip(lower=0)
    out["amount"] = (out["close"] * out["volume"]).fillna(0)
    return out.reset_index(drop=True)


def _split_by_date(df: pd.DataFrame, train_ratio: float, val_ratio: float, test_ratio: float) -> pd.Series:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("split ratios must sum positive")
    train_ratio, val_ratio, test_ratio = train_ratio / total, val_ratio / total, test_ratio / total
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    split = pd.Series("test", index=df.index, dtype="object")
    split.iloc[:train_end] = "train"
    split.iloc[train_end:val_end] = "val"
    if test_ratio == 0:
        split.iloc[val_end:] = "val"
    return split


def _iter_symbols(source_dir: Path, symbols: Iterable[str] | None, limit: int) -> list[Path]:
    wanted = {s.upper().strip() for s in symbols or [] if s.strip()}
    files = sorted(source_dir.glob("*.feather"))
    if wanted:
        files = [p for p in files if p.stem.upper() in wanted]
    # Exclude derivative-style files by default; daily equity/ETF symbols are enough for first finetune.
    files = [p for p in files if not any(tag in p.stem.upper() for tag in ("CE", "PE", "FUT"))]
    return files[:limit] if limit else files


def export_dataset(args: argparse.Namespace) -> dict:
    autotrader_root = Path(args.autotrader_root or os.getenv("AUTOTRADER_ROOT", DEFAULT_AUTOTRADER_ROOT)).expanduser()
    source_dir = Path(args.source_dir).expanduser() if args.source_dir else autotrader_root / "intermediary_files" / "Hist_Data"
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / args.output_name
    manifest_path = out_dir / args.manifest_name

    rows: list[pd.DataFrame] = []
    summaries: list[SymbolSummary] = []
    symbols = args.symbols.split(",") if args.symbols else None

    for fp in _iter_symbols(source_dir, symbols, args.limit):
        symbol = fp.stem.upper()
        try:
            df = _normalize_ohlcv(fp)
            if len(df) < args.min_rows:
                summaries.append(SymbolSummary(symbol, len(df), 0, 0, 0, None, None, f"rows_lt_{args.min_rows}"))
                continue
            df = df.tail(args.max_rows_per_symbol).copy() if args.max_rows_per_symbol else df.copy()
            split = _split_by_date(df, args.train_ratio, args.val_ratio, args.test_ratio)
            df.insert(0, "symbol", symbol)
            df["split"] = split.values
            rows.append(df)
            counts = split.value_counts().to_dict()
            summaries.append(
                SymbolSummary(
                    symbol=symbol,
                    rows=len(df),
                    train_rows=int(counts.get("train", 0)),
                    val_rows=int(counts.get("val", 0)),
                    test_rows=int(counts.get("test", 0)),
                    start=str(pd.Timestamp(df["timestamps"].iloc[0]).date()),
                    end=str(pd.Timestamp(df["timestamps"].iloc[-1]).date()),
                )
            )
        except Exception as exc:
            summaries.append(SymbolSummary(symbol, 0, 0, 0, 0, None, None, str(exc)[:200]))

    if not rows:
        raise SystemExit(f"No symbols exported from {source_dir}")

    dataset = pd.concat(rows, ignore_index=True)
    dataset = dataset.sort_values(["symbol", "timestamps"]).reset_index(drop=True)
    # Kronos README uses string timestamps; keep second precision for deterministic CSVs.
    dataset["timestamps"] = pd.to_datetime(dataset["timestamps"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    dataset.to_csv(out_csv, index=False)

    manifest = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_dir": str(source_dir),
        "output_csv": str(out_csv),
        "rows": int(len(dataset)),
        "symbols_exported": int(sum(1 for s in summaries if s.skipped_reason is None)),
        "symbols_skipped": int(sum(1 for s in summaries if s.skipped_reason is not None)),
        "split_counts": {k: int(v) for k, v in dataset["split"].value_counts().to_dict().items()},
        "args": vars(args),
        "symbols": [asdict(s) for s in summaries],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("output_csv", "rows", "symbols_exported", "symbols_skipped", "split_counts")}, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Kite feather history to Kronos CSV finetune dataset")
    parser.add_argument("--autotrader-root", default="", help="Auto_Trader repo root; defaults to AUTOTRADER_ROOT or ../Stocks")
    parser.add_argument("--source-dir", default="", help="Override Hist_Data source directory")
    parser.add_argument("--output-dir", default=str(LAB_ROOT / "kronos_finetune" / "data"))
    parser.add_argument("--output-name", default="kite_nse_daily_kronos.csv")
    parser.add_argument("--manifest-name", default="kite_nse_daily_kronos_manifest.json")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols; default exports all eligible feather files")
    parser.add_argument("--limit", type=int, default=0, help="Max symbols to export after filtering; 0 = all")
    parser.add_argument("--min-rows", type=int, default=600)
    parser.add_argument("--max-rows-per-symbol", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    args = parser.parse_args()
    export_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
