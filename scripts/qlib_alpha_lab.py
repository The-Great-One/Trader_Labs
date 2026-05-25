#!/usr/bin/env python3
"""Qlib-style Kite alpha/ranking lab for Auto_Trader research.

This is intentionally research-only:
  * reads Kite feather OHLCV from Trader_Labs/Auto_Trader cache or sibling Stocks cache
  * builds point-in-time technical features and forward-return labels
  * trains walk-forward cross-sectional ranking models
  * backtests top-N rotation from model scores
  * writes ignored reports under reports/

It does not import or mutate live Auto_Trader rule configs.  The implementation
keeps Qlib optional because the production/lab hosts already have sklearn and
LightGBM in the existing stack, while qlib is a heavy research dependency.  The
report is Qlib-compatible in spirit: features -> labels -> model score ->
portfolio analysis, with strict walk-forward splits.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
DEFAULT_HIST_DIRS = [
    ROOT / "intermediary_files" / "Hist_Data",
    ROOT.parent / "Stocks" / "intermediary_files" / "Hist_Data",
]


@dataclass
class PortfolioResult:
    name: str
    model: str
    rebalance: str
    top_n: int
    horizon_days: int
    cost_bps: float
    start: str
    end: str
    days: int
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    vol_pct: float
    sharpe_like: float
    worst_year_pct: float
    positive_years: int
    total_years: int
    avg_positions: float
    turnover_monthly_equiv: float
    selection_score: float


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_csv_str(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def parse_symbols(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    vals = {x.strip().upper() for x in raw.split(",") if x.strip()}
    return vals or None


def is_derivative_symbol(symbol: str) -> bool:
    s = symbol.upper()
    return any(tag in s for tag in ("CE", "PE", "FUT")) or any(ch.isdigit() for ch in s[-8:])


def find_hist_dir(value: str | None) -> Path:
    if value:
        p = Path(value).expanduser()
        if p.exists():
            return p
        raise SystemExit(f"hist dir does not exist: {p}")
    for p in DEFAULT_HIST_DIRS:
        if p.exists():
            return p
    raise SystemExit("No Hist_Data directory found")


def normalise_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {str(c).lower(): c for c in df.columns}
    date_col = cmap.get("date")
    close_col = cmap.get("close")
    if date_col is None or close_col is None:
        raise ValueError("missing date/close columns")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "close": pd.to_numeric(df[close_col], errors="coerce"),
    })
    for name in ("open", "high", "low", "volume"):
        col = cmap.get(name)
        out[name] = pd.to_numeric(df[col], errors="coerce") if col is not None else np.nan
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out[["open", "close"]].max(axis=1))
    out["low"] = out["low"].fillna(out[["open", "close"]].min(axis=1))
    out["volume"] = out["volume"].fillna(0)
    out = out.dropna(subset=["date", "close"]).drop_duplicates("date").sort_values("date")
    return out.reset_index(drop=True)


def load_ohlcv(
    hist_dir: Path,
    min_rows: int,
    min_end_date: str,
    symbols: set[str] | None,
    max_symbols: int,
) -> tuple[dict[str, pd.DataFrame], dict]:
    min_end = pd.Timestamp(min_end_date) if min_end_date else None
    data: dict[str, pd.DataFrame] = {}
    skipped: dict[str, int] = {"derivative": 0, "not_requested": 0, "too_short": 0, "stale": 0, "read_error": 0}
    summaries: list[dict] = []
    for fp in sorted(hist_dir.glob("*.feather")):
        symbol = fp.stem.upper()
        if symbols and symbol not in symbols:
            skipped["not_requested"] += 1
            continue
        if is_derivative_symbol(symbol):
            skipped["derivative"] += 1
            continue
        try:
            df = normalise_ohlcv_columns(pd.read_feather(fp))
            if len(df) < min_rows:
                skipped["too_short"] += 1
                continue
            if min_end is not None and df["date"].max() < min_end:
                skipped["stale"] += 1
                continue
            data[symbol] = df
            summaries.append({
                "symbol": symbol,
                "rows": int(len(df)),
                "start": str(df["date"].min().date()),
                "end": str(df["date"].max().date()),
            })
            if max_symbols and len(data) >= max_symbols:
                break
        except Exception:
            skipped["read_error"] += 1
    if not data:
        raise SystemExit(f"No usable symbols loaded from {hist_dir}")
    all_dates = pd.concat([df["date"] for df in data.values()])
    context = {
        "hist_dir": str(hist_dir),
        "symbols_loaded": len(data),
        "skipped": skipped,
        "date_range": [str(all_dates.min().date()), str(all_dates.max().date())],
        "min_rows": min_rows,
        "min_end_date": min_end_date,
        "loaded_symbols": sorted(data),
        "symbol_summaries_sample": summaries[:50],
    }
    return data, context


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_symbol_features(symbol: str, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    x = df.copy()
    close = x["close"]
    high = x["high"]
    low = x["low"]
    volume = x["volume"].replace(0, np.nan)
    ret1 = close.pct_change(fill_method=None)
    x["symbol"] = symbol
    x["ret_1"] = ret1
    x["ret_5"] = close.pct_change(5, fill_method=None)
    x["ret_10"] = close.pct_change(10, fill_method=None)
    x["ret_21"] = close.pct_change(21, fill_method=None)
    x["ret_63"] = close.pct_change(63, fill_method=None)
    x["vol_21"] = ret1.rolling(21, min_periods=21).std()
    x["vol_63"] = ret1.rolling(63, min_periods=63).std()
    x["rsi_14"] = rsi(close, 14)
    x["rsi_22"] = rsi(close, 22)
    x["rsi_44"] = rsi(close, 44)
    x["rsi_66"] = rsi(close, 66)
    x["sma_20_ratio"] = close / close.rolling(20, min_periods=20).mean() - 1
    x["sma_50_ratio"] = close / close.rolling(50, min_periods=50).mean() - 1
    x["sma_100_ratio"] = close / close.rolling(100, min_periods=100).mean() - 1
    x["sma_200_ratio"] = close / close.rolling(200, min_periods=200).mean() - 1
    tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    x["atr_14_pct"] = tr.rolling(14, min_periods=14).mean() / close
    x["dollar_vol_21"] = (close * volume).rolling(21, min_periods=10).mean()
    x["volume_z_21"] = (volume - volume.rolling(21, min_periods=21).mean()) / volume.rolling(21, min_periods=21).std()
    x["label_fwd_return"] = close.shift(-horizon) / close - 1
    return x


def build_feature_frame(data: dict[str, pd.DataFrame], horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    frames = [add_symbol_features(symbol, df, horizon) for symbol, df in data.items()]
    panel = pd.concat(frames, ignore_index=True).sort_values(["date", "symbol"])
    feature_cols = [
        "ret_1", "ret_5", "ret_10", "ret_21", "ret_63",
        "vol_21", "vol_63", "rsi_14", "rsi_22", "rsi_44", "rsi_66",
        "sma_20_ratio", "sma_50_ratio", "sma_100_ratio", "sma_200_ratio",
        "atr_14_pct", "dollar_vol_21", "volume_z_21",
    ]
    # Cross-sectional ranks make the lab closer to Qlib alpha ranking and reduce scale drift.
    ranked_parts = []
    for col in feature_cols:
        rcol = f"xrank_{col}"
        panel[rcol] = panel.groupby("date")[col].rank(pct=True)
        ranked_parts.append(rcol)
    model_cols = feature_cols + ranked_parts
    panel = panel.dropna(subset=model_cols + ["label_fwd_return", "close"]).reset_index(drop=True)
    prices = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    return panel, prices, model_cols


def make_model(name: str, random_state: int):
    name = name.lower()
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(
                n_estimators=250,
                learning_rate=0.035,
                max_depth=4,
                num_leaves=24,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="regression",
                random_state=random_state,
                n_jobs=1,
                verbose=-1,
            )
        except Exception as exc:
            print(f"WARN: lightgbm unavailable ({exc}); falling back to sklearn_hgb", file=sys.stderr)
    if name in {"sklearn_hgb", "lightgbm"}:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, max_leaf_nodes=24, l2_regularization=0.01, random_state=random_state)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=160, max_depth=6, min_samples_leaf=20, random_state=random_state, n_jobs=1)
    raise ValueError(f"unknown model: {name}")


def fold_boundaries(dates: pd.DatetimeIndex, n_folds: int, min_train_days: int, test_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    uniq = pd.DatetimeIndex(sorted(pd.unique(dates)))
    out: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    start_idx = min_train_days
    while start_idx < len(uniq) - 30 and len(out) < n_folds:
        test_start = uniq[start_idx]
        test_end = uniq[min(start_idx + test_days - 1, len(uniq) - 1)]
        train_end = uniq[start_idx - 1]
        out.append((uniq[0], train_end, test_end))
        start_idx += test_days
    return out


def train_walkforward_scores(
    panel: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    n_folds: int,
    min_train_days: int,
    test_days: int,
    horizon_days: int,
    embargo_days: int,
    random_state: int,
) -> tuple[pd.DataFrame, list[dict]]:
    dates = pd.DatetimeIndex(panel["date"])
    folds = fold_boundaries(dates, n_folds=n_folds, min_train_days=min_train_days, test_days=test_days)
    if not folds:
        raise SystemExit("Not enough dates for requested walk-forward folds")
    scored_parts: list[pd.DataFrame] = []
    fold_reports: list[dict] = []
    unique_dates = pd.DatetimeIndex(sorted(pd.unique(panel["date"])))
    for i, (_train_start, train_end_raw, test_end) in enumerate(folds, 1):
        test_start_idx = unique_dates.get_loc(train_end_raw) + 1
        test_start = unique_dates[test_start_idx]
        embargo = max(horizon_days, embargo_days)
        safe_train_end_idx = max(0, test_start_idx - embargo - 1)
        safe_train_end = unique_dates[safe_train_end_idx]
        train = panel[panel["date"] <= safe_train_end]
        test = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)]
        if len(train) < 1000 or len(test) < 100:
            fold_reports.append({"fold": i, "status": "skipped", "train_rows": len(train), "test_rows": len(test)})
            continue
        model = make_model(model_name, random_state + i)
        X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train["label_fwd_return"].clip(-0.5, 0.5)
        X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        part = test[["date", "symbol", "label_fwd_return"]].copy()
        part["score"] = pred
        scored_parts.append(part)
        # Rank IC by day; keep both mean and positive-day count for robustness.
        ic_by_day = []
        for _, g in part.groupby("date"):
            if g["score"].nunique() > 1 and g["label_fwd_return"].nunique() > 1 and len(g) >= 5:
                ic_by_day.append(float(g["score"].corr(g["label_fwd_return"], method="spearman")))
        fold_reports.append({
            "fold": i,
            "status": "ok",
            "train_end": str(safe_train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "rank_ic_mean": round(float(np.nanmean(ic_by_day)), 4) if ic_by_day else 0.0,
            "rank_ic_positive_days": int(np.sum(np.array(ic_by_day) > 0)) if ic_by_day else 0,
            "rank_ic_days": int(len(ic_by_day)),
        })
    if not scored_parts:
        raise SystemExit("No walk-forward fold produced scores")
    return pd.concat(scored_parts, ignore_index=True), fold_reports


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    marker = pd.Series(index=index, dtype=float)
    for d in marker.resample(freq).last().index:
        loc = index[index <= d]
        if len(loc):
            out.append(loc[-1])
    return sorted(set(out))


def portfolio_metrics(
    name: str,
    model_name: str,
    rebalance: str,
    top_n: int,
    horizon_days: int,
    cost_bps: float,
    returns: pd.Series,
    weights: pd.DataFrame,
    turnover: pd.Series,
) -> PortfolioResult:
    active = weights.sum(axis=1) > 0
    r = returns.loc[active].copy()
    if r.empty:
        raise ValueError(f"empty active returns for {name}")
    eq = (1 + r).cumprod()
    years = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / max(years, 0.01)) - 1
    dd = eq / eq.cummax() - 1
    vol = r.std() * math.sqrt(252)
    sharpe = (r.mean() * 252) / vol if vol and not np.isnan(vol) else 0.0
    yearly = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    avg_positions = weights.loc[active].astype(bool).sum(axis=1).mean()
    turnover_monthly_equiv = turnover.sum() / max(len(r) / 21, 1)
    selection_score = (cagr * 100) + (dd.min() * 35) + (sharpe * 2) - (turnover_monthly_equiv * 1.5)
    return PortfolioResult(
        name=name,
        model=model_name,
        rebalance=rebalance,
        top_n=int(top_n),
        horizon_days=int(horizon_days),
        cost_bps=float(cost_bps),
        start=str(r.index[0].date()),
        end=str(r.index[-1].date()),
        days=int(len(r)),
        total_return_pct=round(float((eq.iloc[-1] - 1) * 100), 2),
        cagr_pct=round(float(cagr * 100), 2),
        max_drawdown_pct=round(float(dd.min() * 100), 2),
        vol_pct=round(float(vol * 100), 2),
        sharpe_like=round(float(sharpe), 3),
        worst_year_pct=round(float(yearly.min() * 100), 2),
        positive_years=int((yearly > 0).sum()),
        total_years=int(len(yearly)),
        avg_positions=round(float(avg_positions), 2),
        turnover_monthly_equiv=round(float(turnover_monthly_equiv), 3),
        selection_score=round(float(selection_score), 3),
    )


def run_score_rotation(
    prices_raw: pd.DataFrame,
    scores: pd.DataFrame,
    model_name: str,
    rebalance: str,
    top_n: int,
    horizon_days: int,
    cost_bps: float,
    ffill_limit: int,
) -> tuple[PortfolioResult, dict]:
    prices = prices_raw.ffill(limit=ffill_limit)
    common_idx = prices.index.intersection(scores.index)
    prices = prices.loc[common_idx]
    scores = scores.loc[common_idx]
    returns = prices.pct_change(fill_method=None).fillna(0)
    dates = rebalance_dates(common_idx, rebalance)
    weights = pd.DataFrame(0.0, index=common_idx, columns=prices.columns)
    turnover = pd.Series(0.0, index=common_idx)
    previous = pd.Series(0.0, index=prices.columns)
    picks_log: list[dict] = []
    for i, d in enumerate(dates):
        pos = common_idx.get_loc(d)
        if pos + 1 >= len(common_idx):
            continue
        trade_date = common_idx[pos + 1]
        end_date = dates[i + 1] if i + 1 < len(dates) else common_idx[-1]
        target = pd.Series(0.0, index=prices.columns)
        sc = scores.loc[d].dropna().sort_values(ascending=False)
        picks = [s for s in sc.index if pd.notna(prices.loc[d, s])][:top_n]
        if picks:
            target.loc[picks] = 1.0 / len(picks)
            picks_log.append({
                "signal_date": str(d.date()),
                "trade_date": str(trade_date.date()),
                "picks": picks,
                "scores": {s: round(float(sc.loc[s]), 5) for s in picks[:20]},
            })
        turnover.loc[trade_date] = abs(target - previous).sum()
        previous = target
        weights.loc[(common_idx >= trade_date) & (common_idx <= end_date), :] = target.values
    net = (weights * returns).sum(axis=1) - turnover * (cost_bps / 10000.0)
    name = f"qlib_style_{model_name}_{rebalance}_top{top_n}_h{horizon_days}"
    result = portfolio_metrics(name, model_name, rebalance, top_n, horizon_days, cost_bps, net, weights, turnover)
    return result, {"latest_picks": picks_log[-1] if picks_log else {}, "rebalance_count": len(picks_log)}


def equal_weight_baseline(prices_raw: pd.DataFrame, cost_bps: float, ffill_limit: int) -> PortfolioResult:
    prices = prices_raw.ffill(limit=ffill_limit)
    returns = prices.pct_change(fill_method=None).fillna(0)
    weights = prices.notna().astype(float)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.sum(axis=1))
    net = (weights * returns).sum(axis=1) - turnover * (cost_bps / 10000.0)
    return portfolio_metrics("equal_weight_available_universe", "baseline", "daily_available", prices.shape[1], 0, cost_bps, net, weights, turnover)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kite-only Qlib-style ML alpha/ranking lab")
    parser.add_argument("--hist-dir", default=os.getenv("AT_QLIB_LAB_HIST_DIR", ""))
    parser.add_argument("--symbols", default=os.getenv("AT_QLIB_LAB_SYMBOLS", ""), help="Comma-separated symbols")
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("AT_QLIB_LAB_MAX_SYMBOLS", "0") or "0"))
    parser.add_argument("--min-rows", type=int, default=int(os.getenv("AT_QLIB_LAB_MIN_ROWS", "700") or "700"))
    parser.add_argument("--min-end-date", default=os.getenv("AT_QLIB_LAB_MIN_END_DATE", "2026-04-17"))
    parser.add_argument("--model", default=os.getenv("AT_QLIB_LAB_MODEL", "lightgbm"), choices=["lightgbm", "sklearn_hgb", "random_forest"])
    parser.add_argument("--horizon-days", type=int, default=int(os.getenv("AT_QLIB_LAB_HORIZON_DAYS", "21") or "21"))
    parser.add_argument("--top-n", default=os.getenv("AT_QLIB_LAB_TOP_N", "10,20,30"))
    parser.add_argument("--rebalance", default=os.getenv("AT_QLIB_LAB_REBALANCE", "W-FRI,ME"))
    parser.add_argument("--folds", type=int, default=int(os.getenv("AT_QLIB_LAB_FOLDS", "5") or "5"))
    parser.add_argument("--min-train-days", type=int, default=int(os.getenv("AT_QLIB_LAB_MIN_TRAIN_DAYS", "756") or "756"))
    parser.add_argument("--test-days", type=int, default=int(os.getenv("AT_QLIB_LAB_TEST_DAYS", "252") or "252"))
    parser.add_argument("--embargo-days", type=int, default=int(os.getenv("AT_QLIB_LAB_EMBARGO_DAYS", "21") or "21"))
    parser.add_argument("--cost-bps", type=float, default=float(os.getenv("AT_QLIB_LAB_COST_BPS", "10") or "10"))
    parser.add_argument("--ffill-limit", type=int, default=int(os.getenv("AT_QLIB_LAB_FFILL_LIMIT", "3") or "3"))
    parser.add_argument("--random-state", type=int, default=int(os.getenv("AT_QLIB_LAB_RANDOM_STATE", "42") or "42"))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_dir = find_hist_dir(args.hist_dir)
    data, data_context = load_ohlcv(
        hist_dir=hist_dir,
        min_rows=args.min_rows,
        min_end_date=args.min_end_date,
        symbols=parse_symbols(args.symbols),
        max_symbols=args.max_symbols,
    )
    panel, prices, feature_cols = build_feature_frame(data, args.horizon_days)
    scored, fold_reports = train_walkforward_scores(
        panel=panel,
        feature_cols=feature_cols,
        model_name=args.model,
        n_folds=args.folds,
        min_train_days=args.min_train_days,
        test_days=args.test_days,
        horizon_days=args.horizon_days,
        embargo_days=args.embargo_days,
        random_state=args.random_state,
    )
    score_matrix = scored.pivot(index="date", columns="symbol", values="score").sort_index()
    ranked: list[PortfolioResult] = [equal_weight_baseline(prices.loc[score_matrix.index.min():], args.cost_bps, args.ffill_limit)]
    diagnostics: dict[str, dict] = {}
    for reb in parse_csv_str(args.rebalance):
        for top_n in parse_csv_ints(args.top_n):
            try:
                result, diag = run_score_rotation(
                    prices_raw=prices,
                    scores=score_matrix,
                    model_name=args.model,
                    rebalance=reb,
                    top_n=top_n,
                    horizon_days=args.horizon_days,
                    cost_bps=args.cost_bps,
                    ffill_limit=args.ffill_limit,
                )
                ranked.append(result)
                diagnostics[result.name] = diag
            except Exception as exc:
                diagnostics[f"{args.model}_{reb}_top{top_n}"] = {"error": str(exc)}
    ranked = sorted(ranked, key=lambda r: (r.selection_score, r.cagr_pct, r.max_drawdown_pct), reverse=True)
    best = ranked[0]
    mean_ic = np.nanmean([f.get("rank_ic_mean", np.nan) for f in fold_reports if f.get("status") == "ok"])
    pos_folds = sum(1 for f in fold_reports if f.get("status") == "ok" and f.get("rank_ic_mean", 0) > 0)
    ok_folds = sum(1 for f in fold_reports if f.get("status") == "ok")
    verdict = "needs_more_validation"
    if best.name != "equal_weight_available_universe" and best.cagr_pct >= 30 and best.max_drawdown_pct > -25 and ok_folds >= 4 and pos_folds >= 4:
        verdict = "research_candidate_not_prod"
    recommendation = {
        "generated_at": now_iso(),
        "lab_type": "qlib_style_alpha_ranking",
        "source_idea": "microsoft/qlib-style alpha ranking; optional qlib dependency, Kite-data-first implementation",
        "data_context": data_context,
        "params": vars(args),
        "feature_count": len(feature_cols),
        "rows_after_feature_label_filter": int(len(panel)),
        "folds_ok": ok_folds,
        "rank_ic_mean_oos": round(float(mean_ic), 4) if not np.isnan(mean_ic) else 0.0,
        "rank_ic_positive_folds": pos_folds,
        "baseline": asdict(next(r for r in ranked if r.name == "equal_weight_available_universe")),
        "best": asdict(best),
        "verdict": verdict,
        "promotion_note": "Research-only. Do not put in production unless a follow-up live-parity WF validation beats RS7/RS2 and paper shadow confirms lifecycle behavior.",
        "diagnostics_for_best": diagnostics.get(best.name, {}),
    }
    payload = {"recommendation": recommendation, "ranked": [asdict(r) for r in ranked], "fold_reports": fold_reports, "diagnostics": diagnostics}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"qlib_alpha_lab_{ts}.json"
    csv_path = OUT_DIR / f"qlib_alpha_lab_{ts}.csv"
    latest_path = OUT_DIR / "qlib_alpha_lab_latest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame([asdict(r) for r in ranked]).to_csv(csv_path, index=False)
    print(json.dumps(recommendation, indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
