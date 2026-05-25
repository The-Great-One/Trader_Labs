#!/usr/bin/env python3
"""US market recommendation engine based only on fundamentals and news.

Research-only / no trading. Designed for Sahil's Tickertape US investing flow.
Inputs are a configurable US stocks/ETFs universe and public data. The scorer
intentionally excludes price momentum/technical indicators.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
DEFAULT_UNIVERSE = ROOT / "data" / "us_markets" / "tickertape_us_universe.csv"
CACHE_DIR = ROOT / "reports" / "us_recommender_cache"

POSITIVE_WORDS = {
    "beat", "beats", "upgrade", "upgraded", "outperform", "growth", "surge", "surges", "record",
    "profit", "profitable", "strong", "raises", "raised", "buyback", "dividend", "approval",
    "partnership", "expands", "launch", "wins", "resilient", "accelerate", "accelerates",
}
NEGATIVE_WORDS = {
    "miss", "misses", "downgrade", "downgraded", "underperform", "lawsuit", "probe", "fraud",
    "weak", "cuts", "cut", "decline", "falls", "plunge", "layoffs", "recall", "warning", "risk",
    "slows", "slowing", "loss", "debt", "antitrust", "ban", "tariff", "delay", "delays",
}


@dataclass
class Recommendation:
    symbol: str
    name: str
    asset_class: str
    theme: str
    recommendation: str
    total_score: float
    fundamental_score: float
    news_score: float
    risk_score: float
    valuation_score: float
    quality_score: float
    growth_score: float
    dividend_score: float
    data_quality: str
    reasons: list[str]
    warnings: list[str]
    metrics: dict[str, Any]
    news_headlines: list[str]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if x is None or not math.isfinite(float(x)):
        return lo
    return max(lo, min(hi, float(x)))


def score_range(value: Any, good: float, bad: float, higher_better: bool = True) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            return 50.0
    except Exception:
        return 50.0
    if higher_better:
        return clamp((v - bad) / (good - bad) * 100.0)
    return clamp((bad - v) / (bad - good) * 100.0)


def pct(x: Any) -> float | None:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


def load_universe(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("symbol"):
                rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows[:limit] if limit else rows


def _cache_path(symbol: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", symbol.upper())
    return CACHE_DIR / f"{safe}.json"


def _read_cache(symbol: str, max_age_hours: float) -> tuple[dict[str, Any], list[dict[str, Any]], str] | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        fetched_at = datetime.fromisoformat(payload.get("fetched_at"))
        age_h = (datetime.now(timezone.utc).astimezone() - fetched_at).total_seconds() / 3600.0
        if age_h <= max_age_hours:
            return payload.get("info") or {}, payload.get("news") or [], "ok_cache"
    except Exception:
        return None
    return None


def _write_cache(symbol: str, info: dict[str, Any], news: list[dict[str, Any]]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"fetched_at": now_iso(), "info": info, "news": news[:20]}
        _cache_path(symbol).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def fetch_yahoo(symbol: str, *, sleep_seconds: float, cache_hours: float) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    cached = _read_cache(symbol, cache_hours)
    if cached is not None:
        return cached
    if yf is None:
        return {}, [], "yfinance_unavailable"
    try:
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        ticker = yf.Ticker(symbol)
        info = ticker.get_info() or {}
        try:
            news = ticker.news or []
        except Exception:
            news = []
        _write_cache(symbol, info, news)
        return info, news, "ok"
    except Exception as exc:
        stale = _read_cache(symbol, max_age_hours=24 * 30)
        if stale is not None:
            info, news, _ = stale
            return info, news, f"stale_cache_after_fetch_error:{str(exc)[:80]}"
        return {}, [], f"fetch_error:{str(exc)[:120]}"


def news_title(item: dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""
    content = item.get("content") if isinstance(item.get("content"), dict) else {}
    return str(item.get("title") or content.get("title") or "").strip()


def score_news(news: list[dict[str, Any]]) -> tuple[float, list[str], list[str]]:
    titles = [news_title(n) for n in news if news_title(n)][:8]
    if not titles:
        return 50.0, [], ["no_recent_news_found"]
    raw = 0
    hits = []
    for title in titles:
        words = set(re.findall(r"[a-zA-Z]+", title.lower()))
        pos = words & POSITIVE_WORDS
        neg = words & NEGATIVE_WORDS
        if pos:
            raw += len(pos)
            hits.extend(sorted(pos))
        if neg:
            raw -= len(neg)
            hits.extend([f"-{x}" for x in sorted(neg)])
    score = clamp(50 + raw * 8)
    warnings = []
    if score < 40:
        warnings.append("negative_news_tone")
    return score, titles[:5], sorted(set(hits))[:12]


def stock_scores(info: dict[str, Any]) -> dict[str, float]:
    roe = pct(info.get("returnOnEquity"))
    op_margin = pct(info.get("operatingMargins"))
    profit_margin = pct(info.get("profitMargins"))
    rev_growth = pct(info.get("revenueGrowth"))
    earn_growth = pct(info.get("earningsGrowth"))
    fpe = pct(info.get("forwardPE") or info.get("trailingPE"))
    peg = pct(info.get("pegRatio"))
    debt_to_equity = pct(info.get("debtToEquity"))
    fcf = pct(info.get("freeCashflow"))
    div_yield = pct(info.get("dividendYield"))

    quality = sum([
        score_range(roe, 0.25, 0.05),
        score_range(op_margin, 0.30, 0.05),
        score_range(profit_margin, 0.22, 0.03),
        80.0 if fcf and fcf > 0 else 40.0,
    ]) / 4.0
    growth = sum([
        score_range(rev_growth, 0.20, -0.05),
        score_range(earn_growth, 0.25, -0.10),
    ]) / 2.0
    valuation = sum([
        score_range(fpe, 18, 55, higher_better=False),
        score_range(peg, 1.0, 3.5, higher_better=False),
    ]) / 2.0
    leverage = score_range(debt_to_equity, 50, 250, higher_better=False)
    dividend = score_range(div_yield, 0.025, 0.0) if div_yield is not None else 45.0
    risk = 100.0 - leverage
    return {
        "quality": quality,
        "growth": growth,
        "valuation": valuation,
        "dividend": dividend,
        "risk": risk,
        "fundamental": quality * 0.38 + growth * 0.27 + valuation * 0.22 + leverage * 0.10 + dividend * 0.03,
    }


def etf_scores(info: dict[str, Any]) -> dict[str, float]:
    expense = pct(info.get("annualReportExpenseRatio") or info.get("expenseRatio"))
    assets = pct(info.get("totalAssets") or info.get("netAssets"))
    yld = pct(info.get("yield") or info.get("dividendYield"))
    beta = pct(info.get("beta3Year") or info.get("beta"))
    expense_score = score_range(expense, 0.0003, 0.008, higher_better=False) if expense is not None else 55.0
    asset_score = score_range(math.log10(max(assets or 1, 1)), 11.0, 8.0) if assets is not None else 50.0
    dividend = score_range(yld, 0.035, 0.0) if yld is not None else 45.0
    risk = score_range(beta, 0.8, 1.5, higher_better=False) if beta is not None else 55.0
    # ETF fundamentals = structure quality, not price returns.
    fundamental = expense_score * 0.40 + asset_score * 0.30 + risk * 0.15 + dividend * 0.15
    return {
        "quality": (expense_score + asset_score) / 2.0,
        "growth": 50.0,
        "valuation": expense_score,
        "dividend": dividend,
        "risk": 100.0 - risk,
        "fundamental": fundamental,
    }


def recommendation_label(score: float, risk_score: float, data_quality: str) -> str:
    if data_quality not in {"ok", "ok_cache"}:
        return "WATCH_DATA_GAP"
    if score >= 72 and risk_score <= 60:
        return "BUY_CANDIDATE"
    if score >= 62:
        return "ACCUMULATE_WATCHLIST"
    if score >= 52:
        return "HOLD_WATCH"
    return "AVOID_FOR_NOW"


def evaluate(row: dict[str, str], *, sleep_seconds: float, cache_hours: float) -> Recommendation:
    symbol = row["symbol"].upper()
    asset_class = (row.get("asset_class") or "STOCK").upper()
    info, news, fetch_status = fetch_yahoo(symbol, sleep_seconds=sleep_seconds, cache_hours=cache_hours)
    nscore, headlines, news_hits = score_news(news)
    scores = etf_scores(info) if asset_class == "ETF" else stock_scores(info)
    total = scores["fundamental"] * 0.72 + nscore * 0.28
    risk_score = scores["risk"]
    data_quality = fetch_status if fetch_status != "ok" else "ok"
    label = recommendation_label(total, risk_score, data_quality)

    reasons = []
    warnings = []
    if scores["quality"] >= 70:
        reasons.append("strong_quality_fundamentals")
    if scores["growth"] >= 65:
        reasons.append("healthy_growth_profile")
    if scores["valuation"] >= 65:
        reasons.append("reasonable_valuation_or_low_cost_structure")
    if nscore >= 62:
        reasons.append("positive_recent_news_tone")
    if asset_class == "ETF" and scores["quality"] >= 65:
        reasons.append("ETF_structure_quality")
    if risk_score > 70:
        warnings.append("high_balance_sheet_or_structure_risk_score")
    if scores["valuation"] < 35:
        warnings.append("valuation_or_expense_ratio_unattractive")
    if nscore < 40:
        warnings.append("negative_recent_news_tone")
    if data_quality not in {"ok", "ok_cache"}:
        warnings.append(data_quality)

    metric_keys = [
        "sector", "industry", "marketCap", "forwardPE", "trailingPE", "pegRatio", "returnOnEquity",
        "operatingMargins", "profitMargins", "revenueGrowth", "earningsGrowth", "debtToEquity",
        "freeCashflow", "dividendYield", "annualReportExpenseRatio", "totalAssets", "yield", "beta",
    ]
    metrics = {k: info.get(k) for k in metric_keys if info.get(k) is not None}
    if news_hits:
        metrics["news_keyword_hits"] = news_hits

    return Recommendation(
        symbol=symbol,
        name=row.get("name") or str(info.get("longName") or info.get("shortName") or symbol),
        asset_class=asset_class,
        theme=row.get("theme") or "",
        recommendation=label,
        total_score=round(float(total), 2),
        fundamental_score=round(float(scores["fundamental"]), 2),
        news_score=round(float(nscore), 2),
        risk_score=round(float(risk_score), 2),
        valuation_score=round(float(scores["valuation"]), 2),
        quality_score=round(float(scores["quality"]), 2),
        growth_score=round(float(scores["growth"]), 2),
        dividend_score=round(float(scores["dividend"]), 2),
        data_quality=data_quality,
        reasons=reasons,
        warnings=warnings,
        metrics=metrics,
        news_headlines=headlines,
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# US Fundamental + News Recommender",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "Research-only. This uses fundamentals/news only; no technicals and no live orders.",
        "",
        "## Top candidates",
        "",
    ]
    for r in payload["ranked"][:15]:
        lines.append(f"- **{r['symbol']}** ({r['asset_class']}) — {r['recommendation']} score {r['total_score']} | F {r['fundamental_score']} / News {r['news_score']} / Risk {r['risk_score']} — {', '.join(r['reasons'][:3])}")
    lines.extend(["", "## Avoid / data gaps", ""])
    for r in payload["ranked"][-10:]:
        lines.append(f"- {r['symbol']} — {r['recommendation']} score {r['total_score']} warnings={', '.join(r['warnings'][:3])}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="US fundamentals/news-only recommender")
    p.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    p.add_argument("--limit", type=int, default=int(os.getenv("AT_US_RECOMMENDER_LIMIT", "0") or "0"))
    p.add_argument("--sleep-seconds", type=float, default=float(os.getenv("AT_US_RECOMMENDER_SLEEP_SECONDS", "1.0") or "1.0"))
    p.add_argument("--cache-hours", type=float, default=float(os.getenv("AT_US_RECOMMENDER_CACHE_HOURS", "24") or "24"))
    return p


def main() -> int:
    args = build_parser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_universe(Path(args.universe), args.limit or None)
    results = [asdict(evaluate(r, sleep_seconds=args.sleep_seconds, cache_hours=args.cache_hours)) for r in rows]
    ranked = sorted(results, key=lambda x: (x["data_quality"] in {"ok", "ok_cache"}, x["total_score"]), reverse=True)
    payload = {
        "generated_at": now_iso(),
        "engine": "us_fundamental_news_recommender",
        "data_sources": ["Tickertape public pages for availability hints", "Yahoo Finance via yfinance for fundamentals/news"],
        "exclusions": ["No technical indicators", "No price momentum inputs", "No live orders"],
        "universe_path": str(Path(args.universe).resolve()),
        "universe_count": len(rows),
        "ranked": ranked,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUT_DIR / f"us_fundamental_news_recommender_{ts}.json"
    csv_path = OUT_DIR / f"us_fundamental_news_recommender_{ts}.csv"
    md_path = OUT_DIR / f"us_fundamental_news_recommender_{ts}.md"
    latest_json = OUT_DIR / "us_fundamental_news_recommender_latest.json"
    latest_md = OUT_DIR / "us_fundamental_news_recommender_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(ranked).drop(columns=["metrics", "news_headlines"], errors="ignore").to_csv(csv_path, index=False)
    write_markdown(md_path, payload)
    write_markdown(latest_md, payload)
    print(json.dumps({
        "generated_at": payload["generated_at"],
        "universe_count": len(rows),
        "top_10": [{"symbol": r["symbol"], "asset_class": r["asset_class"], "recommendation": r["recommendation"], "score": r["total_score"]} for r in ranked[:10]],
        "saved": str(latest_json),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
