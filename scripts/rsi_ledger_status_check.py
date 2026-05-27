#!/usr/bin/env python3
"""Status check script for RSI Momentum paper ledger — runs via Hermes cron.
Reads the ledger state from the primary Oracle server and outputs a summary.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Paths
REPORTS_DIR = Path("/Users/sahilgoel/Desktop/Stocks/reports")
LOCAL_LEDGER = REPORTS_DIR / "paper_ledger_rsi_momentum_local.json"
OUTPUT_FILE = Path("/Users/sahilgoel/.hermes/cron/output/rsi_ledger_status.txt")

# Server config
AT_HOST = os.environ.get("AT_SERVER_HOST", "ubuntu@168.138.114.147")
AT_KEY = os.environ.get("AT_SERVER_KEY", "/Users/sahilgoel/Desktop/Sahil_Oracle_Keys/ssh-key-2024-10-12.key")


def fetch_remote_ledger() -> dict | None:
    """SSH into primary server and fetch the latest ledger JSON."""
    try:
        result = subprocess.run(
            [
                "ssh", "-i", AT_KEY,
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                AT_HOST,
                "cat /home/ubuntu/Auto_Trader/reports/paper_ledger_rsi_momentum_latest.json",
            ],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception as e:
        print(f"ERROR fetching remote: {e}", file=sys.stderr)
    return None


def generate_summary(data: dict) -> str:
    """Produce a human-readable Telegram-friendly summary."""
    lines = []

    port = data.get("portfolio", {})
    metrics = data.get("metrics", {})
    signal = data.get("signal", {})

    lines.append("📊 **RSI+Momentum Paper Ledger**")
    lines.append(f"*{datetime.now().strftime('%d %b %Y, %H:%M IST')}*")
    lines.append("")

    # Portfolio
    total = port.get("total_value", 0)
    cash = port.get("cash", 0)
    pos_ct = port.get("positions_count", 0)
    last_reb = port.get("last_rebalance", "N/A")

    lines.append(f"💰 Portfolio: ₹{total:,.0f}")
    lines.append(f"   Cash: ₹{cash:,.0f} | Positions: {pos_ct} | Last rebalance: {last_reb}")

    # Metrics
    if metrics:
        cagr = metrics.get("cagr_pct", 0)
        dd = metrics.get("max_drawdown_pct", 0)
        total_ret = metrics.get("total_return_pct", 0)
        sharpe = metrics.get("sharpe", 0)
        days = metrics.get("days_tracked", 0)
        lines.append("")
        lines.append(f"📈 Return: {total_ret:+.2f}% | CAGR: {cagr:+.2f}% | MaxDD: {dd:+.2f}%")
        lines.append(f"   Sharpe: {sharpe:.3f} | Days tracked: {days}")

    # Positions
    positions = port.get("positions", {})
    if positions:
        lines.append("")
        lines.append("🎯 **Open Positions:**")
        for sym, pos in sorted(positions.items(), key=lambda x: x[1].get("pnl_pct", 0), reverse=True):
            pnl = pos.get("pnl_pct", 0)
            mv = pos.get("market_value", 0)
            emoji = "🟢" if pnl > 0 else ("🔴" if pnl < -5 else "🟡")
            lines.append(f"  {emoji} {sym}: ₹{mv:,.0f} ({pnl:+.1f}%)")

    # Signal
    picks = signal.get("picks", [])
    if picks:
        lines.append("")
        lines.append(f"🔮 Signal ({signal.get('date', 'N/A')}): {', '.join(picks[:6])}" + ("..." if len(picks) > 6 else ""))

    # Monthly returns
    monthly = metrics.get("monthly_returns", {})
    if monthly:
        lines.append("")
        lines.append("📅 Monthly returns:")
        for month, ret in sorted(monthly.items())[-6:]:
            emoji = "🟢" if ret > 0 else "🔴"
            lines.append(f"  {emoji} {month}: {ret:+.1f}%")

    return "\n".join(lines)


def main():
    data = fetch_remote_ledger()
    if data is None:
        print("ERROR: could not fetch remote ledger", file=sys.stderr)
        return 1

    summary = generate_summary(data)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
