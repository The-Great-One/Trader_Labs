#!/usr/bin/env python3
"""One-shot master validation runner for RSI+momentum research.

Runs the full stack in sequence:
- official report
- quick WF scan
- robustness report
- paper-shadow parity check

Produces a master summary JSON in reports/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(exist_ok=True)


def run(cmd: list[str], cwd: Path) -> dict:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run master RSI+momentum validation stack")
    parser.add_argument("--python", default="/Users/sahilgoel/Desktop/Stocks/venv/bin/python3")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    py = args.python
    steps = []

    commands = [
        [py, "scripts/rsi_momentum_report.py", "--top-n", str(args.top_n), "--cost-bps", str(args.cost_bps)],
        [py, "scripts/rsi_momentum_wf_quick.py", "--top-n", str(args.top_n)],
        [py, "scripts/rsi_momentum_robustness_report.py", "--top-n", str(args.top_n), "--cost-bps", str(args.cost_bps)],
        [py, "scripts/rsi_momentum_parity_check.py"],
    ]

    for cmd in commands:
        result = run(cmd, ROOT)
        steps.append(result)
        if result["exit_code"] != 0:
            summary = {
                "generated_at": datetime.utcnow().isoformat(),
                "status": "failed",
                "failed_step": cmd,
                "steps": steps,
            }
            latest = OUT_DIR / "rsi_momentum_master_validation_latest.json"
            latest.write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary, indent=2))
            return 1

    report = read_json(OUT_DIR / "rsi_momentum_latest.json")
    robust = read_json(OUT_DIR / "rsi_momentum_robustness_latest.json")
    parity = read_json(OUT_DIR / "rsi_momentum_parity_check_latest.json")

    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "status": "ok",
        "params": {
            "top_n": args.top_n,
            "cost_bps": args.cost_bps,
            "python": py,
        },
        "headline": {
            "official_xirr_pct": report["in_sample"]["xirr_pct"],
            "official_cagr_pct": report["in_sample"]["cagr_pct"],
            "official_max_drawdown_pct": report["in_sample"]["max_drawdown_pct"],
            "wf_positive_folds": report["walk_forward"]["summary"]["positive_folds"],
            "wf_fold_count": report["walk_forward"]["summary"]["fold_count"],
            "wf_worst_test_cagr_pct": report["walk_forward"]["summary"]["worst_test_cagr_pct"],
            "robust_mc_p50_cagr_pct": robust["monte_carlo_monthly_bootstrap"]["cagr_pct_p50"],
            "robust_mc_p5_cagr_pct": robust["monte_carlo_monthly_bootstrap"]["cagr_pct_p5"],
            "robust_pct_sims_above_30": robust["monte_carlo_monthly_bootstrap"]["pct_sims_above_30_cagr"],
            "paper_shadow_xirr_pct": parity["paper_shadow"]["latest_result"]["backtest_metrics"]["xirr_pct"],
            "paper_shadow_exact_pick_match": parity["latest_pick_parity"]["exact_match"],
            "paper_shadow_pick_overlap": parity["latest_pick_parity"]["overlap_count"],
        },
        "artifact_paths": {
            "official_report": str(OUT_DIR / "rsi_momentum_latest.json"),
            "robustness_report": str(OUT_DIR / "rsi_momentum_robustness_latest.json"),
            "parity_report": str(OUT_DIR / "rsi_momentum_parity_check_latest.json"),
        },
        "steps": steps,
    }

    latest = OUT_DIR / "rsi_momentum_master_validation_latest.json"
    stamped = OUT_DIR / f"rsi_momentum_master_validation_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    latest.write_text(json.dumps(summary, indent=2))
    stamped.write_text(json.dumps(summary, indent=2))

    print(f"Saved: {latest}")
    print(f"Saved: {stamped}")
    print(
        f"Official XIRR={summary['headline']['official_xirr_pct']:.2f}% | "
        f"WF={summary['headline']['wf_positive_folds']}/{summary['headline']['wf_fold_count']} positive | "
        f"MC>30={summary['headline']['robust_pct_sims_above_30']:.1f}% | "
        f"Paper exact match={summary['headline']['paper_shadow_exact_pick_match']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
