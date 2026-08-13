from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def test_research_fixture_matches_versioned_paper_signal_contract() -> None:
    ohlc = json.loads((FIXTURES / "parity_ohlc.json").read_text())
    signal = json.loads((FIXTURES / "expected_signal_v2.json").read_text())
    assert ohlc["schema_version"] == "parity_ohlc_v1"
    assert signal["schema_version"] == "paper_signal_v2_target_weights"
    assert signal["signal_date"] == ohlc["signal_date"]
    assert signal["modeled_execution_date"] == ohlc["modeled_execution_date"]
    assert signal["modeled_execution_open"] == ohlc["modeled_opens"]
    assert signal["target_weights"] == {"AAA": 0.6, "BBB": 0.3}
    assert abs(sum(signal["target_weights"].values()) + signal["target_cash_weight"] - 1.0) < 1e-12
    assert len(signal["params_fingerprint"]) == 64
    assert len(signal["signal_id"]) == 64
