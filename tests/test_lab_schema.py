from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lab_schema import (
    RESULT_SCHEMA_VERSION,
    canonical_json,
    canonical_params,
    params_fingerprint,
)


def test_fingerprint_ignores_labels_and_fills_omitted_defaults() -> None:
    baseline = {"rsi_periods": [22, 44, 66], "enhancement": "baseline"}
    incumbent = {"rsi_periods": (22, 44, 66), "enhancement": "champion_baseline"}
    mutation = {"rsi_periods": [22.0, 44.0, 66.0], "enhancement": "mutation_top_n=8"}

    assert params_fingerprint(baseline) == params_fingerprint(incumbent)
    assert params_fingerprint(baseline) == params_fingerprint(mutation)
    assert params_fingerprint({}) == params_fingerprint(canonical_params({}))
    assert params_fingerprint({}).startswith("sha256:")
    assert params_fingerprint({}) == "sha256:" + hashlib.sha256(canonical_json({}).encode()).hexdigest()


def test_canonicalization_normalizes_lists_numbers_and_booleans() -> None:
    left = canonical_params(
        {"rsi_periods": (22.0, 44, 66), "cost_bps": 10, "vol_weight": 1}
    )
    right = canonical_params(
        {"rsi_periods": [22, 44.0, 66], "cost_bps": 10.0, "vol_weight": True}
    )

    assert left == right
    assert left["rsi_periods"] == [22, 44, 66]
    assert left["cost_bps"] == 10.0
    assert left["vol_weight"] is True
    assert json.loads(canonical_json(left)) == left
    assert RESULT_SCHEMA_VERSION == "nightly_lab_v6_next_open_stateful"


def test_unknown_strategy_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown strategy keys.*mystery"):
        canonical_params({"mystery": 42})
