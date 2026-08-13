from __future__ import annotations

import copy
import json

import pytest

import scripts.auto_iteration_lab as lab


@pytest.fixture
def valid_payload():
    return json.loads(lab.CONFIG.read_text())


@pytest.mark.parametrize("section", ["consistency", "walk_forward", "execution", "schema"])
def test_missing_required_section_is_rejected(valid_payload, section):
    valid_payload.pop(section)
    with pytest.raises(lab.ConfigError, match=section):
        lab.validate_lab_config(valid_payload)


def test_unknown_keys_are_rejected_at_every_level(valid_payload):
    for path in [(None, "mystery"), ("consistency", "mystery"), ("walk_forward", "mystery"), ("execution", "mystery"), ("schema", "mystery")]:
        payload = copy.deepcopy(valid_payload)
        section, key = path
        (payload if section is None else payload[section])[key] = 1
        with pytest.raises(lab.ConfigError, match="unknown"):
            lab.validate_lab_config(payload)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("consistency", "min_complete_years", True),
        ("consistency", "min_sharpe_ratio", float("nan")),
        ("consistency", "max_drawdown_abs_pct", float("inf")),
        ("walk_forward", "test_months", 0),
        ("walk_forward", "step_months", 5),
        ("walk_forward", "min_folds", 0),
        ("execution", "min_execution_open_coverage", 1.1),
        ("execution", "min_held_close_coverage", -0.1),
        ("execution", "max_close_ffill_rows", -1),
        ("schema", "execution_model", 7),
    ],
)
def test_wrong_types_nonfinite_and_impossible_bounds_are_rejected(valid_payload, section, key, value):
    payload = copy.deepcopy(valid_payload)
    payload[section][key] = value
    with pytest.raises(lab.ConfigError, match=key):
        lab.validate_lab_config(payload)


def test_incomplete_or_nonfinite_score_weights_are_rejected(valid_payload):
    payload = copy.deepcopy(valid_payload)
    payload["consistency"]["score_weights"].pop("sharpe")
    with pytest.raises(lab.ConfigError, match="score_weights"):
        lab.validate_lab_config(payload)
    payload = copy.deepcopy(valid_payload)
    payload["consistency"]["score_weights"]["sharpe"] = float("nan")
    with pytest.raises(lab.ConfigError, match="sharpe"):
        lab.validate_lab_config(payload)


def test_invalid_rebalance_enum_is_rejected(valid_payload):
    valid_payload["execution"]["allowed_rebalance_frequencies"] = ["DAILY"]
    with pytest.raises(lab.ConfigError, match="allowed_rebalance_frequencies"):
        lab.validate_lab_config(valid_payload)


def test_validation_happens_before_market_data_load(tmp_path, monkeypatch, valid_payload):
    config = tmp_path / "config.json"
    valid_payload["walk_forward"]["step_months"] = 1
    config.write_text(json.dumps(valid_payload))
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("market data load must not occur")

    monkeypatch.setattr(lab, "CONFIG", config)
    monkeypatch.setattr(lab, "lab_load_ohlc_prices", forbidden)
    with pytest.raises(lab.ConfigError):
        lab.main()
    assert called is False


def test_all_governance_values_are_consumed_into_typed_config(valid_payload):
    parsed = lab.validate_lab_config(valid_payload)
    assert parsed.consistency.min_complete_years == valid_payload["consistency"]["min_complete_years"]
    assert parsed.consistency.score_weights.sharpe == valid_payload["consistency"]["score_weights"]["sharpe"]
    assert parsed.walk_forward.min_folds == valid_payload["walk_forward"]["min_folds"]
    assert parsed.walk_forward.max_fold_to_median_ratio == valid_payload["walk_forward"]["max_fold_to_median_ratio"]
    assert parsed.execution.min_execution_open_coverage == valid_payload["execution"]["min_execution_open_coverage"]
    assert parsed.execution.min_held_close_coverage == valid_payload["execution"]["min_held_close_coverage"]
    assert parsed.execution.max_close_ffill_rows == valid_payload["execution"]["max_close_ffill_rows"]
    assert parsed.schema.result_schema_version == valid_payload["schema"]["result_schema_version"]
