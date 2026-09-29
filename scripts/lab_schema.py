"""Canonical strategy parameter schema and stable identity for the nightly lab."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any

RESULT_SCHEMA_VERSION = "nightly_lab_v6_next_open_stateful"

STRATEGY_DEFAULTS: dict[str, Any] = {
    "rsi_periods": [22, 44, 66],
    "momentum_period": 63,
    "regime_mode": "sma100",
    "use_macd": True,
    "top_n": 8,
    "rebalance_freq": "3W-FRI",
    "cost_bps": 10.0,
    "max_per_sector": 3,
    "blend_weight": 0.3,
    "vol_weight": True,
    "vol_lookback": 10,
    "use_universe_regime": False,
    "universe_regime_mode": "universe_sma100",
    "rsi_min": 0.0,
    "rsi_max": 100.0,
    "use_rsi_accel": False,
    "rsi_accel_weight": 0.15,
    "turnover_penalty": 0.0,
    "dynamic_top_n": False,
    "dd_exit_pct": 0.0,
}

_INT_KEYS = {"momentum_period", "top_n", "max_per_sector", "vol_lookback"}
_FLOAT_KEYS = {
    "cost_bps", "blend_weight", "rsi_min", "rsi_max", "rsi_accel_weight",
    "turnover_penalty", "dd_exit_pct",
}
_BOOL_KEYS = {
    "use_macd", "vol_weight", "use_universe_regime", "use_rsi_accel",
    "dynamic_top_n",
}
_STR_KEYS = {"regime_mode", "rebalance_freq", "universe_regime_mode"}
_METADATA_KEYS = {
    "enhancement", "label", "display_name", "run_id", "run_date",
    "params_fingerprint", "retest_role", "schema_version", "instruments",
}


def _finite_number(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"strategy parameter {key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"strategy parameter {key} must be finite")
    return number


def canonical_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Validate, fill defaults and normalize strategy values only."""
    supplied = dict(params or {})
    unknown = set(supplied) - set(STRATEGY_DEFAULTS) - _METADATA_KEYS
    if unknown:
        raise ValueError(f"unknown strategy keys: {sorted(unknown)}")
    values = {**STRATEGY_DEFAULTS}
    values.update({key: value for key, value in supplied.items() if key in STRATEGY_DEFAULTS})

    periods = values["rsi_periods"]
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("strategy parameter rsi_periods must be a non-empty list")
    normalized_periods: list[int] = []
    for value in periods:
        number = _finite_number("rsi_periods", value)
        if number != int(number) or number <= 0:
            raise ValueError("strategy parameter rsi_periods must contain positive integers")
        normalized_periods.append(int(number))
    values["rsi_periods"] = normalized_periods

    for key in _INT_KEYS:
        number = _finite_number(key, values[key])
        if number != int(number):
            raise ValueError(f"strategy parameter {key} must be an integer")
        values[key] = int(number)
    for key in _FLOAT_KEYS:
        values[key] = _finite_number(key, values[key])
    for key in _BOOL_KEYS:
        value = values[key]
        if value not in (True, False, 0, 1):
            raise ValueError(f"strategy parameter {key} must be boolean")
        values[key] = bool(value)
    for key in _STR_KEYS:
        if not isinstance(values[key], str) or not values[key]:
            raise ValueError(f"strategy parameter {key} must be a non-empty string")
    return {key: values[key] for key in sorted(values)}


def canonical_json(params: dict[str, Any] | None) -> str:
    return json.dumps(canonical_params(params), sort_keys=True, separators=(",", ":"), allow_nan=False)


def params_fingerprint(params: dict[str, Any] | None) -> str:
    digest = hashlib.sha256(canonical_json(params).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
