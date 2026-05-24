"""Trader_Labs lab-only Auto_Trader namespace bridge.

This package hosts lab-only modules (currently rnn_lab) and extends its module
search path to the sibling live Auto_Trader package for runtime rule/util imports.
"""
from __future__ import annotations

import os
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parents[1]
_AUTOTRADER_ROOT = Path(os.environ.get("AUTOTRADER_ROOT", _LAB_ROOT.parent / "Stocks")).expanduser()
_LIVE_PACKAGE = _AUTOTRADER_ROOT / "Auto_Trader"
if _LIVE_PACKAGE.exists():
    live_path = str(_LIVE_PACKAGE)
    if live_path not in __path__:
        __path__.append(live_path)
