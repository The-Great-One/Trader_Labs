"""Local import bridge from Trader_Labs to the sibling Auto_Trader repo."""
from __future__ import annotations

import os
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent
AUTOTRADER_ROOT = Path(os.environ.get('AUTOTRADER_ROOT', LAB_ROOT.parent / 'Stocks')).expanduser()
if AUTOTRADER_ROOT.exists():
    p = str(AUTOTRADER_ROOT)
    if p not in sys.path:
        sys.path.append(p)
