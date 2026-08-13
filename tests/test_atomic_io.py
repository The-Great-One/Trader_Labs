from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.atomic_io import atomic_write_json, atomic_write_text


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    target = tmp_path / "latest.json"
    target.write_text('{"old": true}\n')
    atomic_write_json(target, {"schema_version": "v6", "run_id": "run-1"})
    assert json.loads(target.read_text()) == {"schema_version": "v6", "run_id": "run-1"}
    assert not list(tmp_path.glob(".*.tmp"))


def test_replace_failure_preserves_prior_file(tmp_path: Path) -> None:
    target = tmp_path / "history.jsonl"
    target.write_text('{"old": true}\n')
    with mock.patch("scripts.atomic_io.os.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError, match="replace failed"):
            atomic_write_text(target, '{"new": true}\n')
    assert target.read_text() == '{"old": true}\n'
