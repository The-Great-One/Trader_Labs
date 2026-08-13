from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.auto_iteration_lab import (
    HistoryFormatError,
    _archive_legacy_history,
    _read_history_path,
    _write_history,
)
from scripts.lab_schema import RESULT_SCHEMA_VERSION


def _row(run_id: str = "run-1") -> dict:
    return {"schema_version": RESULT_SCHEMA_VERSION, "run_id": run_id, "params": {}}


def test_legacy_history_is_archived_once_and_new_history_is_current_only(tmp_path: Path) -> None:
    history = tmp_path / "auto_iteration_history.jsonl"
    history.write_text('{"legacy": true}\n')
    archived = _archive_legacy_history(history, tmp_path / "archive", "20260813T120000")
    assert archived is not None and archived.exists()
    assert not history.exists()
    assert _archive_legacy_history(history, tmp_path / "archive", "20260813T120001") is None
    _write_history(history, [_row()])
    assert _read_history_path(history) == [_row()]


def test_malformed_current_history_reports_exact_line_and_is_unchanged(tmp_path: Path) -> None:
    history = tmp_path / "auto_iteration_history.jsonl"
    original = json.dumps(_row()) + "\n{broken\n"
    history.write_text(original)
    with pytest.raises(HistoryFormatError, match="line 2"):
        _read_history_path(history)
    assert history.read_text() == original


def test_history_write_requires_current_schema_and_matching_run_id(tmp_path: Path) -> None:
    history = tmp_path / "auto_iteration_history.jsonl"
    with pytest.raises(HistoryFormatError):
        _write_history(history, [{"schema_version": "legacy", "run_id": "run-1"}])
    _write_history(history, [_row("same"), _row("same")])
    assert {row["run_id"] for row in _read_history_path(history)} == {"same"}
