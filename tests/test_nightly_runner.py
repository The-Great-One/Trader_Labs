from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_lab_worker import start_worker, stop_group


def test_worker_exports_run_id_and_records_actual_identity(tmp_path: Path) -> None:
    metadata = tmp_path / "current.json"
    observed = tmp_path / "observed.txt"
    code = "import os,pathlib; pathlib.Path(%r).write_text(os.environ['AT_LAB_RUN_ID'])" % str(observed)
    assert start_worker([sys.executable, "-c", code], metadata, "run-123") == 0
    record = json.loads(metadata.read_text())
    assert record["run_id"] == "run-123"
    assert record["pid"] > 0 and record["pgid"] > 0
    assert observed.read_text() == "run-123"


def test_stop_terminates_worker_process_group_and_clears_metadata(tmp_path: Path) -> None:
    metadata = tmp_path / "current.json"
    child = tmp_path / "child.pid"
    code = (
        "import subprocess,sys,time,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(child)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    import subprocess
    runner = subprocess.Popen([
        sys.executable, str(ROOT / "scripts" / "run_lab_worker.py"),
        "run", str(metadata), "run-stop", sys.executable, "-c", code,
    ], start_new_session=True)
    deadline = time.time() + 5
    while time.time() < deadline and (not metadata.exists() or not child.exists()):
        time.sleep(0.05)
    assert metadata.exists() and child.exists()
    assert stop_group(metadata, timeout_sec=1.0)
    runner.wait(timeout=5)
    assert not metadata.exists()
