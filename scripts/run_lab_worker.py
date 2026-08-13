"""Process-group supervisor for one nightly lab worker."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.atomic_io import atomic_write_json


def start_worker(command: list[str], metadata_path: Path, run_id: str, env: dict[str, str] | None = None) -> int:
    child_env = dict(os.environ)
    child_env.update(env or {})
    child_env["AT_LAB_RUN_ID"] = run_id
    process = subprocess.Popen(command, start_new_session=True, env=child_env)
    atomic_write_json(metadata_path, {
        "pid": process.pid,
        "pgid": os.getpgid(process.pid),
        "command": command,
        "run_id": run_id,
        "started_at": datetime.now().astimezone().isoformat(),
    })
    return process.wait()


def stop_group(metadata_path: Path, timeout_sec: float = 5.0) -> bool:
    if not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    pgid = int(metadata["pgid"])
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        metadata_path.unlink(missing_ok=True)
        return False
    import time
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            metadata_path.unlink(missing_ok=True)
            return True
        time.sleep(0.05)
    os.killpg(pgid, signal.SIGKILL)
    metadata_path.unlink(missing_ok=True)
    return True


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "stop":
        return 0 if stop_group(Path(sys.argv[2])) else 1
    if len(sys.argv) < 5 or sys.argv[1] != "run":
        raise SystemExit("usage: run_lab_worker.py {run METADATA RUN_ID COMMAND...|stop METADATA}")
    return start_worker(sys.argv[4:], Path(sys.argv[2]), sys.argv[3])


if __name__ == "__main__":
    raise SystemExit(main())
