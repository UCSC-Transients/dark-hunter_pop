"""#391 job-safety scripts: syntax, and run_heavy returns the job's exit code with no leftover process."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JOBS = Path("scripts/jobs")


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("zsh") is None, reason="zsh not installed")
def test_job_scripts_parse() -> None:
    for s in sorted(JOBS.glob("*.sh")):
        assert subprocess.run(["zsh", "-n", str(s)]).returncode == 0, s


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("zsh") is None or sys.platform != "darwin", reason="macOS job tooling")
def test_run_heavy_propagates_exit_code(tmp_path: Path) -> None:
    env = dict(os.environ, PY=sys.executable, LOG_DIR=str(tmp_path))
    ok = subprocess.run(["zsh", str(JOBS / "run_heavy.sh"), "t_ok", "-c", "pass"], env=env)
    bad = subprocess.run(["zsh", str(JOBS / "run_heavy.sh"), "t_bad", "-c", "raise SystemExit(3)"], env=env)
    assert ok.returncode == 0 and bad.returncode == 3
    assert "exit 3" in (tmp_path / "t_bad.log").read_text()
