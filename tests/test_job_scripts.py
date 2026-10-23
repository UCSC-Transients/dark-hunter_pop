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


def _run_guard(tmp_path: Path, env_over: dict, wait_s: float) -> tuple[str, bool]:
    import time

    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    pf, log = tmp_path / "job.pid", tmp_path / "job.log"
    pf.write_text(str(job.pid))
    env = dict(os.environ, INTERVAL="1", **env_over)
    g = subprocess.Popen(["zsh", str(JOBS / "mem_guard.sh"), str(pf), "time.sleep", str(log)], env=env)
    time.sleep(wait_s)
    alive = job.poll() is None
    job.kill()
    g.kill()
    return log.read_text() if log.exists() else "", alive


BASE = {"MG_SWAP_MB": "1000", "MG_FREE_PCT": "50", "MG_PRESSURE": "1", "MG_TREE_RSS_MB": "100", "MG_RAM_MB": "65536"}


@pytest.mark.unit
@pytest.mark.skipif(shutil.which("zsh") is None or sys.platform != "darwin", reason="macOS job tooling")
def test_mem_guard_v4_decisions(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    log, alive = _run_guard(tmp_path / "a", BASE, 3)
    assert alive and "MEM_GUARD STOP" not in log  # benign: no stop
    (tmp_path / "b").mkdir()
    log, alive = _run_guard(tmp_path / "b", {**BASE, "MG_FREE_PCT": "5"}, 3)
    assert not alive and "free memory 5%" in log
    (tmp_path / "c").mkdir()
    log, alive = _run_guard(tmp_path / "c", {**BASE, "MG_PRESSURE": "4"}, 3)
    assert not alive and "pressure critical" in log
    for name, rss_late, expect_stop in (("d", "100", False), ("e", "40000", True), ("f", "5000", True)):
        d = tmp_path / name
        d.mkdir()
        sw, rs = d / "swap", d / "rss"
        sw.write_text("1000")
        rs.write_text("100")
        import threading
        import time as _t

        def grow() -> None:
            _t.sleep(1.5)
            sw.write_text("6000")  # +5 GB of swap
            rs.write_text(rss_late)

        threading.Thread(target=grow, daemon=True).start()
        env = {k: v for k, v in BASE.items() if k not in ("MG_SWAP_MB", "MG_TREE_RSS_MB")}
        log, alive = _run_guard(d, {**env, "MG_SWAP_FILE": str(sw), "MG_RSS_FILE": str(rs)}, 4)
        # d: swap from other apps only -> keep running; e: tree > 50% RAM -> stop; f: tree grew > 4 GB -> stop
        assert alive is (not expect_stop), (name, log)
        if expect_stop:
            assert "swap +5000 MB" in log
