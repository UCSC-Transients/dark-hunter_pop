"""Regression coverage for scripts/fetch_flame_enrichment.py's network-timeout
fix (#257 follow-up).

The installed astroquery (0.4.11) TAP+ client builds its HTTP(S) connection
via plain ``http.client`` with no ``timeout`` argument, so a stalled/dead
connection to the Gaia archive hangs forever (observed live: 3 consecutive
production launches never even returned a job id). ``_bounded_socket_timeout``
sets the process-wide ``socket`` default timeout for the duration of one
archive call — the only lever astroquery's own connection code actually
reads. These tests simulate a stalled connection with a local TCP server that
accepts but never responds, so they exercise the *real* timeout mechanism
without touching the live Gaia archive (no ``network`` marker needed).
"""

from __future__ import annotations

import http.client
import importlib.util
import socket
import threading
import time
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_flame_enrichment.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_flame_enrichment", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fetch_flame_mod() -> ModuleType:
    return _load_module()


class _StalledServer:
    """Accepts one TCP connection and never writes a response."""

    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.host, self.port = self.sock.getsockname()
        self._accepted: list[socket.socket] = []
        self._thread = threading.Thread(target=self._accept_forever, daemon=True)
        self._thread.start()

    def _accept_forever(self) -> None:
        try:
            while True:
                conn, _ = self.sock.accept()
                self._accepted.append(conn)
        except OSError:
            pass

    def close(self) -> None:
        for conn in self._accepted:
            try:
                conn.close()
            except OSError:
                pass
        try:
            self.sock.close()
        except OSError:
            pass


@pytest.fixture
def stalled_server() -> _StalledServer:
    server = _StalledServer()
    try:
        yield server
    finally:
        server.close()


def test_bounded_socket_timeout_bounds_a_stalled_read(
    fetch_flame_mod: ModuleType, stalled_server: _StalledServer
) -> None:
    """A connection that accepts but never responds must raise within the
    configured timeout, not hang — this is the exact failure mode observed
    against the live Gaia archive (no traceback, no job id, indefinite hang).
    """
    prior_default = socket.getdefaulttimeout()
    start = time.monotonic()
    with pytest.raises(fetch_flame_mod.GaiaArchiveTimeoutError):
        with fetch_flame_mod._bounded_socket_timeout(0.3):
            conn = http.client.HTTPConnection(stalled_server.host, stalled_server.port)
            conn.request("GET", "/")
            conn.getresponse()  # never returns from the stalled peer
    elapsed = time.monotonic() - start
    # Generous upper bound so this is robust on a loaded CI runner while still
    # proving it did not hang indefinitely (the bug being fixed).
    assert elapsed < 10.0
    assert socket.getdefaulttimeout() == prior_default


def test_bounded_socket_timeout_restores_default_after_success() -> None:
    mod = _load_module()
    prior_default = socket.getdefaulttimeout()
    with mod._bounded_socket_timeout(5.0):
        assert socket.getdefaulttimeout() == 5.0
    assert socket.getdefaulttimeout() == prior_default


def test_bounded_socket_timeout_restores_default_after_unrelated_error() -> None:
    mod = _load_module()
    prior_default = socket.getdefaulttimeout()
    with pytest.raises(ValueError):
        with mod._bounded_socket_timeout(5.0):
            raise ValueError("unrelated failure inside the block")
    assert socket.getdefaulttimeout() == prior_default


def test_launch_returns_error_code_without_hanging_on_stalled_connection(
    fetch_flame_mod: ModuleType, stalled_server: _StalledServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: launch() must return a clean nonzero exit code, bounded in
    wall-clock time, when the archive call stalls -- not hang the process.
    """
    import astroquery.gaia as gaia_module

    def _stalled_launch_job_async(adql: str, **kwargs: object) -> None:
        # Mirrors exactly what astroquery's own ConnectionHandler does: an
        # http.client connection built with no explicit timeout, against our
        # stalled local server standing in for a dead archive connection.
        conn = http.client.HTTPConnection(stalled_server.host, stalled_server.port)
        conn.request("GET", "/")
        conn.getresponse()  # never returns
        raise AssertionError("unreachable — the stalled read must raise first")

    monkeypatch.setattr(gaia_module.Gaia, "launch_job_async", _stalled_launch_job_async)

    start = time.monotonic()
    result = fetch_flame_mod.launch(timeout_s=0.3)
    elapsed = time.monotonic() - start

    assert result == 1
    assert elapsed < 10.0


def test_default_network_timeout_is_a_positive_finite_seconds_value(
    fetch_flame_mod: ModuleType,
) -> None:
    assert fetch_flame_mod.DEFAULT_NETWORK_TIMEOUT_S > 0.0


# --- #257 follow-up: sync-mode fallback (option 3) -------------------------
#
# The async submit/poll/retrieve sequence stalled twice more against the live
# archive even with the timeout in place (initial HTTP response fine, chunked
# body read hung). Gaia.launch_job (sync) with an explicit large TOP n
# sidesteps that path entirely -- confirmed live: 443205 real NSS rows in
# ~37s via TOP 500000, vs. a bare SELECT silently capped at 2000 rows.


class _FakeSyncJob:
    def __init__(self, table: "Table") -> None:
        self._table = table

    def get_results(self) -> "Table":
        return self._table


def _make_table(n_rows: int, *, with_flame: bool = True) -> "Table":
    from astropy.table import Table

    return Table(
        {
            "source_id": list(range(n_rows)),
            "nss_solution_type": ["Orbital"] * n_rows,
            "mass_flame": [0.9 if with_flame else None] * n_rows,
            "mass_flame_upper": [0.95 if with_flame else None] * n_rows,
            "mass_flame_lower": [0.85 if with_flame else None] * n_rows,
        }
    )


def test_sync_fetch_writes_query_and_meta(
    fetch_flame_mod: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import astroquery.gaia as gaia_module

    table = _make_table(50)
    monkeypatch.setattr(fetch_flame_mod, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        gaia_module.Gaia, "launch_job", lambda adql, **kw: _FakeSyncJob(table)
    )

    result = fetch_flame_mod.sync_fetch(top_n=1000, timeout_s=5.0)

    assert result == 0
    out_dir = tmp_path / "data" / "dr3" / "gaia_snapshots" / "flame_enrichment"
    assert (out_dir / "query.ecsv").is_file()
    meta = (out_dir / "meta.yaml").read_text(encoding="utf-8")
    assert "row_count: 50" in meta
    assert "n_with_mass_flame: 50" in meta
    assert "TOP 1000" in meta


def test_sync_fetch_warns_when_result_size_reaches_top_n(
    fetch_flame_mod: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the result size reaches top_n, the true count may be larger and
    truncated -- must warn rather than silently write a partial cache.
    """
    import astroquery.gaia as gaia_module

    table = _make_table(1000)
    monkeypatch.setattr(fetch_flame_mod, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        gaia_module.Gaia, "launch_job", lambda adql, **kw: _FakeSyncJob(table)
    )

    result = fetch_flame_mod.sync_fetch(top_n=1000, timeout_s=5.0)

    assert result == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "may be truncated" in captured.err


def test_sync_fetch_returns_error_code_without_hanging_on_stalled_connection(
    fetch_flame_mod: ModuleType,
    stalled_server: _StalledServer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import astroquery.gaia as gaia_module

    def _stalled_launch_job(adql: str, **kwargs: object) -> None:
        conn = http.client.HTTPConnection(stalled_server.host, stalled_server.port)
        conn.request("GET", "/")
        conn.getresponse()  # never returns
        raise AssertionError("unreachable — the stalled read must raise first")

    monkeypatch.setattr(fetch_flame_mod, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(gaia_module.Gaia, "launch_job", _stalled_launch_job)

    start = time.monotonic()
    result = fetch_flame_mod.sync_fetch(top_n=1000, timeout_s=0.3)
    elapsed = time.monotonic() - start

    assert result == 1
    assert elapsed < 10.0


def test_default_sync_top_n_comfortably_exceeds_known_nss_row_count(
    fetch_flame_mod: ModuleType,
) -> None:
    # Live measurement 2026-09-26: 443205 rows. Default must clear that with
    # ample headroom for catalog growth.
    assert fetch_flame_mod.DEFAULT_SYNC_TOP_N > 443_205 * 2
