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
