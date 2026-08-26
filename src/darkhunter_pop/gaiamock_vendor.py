"""Locate, verify, and import the vendored modified-RUWE ``gaiamock_mod``.

Science code must use :func:`import_gaiamock_mod` (never stock ``gaiamock``) for RUWE /
selection-function work — see ``docs/ARCHITECTURE.md`` §1.1.

Version triple for config / run files:

* ``gaiamock_mod_release`` — immutable GitHub Release tag (default ``gaiamock-mod-v1``)
* ``gaiamock_mod_sha256`` — SHA256 of the installed ``gaiamock_mod.py``
* ``gaiamock_git_commit`` — submodule commit at ``vendor/gaiamock``
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

DEFAULT_GAIAMOCK_MOD_RELEASE = "gaiamock-mod-v1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_DIR = _REPO_ROOT / "vendor" / "gaiamock"
_OVERLAY_SRC = _REPO_ROOT / "vendor" / "overlays" / "gaiamock_mod.py"
_MANIFEST_PATH = _REPO_ROOT / "vendor" / "DATA_MANIFEST.md"


@dataclass(frozen=True)
class GaiamockModVersions:
    """Version triple recorded in config and every gaiamock-using stage."""

    gaiamock_mod_release: str
    gaiamock_mod_sha256: str
    gaiamock_git_commit: str


def repo_root() -> Path:
    return _REPO_ROOT


def vendor_dir() -> Path:
    return _VENDOR_DIR


def overlay_source_path() -> Path:
    return _OVERLAY_SRC


def installed_mod_path() -> Path:
    return _VENDOR_DIR / "gaiamock_mod.py"


def expected_sha256(filename: str) -> str:
    """Parse ``vendor/DATA_MANIFEST.md`` for the SHA256 of ``filename``."""
    text = _MANIFEST_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 4:
            continue
        if cells[1] == filename and cells[2] and cells[2] != "sha256":
            return cells[2]
    raise KeyError(f"no SHA256 for {filename!r} in {_MANIFEST_PATH}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_installed_overlay() -> str:
    """Verify installed ``gaiamock_mod.py`` matches the manifest; return its SHA256."""
    path = installed_mod_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/install_gaiamock_mod.sh"
        )
    got = sha256_file(path)
    want = expected_sha256("gaiamock_mod.py")
    if got != want:
        raise ValueError(
            f"SHA256 mismatch for gaiamock_mod.py: got {got}, want {want}"
        )
    return got


def submodule_commit() -> str:
    if not (_VENDOR_DIR / ".git").exists() and not (_VENDOR_DIR / "gaiamock.py").is_file():
        raise FileNotFoundError(f"vendor/gaiamock is missing: {_VENDOR_DIR}")
    result = subprocess.run(
        ["git", "-C", str(_VENDOR_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_versions(
    release: str = DEFAULT_GAIAMOCK_MOD_RELEASE,
) -> GaiamockModVersions:
    """Return the version triple after verifying the installed overlay checksum."""
    return GaiamockModVersions(
        gaiamock_mod_release=release,
        gaiamock_mod_sha256=verify_installed_overlay(),
        gaiamock_git_commit=submodule_commit(),
    )


def assert_versions_match(expected: GaiamockModVersions) -> None:
    """Refuse (raise) if installed gaiamock differs from ``expected`` (run-file check)."""
    current = read_versions(release=expected.gaiamock_mod_release)
    if current != expected:
        raise ValueError(
            "gaiamock version mismatch:\n"
            f"  installed={current}\n"
            f"  expected={expected}"
        )


def import_gaiamock_mod(*, verify: bool = True) -> ModuleType:
    """Import ``gaiamock_mod`` from ``vendor/gaiamock`` after optional checksum verify.

    Adds the vendor directory to ``sys.path`` if needed. Raises if the overlay or
    compiled ``kepler_solve_astrometry.so`` is missing.
    """
    if verify:
        verify_installed_overlay()
    so_path = _VENDOR_DIR / "kepler_solve_astrometry.so"
    if not so_path.is_file():
        raise FileNotFoundError(
            f"missing {so_path}; run scripts/install_gaiamock_mod.sh "
            "(requires GSL + gcc)"
        )
    vendor = str(_VENDOR_DIR)
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    return importlib.import_module("gaiamock_mod")


def is_overlay_ready() -> bool:
    """True if overlay + compiled ``.so`` are present (does not verify checksum)."""
    return installed_mod_path().is_file() and (
        _VENDOR_DIR / "kepler_solve_astrometry.so"
    ).is_file()
