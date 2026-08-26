"""Tests for gaiamock vendor helpers.

``unit`` tests never require the overlay install. ``gaiamock`` tests skip unless
``scripts/install_gaiamock_mod.sh`` has been run locally (or CI downloads Release assets).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from darkhunter_pop import gaiamock_vendor

pytestmark_unit = pytest.mark.unit
pytestmark_gaiamock = pytest.mark.gaiamock


@pytest.mark.unit
def test_manifest_lists_required_assets() -> None:
    for name in ("gaiamock_mod.py", "healpix_16_med_ruwe.npz", "individual_ccds.zip"):
        digest = gaiamock_vendor.expected_sha256(name)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)


@pytest.mark.unit
def test_overlay_source_tracked() -> None:
    path = gaiamock_vendor.overlay_source_path()
    assert path.is_file()
    assert gaiamock_vendor.sha256_file(path) == gaiamock_vendor.expected_sha256(
        "gaiamock_mod.py"
    )


@pytest.mark.unit
def test_submodule_present() -> None:
    vendor = gaiamock_vendor.vendor_dir()
    assert (vendor / "gaiamock.py").is_file()
    assert (vendor / "kepler_solve_astrometry.c").is_file()
    commit = gaiamock_vendor.submodule_commit()
    assert len(commit) == 40


@pytest.mark.unit
def test_version_mismatch_detection(tmp_path: Path) -> None:
    if not gaiamock_vendor.is_overlay_ready():
        pytest.skip("gaiamock overlay not installed")
    current = gaiamock_vendor.read_versions()
    bogus = gaiamock_vendor.GaiamockModVersions(
        gaiamock_mod_release=current.gaiamock_mod_release,
        gaiamock_mod_sha256="0" * 64,
        gaiamock_git_commit=current.gaiamock_git_commit,
    )
    with pytest.raises(ValueError, match="gaiamock version mismatch"):
        gaiamock_vendor.assert_versions_match(bogus)


@pytest.mark.gaiamock
def test_import_gaiamock_mod_smoke() -> None:
    if not gaiamock_vendor.is_overlay_ready():
        pytest.skip("run scripts/install_gaiamock_mod.sh first")
    mod = gaiamock_vendor.import_gaiamock_mod(verify=True)
    assert hasattr(mod, "al_uncertainty_per_ccd_interp")
    assert hasattr(mod, "check_ruwe")
    assert hasattr(mod, "run_full_astrometric_cascade")
    versions = gaiamock_vendor.read_versions()
    assert versions.gaiamock_mod_release == "gaiamock-mod-v1"
    assert versions.gaiamock_mod_sha256 == gaiamock_vendor.expected_sha256(
        "gaiamock_mod.py"
    )
