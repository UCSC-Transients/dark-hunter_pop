#!/usr/bin/env bash
# Install the modified-RUWE gaiamock overlay into vendor/gaiamock.
# See docs/ARCHITECTURE.md §1.1 and vendor/gaiamock/DATA_MANIFEST.md.
#
# Expects either:
#   - local staging at mod_files/ (gaiamock_mod.py, healpix_16_med_ruwe.npz, individual_ccds/), or
#   - DOWNLOAD_FROM_RELEASE=1 with gh auth, pulling immutable tag gaiamock-mod-v1
#
# Does not yet pin the git submodule (Foundation F0). Stub install paths only.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${ROOT}/vendor/gaiamock"
MOD_FILES="${ROOT}/mod_files"
RELEASE_TAG="${GAIAMOCK_MOD_RELEASE:-gaiamock-mod-v1}"
REPO="${GAIAMOCK_MOD_REPO:-UCSC-Transients/dark-hunter_pop}"

mkdir -p "${VENDOR}/healpix_scans"

if [[ "${DOWNLOAD_FROM_RELEASE:-0}" == "1" ]]; then
  echo "Downloading ${RELEASE_TAG} assets from ${REPO} ..."
  tmp="$(mktemp -d)"
  gh release download "${RELEASE_TAG}" -R "${REPO}" -D "${tmp}"
  # Placeholders until release exists; script exits clearly if missing.
  test -f "${tmp}/gaiamock_mod.py"
  test -f "${tmp}/healpix_16_med_ruwe.npz"
  test -f "${tmp}/individual_ccds.zip"
  cp "${tmp}/gaiamock_mod.py" "${VENDOR}/"
  cp "${tmp}/healpix_16_med_ruwe.npz" "${VENDOR}/"
  unzip -qo "${tmp}/individual_ccds.zip" -d "${VENDOR}/healpix_scans"
  rm -rf "${tmp}"
elif [[ -d "${MOD_FILES}" ]]; then
  echo "Installing overlay from ${MOD_FILES} ..."
  cp "${MOD_FILES}/gaiamock_mod.py" "${VENDOR}/"
  cp "${MOD_FILES}/healpix_16_med_ruwe.npz" "${VENDOR}/"
  if [[ -d "${MOD_FILES}/individual_ccds" ]]; then
    # Copy FITS into healpix_scans/ (may already be partially populated by upstream submodule).
    cp -n "${MOD_FILES}/individual_ccds/"*.fits "${VENDOR}/healpix_scans/" 2>/dev/null || \
      cp "${MOD_FILES}/individual_ccds/"*.fits "${VENDOR}/healpix_scans/"
  else
    echo "error: ${MOD_FILES}/individual_ccds missing" >&2
    exit 1
  fi
else
  echo "error: no mod_files/ and DOWNLOAD_FROM_RELEASE!=1" >&2
  exit 1
fi

echo "Overlay files in place under ${VENDOR}."
echo "Next (Foundation): ensure git submodule checked out, then compile kepler_solve_astrometry.so"
echo "  (see upstream gaiamock README)."
