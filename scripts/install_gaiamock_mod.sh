#!/usr/bin/env bash
# Install the modified-RUWE gaiamock overlay into vendor/gaiamock (git submodule).
# See docs/ARCHITECTURE.md §1.1 and vendor/DATA_MANIFEST.md.
#
# Usage:
#   scripts/install_gaiamock_mod.sh              # from mod_files/ if present, else Release
#   DOWNLOAD_FROM_RELEASE=1 scripts/install_gaiamock_mod.sh
#   SKIP_COMPILE=1 scripts/install_gaiamock_mod.sh
#
# Env:
#   GAIAMOCK_MOD_RELEASE  default gaiamock-mod-v1
#   GAIAMOCK_MOD_REPO     default UCSC-Transients/dark-hunter_pop
#   GSL_PREFIX            optional prefix containing include/ and lib/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${ROOT}/vendor/gaiamock"
OVERLAY_PY="${ROOT}/vendor/overlays/gaiamock_mod.py"
MOD_FILES="${ROOT}/mod_files"
MANIFEST="${ROOT}/vendor/DATA_MANIFEST.md"
RELEASE_TAG="${GAIAMOCK_MOD_RELEASE:-gaiamock-mod-v1}"
REPO="${GAIAMOCK_MOD_REPO:-UCSC-Transients/dark-hunter_pop}"

die() { echo "error: $*" >&2; exit 1; }

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

expected_sha() {
  local fname="$1"
  # Manifest table rows: | file | sha256 | notes |
  awk -F'|' -v f="$fname" '
    $0 ~ /^\|/ && $2 ~ f {
      gsub(/^[ \t]+|[ \t]+$/, "", $3);
      print $3;
      exit
    }' "${MANIFEST}"
}

verify_sha() {
  local path="$1" fname="$2"
  local want got
  want="$(expected_sha "$fname")"
  [[ -n "$want" ]] || die "no SHA256 for ${fname} in ${MANIFEST}"
  got="$(sha256_file "$path")"
  [[ "$got" == "$want" ]] || die "SHA256 mismatch for ${fname}: got ${got}, want ${want}"
  echo "verified ${fname}: ${got}"
}

[[ -d "${VENDOR}/.git" || -f "${VENDOR}/.git" ]] || die "vendor/gaiamock is not a git submodule; run: git submodule update --init"
[[ -f "${VENDOR}/kepler_solve_astrometry.c" ]] || die "upstream gaiamock incomplete under ${VENDOR}"
[[ -f "${OVERLAY_PY}" ]] || die "missing tracked overlay ${OVERLAY_PY}"

mkdir -p "${VENDOR}/healpix_scans"

tmp=""
cleanup() { [[ -n "${tmp}" && -d "${tmp}" ]] && rm -rf "${tmp}"; }
trap cleanup EXIT

if [[ "${DOWNLOAD_FROM_RELEASE:-0}" == "1" ]] || [[ ! -d "${MOD_FILES}/individual_ccds" ]]; then
  command -v gh >/dev/null 2>&1 || die "gh CLI required for Release download"
  echo "Downloading ${RELEASE_TAG} assets from ${REPO} ..."
  tmp="$(mktemp -d)"
  gh release download "${RELEASE_TAG}" -R "${REPO}" -D "${tmp}"
  test -f "${tmp}/gaiamock_mod.py" || die "release missing gaiamock_mod.py"
  test -f "${tmp}/healpix_16_med_ruwe.npz" || die "release missing healpix_16_med_ruwe.npz"
  test -f "${tmp}/individual_ccds.zip" || die "release missing individual_ccds.zip"
  verify_sha "${tmp}/gaiamock_mod.py" "gaiamock_mod.py"
  verify_sha "${tmp}/healpix_16_med_ruwe.npz" "healpix_16_med_ruwe.npz"
  verify_sha "${tmp}/individual_ccds.zip" "individual_ccds.zip"
  cp "${tmp}/gaiamock_mod.py" "${VENDOR}/gaiamock_mod.py"
  cp "${tmp}/healpix_16_med_ruwe.npz" "${VENDOR}/healpix_16_med_ruwe.npz"
  unzip -qo "${tmp}/individual_ccds.zip" -d "${VENDOR}/healpix_scans"
else
  echo "Installing overlay from ${MOD_FILES} ..."
  verify_sha "${MOD_FILES}/gaiamock_mod.py" "gaiamock_mod.py"
  verify_sha "${MOD_FILES}/healpix_16_med_ruwe.npz" "healpix_16_med_ruwe.npz"
  # Prefer tracked overlay as install source of truth when present.
  cp "${OVERLAY_PY}" "${VENDOR}/gaiamock_mod.py"
  cp "${MOD_FILES}/healpix_16_med_ruwe.npz" "${VENDOR}/healpix_16_med_ruwe.npz"
  shopt -s nullglob
  fits=( "${MOD_FILES}/individual_ccds/"*.fits )
  [[ ${#fits[@]} -gt 0 ]] || die "${MOD_FILES}/individual_ccds has no FITS"
  cp "${fits[@]}" "${VENDOR}/healpix_scans/"
  shopt -u nullglob
fi

# Prefer tracked overlay over release copy so repo edits win.
cp "${OVERLAY_PY}" "${VENDOR}/gaiamock_mod.py"
verify_sha "${VENDOR}/gaiamock_mod.py" "gaiamock_mod.py"
verify_sha "${VENDOR}/healpix_16_med_ruwe.npz" "healpix_16_med_ruwe.npz"

n_fits="$(find "${VENDOR}/healpix_scans" -maxdepth 1 -name 'healpix_16_*.fits' | wc -l | tr -d ' ')"
[[ "${n_fits}" -ge 3072 ]] || die "expected >=3072 healpix_16_*.fits in healpix_scans, found ${n_fits}"
echo "healpix_scans: ${n_fits} FITS files"

compile_so() {
  local c_src="${VENDOR}/kepler_solve_astrometry.c"
  local so_out="${VENDOR}/kepler_solve_astrometry.so"
  local -a flags=(-shared -o "${so_out}" "${c_src}" -lgsl -lgslcblas -lm -fPIC)
  if [[ -n "${GSL_PREFIX:-}" ]]; then
    flags=(-shared -o "${so_out}" "${c_src}" "-I${GSL_PREFIX}/include" "-L${GSL_PREFIX}/lib" -lgsl -lgslcblas -lm -fPIC)
  elif command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gsl; then
    # shellcheck disable=SC2207
    flags=(-shared -o "${so_out}" "${c_src}" $(pkg-config --cflags --libs gsl) -lm -fPIC)
  elif [[ -d /opt/homebrew/opt/gsl ]]; then
    flags=(-shared -o "${so_out}" "${c_src}" -I/opt/homebrew/opt/gsl/include -L/opt/homebrew/opt/gsl/lib -lgsl -lgslcblas -lm -fPIC)
  elif [[ -d /usr/local/opt/gsl ]]; then
    flags=(-shared -o "${so_out}" "${c_src}" -I/usr/local/opt/gsl/include -L/usr/local/opt/gsl/lib -lgsl -lgslcblas -lm -fPIC)
  fi
  echo "Compiling kepler_solve_astrometry.so ..."
  (cd "${VENDOR}" && gcc "${flags[@]}") || die "gcc compile failed; install GSL and/or set GSL_PREFIX"
  [[ -f "${so_out}" ]] || die "compile produced no ${so_out}"
  echo "compiled ${so_out}"
}

if [[ "${SKIP_COMPILE:-0}" != "1" ]]; then
  compile_so
else
  echo "SKIP_COMPILE=1: leaving .so untouched"
fi

echo "gaiamock_mod_release=${RELEASE_TAG}"
echo "gaiamock_git_commit=$(git -C "${VENDOR}" rev-parse HEAD)"
echo "Install complete. Import via darkhunter_pop.gaiamock_vendor.import_gaiamock_mod()"
