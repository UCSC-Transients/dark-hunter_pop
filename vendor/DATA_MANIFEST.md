# gaiamock modified-RUWE data manifest
#
# Upstream submodule: vendor/gaiamock → https://github.com/kareemelbadry/gaiamock (MIT)
# Tracked overlay source: vendor/overlays/gaiamock_mod.py
# Install copies overlay + Release/mod_files data into the submodule working tree.
#
# Release tag (immutable): gaiamock-mod-v1
# https://github.com/UCSC-Transients/dark-hunter_pop/releases/tag/gaiamock-mod-v1
#
# Do NOT host the default healpix_scans.zip (~984 MB).
#
# SHA256:

| file | sha256 | notes |
|---|---|---|
| individual_ccds.zip | c3ba58c51be373ee7d1e3ddf6c30e4a286d417684e4a2639c2f3727ee0c8b09f | ~108 MB zip; 3072 FITS unpacked |
| healpix_16_med_ruwe.npz | bd27146c02e90cecc97348f15bdf74f0d4888b7aa2313fd10c8b79dc0c0a7cca | |
| gaiamock_mod.py | 71454d768f5eb5514555d1cee716a3eab89121815d55fb561ba13f013d7ef028 | tracked at `vendor/overlays/gaiamock_mod.py` |

Install: `scripts/install_gaiamock_mod.sh` (or `DOWNLOAD_FROM_RELEASE=1`).
Config/run must record: `gaiamock_mod_release`, `gaiamock_mod_sha256`, `gaiamock_git_commit`
(see `darkhunter_pop.gaiamock_vendor.GaiamockModVersions`).
