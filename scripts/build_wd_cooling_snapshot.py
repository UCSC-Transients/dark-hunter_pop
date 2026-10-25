#!/usr/bin/env python3
"""Snapshot the Montreal WD cooling sequences (Bédard et al. 2020) into pop's data root (#552, spec §13.2).

Reads the per-mass synthetic-photometry tables ``Table_Mass_<M>`` (P. Bergeron's Montreal grid, files dated
2021-01-13, already on this host at ``--src``; nothing is downloaded) and keeps the pure-hydrogen (DA) block
of each: Teff, log g, Gaia DR3 G / BP / RP absolute magnitudes (columns ``G3``, ``G3_BP``, ``G3_RP``) and the
cooling age. Writes ``<data_root>/reference/wd_cooling/<UTC stamp>_montreal_bedard2020_DA/tracks.csv`` with
the columns ``mass_msun, age_gyr, Mg, Mbp, Mrp, teff_k, logg`` (the first three are what
``companion_nature.load_cooling_tracks`` reads) and ``meta.yaml`` with the SHA256 of every source table and of
``tracks.csv``. The source files are snapshot, never read live (CLAUDE.md).

Usage::

    .venv/bin/python scripts/build_wd_cooling_snapshot.py --src ~/stellar/wd [--data-root data]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

#: Column names of the Montreal ``Table_Mass_*`` files after joining "log g" (43 columns).
EXPECTED_COLUMNS: tuple[str, ...] = (
    "Teff", "logg", "Mbol", "BC", "U", "B", "V", "R", "I", "J", "H", "Ks", "Y", "J", "H", "K", "W1", "W2", "W3", "W4",
    "S3.6", "S4.5", "S5.8", "S8.0", "u", "g", "r", "i", "z", "g", "r", "i", "z", "y", "G2", "G2_BP", "G2_RP",
    "G3", "G3_BP", "G3_RP", "FUV", "NUV", "Age",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_da_block(path: Path) -> tuple[float, np.ndarray]:
    """(mass, rows) of the pure-hydrogen block of one ``Table_Mass_<M>`` file; rows are the 43 numeric columns."""
    lines = path.read_text().splitlines()
    head = lines[0].split()
    n_rows, mass = int(head[0]), float(head[1])
    if "Pure-hydrogen" not in lines[0]:
        raise ValueError(f"{path}: first block is not the pure-hydrogen grid: {lines[0]!r}")
    cols = tuple(lines[1].replace("log g", "logg").split())
    if cols != EXPECTED_COLUMNS:
        raise ValueError(f"{path}: unexpected columns {cols}")
    rows = np.array([[float(v) for v in ln.split()] for ln in lines[2:2 + n_rows]])
    if rows.shape != (n_rows, len(EXPECTED_COLUMNS)):
        raise ValueError(f"{path}: DA block shape {rows.shape} != ({n_rows}, {len(EXPECTED_COLUMNS)})")
    return mass, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, required=True, help="directory holding the Montreal Table_Mass_* files")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    args = ap.parse_args(argv)

    files = sorted(args.src.expanduser().glob("Table_Mass_*"), key=lambda p: float(re.sub(r"^Table_Mass_", "", p.name)))
    if not files:
        print(f"no Table_Mass_* files under {args.src}", file=sys.stderr)
        return 2
    idx = {name: EXPECTED_COLUMNS.index(name) for name in ("Teff", "logg", "G3", "G3_BP", "G3_RP", "Age")}
    out_rows = []
    for f in files:
        mass, rows = parse_da_block(f)
        for r in rows:
            out_rows.append((mass, r[idx["Age"]] / 1e9, r[idx["G3"]], r[idx["G3_BP"]], r[idx["G3_RP"]], r[idx["Teff"]], r[idx["logg"]]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.data_root / "reference" / "wd_cooling" / f"{stamp}_montreal_bedard2020_DA"
    out.mkdir(parents=True, exist_ok=False)
    csv = out / "tracks.csv"
    with csv.open("w") as fh:
        fh.write("mass_msun,age_gyr,Mg,Mbp,Mrp,teff_k,logg\n")
        for row in out_rows:
            fh.write(",".join(f"{v:.6g}" for v in row) + "\n")
    meta = {
        "issue": 552,
        "description": "Montreal synthetic photometry on the Bedard et al. (2020) cooling sequences, pure-H (DA) block; "
                       "Gaia DR3 G/BP/RP absolute magnitudes (G3, G3_BP, G3_RP) and cooling age",
        "source_dir": str(args.src.expanduser()),
        "source_files": {f.name: {"sha256": _sha256(f), "mtime_utc": datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).isoformat()}
                         for f in files},
        "masses_msun": sorted({r[0] for r in out_rows}),
        "n_rows": len(out_rows),
        "tracks_csv_sha256": _sha256(csv),
        "created_utc": stamp,
    }
    (out / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False))
    print(f"wrote {csv} ({len(out_rows)} rows, {len(files)} masses); sha256 {meta['tracks_csv_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
