#!/usr/bin/env python3
"""#646: π̂ per real DR3 orbit (Orbital + AstroSpectroSB1) from a §12.16 calibration, for rung 5 (``--pihat``).

π̂ is evaluated exactly as the rung-3 fit does (``rung3_model.spurious_probability``, clamp to the calibrated range,
median fill for a non-finite covariate) on the calibration's own covariates: log10(ϖ/σ_ϖ) and observed BP−RP for the
four-sample template. Writes an npz with ``source_id`` and ``pihat`` (plus the calibration's SHA256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

from darkhunter_pop.rung3_model import spurious_probability

COV = {"log10_parallax_snr": lambda c: np.log10(c("parallax") / c("parallax_error")),
       "bp_rp": lambda c: c("bp_mag") - c("rp_mag"),
       "log10_a0_snr": lambda c: np.log10(c("significance")),
       "g_mag": lambda c: c("g_mag")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--types", nargs="+", default=["Orbital", "AstroSpectroSB1"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    cal = json.loads(a.calibration.read_text())
    missing = [k for k in cal["pi_keys"] if k not in COV]
    if missing:
        raise SystemExit(f"no definition for π covariates {missing}")
    t = Table.read(a.real_snapshot, format="ascii.ecsv")
    t = t[np.isin(np.asarray(t["nss_solution_type"]).astype(str), a.types)]

    def col(n: str) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(t[n], float), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        cov = {k: COV[k](col) for k in cal["pi_keys"]}
    pi = spurious_probability(cal, cov, extrapolation="clamp")
    sid = np.asarray(t["source_id"], np.int64)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, source_id=sid, pihat=pi, calibration_sha256=np.asarray(hashlib.sha256(a.calibration.read_bytes()).hexdigest()),
                        pi_keys=np.asarray(cal["pi_keys"]))
    print(f"{sid.size} orbits ({', '.join(a.types)}); π̂ covariates {cal['pi_keys']}; Σπ̂ {pi.sum():.0f}; mean {pi.mean():.4f} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
