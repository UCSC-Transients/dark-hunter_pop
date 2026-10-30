"""Shared loaders for the #589–#591 diagnostics on the colour_v2.1+tr1 luminous generations 47–50.

Mock: every draw of gens 47–50 with the default-target (MdS17) deterministic-mixture weight from
``weights_gen47_50.npz`` and selected truth columns read from the artifacts (same order as the cache). Real: DR3 C1
orbits and published accelerations from the #587 (a) plot dump (same selection and CMD as the rung-3 fit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

D = Path("/Users/rfoley/darkhunter/pop/dark-hunter_pop")
V = D / "output/proposal_set/v21tr1"
DUMP = D.parent / "dark-hunter_pop-worktrees/noise-check-587/output/nc587/a_v3theta/plot_dump.npz"
TRUTH = ("sys_mg0", "sys_colour0", "sys_evolved", "primary_mg0", "primary_colour0", "primary_log_teff", "iso_phase", "m1_msun", "m2_msun",
         "flux_ratio", "period_days", "eccentricity", "parallax_mas", "phot_g_mean_mag", "is_giant", "resolved_companion")
C1_PMAX = 830.4


def load_mock(gens=(47, 48, 49, 50)) -> dict:
    import h5py

    c = np.load(V / "weights_gen47_50.npz")
    nn = c["n_per_gen"].astype(float)
    lq = np.logaddexp.reduce(c["lqs"] + np.log(nn / nn.sum())[:, None], axis=0)
    lt = c["ll"] + c["lw"] - lq
    w = np.exp(lt - np.nanmax(lt))
    w[~np.isfinite(w)] = 0.0
    t = {k: [] for k in TRUTH}
    st = []
    for g in gens:
        with h5py.File(V / f"single_gen{g}.h5", "r") as f:
            for k in TRUTH:
                t[k].append(f["truth"][k][()])
            st.append(f["outcome/solution_type"][()])
    t = {k: np.concatenate(v) for k, v in t.items()}
    if t["sys_mg0"].size != w.size:
        raise SystemExit("artifacts do not match the weight cache")
    cas = np.asarray(c["cascade"], float)
    out = {"w": w, "cascade": cas, "accepted": np.asarray(c["accepted"], bool), "published_acceleration": np.asarray(c["published_acceleration"], bool),
           "generation": np.asarray(c["generation"]), "solution_type": np.concatenate(st), **t}
    with np.errstate(divide="ignore", invalid="ignore"):
        out["orbit"] = out["accepted"] & (cas[:, 10] <= C1_PMAX) & (out["phot_g_mean_mag"] >= 10) & (out["phot_g_mean_mag"] < 19.5) & (1 / cas[:, 0] < 6)
        out["o_P"], out["o_e"], out["o_f2"], out["o_ruwe"] = cas[:, 10], cas[:, 14], cas[:, 21], cas[:, 22]
        out["o_A"] = (cas[:, 17] / cas[:, 0]) / (cas[:, 10] / 365.25) ** (2 / 3)
        out["o_a0snr"], out["o_plxsnr"], out["o_d"] = cas[:, 17] / cas[:, 18], cas[:, 0] / cas[:, 1], 1 / cas[:, 0]
        nine = np.isclose(cas[:, 0], -9.0)
        out["a_f2"] = np.where(nine, cas[:, 13], cas[:, 9])
        out["a_ruwe"] = np.where(nine, cas[:, 12], cas[:, 8])
        out["a_d"] = 1 / cas[:, 2]
        out["accel"] = out["published_acceleration"] & (out["phot_g_mean_mag"] >= 10) & (out["phot_g_mean_mag"] < 19.5) & (out["a_d"] < 6)
    return out


def load_real() -> dict:
    z = np.load(DUMP)
    o = {k[7:]: np.asarray(z[k], float) for k in z.files if k.startswith("o_real_")}
    a = {k[7:]: np.asarray(z[k], float) for k in z.files if k.startswith("a_real_")}
    return {"orbits": o, "accelerations": a}


def ridge_and_nsigma():
    from darkhunter_pop import giants
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config

    cfg = load_config()
    prop = ps.load_proposal_set_fragment("config/population/proposal_set_single50.yaml").proposal
    parent = ps.load_parent_snapshot(D / "data/dr3/gaia_snapshots/20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj", cfg, prop)
    rd = parent.meta["ms_ridge"]
    ridge = giants.MSRidge(colour=np.asarray(rd["colour"], float), mag=np.asarray(rd["mag"], float), sigma=np.asarray(rd["sigma"], float),
                           n_rows=np.asarray(rd["n_rows"], np.int64))
    ns = float(giants.load_giants_config(Path("config/population/giants.yaml")).provisional_n_sigma)
    return ridge, ns


def classes(mg0, c0, ridge, n_sigma):
    """dwarf / subgiant (evolved, M_G0 >= 2) / giant (evolved, M_G0 < 2) / unclassified, on the dereddened CMD."""
    from darkhunter_pop import giants

    ev = giants.classify_evolved(mg0, c0, np.full(np.size(mg0), np.nan), ridge, n_sigma)
    cl = np.full(np.size(mg0), "unclassified", dtype=object)
    cl[ev.classified & ~ev.evolved] = "dwarf"
    cl[ev.evolved & (np.asarray(mg0) >= 2)] = "subgiant"
    cl[ev.evolved & (np.asarray(mg0) < 2)] = "giant"
    return cl


def kish(w):
    w = np.asarray(w, float)
    s2 = float((w * w).sum())
    return float(w.sum() ** 2 / s2) if s2 > 0 else 0.0
