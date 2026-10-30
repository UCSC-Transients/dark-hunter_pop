#!/usr/bin/env python3
"""#589 (part 2): is the a0/σ_a0–RUWE edge a median offset or a missing per-orbit dispersion?

Same orbits (#390 published orbits, ``reinject_v21tr1``, 2 realizations each under colour_v2.1+tr1):

* per-orbit scatter of each term of log z = log a0 − log σ_a0 − ½ log(RUWE² − 1) (and log σ_ϖ): the mock realization
  noise σ_mock = MAD(m0 − m1)/√2, and the DR3 dispersion about the mock mean, σ_DR3 = √(MAD(d − m̄)² − σ_mock²/2);
  the extra per-source dispersion DR3 has and the mock lacks is √(σ_DR3² − σ_mock²);
* how much of the DR3−mock σ_a0 scatter the F2 inflation factor c, the coupling to (e, P, T_p) (DR3 f_nl, Δ log σ_P,
  Δ log σ_e) and the orbit covariates explain (linear regression);
* closure: smear the gens 47–50 population z by the measured extra dispersion (Gaussian in log, per G range) and
  compare the 50th / 90th / 99th percentiles of z with DR3 per G bin.

Writes ``dispersion.json`` to ``--out-dir``. Analysis only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_edge_589 as de  # noqa: E402

MAD = lambda x: float(1.4826 * np.nanmedian(np.abs(x - np.nanmedian(x))))  # noqa: E731


def logc(F2, nu):
    """log10 of the Halbwachs et al. (2023) goodness-of-fit inflation c from F2 and ν."""
    x = (F2 * np.sqrt(2 / (9 * nu)) + 1 - 2 / (9 * nu)) ** 3
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * np.log10(x / (1 - 2 / (9 * nu)) ** 3)


def wq(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), x.size - 1)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--published", type=Path, default=de.D / "output/gate390/published_orbits.h5")
    ap.add_argument("--reinject", type=Path, default=de.D.parent / "dark-hunter_pop-worktrees/epoch-581/output/gate_noise_colour/reinject_v21tr1/inject.jsonl")
    ap.add_argument("--cache", type=Path, default=de.D / "output/proposal_set/v21tr1/weights_gen47_50.npz")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    import h5py

    with h5py.File(a.published, "r") as f:
        pub = {k: f["published"][k][()] for k in f["published"]}
    pub["nss_solution_type"] = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in pub["nss_solution_type"]])
    nrow = pub["source_id"].size
    ix = {int(s): i for i, s in enumerate(pub["source_id"])}
    by: dict = {}
    for line in a.reinject.read_text().splitlines():
        r = json.loads(line)
        if r["accepted"]:
            by.setdefault(r["source_id"], {})[r["realization"]] = r
    pairs = [(s, v[0], v[1]) for s, v in by.items() if 0 in v and 1 in v]
    i = np.array([ix[s] for s, _, _ in pairs])
    sub = {k: (v[i] if np.ndim(v) and len(v) == nrow else v) for k, v in pub.items()}
    cb = de.dr3_covariance_block(sub)
    a0d, _ = de.halbwachs_t(*(sub[x] for x in de.TI))
    g = sub["g_mag"]

    def comp(r):
        return [np.log10(r["a0_mas"]), np.log10(r["sigma_a0_mas"]), 0.5 * np.log10(r["ruwe"] ** 2 - 1), np.log10(r["parallax_error"]),
                np.log10(r["period_error"]), np.log10(r["eccentricity_error"]), r["goodness_of_fit"], r["n_obs"]]

    m0 = np.array([comp(p[1]) for p in pairs], float)
    m1 = np.array([comp(p[2]) for p in pairs], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.column_stack([np.log10(a0d), np.log10(a0d / sub["significance"]), 0.5 * np.log10(sub["ruwe"] ** 2 - 1), np.log10(sub["parallax_error"]),
                             np.log10(sub["period_error"]), np.log10(sub["eccentricity_error"])])
    z = lambda x: x[:, 0] - x[:, 1] - x[:, 2]  # noqa: E731
    terms = {"z": None, "log_a0": 0, "log_sigma_a0": 1, "log_sqrt_ruwe2m1": 2, "log_sigma_plx": 3}
    res = {"n_pairs": len(pairs), "scatter": {}}
    for name, sl in (("all", np.ones(len(pairs), bool)), ("G<13", g < 13), ("G>=13", g >= 13)):
        res["scatter"][name] = {"n": int(sl.sum())}
        for lab, k in terms.items():
            xm0, xm1, xd = (z(m0), z(m1), z(d)) if k is None else (m0[:, k], m1[:, k], d[:, k])
            sm = MAD((xm0 - xm1)[sl]) / np.sqrt(2)
            sdm = MAD((xd - 0.5 * (xm0 + xm1))[sl])
            sd = float(np.sqrt(max(sdm**2 - sm**2 / 2, 0)))
            res["scatter"][name][lab] = {"sigma_mock": sm, "sigma_dr3": sd, "sigma_extra": float(np.sqrt(max(sd**2 - sm**2, 0))),
                                         "median_dr3_minus_mock": float(np.nanmedian((xd - 0.5 * (xm0 + xm1))[sl]))}
    # what explains the DR3 - mock sigma_a0 / geometry residuals
    mm = lambda k: 0.5 * (m0[:, k] + m1[:, k])  # noqa: E731
    nu = 0.5 * (m0[:, 7] + m1[:, 7]) - 12
    dsa, dsp = d[:, 1] - mm(1), d[:, 3] - mm(3)
    dgeo = dsa - dsp
    dc = logc(sub["goodness_of_fit"], nu) - 0.5 * (logc(m0[:, 6], nu) + logc(m1[:, 6], nu))
    A_d = (a0d / sub["parallax"]) / (sub["period"] / 365.25) ** (2 / 3)
    X = {"dlog_sigma_P": d[:, 4] - mm(4), "dlog_sigma_e": d[:, 5] - mm(5), "f_nl": cb["f_nl"], "log_P": np.log10(sub["period"]), "e": sub["eccentricity"],
         "log_A": np.log10(A_d), "G": g, "dlog_c": dc}
    ok = np.all([np.isfinite(v) for v in X.values()], 0) & np.isfinite(dgeo)
    M = np.column_stack([np.ones(ok.sum())] + [X[k][ok] for k in X])
    b, *_ = np.linalg.lstsq(M, dgeo[ok], rcond=None)
    res["geometry_residual"] = {"corr": {k: float(np.corrcoef(v[ok], dgeo[ok])[0, 1]) for k, v in X.items()}, "mad_before": MAD(dgeo[ok]),
                                "mad_after_regression": MAD(dgeo[ok] - M @ b), "coefficients": dict(zip(["const"] + list(X), map(float, b))),
                                "mad_sigma_a0_after_removing_dlog_c": {nm: MAD((dsa - dc)[ok & sl]) for nm, sl in (("G<13", g < 13), ("G>=13", g >= 13))}}
    # closure on the population
    with h5py.File(a.published, "r") as f:
        st_all = pub["nss_solution_type"]
    sel = np.isin(st_all, ["Orbital", "AstroSpectroSB1"]) & (pub["ruwe"] > 1.4) & (pub["period"] < 1000)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_d = np.log10(pub["significance"] / np.sqrt(pub["ruwe"] ** 2 - 1))
    c = np.load(a.cache)
    acc = np.asarray(c["accepted"], bool)
    cas = np.asarray(c["cascade"], float)[acc]
    nn = c["n_per_gen"].astype(float)
    lq = np.logaddexp.reduce(c["lqs"][:, acc] + np.log(nn / nn.sum())[:, None], axis=0)
    lt = c["ll"][acc] + c["lw"][acc] - lq
    w = np.exp(lt - np.nanmax(lt))
    w[~np.isfinite(w)] = 0
    gm = np.asarray(c["phot_g_mean_mag"], float)[acc]
    mk = (cas[:, 22] > 1.4) & (cas[:, 10] < 1000)
    with np.errstate(invalid="ignore", divide="ignore"):
        z_m = np.log10((cas[:, 17] / cas[:, 18]) / np.sqrt(cas[:, 22] ** 2 - 1))
    s_b, s_f = res["scatter"]["G<13"]["z"]["sigma_extra"], res["scatter"]["G>=13"]["z"]["sigma_extra"]
    rng = np.random.default_rng(589)
    K = 20
    zs = np.concatenate([z_m + rng.normal(0, np.where(gm < 13, s_b, s_f)) for _ in range(K)])
    ws, ms, gs = np.tile(w, K), np.tile(mk, K), np.tile(gm, K)
    clos = {}
    for lo, hi in ((9, 11.5), (11.5, 13), (13, 15), (15, 17), (17, 19)):
        aa = sel & (pub["g_mag"] >= lo) & (pub["g_mag"] < hi) & np.isfinite(z_d)
        bb = mk & (gm >= lo) & (gm < hi) & np.isfinite(z_m)
        bs = ms & (gs >= lo) & (gs < hi) & np.isfinite(zs)
        clos[f"G{lo:g}-{hi:g}"] = {q: {"dr3": float(np.percentile(z_d[aa], 100 * q)), "mock": wq(z_m[bb], w[bb], q), "mock_smeared": wq(zs[bs], ws[bs], q)}
                                   for q in (0.5, 0.9, 0.99)}
        clos[f"G{lo:g}-{hi:g}"]["mock_ess"] = float(w[bb].sum() ** 2 / (w[bb] ** 2).sum())
    res["closure_smear"] = {"sigma_extra_z": {"G<13": s_b, "G>=13": s_f}, "by_G": clos}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "dispersion.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["scatter"], indent=0, default=float)[:2500])
    print(json.dumps(res["closure_smear"], default=float)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
