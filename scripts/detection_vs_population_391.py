#!/usr/bin/env python3
"""Detection vs population test for the #391 distance deficit (0.7-1.5 kpc, G 13-16).

Ryan, 2026-10-09. Analysis plus one small targeted re-injection; no new proposal draws.

1. **Mock acceptance for matched true orbits.** Real DR3 orbits in the bin (ZP-corrected
   1/parallax in [0.7, 1.5] kpc, G in [13, 16], published P <= 830 d = C1) are matched to
   generation 23-27 mock draws by k nearest neighbours (k = ``--k``) in the standardized
   *true* space (log10 a0, log10 P, e, G, d, |ecliptic latitude|); the mock's true
   photocentre a0 is ``gaiamock.get_a0_mas`` of the draw's truth. Mock acceptance of a real
   system = the unweighted fraction of its k neighbours that are accepted with recovered
   P <= 830 d.
2. **Targeted re-injection** (rung-1 style, current forward model). A seeded random subset
   of the same real systems (``--n-inject``) x ``--n-real`` realizations: the published
   photocentre orbit at the real position, parallax, proper motion and G via
   ``predict_astrometry_binary_in_terms_of_a0`` inside ``epoch_model.run_cascade`` with the
   current ``dr3.epoch_model`` (v2 + N2d + VP loss, ``enabled`` forced on as the proposal
   runner does) and the same visibility gate and ``orbital_solution_cuts``; C1 on the
   recovered period. The #390 (pre-epoch-model) acceptance in the same region is reported
   for reference.
3. **Verdict** from the paired comparison (same systems both ways), the outcome / cut
   decomposition of the failures on each side, and accepted-count ratios in P, f_m (a q
   proxy) and dereddened M_G (an M1 proxy) with the mock's true M1 and q in the bin.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BASELINE = 1038.0
P_MAX = 0.8 * BASELINE
D_LO, D_HI, G_LO, G_HI = 0.7, 1.5, 13.0, 16.0
GEN_ARTIFACTS = ("restart_gen23_tune.h5", "restart_gen24_full.h5", "restart_gen25_tune2.h5",
                 "restart_gen26_topup.h5", "restart_gen27_evolved.h5")
# kNN feature scales: log10 a0, log10 P, e, G, d (kpc), |beta| (deg), |cos i|
FEATURES = ("log10 a0", "log10 P", "e", "G", "d", "|beta|", "|cos i|")
SCALES = np.array([0.1, 0.1, 0.1, 0.5, 0.15, 20.0, 0.1])
MOCK_ID_OFFSET = 10**15  # keeps cross-check seeds disjoint from DR3 source_ids used for re-injection
_W: dict[str, Any] = {}


# ---------------------------------------------------------------- re-injection worker
def _init(base_seed: int, cuts_json: str, emc_json: str) -> None:
    os.nice(10)
    from threadpoolctl import threadpool_limits

    import dataclasses
    from darkhunter_pop import epoch_model as em
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod

    _W["blas"] = threadpool_limits(limits=1)
    cfg = load_config()
    gm = import_gaiamock_mod()
    emc = dataclasses.replace(em.epoch_model_config_from_mapping(cfg.active_dr().epoch_model), enabled=True)
    if json.dumps(dataclasses.asdict(emc), default=str, sort_keys=True) != emc_json:
        raise RuntimeError("epoch-model config differs between parent and worker")
    _W.update(gm=gm, c=gm.read_in_C_functions(), em=em, emc=emc,
              gaps=em.gap_intervals_jd(emc), base_seed=base_seed,
              cuts=cfg.active_dr().selection_function_astrometric.orbital_solution_cuts,
              pop=cfg.selection_function_astrometric.mock_population, dr=cfg.active_dr_mode.value)


def _inject(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    from darkhunter_pop import injection_test as it
    from darkhunter_pop.cascade_replay import GAIAMOCK_MIN_OBSERVATIONS, GAIAMOCK_MIN_VISIBILITY_PERIODS
    from darkhunter_pop.forward_model import seeded_global_rng

    row, r = task
    gm, c, em = _W["gm"], _W["c"], _W["em"]
    if "seed_id" in row:  # a mock draw (its truth, or its fitted orbit) through the rung-1 path
        v = row
        sid = int(row["seed_id"])
    else:
        truth = it.published_truth_from_row(row, gm)
        v = truth.values
        sid = truth.source_id
    seeds = it.injection_rng_seeds(_W["base_seed"], sid, r)
    t0 = time.process_time()

    def predict() -> Any:
        return gm.predict_astrometry_binary_in_terms_of_a0(
            ra=v["ra"], dec=v["dec"], parallax=v["parallax"], pmra=v["pmra"], pmdec=v["pmdec"],
            period=v["period"], Tp=v["t_periastron"], ecc=v["eccentricity"], omega=v["Omega_rad"],
            inc=v["inc_rad"], w=v["omega_rad"], a0_mas=v["a0_mas"], phot_g_mean_mag=v["g_mag"],
            data_release=_W["dr"], c_funcs=c)

    with seeded_global_rng(seeds, c):
        run = em.run_cascade(
            gm, c, predict, _W["emc"], em.source_context(v["ra"], v["dec"], v["g_mag"]),
            epoch_rng=em.epoch_model_rng(_W["base_seed"], it.INJECTION_RNG_STREAM, sid, r),
            noise_rng=em.epoch_model_rng(_W["base_seed"], it.INJECTION_RNG_STREAM, sid, r, tag=em.PER_CCD_NOISE_RNG_TAG),
            ruwe_min=_W["pop"].ruwe_min, skip_acceleration=_W["pop"].skip_acceleration, gaps_jd=_W["gaps"])
    if run.n_visibility_periods < GAIAMOCK_MIN_VISIBILITY_PERIODS or run.n_obs < GAIAMOCK_MIN_OBSERVATIONS:
        casc = [0.0] * 23
    else:
        casc = [float(x) for x in run.cascade]
    rec = it.parse_cascade_result(casc, n_visibility_periods=run.n_visibility_periods, n_obs=run.n_obs, cuts=_W["cuts"])
    out = {k: (float(x) if isinstance(x, (float, np.floating)) else (bool(x) if isinstance(x, (bool, np.bool_)) else x))
           for k, x in rec.items()}
    out.update(source_id=sid, realization=int(r), cpu_s=time.process_time() - t0,
               n_obs_sim=int(run.n_obs), n_vis_sim=int(run.n_visibility_periods), ruwe_scale=float(run.ruwe_scale),
               truth_a0_mas=float(v["a0_mas"]))
    return out


# ---------------------------------------------------------------- helpers
def sky_beta(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    from astropy.coordinates import BarycentricTrueEcliptic, SkyCoord
    import astropy.units as u

    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to(BarycentricTrueEcliptic()).lat.deg


def outcome_class(rec_outcome: np.ndarray, accepted: np.ndarray, period: np.ndarray, flags: dict[str, np.ndarray]) -> np.ndarray:
    """String class per realization: the first failure in cascade order."""
    out = np.full(rec_outcome.size, "orbital: other", dtype=object)
    out[rec_outcome == 0] = "insufficient visibility"
    out[rec_outcome == 5] = "5-par"
    out[(rec_outcome == 7) | (rec_outcome == 9)] = "7/9-par"
    orb = rec_outcome == 12
    for name in ("cut_a0_over_err", "cut_a0_over_err_sqrt_p", "cut_parallax_over_error", "cut_sigma_e", "cut_f2"):
        out[orb & ~flags[name] & (out == "orbital: other")] = f"fails {name[4:]}"
    out[accepted & (period > P_MAX)] = "accepted, P > 830 d"
    out[accepted & (period <= P_MAX)] = "accepted"
    return out


def parse_many(casc: np.ndarray, cuts: Any) -> dict[str, np.ndarray]:
    from darkhunter_pop import injection_test as it

    recs = [it.parse_cascade_result(c, n_visibility_periods=0, n_obs=0, cuts=cuts) for c in casc]
    keys = ("outcome", "accepted", "period", "significance", "ruwe", "cut_a0_over_err", "cut_a0_over_err_sqrt_p",
            "cut_parallax_over_error", "cut_sigma_e", "cut_f2", "parallax", "parallax_error", "eccentricity_error")
    return {k: np.array([r[k] for r in recs]) for k in keys}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifact-dir", type=Path, default=Path("output/proposal_set"))
    ap.add_argument("--cache", type=Path, default=Path("output/proposal_set/weights_gen23_27.npz"))
    ap.add_argument("--parent-dir", type=Path, required=True)
    ap.add_argument("--real-snapshot", type=Path, required=True)
    ap.add_argument("--real-input-columns", type=Path, required=True)
    ap.add_argument("--gate390", type=Path, required=True, help="output/gate390 (published_orbits.h5, injection_test_full.h5)")
    ap.add_argument("--fragment", type=Path, default=Path("config/population/proposal_set_restart_gen26.yaml"))
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--n-inject", type=int, default=300)
    ap.add_argument("--n-real", type=int, default=3)
    ap.add_argument("--n-cross", type=int, default=2, help="mock neighbours per re-injected system for the engine cross-check")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=391)
    ap.add_argument("--inject-log", type=Path, required=True, help="resumable JSONL of re-injection realizations")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    import dataclasses
    import h5py
    from astropy.table import Table
    from scipy.spatial import cKDTree

    from darkhunter_pop import epoch_model as em
    from darkhunter_pop import injection_test as it
    from darkhunter_pop import proposal_set as ps
    from darkhunter_pop.config_loader import load_config
    from darkhunter_pop.gaiamock_vendor import import_gaiamock_mod
    from darkhunter_pop.giants import cmd_for_rows, load_giants_config, parent_row_cmd
    from darkhunter_pop.physics_utils import astrometric_mass_function
    from darkhunter_pop.plotting import apply_axes_style, require_pyplot, resolve_plotting_style, save_figure, series_style

    cfg = load_config()
    gm = import_gaiamock_mod()
    cuts = cfg.active_dr().selection_function_astrometric.orbital_solution_cuts
    prop = ps.load_proposal_set_fragment(args.fragment).proposal
    parent = ps.load_parent_snapshot(args.parent_dir, cfg, prop)
    gcfg = load_giants_config(Path("config/population/giants.yaml"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rep: list[str] = ["#391 detection vs population test: 0.7-1.5 kpc, G 13-16, C1 (P <= 830 d) on both sides", ""]

    # ================= mock truth (gens 23-27, cache order) =================
    keys = ("m1_msun", "m2_msun", "flux_ratio", "period_days", "eccentricity", "parallax_mas", "phot_g_mean_mag",
            "ra_deg", "dec_deg", "parent_row", "inc_deg", "Omega_rad", "omega_rad", "Tp_days", "pmra_masyr", "pmdec_masyr",
            "draw_index")
    parts = [ps.read_proposal_artifact(args.artifact_dir / a) for a in GEN_ARTIFACTS]
    t = {k: np.concatenate([np.asarray(p[0][k]) for p in parts]) for k in keys}
    z = np.load(args.cache, allow_pickle=True)
    if not np.array_equal(t["period_days"], np.asarray(z["period_days"])):
        raise SystemExit("artifact order differs from the weight cache")
    casc = np.asarray(z["cascade"], float)
    acc = np.asarray(z["accepted"], bool)
    W = {"noW": np.asarray(z["w_noW"], float), "cmdW": np.asarray(z["w_cmdW"], float)}
    m_a0 = np.abs(np.asarray(gm.get_a0_mas(t["period_days"], t["m1_msun"], t["m2_msun"], t["parallax_mas"], t["flux_ratio"]), float))
    m_d = 1.0 / t["parallax_mas"]
    row = t["parent_row"].astype(np.int64)
    pc = parent.columns
    p_beta = sky_beta(np.asarray(pc["ra"], float), np.asarray(pc["dec"], float))
    m_beta = np.abs(p_beta[row])
    m_P_rec = casc[:, 10]
    m_acc_c1 = acc & (m_P_rec <= P_MAX)
    pcmd = parent_row_cmd(parent, cfg, gcfg)
    m_mg0 = np.asarray(pcmd.mg0, float)[row]
    m_q = t["m2_msun"] / t["m1_msun"]
    with np.errstate(divide="ignore", invalid="ignore"):
        m_fm_true = astrometric_mass_function(m_a0, t["parallax_mas"], t["period_days"])
    mfeat = np.column_stack([np.log10(m_a0), np.log10(t["period_days"]), t["eccentricity"], t["phot_g_mean_mag"], m_d, m_beta,
                             np.abs(np.cos(np.radians(t["inc_deg"])))])
    # restrict the tree to a padded box around the bin (true quantities)
    box = (m_d > D_LO - 0.3) & (m_d < D_HI + 0.4) & (t["phot_g_mean_mag"] > G_LO - 1.0) & (t["phot_g_mean_mag"] < G_HI + 1.0) \
        & np.all(np.isfinite(mfeat), axis=1) & (m_a0 > 0)
    bidx = np.flatnonzero(box)
    tree = cKDTree(mfeat[bidx] / SCALES)
    rep.append(f"mock draws (gens 23-27): {acc.size}; in the padded true-space box: {bidx.size} ({acc[bidx].sum()} accepted)")

    # ================= real bin =================
    pg = np.asarray(pc["phot_g_mean_mag"], float)
    snr_p = np.asarray(pc["parallax"], float) / np.asarray(pc["parallax_error"], float)
    dzp = parent.truth_parallax_mas - np.asarray(pc["parallax"], float)
    gedges = np.array([0, 11, 13, 15, 17, 25.0])
    zp = np.array([np.median(dzp[parent.usable & (pg >= a) & (pg < b) & (snr_p > 20)]) for a, b in zip(gedges[:-1], gedges[1:])])
    rt = Table.read(args.real_snapshot, format="ascii.ecsv")
    types = list(cfg.active_dr().selection_function_astrometric.elbadry2024_comparison_nss_solution_types)
    rt = rt[np.isin(np.asarray(rt["nss_solution_type"]).astype(str), types)]
    keep, _ = ps.real_comparison_keep(rt, prop, ps.load_real_input_columns(args.real_input_columns))
    rt = rt[keep]

    def col(name: str) -> np.ndarray:
        return np.ma.filled(np.ma.asarray(rt[name], float), np.nan)

    r_g = col("g_mag")
    r_plx = col("parallax") + zp[np.clip(np.digitize(r_g, gedges) - 1, 0, zp.size - 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        r_d = np.where(r_plx > 0, 1.0 / r_plx, np.nan)
    r_P, r_e = col("period"), col("eccentricity")
    r_a0 = np.asarray(gm.get_Campbell_elements(col("A"), col("B"), col("F"), col("G"))[0], float)
    r_beta = np.abs(sky_beta(col("ra"), col("dec")))
    r_cosi = np.abs(np.cos(np.asarray(gm.get_Campbell_elements(col("A"), col("B"), col("F"), col("G"))[3], float)))
    inbin = (r_d >= D_LO) & (r_d <= D_HI) & (r_g >= G_LO) & (r_g <= G_HI) & (r_P <= P_MAX)
    rfeat = np.column_stack([np.log10(r_a0), np.log10(r_P), r_e, r_g, r_d, r_beta, r_cosi])
    inbin &= np.all(np.isfinite(rfeat), axis=1)
    rb = np.flatnonzero(inbin)
    r_sid = np.asarray(rt["source_id"], np.int64)
    r_type = np.asarray(rt["nss_solution_type"]).astype(str)
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    gal = SkyCoord(ra=col("ra") * u.deg, dec=col("dec") * u.deg).galactic
    eplx = col("parallax_error")
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = np.where(r_plx > 0, 1000 / r_plx, np.nan)
        rlo = np.where(r_plx > 0, 1000 / (r_plx + eplx), np.nan)
        rhi = np.where(r_plx - eplx > 0, 1000 / (r_plx - eplx), np.nan)
        r_fm = astrometric_mass_function(r_a0, r_plx, r_P)
    r_mg0 = np.asarray(cmd_for_rows(r_g, col("bp_mag") - col("rp_mag"), gal.l.deg, gal.b.deg, rr, rlo, rhi, cfg, gcfg).mg0, float)
    rep.append(f"real orbits after mirror filters: {len(rt)}; in the bin (ZP-corrected d, G, published P <= 830 d): {rb.size}")

    # ================= 1. matched mock acceptance =================
    dist, nn = tree.query(rfeat[rb] / SCALES, k=args.k)
    nn_g = bidx[nn]
    p_mock = m_acc_c1[nn_g].mean(axis=1)
    rep.append(f"kNN match: k = {args.k} in {FEATURES} scaled by {SCALES.tolist()}; "
               f"median scaled distance to the k-th neighbour {np.median(dist[:, -1]):.2f} "
               f"(per-dim median |offset| in scale units: "
               + ", ".join(f"{np.median(np.abs(mfeat[nn_g][..., j] - rfeat[rb][:, None, j])) / SCALES[j]:.2f}" for j in range(SCALES.size)) + ")")
    rep.append(f"mock acceptance of matched true orbits, mean over all {rb.size} real systems: {p_mock.mean():.3f}")
    # neighbour-offset bias: acceptance gradient x mean signed offset (pooled local-linear correction)
    off = (mfeat[nn_g] - rfeat[rb][:, None, :]) / SCALES
    yy = m_acc_c1[nn_g].astype(float).ravel()
    X = np.column_stack([np.ones(yy.size), off.reshape(-1, SCALES.size)])
    beta = np.linalg.lstsq(X, yy, rcond=None)[0]
    corr = off.mean(axis=1) @ beta[1:]
    p_mock_lin = np.clip(p_mock - corr, 0.0, 1.0)
    rep.append("  mean signed neighbour offset (scale units): " + ", ".join(f"{v:+.2f}" for v in off.mean(axis=(0, 1)))
               + "; pooled acceptance slopes per scale unit: " + ", ".join(f"{v:+.3f}" for v in beta[1:]))
    rep.append(f"  local-linear (offset-corrected) mock acceptance, all real systems: {p_mock_lin.mean():.3f}")

    # ================= 2. targeted re-injection =================
    with h5py.File(args.gate390 / "published_orbits.h5", "r") as h:
        pub = {k: h["published"][k][()] for k in h["published"]}
    pub["nss_solution_type"] = np.array([s.decode().strip('"') for s in pub["nss_solution_type"]])
    pkey = {(int(s), ty): i for i, (s, ty) in enumerate(zip(pub["source_id"], pub["nss_solution_type"]))}
    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(rb.size, size=min(args.n_inject, rb.size), replace=False))
    sub_rows = []
    for j in sel:
        i = pkey[(int(r_sid[rb[j]]), r_type[rb[j]])]
        rowmap = {k: pub[k][i] for k in pub if k not in ("corr_vec", "nss_solution_type")}
        rowmap["nss_solution_type"] = str(pub["nss_solution_type"][i])
        cv = np.asarray(pub["corr_vec"][i], np.float64)
        rowmap["corr_vec"] = cv[np.isfinite(cv)]
        rowmap["bit_index"] = int(pub["bit_index"][i])
        rowmap["source_id"] = int(pub["source_id"][i])
        sub_rows.append({k: (x.item() if isinstance(x, np.generic) else x) for k, x in rowmap.items()})
    done: dict[tuple[int, int], dict[str, Any]] = {}
    if args.inject_log.exists():
        for line in args.inject_log.read_text().splitlines():
            d = json.loads(line)
            done[(d["source_id"], d["realization"])] = d
    tasks = [(rm, r) for rm in sub_rows for r in range(args.n_real) if (rm["source_id"], r) not in done]
    # engine cross-check: the --n-cross nearest mock neighbours of each re-injected system, their *mock truth*
    # re-simulated through the rung-1 path (predict_astrometry_binary_in_terms_of_a0 + run_cascade), 1 realization
    xdraw = np.unique(nn_g[sel, : args.n_cross].ravel())
    xrows = []
    for gidx in xdraw:
        xrows.append({"ra": float(t["ra_deg"][gidx]), "dec": float(t["dec_deg"][gidx]),
                      "parallax": float(t["parallax_mas"][gidx]), "pmra": float(t["pmra_masyr"][gidx]),
                      "pmdec": float(t["pmdec_masyr"][gidx]), "period": float(t["period_days"][gidx]),
                      "t_periastron": float(t["Tp_days"][gidx]), "eccentricity": float(t["eccentricity"][gidx]),
                      "Omega_rad": float(t["Omega_rad"][gidx]), "inc_rad": float(np.radians(t["inc_deg"][gidx])),
                      "omega_rad": float(t["omega_rad"][gidx]), "a0_mas": float(m_a0[gidx]),
                      "g_mag": float(t["phot_g_mean_mag"][gidx]), "seed_id": int(t["draw_index"][gidx]) + MOCK_ID_OFFSET})
    tasks += [(xr, 0) for xr in xrows if (xr["seed_id"], 0) not in done]
    # symmetric rung 1 on the mock: the mock's own *accepted* orbits matched to the re-injected real systems on
    # recovered quantities (1 nearest each), re-injected at their *fitted* orbit (A, B, F, G -> Campbell; period,
    # e, Tp = phi_p P / 2 pi as gaiamock's predict_astrometry_epochs; fitted parallax) through the rung-1 path.
    with np.errstate(divide="ignore", invalid="ignore"):
        fa0, fOm, fom, finc = (np.asarray(x, float) for x in gm.get_Campbell_elements(casc[:, 2], casc[:, 4], casc[:, 6], casc[:, 8]))
        mrfeat = np.column_stack([np.log10(fa0), np.log10(m_P_rec), casc[:, 14], t["phot_g_mean_mag"], 1.0 / casc[:, 0],
                                  m_beta, np.abs(np.cos(finc))])
    macc_in = np.flatnonzero(m_acc_c1 & np.all(np.isfinite(mrfeat), axis=1) & (casc[:, 0] > 0))
    rtree = cKDTree(mrfeat[macc_in] / SCALES)
    r_rec_feat = rfeat[rb[sel]]
    rdist, rnn = rtree.query(r_rec_feat / SCALES, k=1)
    sdraw = macc_in[rnn]
    srows = []
    for gidx in np.unique(sdraw):
        srows.append({"ra": float(t["ra_deg"][gidx]), "dec": float(t["dec_deg"][gidx]),
                      "parallax": float(casc[gidx, 0]), "pmra": float(t["pmra_masyr"][gidx]),
                      "pmdec": float(t["pmdec_masyr"][gidx]), "period": float(m_P_rec[gidx]),
                      "t_periastron": float(casc[gidx, 12] * m_P_rec[gidx] / (2 * np.pi)),
                      "eccentricity": float(casc[gidx, 14]), "Omega_rad": float(fOm[gidx]), "inc_rad": float(finc[gidx]),
                      "omega_rad": float(fom[gidx]), "a0_mas": float(fa0[gidx]), "g_mag": float(t["phot_g_mean_mag"][gidx]),
                      "seed_id": int(t["draw_index"][gidx]) + 2 * MOCK_ID_OFFSET})
    tasks += [(sr, r) for sr in srows for r in range(args.n_real) if (sr["seed_id"], r) not in done]
    base_seed = int(cfg.selection_function_astrometric.mock_population.random_seed)
    emc = dataclasses.replace(em.epoch_model_config_from_mapping(cfg.active_dr().epoch_model), enabled=True)
    emc_json = json.dumps(dataclasses.asdict(emc), default=str, sort_keys=True)
    if tasks:
        print(f"re-injecting {len(tasks)} realizations on {args.workers} workers", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, initializer=_init, initargs=(base_seed, cuts.model_dump_json(), emc_json)) as pool, \
                args.inject_log.open("a") as fh:
            for n, rec in enumerate(pool.imap_unordered(_inject, tasks, chunksize=2), 1):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                done[(rec["source_id"], rec["realization"])] = rec
                if n % 100 == 0:
                    print(f"  {n}/{len(tasks)}", flush=True)
    sid_sub = np.array([rm["source_id"] for rm in sub_rows])
    recs = [done[(s, r)] for s in sid_sub for r in range(args.n_real)]
    inj = {k: np.array([d[k] for d in recs]) for k in recs[0]}
    inj_acc = inj["accepted"].astype(bool) & (inj["period"] <= P_MAX)
    inj_acc = np.where(np.isfinite(inj["period"]), inj_acc, False)
    p_inj_sys = inj_acc.reshape(-1, args.n_real).mean(axis=1)
    p_mock_sub = p_mock[sel]
    k_inj, n_inj = int(inj_acc.sum()), inj_acc.size
    f_inj, lo_inj, hi_inj = it.binomial_fraction_interval(k_inj, n_inj)
    # paired difference with a system bootstrap
    boot = np.random.default_rng(args.seed + 1).integers(0, sel.size, size=(4000, sel.size))
    diff = p_mock_sub - p_inj_sys
    dboot = diff[boot].mean(axis=1)
    rep += ["", f"re-injection: {sel.size} real systems (seed {args.seed} random subset of the bin) x {args.n_real} realizations, "
            f"current epoch model (v2 + N2d + VP loss), base_seed {base_seed}; median CPU {np.median(inj['cpu_s']):.2f} s",
            f"  re-injection acceptance (C1): {k_inj}/{n_inj} = {f_inj:.3f} (1-sigma {lo_inj:.3f}-{hi_inj:.3f})",
            f"  mock acceptance of matched true orbits, same {sel.size} systems: {p_mock_sub.mean():.3f}",
            f"  paired difference mock - re-injection: {diff.mean():+.3f} (bootstrap 1-sigma {dboot.std():.3f}; "
            f"95% {np.percentile(dboot, 2.5):+.3f} to {np.percentile(dboot, 97.5):+.3f})",
            f"  offset-corrected mock acceptance, same systems: {p_mock_lin[sel].mean():.3f}; paired difference "
            f"{(p_mock_lin[sel] - p_inj_sys).mean():+.3f} (bootstrap 1-sigma {(p_mock_lin[sel] - p_inj_sys)[boot].mean(axis=1).std():.3f})"]

    # engine cross-check result
    xrec = [done[(xr["seed_id"], 0)] for xr in xrows]
    x_acc = np.array([bool(d["accepted"]) and np.isfinite(d["period"]) and d["period"] <= P_MAX for d in xrec])
    x_stored = m_acc_c1[xdraw]
    fx = it.binomial_fraction_interval(int(x_acc.sum()), x_acc.size)
    fs = it.binomial_fraction_interval(int(x_stored.sum()), x_stored.size)
    both = np.mean(x_acc == x_stored)
    rep += [f"  engine cross-check: {xdraw.size} matched mock draws ({args.n_cross} nearest per re-injected system), "
            f"mock truth re-simulated through the rung-1 path: accepted {fx[0]:.3f} ({fx[1]:.3f}-{fx[2]:.3f}) vs their stored "
            f"proposal-runner outcome {fs[0]:.3f} ({fs[1]:.3f}-{fs[2]:.3f}); per-draw agreement {both:.3f}"]
    p_mock1 = m_acc_c1[nn_g[sel, : args.n_cross]].mean()
    smap = {sr["seed_id"]: sr for sr in srows}
    srec = [done[(int(t["draw_index"][g_]) + 2 * MOCK_ID_OFFSET, r)] for g_ in sdraw for r in range(args.n_real)]
    s_acc = np.array([bool(d["accepted"]) and np.isfinite(d["period"]) and d["period"] <= P_MAX for d in srec])
    p_sym_sys = s_acc.reshape(-1, args.n_real).mean(axis=1)
    fsy = it.binomial_fraction_interval(int(s_acc.sum()), s_acc.size)
    dsym = p_sym_sys - p_inj_sys
    dsb = dsym[boot].mean(axis=1)
    s_inj = {k: np.array([d[k] for d in srec]) for k in ("outcome", "significance", "ruwe")}
    rep += [f"  SYMMETRIC rung 1: {np.unique(sdraw).size} mock accepted orbits (1 nearest per real system on recovered "
            f"quantities; median scaled distance {np.median(rdist):.2f}) re-injected at their fitted orbit x {args.n_real}: "
            f"acceptance {fsy[0]:.3f} ({fsy[1]:.3f}-{fsy[2]:.3f}) vs real {f_inj:.3f}; paired difference {dsym.mean():+.3f} "
            f"(bootstrap 1-sigma {dsb.std():.3f}); outcomes 5-par {np.mean(s_inj['outcome'] == 5):.3f} | "
            f"7/9-par {np.mean(np.isin(s_inj['outcome'], (7, 9))):.3f} (real {np.mean(inj['outcome'] == 5):.3f} | "
            f"{np.mean(np.isin(inj['outcome'], (7, 9))):.3f}); median recovered significance "
            f"{np.nanmedian(s_inj['significance']):.2f} (real {np.nanmedian(inj['significance']):.2f})"]
    rep.append(f"  (the {args.n_cross} nearest neighbours alone give mock acceptance {p_mock1:.3f} for the re-injected systems)")

    # real vs symmetric by significance and period (real: published; mock: fitted), with the 7/9-par fallback share
    r_sig_pub = np.array([pub["significance"][pkey[(int(r_sid[rb[j]]), r_type[rb[j]])]] for j in sel])
    r_sig_r = np.repeat(r_sig_pub, args.n_real)
    r_P_r = np.repeat(r_P[rb[sel]], args.n_real)
    s_draw_r = np.repeat(sdraw, args.n_real)
    with np.errstate(divide="ignore", invalid="ignore"):
        s_sig_r = casc[s_draw_r, 17] / casc[s_draw_r, 18]
    s_P_r = m_P_rec[s_draw_r]
    i_out = inj["outcome"]
    s_out = s_inj["outcome"]
    rep.append("  real re-injection | symmetric mock re-injection, acceptance (7/9-par share) [n realizations]:")
    for lab, rv, sv, edges in (("significance", r_sig_r, s_sig_r, (0, 10, 15, 25, 1e9)), ("P (d)", r_P_r, s_P_r, (0, 300, 500, 650, P_MAX))):
        parts_ = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            a_ = (rv > lo) & (rv <= hi)
            b_ = (sv > lo) & (sv <= hi)
            parts_.append(f"{lo:g}-{hi:g}: {inj_acc[a_].mean():.2f} ({np.isin(i_out[a_], (7, 9)).mean():.2f}) [{a_.sum()}] | "
                          f"{s_acc[b_].mean():.2f} ({np.isin(s_out[b_], (7, 9)).mean():.2f}) [{b_.sum()}]")
        rep.append(f"    by {lab}: " + "; ".join(parts_))

    # #390 (pre-epoch-model) acceptance in the same region, for reference
    with h5py.File(args.gate390 / "injection_test_full.h5", "r") as h:
        s390 = h["systems/source_id"][()]
        tr = {k: h[f"systems/truth/{k}"][()] for k in ("parallax", "g_mag", "period")}
        r390 = {k: h[f"realizations/{k}"][()] for k in ("system_index", "accepted")}
        rp390 = h["realizations/recovered/period"][()]
    gz = zp[np.clip(np.digitize(tr["g_mag"], gedges) - 1, 0, zp.size - 1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        d390 = 1.0 / (tr["parallax"] + gz)
    in390 = (d390 >= D_LO) & (d390 <= D_HI) & (tr["g_mag"] >= G_LO) & (tr["g_mag"] <= G_HI) & (tr["period"] <= P_MAX)
    rsel = in390[r390["system_index"]]
    a390 = r390["accepted"].astype(bool) & (rp390 <= P_MAX)
    f390 = it.binomial_fraction_interval(int(a390[rsel].sum()), int(rsel.sum()))
    rep.append(f"  #390 (no epoch model) in the same region: {int(in390.sum())} systems, {int(a390[rsel].sum())}/{int(rsel.sum())} "
               f"realizations accepted = {f390[0]:.3f} (1-sigma {f390[1]:.3f}-{f390[2]:.3f}); "
               f"#390 is stratified, so this is not the bin's population mix")

    # ---- failure decomposition, both sides ----
    nm = nn_g[sel].ravel()
    mp_ = parse_many(casc[nm], cuts)
    mclass = outcome_class(mp_["outcome"], mp_["accepted"].astype(bool), mp_["period"], mp_)
    iflags = {k: inj[k].astype(bool) for k in ("cut_a0_over_err", "cut_a0_over_err_sqrt_p", "cut_parallax_over_error", "cut_sigma_e", "cut_f2")}
    iclass = outcome_class(inj["outcome"], inj["accepted"].astype(bool), inj["period"], iflags)
    cats = ["accepted", "accepted, P > 830 d", "insufficient visibility", "5-par", "7/9-par", "fails a0_over_err",
            "fails a0_over_err_sqrt_p", "fails parallax_over_error", "fails sigma_e", "fails f2", "orbital: other"]
    rep += ["", "outcome / first-failing-cut fractions (matched mock neighbours of the re-injected systems | re-injection):"]
    for c_ in cats:
        rep.append(f"  {c_:28s} {np.mean(mclass == c_):.3f} | {np.mean(iclass == c_):.3f}")
    for lab, (mx, ix) in {"recovered significance a0/sigma_a0 (orbital)": (mp_["significance"], inj["significance"]),
                          "RUWE (all outcomes)": (mp_["ruwe"], inj["ruwe"])}.items():
        mf, if_ = mx[np.isfinite(mx)], ix[np.isfinite(ix)]
        rep.append(f"  median {lab}: mock {np.median(mf):.2f} (n={mf.size}) | re-injection {np.median(if_):.2f} (n={if_.size})")

    # acceptance vs truth variables, both sides (paired systems)
    style = resolve_plotting_style(cfg.plotting)
    plt = require_pyplot()
    rsub = rb[sel]
    prof = {"log10 true a0 (mas)": (np.log10(r_a0[rsub]), np.linspace(-1.0, 1.0, 9)),
            "log10 P (d)": (np.log10(r_P[rsub]), np.linspace(1.0, 3.0, 9)),
            "eccentricity": (r_e[rsub], np.linspace(0, 1, 6)),
            "G (mag)": (r_g[rsub], np.linspace(13, 16, 7)),
            "d (kpc, ZP-corrected)": (r_d[rsub], np.linspace(0.7, 1.5, 5)),
            "|ecliptic latitude| (deg)": (r_beta[rsub], np.linspace(0, 90, 7)),
            "log10 f_m (Msun; published)": (np.log10(r_fm[rsub]), np.array([-3.5, -2.0, -1.5, -1.0, 0.5])),
            "dereddened M_G (mag)": (r_mg0[rsub], np.array([-1.0, 2.5, 3.5, 4.5, 7.5])),
            "published RUWE": (col("ruwe")[rsub], np.array([1.0, 1.4, 2.0, 3.0, 50.0]))}
    fig, axs = plt.subplots(3, 3, figsize=(17, 15))
    rep += ["", "acceptance vs real published variable (bin centre: mock matched truth | real re-injection | symmetric mock re-injection [n systems]):"]
    for ax, (name, (x, edges)) in zip(np.ravel(axs), prof.items()):
        b = np.digitize(x, edges) - 1
        cen, pm, pi, em_, ei, nn_, ps_, es_ = [], [], [], [], [], [], [], []
        for i in range(edges.size - 1):
            s = b == i
            if s.sum() < 5:
                continue
            cen.append(0.5 * (edges[i] + edges[i + 1])); nn_.append(int(s.sum()))
            pm.append(p_mock_sub[s].mean()); pi.append(p_inj_sys[s].mean())
            em_.append(p_mock_sub[s].std() / np.sqrt(s.sum())); ei.append(p_inj_sys[s].std() / np.sqrt(s.sum()))
            ps_.append(p_sym_sys[s].mean()); es_.append(p_sym_sys[s].std() / np.sqrt(s.sum()))
        for j, (lab, y, e) in enumerate((("mock, matched true orbits", pm, em_), ("real re-injection, current model", pi, ei),
                                         ("mock accepted orbits re-injected (symmetric)", ps_, es_))):
            st = series_style(j, style)
            ax.errorbar(cen, y, yerr=e, color=st["color"], ls=st["linestyle"], marker="o", lw=2, label=lab)
        ax.set_ylim(0, 1.05)
        apply_axes_style(ax, style, xlabel=name, ylabel="acceptance (C1)")
        if ax is np.ravel(axs)[0]:
            ax.legend(fontsize=style.tick_label_fontsize, loc="lower right")
        rep.append(f"  {name}: " + " ".join(f"{c:.2f}:{a:.2f}|{b_:.2f}|{c_:.2f}[{n}]" for c, a, b_, c_, n in zip(cen, pm, pi, ps_, nn_)))
    fig.suptitle(f"Acceptance in the bin ({sel.size} real systems): matched mock truth, real re-injection, symmetric mock re-injection",
                 fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "acceptance_mock_vs_reinjection.png", dpi=int(cfg.diagnostics.figure_dpi))

    # ================= 3. population: accepted-count ratios in the bin =================
    with np.errstate(divide="ignore", invalid="ignore"):
        mplx_rec = casc[:, 0]
        m_d_rec_c = 1.0 / mplx_rec
        m_fm_rec = astrometric_mass_function(casc[:, 17], mplx_rec, m_P_rec)
    m_g = t["phot_g_mean_mag"]
    m_in = m_acc_c1 & (m_d_rec_c >= D_LO) & (m_d_rec_c <= D_HI) & (m_g >= G_LO) & (m_g <= G_HI)
    m_true_in = (m_d >= D_LO) & (m_d <= D_HI) & (m_g >= G_LO) & (m_g <= G_HI) & (t["period_days"] <= P_MAX)
    nreal = rb.size
    rep += ["", f"population: accepted orbits in the bin (observed d for both, mock recovered P <= 830 d): real {nreal}; "
            + "; ".join(f"mock {k} {np.sum(w[m_in]):.0f} (ratio {np.sum(w[m_in]) / nreal:.2f}, ESS {ps.kish_ess(w[m_in]):.0f})" for k, w in W.items())]
    dims = {"log10 P (d)": (np.log10(m_P_rec), np.log10(r_P), np.linspace(1.0, np.log10(P_MAX), 8)),
            "log10 f_m (Msun)": (np.log10(m_fm_rec), np.log10(r_fm), np.linspace(-4, 0, 9)),
            "dereddened M_G (mag; M1 proxy)": (m_mg0, r_mg0, np.linspace(-1, 7, 9))}
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    rep.append("mock/real accepted-count ratio per bin (noW | cmdW) [real n]:")
    for ax, (name, (mv, rv, edges)) in zip(axs[0], dims.items()):
        rn = np.histogram(rv[rb][np.isfinite(rv[rb])], bins=edges)[0].astype(float)
        cen = 0.5 * (edges[1:] + edges[:-1])
        rr_ = {}
        for j, (k, w) in enumerate(W.items()):
            s = m_in & np.isfinite(mv)
            mn = np.histogram(mv[s], bins=edges, weights=w[s])[0]
            mn2 = np.histogram(mv[s], bins=edges, weights=w[s] ** 2)[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                rr_[k] = mn / rn
                st = series_style(j, style)
                ax.errorbar(cen, mn / rn, yerr=np.sqrt(mn2) / rn, color=st["color"], ls=st["linestyle"], marker="o", lw=2, label=f"mock {k} / real")
        ax.axhline(1, color="0.4", ls=":")
        ax.set_ylim(0, 2)
        apply_axes_style(ax, style, xlabel=name, ylabel="mock / real accepted counts")
        ax.legend(fontsize=style.tick_label_fontsize)
        rep.append(f"  {name}: " + " ".join(f"{c:.2f}:{a:.2f}|{b_:.2f}[{int(n)}]" for c, a, b_, n in zip(cen, rr_["noW"], rr_["cmdW"], rn)))
    # mock true M1, q, P in the bin: accepted vs all (pre-detection), weighted noW
    w0 = W["noW"]
    tdims = {"true M1 (Msun)": (t["m1_msun"], np.linspace(0.3, 2.1, 10)),
             "true q = M2/M1": (m_q, np.linspace(0, 1.2, 13)),
             "true log10 P (d)": (np.log10(t["period_days"]), np.linspace(1.0, np.log10(P_MAX), 8))}
    rep.append("mock (noW) true-parameter distributions in the bin (true d, G, P <= 830 d): pre-detection share | accepted share | acceptance:")
    for ax, (name, (x, edges)) in zip(axs[1], tdims.items()):
        pre = np.histogram(x[m_true_in], bins=edges, weights=w0[m_true_in])[0]
        s = m_true_in & m_acc_c1
        det = np.histogram(x[s], bins=edges, weights=w0[s])[0]
        cen = 0.5 * (edges[1:] + edges[:-1])
        for j, (lab, y) in enumerate((("pre-detection", pre / pre.sum()), ("accepted (C1)", det / det.sum()))):
            st = series_style(j, style)
            ax.step(cen, y, where="mid", color=st["color"], ls=st["linestyle"], lw=2, label=lab)
        apply_axes_style(ax, style, xlabel=name, ylabel="weighted fraction (mock noW)")
        ax.legend(fontsize=style.tick_label_fontsize)
        with np.errstate(divide="ignore", invalid="ignore"):
            rep.append(f"  {name}: " + " ".join(f"{c:.2f}:{a:.3f}|{b_:.3f}|{d_:.4f}" for c, a, b_, d_ in
                                                zip(cen, pre / pre.sum(), det / det.sum(), det / pre)))
    fig.suptitle("Population in the bin: mock / real accepted counts (top); mock true M1, q, P before and after detection (bottom)",
                 fontsize=style.title_fontsize)
    fig.tight_layout()
    save_figure(fig, args.out_dir / "population_ratios.png", dpi=int(cfg.diagnostics.figure_dpi))

    summary = {"n_real_bin": int(nreal), "k": args.k, "p_mock_all": float(p_mock.mean()),
               "p_mock_sub": float(p_mock_sub.mean()), "p_inj": [f_inj, lo_inj, hi_inj], "n_inj": n_inj,
               "paired_diff": float(diff.mean()), "paired_diff_sd": float(dboot.std()),
               "cross_check_rung1_path": list(map(float, fx)), "symmetric_rung1_mock": list(map(float, fsy)),
               "symmetric_paired_diff": float(dsym.mean()), "symmetric_paired_diff_sd": float(dsb.std()), "cross_check_stored": list(map(float, fs)),
               "p_mock_lin_all": float(p_mock_lin.mean()), "p_mock_lin_sub": float(p_mock_lin[sel].mean()), "p_390": list(map(float, f390)),
               "n_390_systems": int(in390.sum())}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (args.out_dir / "report.txt").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
