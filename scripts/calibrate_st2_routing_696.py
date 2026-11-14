#!/usr/bin/env python3
"""#696 (EPOCH_MODEL_SPEC §8.16 st2): diagnose and recalibrate the st1 AstroSpectroSB1 routing against DR3, post hoc.

The st routing draw is ``u < P(ASB1 | x)``, ``u`` the first child of the per-CCD noise Generator and P read from the
fitted a0, ϖ, P, e and i, which the type offsets do not change. So for every stored orbital-branch draw the pre-routing
cascade is recovered exactly (``epoch_model.solution_type_properties_array(..., inverse=True)``) and the type, the
type-dependent σ / F2 and the acceptance under ANY routing version follow without gaiamock. The script refuses unless the
stored labels and acceptance are reproduced bit for bit under the stored (st1) routing.

Versions compared: ``st1`` (adopted 2026-10-31: DR3 logistic + iteration-2 intercept δ(G), taken from
``--st1-iter2``) and ``st2`` (the same logistic with δ ≡ 0; every other st1 parameter unchanged).

Outputs (JSON):

* **diagnosis**: the DR3 logistic on DR3 rows (no δ) by G bin; covariate quantiles at G 10–11 / 11–12 / 12–13 for the
  weighted mock, the unweighted mock and DR3; the constant intercept per G bin (and pooled) that makes the weighted
  post-cut AstroSpectroSB1 share equal DR3's, with a bootstrap error (mock draws, Poisson, + DR3 binomial);
* **validation** for both versions: weighted AstroSpectroSB1 share among accepted orbits by G, by the G = 13 split, by
  log P (all G and G < 13), log K1 and BP−RP, against DR3; the luminous gens with the rung-3 v5 weights and the
  compact-object gens with the co_v1 target weights per component; F2 and σ_ϖ medians by type at G 10–13; the change in
  the weighted accepted count.

Domain: accepted orbits with ``--g-min`` ≤ G < ``--g-max`` and fitted P ≤ ``--c1-period-max`` on both sides (DR3:
Orbital + AstroSpectroSB1 published orbits). Luminous draws accepted under a version but outside the exported v5 weights
(not accepted under st1) get a k-nearest-neighbour proxy weight (same generation; G, BP−RP, log M1, log M2, log P);
their number and the share with zero weight instead are reported.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from darkhunter_pop import epoch_model as em
from darkhunter_pop import joint_refit as jr
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config

G_EDGES = [10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 19.5]
LP_EDGES = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
K1_EDGES = [-1.0, 0.0, 0.5, 1.0, 1.5, 3.0]
C_EDGES = [-1.0, 0.6, 0.8, 1.0, 1.3, 5.0]
DIAG_G = [(10.0, 11.0), (11.0, 12.0), (12.0, 13.0), (10.0, 13.0)]
COVARIATES = ("grvs", "bp_rp", "logK1", "logP", "e")
FArr = NDArray[np.float64]


def accept(c: FArr, cuts: Any) -> NDArray[np.bool_]:
    """Vectorized ``forward_model.classify_cascade_result`` acceptance for orbital-branch cascades."""
    plx, splx, P, se, a0, sa0, f2 = (c[:, k] for k in (0, 1, 10, 15, 17, 18, 21))
    with np.errstate(invalid="ignore", divide="ignore"):
        a0s = a0 / sa0
        return ((plx > 0) & (P > 0) & (splx > 0) & (sa0 > 0) & (a0s > cuts.a0_over_err_min) & (f2 < cuts.goodness_of_fit_f2_max)
                & (plx / splx > cuts.parallax_over_error_times_period_min_days / P) & (a0s > cuts.a0_over_err_times_sqrt_period_min / np.sqrt(P))
                & (se < cuts.sigma_e_ln_period_slope * np.log(P) + cuts.sigma_e_intercept))


def load_draws(paths: list[Path], keep_truth: bool = False) -> dict[str, Any]:
    """Orbital-branch draws of the artifacts with stored labels (outcome or #690 sidecar) and routing uniforms; with
    ``keep_truth`` also the full truth tables (``"_truth"``, all draws), for the compact-object target weights."""
    keys = ("phot_g_mean_mag", "sys_bp_rp", "generation", "draw_index", "m1_msun", "m2_msun", "period_days")
    parts: dict[str, list[Any]] = {k: [] for k in (*keys, "cascade", "acc", "label", "u", "comp")}
    full: list[dict[str, Any]] = []
    for p in paths:
        truth, out, attrs = ps.read_proposal_artifact(p)
        if keep_truth:
            full.append(truth)
        cas = np.asarray(out["cascade"], float)
        ob = ~np.isin(cas[:, 0], list(em.CASCADE_NON_ORBITAL_FLAGS)) & (cas[:, em.CASCADE_SLOTS["period"]] > 0)
        if "nss_solution_type" in out:
            lab = np.char.decode(np.asarray(out["nss_solution_type"]).astype("S32"), "ascii")
        else:
            side = ps.read_nss_type_sidecar(p)
            if side is None:
                raise SystemExit(f"{p}: no stored solution type (run scripts/recover_nss_type_690.py)")
            if not np.array_equal(side[0], truth["draw_index"]):
                raise SystemExit(f"{p}: sidecar draw order differs")
            lab = side[1]
        primary = str(attrs.get("primary_model", ""))
        if "+ss1" in primary or "+sa1" in primary:
            raise SystemExit(f"{p}: per-source noise terms spawn before the routing draw; post-hoc routing not exact")
        base = int(json.loads(attrs["proposal_config_json"])["base_seed"])
        for k in keys:
            parts[k].append(np.asarray(truth[k])[ob])
        parts["cascade"].append(cas[ob])
        parts["acc"].append(np.asarray(out["accepted_orbital"], bool)[ob])
        parts["label"].append(np.asarray(lab)[ob].astype("U16"))
        parts["u"].append(jr.routing_uniforms(truth["generation"][ob], truth["draw_index"][ob], base))
        parts["comp"].append(np.asarray(truth["co_component"])[ob] if "co_component" in truth else np.full(int(ob.sum()), -1))
        print(f"{p.name}: {int(ob.sum())} orbital-branch draws, {int(out['accepted_orbital'].sum())} accepted", flush=True)
    res: dict[str, Any] = {k: np.concatenate(v) for k, v in parts.items()}
    if keep_truth:
        tk = set.intersection(*(set(t) for t in full))
        res["_truth"] = {k: np.concatenate([np.asarray(t[k]) for t in full]) for k in tk}
    return res


def dr3_rows(published: Path, input_columns: Path) -> dict[str, FArr]:
    import h5py

    with h5py.File(published, "r") as f:
        P = f["published"]
        pub = {k: P[k][()] for k in ("source_id", "g_mag", "period", "eccentricity", "parallax", "parallax_error", "goodness_of_fit",
                                     "a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes")}
        typ = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in P["nss_solution_type"][()]])
    cols = ps.load_real_input_columns(input_columns)
    sid, bpc = np.asarray(cols["source_id"], np.int64), np.asarray(cols["bp_rp"], float)
    o = np.argsort(sid)
    pos = np.clip(np.searchsorted(sid[o], pub["source_id"]), 0, sid.size - 1)
    bp = np.where(sid[o][pos] == pub["source_id"], bpc[o][pos], np.nan)
    A_, B_, F_, G_ = (pub[k] for k in ("a_thiele_innes", "b_thiele_innes", "f_thiele_innes", "g_thiele_innes"))
    u = 0.5 * (A_**2 + B_**2 + F_**2 + G_**2)
    v = A_ * G_ - B_ * F_
    a0 = np.sqrt(u + np.sqrt(np.clip(u**2 - v**2, 0, None)))
    inc = np.degrees(np.arccos(np.clip(v / a0**2, -1, 1)))
    keep = np.isin(typ, ["Orbital", "AstroSpectroSB1"])
    return {"g": pub["g_mag"][keep], "bp": bp[keep], "P": pub["period"][keep], "e": pub["eccentricity"][keep],
            "k1": em.predicted_k1_kms_array(a0, pub["parallax"], pub["period"], pub["eccentricity"], inc)[keep],
            "isA": (typ == "AstroSpectroSB1")[keep], "f2": pub["goodness_of_fit"][keep], "splx": pub["parallax_error"][keep]}


def features(st: em.SolutionTypeConfig, g: FArr, bp: FArr, P: FArr, e: FArr, k1: FArr) -> dict[str, FArr]:
    c = np.clip(np.where(np.isfinite(bp), bp, 1.0), *st.colour_clip)
    return {"grvs": g - np.polyval(st.grvs_poly, c), "bp_rp": c, "logK1": np.log10(np.maximum(k1, 1e-3)), "logP": np.log10(P), "e": e}


def knn_weights(M: dict[str, Any], w: FArr, have: NDArray[np.bool_], k: int) -> FArr:
    from scipy.spatial import cKDTree

    out = w.copy()
    X = np.column_stack([M["phot_g_mean_mag"], np.nan_to_num(M["sys_bp_rp"], nan=1.0), np.log10(M["m1_msun"]), np.log10(M["m2_msun"]),
                         np.log10(M["period_days"])])
    X = X / np.std(X[have], axis=0)
    for gg in np.unique(M["generation"]):
        a = (M["generation"] == gg) & have
        b = (M["generation"] == gg) & ~have
        if a.sum() >= k and b.any():
            _, j = cKDTree(X[a]).query(X[b], k=k)
            out[b] = np.median(w[a][j], axis=1)
    return out


def share(isA: NDArray[np.bool_], w: FArr, sel: NDArray[np.bool_], n_boot: int, rng: np.random.Generator) -> dict[str, float]:
    """Weighted share with a Poisson-bootstrap error, the Kish ESS and the number of draws."""
    m = sel & (w > 0)
    ww, aa = w[m], isA[m].astype(float)
    if ww.sum() <= 0:
        return {"share": float("nan"), "err": float("nan"), "ess": 0.0, "n": 0}
    s = float((ww * aa).sum() / ww.sum())
    if ww.size > 1:
        bw = rng.poisson(1.0, (n_boot, ww.size)) * ww
        tot = bw.sum(axis=1)
        err = float(np.std((bw @ aa)[tot > 0] / tot[tot > 0]))
    else:
        err = float("nan")
    return {"share": s, "err": err, "ess": float(ww.sum() ** 2 / (ww**2).sum()), "n": int(ww.size)}


def binom(isA: NDArray[np.bool_], sel: NDArray[np.bool_]) -> dict[str, float]:
    n = int(sel.sum())
    s = float(isA[sel].mean()) if n else float("nan")
    return {"share": s, "err": float(np.sqrt(s * (1 - s) / n)) if n else float("nan"), "n": n}


def zrow(m: dict[str, float], r: dict[str, float]) -> float:
    den = np.hypot(m["err"], r["err"])
    return float((m["share"] - r["share"]) / den) if np.isfinite(den) and den > 0 else float("nan")


def table(xm: FArr, isA: NDArray[np.bool_], w: FArr, base_m: NDArray[np.bool_], xr: FArr, rA: NDArray[np.bool_], base_r: NDArray[np.bool_],
          edges: list[float], n_boot: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = share(isA, w, base_m & (xm >= lo) & (xm < hi), n_boot, rng)
        r = binom(rA, base_r & (xr >= lo) & (xr < hi))
        rows.append({"bin": [lo, hi], "mock": m, "dr3": r, "z": zrow(m, r)})
    return rows


def wmedian(x: FArr, w: FArr) -> float:
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(np.interp(0.5 * c[-1], c, x[o])) if c.size and c[-1] > 0 else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lum-artifacts", type=Path, nargs="+", required=True)
    ap.add_argument("--lum-weights", type=Path, required=True, help="npz: generation, draw_index, weight (rung-3 v5 θ)")
    ap.add_argument("--co-artifacts", type=Path, nargs="*", default=[])
    ap.add_argument("--co-fragments", type=Path, nargs="*", default=[])
    ap.add_argument("--parent-dir", type=Path, default=None, help="parent snapshot (needed for the CO target weights)")
    ap.add_argument("--published-orbits", type=Path, required=True)
    ap.add_argument("--input-columns", type=Path, required=True, help="DR3 NSS input-columns snapshot (bp_rp)")
    ap.add_argument("--st1-iter2", type=Path, default=Path("docs/gate_noise_colour/st1/iter2.json"),
                    help="the st1 iteration-2 intercept δ(G) the stored draws were routed with")
    ap.add_argument("--g-min", type=float, default=10.0)
    ap.add_argument("--g-max", type=float, default=19.5)
    ap.add_argument("--c1-period-max", type=float, default=830.4)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-boot-delta", type=int, default=100)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--seed", type=int, default=696)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    cfg = load_config()
    cuts = cfg.dr3.selection_function_astrometric.orbital_solution_cuts
    st_cfg = em.epoch_model_config_from_mapping(cfg.dr3.epoch_model).solution_type_model
    if st_cfg is None:
        raise SystemExit("dr3.epoch_model.solution_type_model is off")
    it2 = json.loads(a.st1_iter2.read_text())["config"]
    versions = {
        "st1": dataclasses.replace(st_cfg, version="st1", routing_delta_knots_g=tuple(it2["routing_delta_knots_g"]),
                                   routing_delta=tuple(it2["routing_delta"])),
        "st2": dataclasses.replace(st_cfg, version="st2", routing_delta_knots_g=(), routing_delta=()),
    }
    R = dr3_rows(a.published_orbits, a.input_columns)
    rbase = (R["g"] >= a.g_min) & (R["g"] < a.g_max) & (R["P"] <= a.c1_period_max)
    rfeat = features(st_cfg, R["g"], R["bp"], R["P"], R["e"], R["k1"])
    res: dict[str, Any] = {"versions": {k: {"routing_delta_knots_g": list(v.routing_delta_knots_g), "routing_delta": list(v.routing_delta)}
                                        for k, v in versions.items()},
                           "domain": {"g": [a.g_min, a.g_max], "p_fit_max_days": a.c1_period_max}, "samples": {}}

    samples = [("luminous", a.lum_artifacts)] + ([("co", a.co_artifacts)] if a.co_artifacts else [])
    for name, paths in samples:
        M = load_draws(paths, keep_truth=(name == "co"))
        g, c = M["phot_g_mean_mag"], M["cascade"]
        stored_A = M["label"] == ps.NSS_TYPE_ASB1
        if not np.all(np.isin(M["label"], [ps.NSS_TYPE_ASB1, ps.NSS_TYPE_ORBITAL])):
            raise SystemExit(f"{name}: an orbital-branch draw has no type label")
        k1 = em.predicted_k1_kms_array(c[:, 17], c[:, 0], c[:, 10], c[:, 14], c[:, 16])
        c_pre = em.solution_type_properties_array(c, g, stored_A, st_cfg, inverse=True)
        p1 = versions["st1"].p_asb1_array(g, M["sys_bp_rp"], c[:, 10], c[:, 14], k1)
        if not np.array_equal(M["u"] < p1, stored_A):
            raise SystemExit(f"{name}: st1 routing does not reproduce the stored labels ({int(np.sum((M['u'] < p1) != stored_A))} differ)")
        if not np.array_equal(accept(c, cuts), M["acc"]):
            raise SystemExit(f"{name}: vectorized acceptance does not reproduce the stored accepted_orbital")
        if not np.allclose(em.solution_type_properties_array(c_pre, g, stored_A, st_cfg), c, rtol=1e-12, atol=0.0):
            raise SystemExit(f"{name}: type-property inverse does not round-trip")
        # weights
        if name == "luminous":
            lw = np.load(a.lum_weights, allow_pickle=True)
            key = {(int(x), int(y)): float(z) for x, y, z in zip(lw["generation"], lw["draw_index"], lw["weight"])}
            w = np.array([key.get((int(x), int(y)), np.nan) for x, y in zip(M["generation"], M["draw_index"])])
            have = np.isfinite(w)
            w = np.where(have, w, 0.0)
            wk = knn_weights(M, w, have, a.knn)
            wsets = {"all": wk}
            wnote = {"weights": str(a.lum_weights), "n_exported": int(have.sum()), "primary_model": str(lw["primary_model"])}
        else:
            from darkhunter_pop import compact_objects as cob

            if a.parent_dir is None or not a.co_fragments:
                raise SystemExit("--parent-dir and --co-fragments are required with --co-artifacts")
            frags = [ps.load_proposal_set_fragment(f) for f in a.co_fragments]
            gens = [f.proposal for f in frags]
            parent = ps.load_parent_snapshot(a.parent_dir, cfg, gens[0])
            T = M.pop("_truth")
            wf = cob.co_importance_weights(T, parent, gens, frags[0].target_co, partition="all")
            per = {(int(x), int(y)): float(z) for x, y, z in zip(T["generation"], T["draw_index"], wf)}
            del T
            wk = np.array([per[(int(x), int(y))] for x, y in zip(M["generation"], M["draw_index"])])
            have = np.ones(wk.size, bool)
            wsets = {"all": wk}
            for cname, code in cob.COMPONENT_CODES.items():
                wsets[cname] = np.where(M["comp"] == code, wk, 0.0)
            wnote = {"weights": "compact_objects.co_importance_weights(partition='all') at the co_v1 target", "target": frags[0].target_co.version}
        base = (g >= a.g_min) & (g < a.g_max) & (c[:, 10] <= a.c1_period_max)
        sres: dict[str, Any] = {"n_orbital_branch": int(g.size), "weights": wnote, "versions": {}}
        mfeat = features(st_cfg, g, M["sys_bp_rp"], c[:, 10], c[:, 14], k1)
        out_v: dict[str, Any] = {}
        for vn, st in versions.items():
            isA = M["u"] < st.p_asb1_array(g, M["sys_bp_rp"], c[:, 10], c[:, 14], k1)
            cv = em.solution_type_properties_array(c_pre, g, isA, st)
            acc = accept(cv, cuts)
            out_v[vn] = (isA, acc, cv)
            vres: dict[str, Any] = {"n_flipped_vs_stored": int(np.sum(isA != stored_A)), "n_acceptance_changed_vs_stored": int(np.sum(acc != M["acc"])),
                                    "n_accepted_without_exported_weight": int(np.sum(acc & base & ~have))}
            for wn, w in wsets.items():
                mb = base & acc
                t = {"G": table(g, isA, w, mb, R["g"], R["isA"], rbase, G_EDGES, a.n_boot, rng),
                     "G_split_13": table(g, isA, w, mb, R["g"], R["isA"], rbase, [a.g_min, 13.0, a.g_max], a.n_boot, rng),
                     "logP": table(mfeat["logP"], isA, w, mb, rfeat["logP"], R["isA"], rbase, LP_EDGES, a.n_boot, rng),
                     "logP_G_lt_13": table(mfeat["logP"], isA, w, mb & (g < 13), rfeat["logP"], R["isA"], rbase & (R["g"] < 13), LP_EDGES, a.n_boot, rng),
                     "logK1_G_lt_13": table(mfeat["logK1"], isA, w, mb & (g < 13), rfeat["logK1"], R["isA"], rbase & (R["g"] < 13), K1_EDGES, a.n_boot, rng),
                     "bp_rp_G_lt_13": table(M["sys_bp_rp"], isA, w, mb & (g < 13), R["bp"], R["isA"], rbase & (R["g"] < 13), C_EDGES, a.n_boot, rng),
                     "overall": table(g, isA, w, mb, R["g"], R["isA"], rbase, [a.g_min, a.g_max], a.n_boot, rng)[0]}
                if name == "luminous" and wn == "all":  # sensitivity: accepted draws without an exported weight at 0 instead of the kNN proxy
                    t["G_split_13_proxy_weights_zero"] = table(g, isA, np.where(have, wk, 0.0), mb, R["g"], R["isA"], rbase, [a.g_min, 13.0, a.g_max], a.n_boot, rng)
                    by_type = {}
                    for typ, mm, rr in (("AstroSpectroSB1", isA, R["isA"]), ("Orbital", ~isA, ~R["isA"])):
                        s = mb & (g < 13) & mm & (w > 0)
                        rs = rbase & (R["g"] < 13) & rr
                        by_type[typ] = {"f2_median": [wmedian(cv[s, 21], w[s]), float(np.median(R["f2"][rs]))],
                                        "log_sigma_plx_median": [wmedian(np.log10(cv[s, 1]), w[s]), float(np.median(np.log10(R["splx"][rs])))]}
                    t["G_10_13_by_type_mock_vs_dr3"] = by_type
                    t["weighted_accepted_G_lt_13"] = float(np.sum(w * (mb & (g < 13))))
                vres[wn] = t
            sres["versions"][vn] = vres
        # diagnosis (luminous: rung-3 v5 weights; CO: target weights)
        w = wsets["all"]
        isA1, acc1, _ = out_v["st1"]
        diag: dict[str, Any] = {"dr3_logistic_no_delta_on_dr3": [], "covariates": {}, "delta_solve": []}
        p_dr3 = versions["st2"].p_asb1_array(R["g"], R["bp"], R["P"], R["e"], R["k1"])
        for lo, hi in DIAG_G:
            rs = rbase & (R["g"] >= lo) & (R["g"] < hi)
            diag["dr3_logistic_no_delta_on_dr3"].append({"bin": [lo, hi], "dr3_share": float(R["isA"][rs].mean()), "model_mean_p": float(p_dr3[rs].mean())})
            ms = base & acc1 & (g >= lo) & (g < hi) & (w > 0)
            qs = [0.1, 0.5, 0.9]
            diag["covariates"][f"{lo}-{hi}"] = {
                k: {"mock_weighted": [float(np.interp(q * w[ms].sum(), np.cumsum(w[ms][np.argsort(mfeat[k][ms])]), np.sort(mfeat[k][ms]))) for q in qs],
                    "mock_unweighted": np.quantile(mfeat[k][ms], qs).tolist(), "dr3": np.quantile(rfeat[k][rs], qs).tolist()} for k in COVARIATES}
        p0 = np.clip(versions["st2"].p_asb1_array(g, M["sys_bp_rp"], c[:, 10], c[:, 14], k1), 1e-300, 1 - 1e-16)
        z0 = np.log(p0) - np.log1p(-p0)  # logit of the δ-free P
        sub = base & (g < 13)
        zs, us, gs, cs, ws = z0[sub], M["u"][sub], g[sub], c_pre[sub], w[sub]

        def post_share(d: float, lo: float, hi: float, wv: FArr) -> float:
            isA = us < 1.0 / (1.0 + np.exp(-(zs + d)))
            acc = accept(em.solution_type_properties_array(cs, gs, isA, st_cfg), cuts)
            s = acc & (gs >= lo) & (gs < hi)
            return float((wv * s * isA).sum() / (wv * s).sum())

        def solve(lo: float, hi: float, target: float, wv: FArr) -> float:
            x0, x1 = -3.0, 3.0
            for _ in range(18):
                mid = 0.5 * (x0 + x1)
                x0, x1 = (x0, mid) if post_share(mid, lo, hi, wv) > target else (mid, x1)
            return 0.5 * (x0 + x1)

        for lo, hi in DIAG_G:
            rs = rbase & (R["g"] >= lo) & (R["g"] < hi)
            tgt, nr = float(R["isA"][rs].mean()), int(rs.sum())
            d0 = solve(lo, hi, tgt, ws)
            bs = [solve(lo, hi, rng.binomial(nr, tgt) / nr, ws * rng.poisson(1.0, ws.size)) for _ in range(a.n_boot_delta)]
            sd = float(np.std(bs))
            diag["delta_solve"].append({"bin": [lo, hi], "dr3_share": tgt, "delta": d0, "err": sd, "z_vs_0": d0 / sd if sd > 0 else float("nan"),
                                        "share_at_delta0": post_share(0.0, lo, hi, ws),
                                        "st1_delta_at_bin_centre": float(np.interp(0.5 * (lo + hi), it2["routing_delta_knots_g"], it2["routing_delta"]))})
            print(f"{name} δ solve G {lo}-{hi}: {d0:+.3f} ± {sd:.3f} (DR3 {tgt:.3f})", flush=True)
        sres["diagnosis"] = diag
        res["samples"][name] = sres
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))
    for name, s in res["samples"].items():
        for vn, v in s["versions"].items():
            for wn in [k for k in v if isinstance(v[k], dict)]:
                rows = v[wn]["G_split_13"] + v[wn]["G"][:3]
                print(f"{name} {vn} {wn}: " + " | ".join(f"G{r['bin'][0]:g}-{r['bin'][1]:g} {r['mock']['share']:.3f}±{r['mock']['err']:.3f} vs {r['dr3']['share']:.3f} z{r['z']:+.1f}" for r in rows))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
