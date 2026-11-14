#!/usr/bin/env python3
"""Reproduce Shahaf et al. (2023b) class-III membership on real DR3 with the reimplemented triage (#682).

Runs every step of :mod:`darkhunter_pop.shahaf2023b_triage` on the real DR3 NSS rows and compares each step with the
authors' published per-source tables, so any disagreement is localised to one step:

1. parent cuts (§3.1)      vs the published step counts and the Table 1 source set (101,380);
2. M1 and AMRF (Eq. 1)     vs Table 1 ``M1`` / ``A`` (and ``e_A`` against first-order propagation);
3. Pr III / Pr II (§3.3)   vs Table 1 ``PIII`` / ``PII`` (Monte-Carlo noise expected: pull per source);
4. membership (§4, BH FDR) vs Table 2 (177), and the published-Pr III BH replay (decision step alone).

Inputs (read-only): the NSS enrichment ECSV (``corr_vec`` / ``bit_index`` / ``significance`` / errors), a
``gaiadr3.binary_masses`` ECSV and the VizieR ``J/MNRAS/518/2991`` ``table1`` / ``table2`` ECSVs
(``scripts/fetch_shahaf2023b_triage_inputs.py``). Writes ``summary.yaml`` and ``per_source.csv`` to ``--out-dir``.

Heavy: ~1e5 sources x 1e5 draws; run under ``nice`` with ``--workers`` (each worker pins BLAS to one thread).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from darkhunter_pop import shahaf2023b_triage as tri  # noqa: E402
from darkhunter_pop.nss_covariance import reconstruct_nss_covariance  # noqa: E402

_DEF: tri.TriageDefinition | None = None


def _read_ecsv(path: Path) -> pd.DataFrame:
    """Large NSS ECSV via pandas (string columns arrive ECSV-quoted; callers strip them)."""
    return pd.read_csv(path, sep=" ", comment="#", quotechar='"', low_memory=False)


def _read_table(path: Path) -> pd.DataFrame:
    """Small ECSV via astropy (proper string / masked handling)."""
    from astropy.table import Table

    t = Table.read(path, format="ascii.ecsv")
    return t.filled().to_pandas() if t.has_masked_columns else t.to_pandas()


def _init(def_path: str) -> None:
    global _DEF
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)  # 7x7 products per chunk: BLAS threads only oversubscribe the pool (#408)
    _DEF = tri.load_definition(def_path)


def _work(rec: dict[str, Any]) -> dict[str, Any]:
    assert _DEF is not None
    res = reconstruct_nss_covariance(rec)
    if not res.ok:
        return {"source_id": rec["source_id"], "status": f"covariance:{res.status}"}
    mean, cov = tri.orbit_mean_covariance(res.parameter_set)
    c = tri.class_probabilities(
        mean, cov, rec["m1"], rec["m1_lower"], rec["m1_upper"], source_id=int(rec["source_id"]), definition=_DEF
    )
    return {"source_id": rec["source_id"], "status": "ok", "n_draws": c.n_draws, "n_class2": c.n_class2,
            "n_class3": c.n_class3, "p_iii": c.p_iii, "p_ii": c.p_ii}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nss-ecsv", type=Path, required=True)
    ap.add_argument("--binary-masses-ecsv", type=Path, required=True)
    ap.add_argument("--table1-ecsv", type=Path, required=True)
    ap.add_argument("--table2-ecsv", type=Path, required=True)
    ap.add_argument("--definition", default=tri.DEFAULT_DEFINITION_PATH)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="smoke: classify only the first N clean rows")
    args = ap.parse_args(argv)
    definition = tri.load_definition(args.definition)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"issue": 682, "definition_sha256": tri.definition_sha256(args.definition)}

    nss = _read_ecsv(args.nss_ecsv).rename(columns={"SOURCE_ID": "source_id"})
    nss["nss_solution_type"] = nss["nss_solution_type"].astype(str).str.strip('"')
    bm = _read_table(args.binary_masses_ecsv)
    for c in ("combination_method", "m1_ref"):
        bm[c] = bm[c].fillna("").astype(str).str.strip('"')
    bm_sel = bm[bm.m1_ref == definition.parent.binary_masses_m1_ref]
    bm_sel = bm_sel[bm_sel.combination_method.isin(definition.parent.binary_masses_combination_methods)]
    if bm_sel.source_id.duplicated().any():
        raise SystemExit("duplicate IsocLum binary_masses rows for one source")
    d = nss.merge(bm_sel[["source_id", "m1", "m1_lower", "m1_upper", "m1_ref", "combination_method"]],
                  on="source_id", how="left")
    d["binary_masses_m1_ref"] = d["m1_ref"].fillna("")
    d["binary_masses_combination_method"] = d["combination_method"].fillna("")
    cols = {k: d[k].to_numpy() for k in d.columns}
    masks = tri.parent_cut_masks(cols, definition)
    acc = np.ones(len(d), dtype=bool)
    steps = {}
    for cut_id, m in masks.items():
        acc &= m
        steps[cut_id] = int(acc.sum())
    summary["parent_cumulative_counts"] = steps
    summary["parent_expected_counts"] = definition.parent.expected_counts
    clean = d[acc].copy()
    t1 = _read_table(args.table1_ecsv)
    t2 = _read_table(args.table2_ecsv)
    ours, theirs = set(clean.source_id.astype(np.int64)), set(t1.GaiaDR3.astype(np.int64))
    summary["parent_vs_table1"] = {"ours": len(ours), "table1": len(theirs), "ours_only": len(ours - theirs),
                                   "table1_only": len(theirs - ours)}

    # Step 2: M1 and AMRF.
    clean["amrf"] = tri.amrf(clean.a_thiele_innes, clean.b_thiele_innes, clean.f_thiele_innes, clean.g_thiele_innes,
                             clean.parallax, clean.period, clean.m1, definition)
    j = clean.merge(t1, left_on="source_id", right_on="GaiaDR3", how="inner")
    summary["m1_max_abs_diff"] = float(np.max(np.abs(j.m1 - j.M1)))
    summary["amrf_max_rel_diff"] = float(np.max(np.abs(j.amrf / j.A - 1.0)))

    # Step 4a: decision alone, on the published Pr III.
    n = definition.monte_carlo.n_draws
    pub_counts = [tri.ClassCounts(n, 0, int(round(p * (n + 1))) - 1, True) for p in t1.PIII.to_numpy()]
    pub_p = np.array([c.p_value(definition.decision.p_value) for c in pub_counts])
    pub_sel = tri.benjamini_hochberg(pub_p, definition.decision.alpha)
    t2_ids = set(t2.GaiaDR3.astype(np.int64))
    pub_ids = set(t1.GaiaDR3.to_numpy()[pub_sel].astype(np.int64))
    summary["decision_on_published_pr_iii"] = {"n": len(pub_ids), "equals_table2": pub_ids == t2_ids}

    # Step 3: Monte Carlo on every clean row.
    recs = clean.to_dict("records")
    if args.limit:
        recs = recs[: args.limit]
    t0 = time.time()
    with mp.Pool(args.workers, initializer=_init, initargs=(str(args.definition),)) as pool:
        out = pool.map(_work, recs, chunksize=64)
    summary["mc_seconds"] = round(time.time() - t0, 1)
    res = pd.DataFrame(out)
    res = res.merge(j[["source_id", "M1", "A", "e_A", "PII", "PIII", "amrf", "m1", "m1_lower", "m1_upper"]],
                    on="source_id", how="left")
    ok = res.status == "ok"
    summary["mc_status_counts"] = res.status.value_counts().to_dict()
    nb_pub = np.round((1.0 - res.PIII) * (n + 1))
    nb_our = res.n_draws - res.n_class3
    var = np.maximum(nb_pub * (1 - nb_pub / n), 1.0) * 2.0
    res["pull_n_below"] = (nb_our - nb_pub) / np.sqrt(var)
    res["n_below_published"] = nb_pub
    inside = (res.m1 - 4 * (res.m1 - res.m1_lower) > definition.curve_m1[0]) & (
        res.m1 + 4 * (res.m1_upper - res.m1) < definition.curve_m1[-1])
    for lab, sel in (("all", ok), ("m1_inside_grid_4sigma", ok & inside), ("m1_near_grid_edge", ok & ~inside)):
        pz = res.pull_n_below[sel]
        summary[f"pr_iii_pull_{lab}"] = {"n": int(sel.sum()), "chi2_per_row": float(np.mean(pz**2)),
                                         "frac_abs_lt3": float(np.mean(np.abs(pz) < 3))}

    # Step 4: BH over our Pr III.
    counts = [tri.ClassCounts(int(r.n_draws), int(r.n_class2), int(r.n_class3), True) for r in res[ok].itertuples()]
    p = np.array([c.p_value(definition.decision.p_value) for c in counts])
    sel = tri.benjamini_hochberg(p, definition.decision.alpha) if not args.limit else np.zeros(len(p), bool)
    our_ids = set(res[ok].source_id.to_numpy()[sel].astype(np.int64))
    res["class3_reimpl"] = False
    res.loc[res[ok].index[sel], "class3_reimpl"] = True
    res["in_table2"] = res.source_id.isin(t2_ids)
    fixed = np.array([tri.fixed_threshold_class3(c, definition) for c in counts])
    summary["membership"] = {
        "ours_bh": len(our_ids), "table2": len(t2_ids), "both": len(our_ids & t2_ids),
        "ours_only": sorted(int(x) for x in our_ids - t2_ids),
        "table2_only": sorted(int(x) for x in t2_ids - our_ids),
        "ours_fixed_threshold": int(fixed.sum()),
    }
    diff_ids = (our_ids ^ t2_ids)
    summary["membership_differences"] = [
        {k: (int(v) if isinstance(v, (np.integer,)) else (float(v) if isinstance(v, (float, np.floating)) else v))
         for k, v in r.items()}
        for r in res[res.source_id.isin(diff_ids)][
            ["source_id", "m1", "A", "n_below_published", "n_draws", "n_class3", "pull_n_below", "in_table2"]
        ].to_dict("records")
    ]
    res.to_csv(args.out_dir / "per_source.csv", index=False)
    (args.out_dir / "summary.yaml").write_text(yaml.safe_dump(json.loads(json.dumps(summary, default=str)),
                                                              sort_keys=False))
    print(yaml.safe_dump(json.loads(json.dumps(summary, default=str)), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
