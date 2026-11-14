#!/usr/bin/env python3
"""#704: validate the luminous-covariance pre-filter on compact-object generations (which carry covariances).

For every accepted orbital draw of the given CO generations (current noise model, st2) this runs each published
sample's FULL forward_model cut chain on the mock rows with their covariances, exactly as rung 5 will:

* Andrews (``andrews2022`` and ``andrews2022_modified``, forward_model): ``andrews_atf_*`` columns from
  ``mock_atf_columns.mock_rows_atf_columns`` (pass-2 M1 = the parent row's TAG10 posterior);
* El-Badry 2024, A22 part: Andrews members with G < the El-Badry 2024 ``g_mag_limit`` (its Table 3 pool is external
  membership; a mock row is never a member);
* El-Badry 2026 (forward_model, ``membership['andrews2022']`` bound): ``primary_ns_bh``, ``andrews2022_import``,
  ``sub_chandrasekhar`` survivors (dust maps per the config).

and the pre-filter (``luminous_covariance.evaluate_prefilter``, covariance-free) on the same rows. The gate: the
number of draws OUTSIDE the filter that pass ANY of these chains must be <= ``validation.max_outside_filter_passing``
(0). Margins (each filter quantity over its threshold, on the draws that pass a chain) are reported. Exit 1 on a gate
failure. Writes a JSON report (never overwrites).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from darkhunter_pop import luminous_covariance as lc
from darkhunter_pop import proposal_set as ps
from darkhunter_pop.config_loader import load_config, repo_root
from darkhunter_pop.mock_atf_columns import mock_rows_atf_columns
from darkhunter_pop.mock_selection_rows import forward_model_selection, mock_draws_to_selection_rows
from darkhunter_pop.sample_selection import SampleSelectionRegistry

ANDREWS_SAMPLES = ("andrews2022", "andrews2022_modified")
ELBADRY2026_SUBSAMPLES = ("primary_ns_bh", "andrews2022_import", "sub_chandrasekhar")


def elbadry2024_g_limit(registry: Any) -> float:
    """``g_mag_faint_limit`` of El-Badry 2024's ``g_mag_limit`` cut (frozen file)."""
    spec = registry.resolved("elbadry2024")
    vals = [float(c.parameters["g_mag_faint_limit"]) for b in spec.branches or [] for s in b.subsamples or [] for c in s.cuts
            if (c.parameters or {}).get("g_mag_faint_limit") is not None]
    if len(set(vals)) != 1:
        raise ValueError(f"elbadry2024: expected one g_mag_faint_limit, got {vals}")
    return vals[0]


def chain_members(rows: list[dict[str, Any]], registry: Any, *, workers: int, extinction_lookup: Any = None) -> dict[str, set[int]]:
    """Survivor ids of every covariance-dependent (and M~2) sample chain on ``rows`` (module docstring)."""
    spec_a = registry.resolved("andrews2022")
    orb = [r for r in rows if r["nss_solution_type"] == "Orbital"]
    atf = mock_rows_atf_columns(orb, spec_a, m1_msun=np.array([r["pipeline_m1_msun"] for r in orb], float),
                                m1_sigma_msun=np.array([r.get("pipeline_m1_sigma_msun", np.nan) for r in orb], float), workers=workers)
    out: dict[str, set[int]] = {}
    for name in ANDREWS_SAMPLES:
        out[name] = set(int(s) for s in forward_model_selection(registry, name, reproduction_columns=atf).evaluate(rows).surviving_source_ids)
    g_lim = elbadry2024_g_limit(registry)
    gmag = {int(r["source_id"]): float(r["phot_g_mean_mag"]) for r in rows}
    out["elbadry2024_a22"] = {s for s in out["andrews2022"] if gmag[s] < g_lim}
    res = forward_model_selection(registry, "elbadry2026", extinction_lookup=extinction_lookup).evaluate(
        rows, membership={"andrews2022": frozenset(out["andrews2022"])})
    for sub in ELBADRY2026_SUBSAMPLES:
        out[f"elbadry2026_{sub}"] = set(int(s) for s in res.subsample_surviving.get(sub, ()))
    return out


def margins(pf: lc.PrefilterResult, members: set[int], cfg: lc.LuminousCovarianceConfig, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Filter quantities of the chain passers relative to the thresholds (> 1 / < 1 = inside with margin)."""
    k = np.array([int(s) in members for s in pf.source_id], dtype=bool)
    if not k.any():
        return {"n": 0}
    e = cfg.prefilter.elbadry2026
    g = np.array([float(r["phot_g_mean_mag"]) for r in rows])[k]
    p = np.array([float(r["period_day"]) for r in rows])[k]
    with np.errstate(invalid="ignore"):
        return {
            "n": int(k.sum()),
            "min_andrews_m2_point_over_threshold": float(np.nanmin(pf.andrews_m2_point_msun[k]) / cfg.prefilter.andrews.m2_point_min_msun)
            if np.any(np.isfinite(pf.andrews_m2_point_msun[k])) else None,
            "min_m2_tilde_over_threshold": float(np.nanmin(pf.m2_tilde_msun[k]) / e.m2_tilde_min_msun) if np.any(np.isfinite(pf.m2_tilde_msun[k])) else None,
            "max_period_over_threshold": float(np.max(p) / e.period_days_max),
            "max_g_minus_limit": float(np.max(g) - e.g_mag_faint_limit),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--artifacts", type=Path, nargs="+", required=True, help="CO generation artifacts (current noise model), read only")
    ap.add_argument("--parent-dir", type=Path, required=True, help="gaia_source parent snapshot (pass-2 TAG10 M1 of the parent rows)")
    ap.add_argument("--config", type=Path, default=Path("config/config.yaml"))
    ap.add_argument("--cov-config", type=Path, default=Path(lc.DEFAULT_CONFIG_PATH))
    ap.add_argument("--out", type=Path, default=Path("docs/gate704/prefilter_validation.json"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=None, help="smoke only: first N accepted rows per generation")
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists; never overwritten")
    cfg = load_config(a.config)
    ccfg = lc.load_luminous_covariance_config(a.cov_config)
    registry = SampleSelectionRegistry(cfg, repo=repo_root())
    bad = lc.check_prefilter_looser(registry, ccfg)
    if bad:
        raise SystemExit("pre-filter not looser than the frozen cuts: " + "; ".join(bad))
    t0 = time.time()
    per_gen: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    all_margins: dict[str, list[dict[str, Any]]] = {}
    for path in a.artifacts:
        truth, outcome, attrs = ps.read_proposal_artifact(path)
        if "orbit_cov" not in outcome:
            raise SystemExit(f"{path}: no orbit_cov (validation needs compact-object generations)")
        if "nss_solution_type" not in outcome:
            raise SystemExit(f"{path}: no outcome/nss_solution_type")
        gen = int(np.unique(truth["generation"])[0])
        if gen not in ccfg.validation.generations:
            raise SystemExit(f"{path}: generation {gen} not in validation.generations {list(ccfg.validation.generations)}")
        prop = ps.ProposalConfig.model_validate_json(attrs["proposal_config_json"])
        parent = ps.load_parent_snapshot(a.parent_dir, cfg, prop)
        if str(parent.meta["parent_h5_sha256"]) != str(attrs.get("parent_h5_sha256")):
            raise SystemExit(f"{path}: --parent-dir checksum differs from the artifact's parent ({attrs.get('parent_snapshot_dir')})")
        rows = mock_draws_to_selection_rows(truth, outcome, parent_columns=parent.columns, config=cfg, include_orbit=True,
                                            max_rows=a.max_rows)
        del truth, outcome
        pf = lc.evaluate_prefilter(rows, ccfg, registry=registry)
        members = chain_members(rows, registry, workers=a.workers)
        union = set().union(*members.values())
        inside = {int(s) for s in pf.source_id[pf.pass_any]}
        outside_pass = sorted(union - inside)
        per_sample_outside = {k: len(v - inside) for k, v in members.items()}
        per_gen.append({
            "artifact": str(path), "generation": gen, "noise_model": prop.noise_model, "n_rows": len(rows),
            "n_pass_andrews": int(pf.pass_andrews.sum()), "n_pass_elbadry2026": int(pf.pass_elbadry2026.sum()),
            "n_pass_any": int(pf.pass_any.sum()), "n_chain_members": {k: len(v) for k, v in members.items()},
            "n_union_members": len(union), "n_outside_filter_passing": len(outside_pass), "outside_by_sample": per_sample_outside,
        })
        for d in outside_pass[:50]:
            j = int(np.flatnonzero(pf.source_id == d)[0])
            violations.append({"generation": gen, "draw_index": d, "andrews_m2_point_msun": float(pf.andrews_m2_point_msun[j]),
                               "m2_tilde_msun": float(pf.m2_tilde_msun[j]), "samples": sorted(k for k, v in members.items() if d in v)})
        for k, v in members.items():
            all_margins.setdefault(k, []).append(margins(pf, v, ccfg, rows))
        print(json.dumps(per_gen[-1]), flush=True)
    n_out = sum(g["n_outside_filter_passing"] for g in per_gen)
    ok = n_out <= ccfg.validation.max_outside_filter_passing
    try:
        commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    rep = {"gate": "PASS" if ok else "FAIL", "n_outside_filter_passing": n_out,
           "max_outside_filter_passing": ccfg.validation.max_outside_filter_passing, "prefilter": ccfg.prefilter.model_dump(mode="json"),
           "frozen_bounds": lc.frozen_bounds(registry, ccfg).__dict__, "generations": per_gen, "violations": violations,
           "margins": all_margins, "smoke_max_rows": a.max_rows, "wall_minutes": (time.time() - t0) / 60, "code_commit": commit}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + ".tmp")
    tmp.write_text(json.dumps(rep, indent=1, default=list))
    tmp.rename(a.out)
    print(f"{rep['gate']}: {n_out} draws outside the pre-filter pass a sample chain (allowed {ccfg.validation.max_outside_filter_passing}); {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
