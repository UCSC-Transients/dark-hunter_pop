#!/usr/bin/env python3
"""#581 (EPOCH_MODEL_SPEC §8.14): calibrate and validate the transit keep scale kappa_T(G) and the CCD-row keep rho(G)
on the looks-single single stars against DR3 (same rows), and check the #390 NSS set. Light analysis.

``calibrate``  kappa_T(G) = DR3 mean matched transits / mock mean transits per G bin (same rows; marginal, so GOST-grid
               scatter cancels); rho(G) = DR3 n_good_obs_al per matched transit / mock CCD rows per transit. Divides out
               the run's own kappa/rho when it had transit_recal on. Writes the config block (enabled: false).
``validate``   paired (median, mean, 10/90/99%) and marginal ratios of transits, CCD obs and N_vis by G, for any number of
               single-star runs; the #390 set (re-injected sim_n_obs / published matched transits).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_colour_analysis_572 as n  # noqa: E402

P = "/Users/rfoley/darkhunter/pop/dark-hunter_pop/data/dr3/gaia_snapshots/"
PARENT, QUALITY = P + "20261019T160642Z_gaia_source_parent_K1000000_plx0p2_bj", P + "20261027T054741Z_gaia_source_astrometric_quality_K1000000"
EDGES = np.array([6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])  # bright rows are few: one bin below 11


def load(run: Path) -> tuple[dict, dict, dict]:
    d = n.load_dr3(Path(PARENT), Path(QUALITY))
    m, meta = n.load_mock(run)
    m["n_transits"] = np.array([json.loads(x)["n_transits"] for x in (run / "singles.jsonl").read_text().splitlines()], float)
    r = m["row"]
    x = {"g": d["phot_g_mean_mag"][r], "mt": d["q_astrometric_matched_transits"][r], "ng": d["q_astrometric_n_good_obs_al"][r],
         "no": d["q_astrometric_n_obs_al"][r], "nv": d["visibility_periods_used"][r], "row": r}
    x["ok"] = np.isfinite(x["mt"]) & (x["mt"] > 0) & (m["n_obs"] > 0)
    return m, meta, x


def run_recal(meta: dict) -> dict | None:
    rc = meta["epoch_model"].get("transit_recal") or {}
    return rc if rc.get("enabled") else None


def cmd_calibrate(a: argparse.Namespace) -> None:
    m, meta, x = load(Path(a.run))
    rc = run_recal(meta)
    kap, rho, knots, rows = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        b = x["ok"] & (x["g"] >= lo) & (x["g"] < hi)
        gm = float(np.median(x["g"][b]))
        k0 = np.interp(gm, rc["transit_keep_scale_knots_g"], rc["transit_keep_scale"]) if rc else 1.0
        r0 = np.interp(gm, rc["ccd_row_keep_knots_g"], rc["ccd_row_keep"]) if rc else 1.0
        # mean over DR3 rows (each row once) vs mean over mock realizations of the same rows
        _, first = np.unique(x["row"][b], return_index=True)
        dr3_tr = float(np.mean(x["mt"][b][first]))
        dr3_rows = float(np.sum(x["ng"][b][first]) / np.sum(x["mt"][b][first]))
        mk_tr = float(np.mean(m["n_transits"][b]))
        mk_rows = float(np.sum(m["n_obs"][b]) / np.sum(m["n_transits"][b]))
        knots.append(gm)
        kap.append(k0 * dr3_tr / mk_tr)
        rho.append(r0 * dr3_rows / mk_rows)
        rows.append({"g_bin": [lo, hi], "g_median": gm, "n_rows": int(first.size), "dr3_mean_transits": dr3_tr, "mock_mean_transits": mk_tr,
                     "dr3_rows_per_transit": dr3_rows, "mock_rows_per_transit": mk_rows, "kappa": kap[-1], "rho": rho[-1]})
    res = {"run": str(a.run), "bins": rows, "knots": knots, "kappa": kap, "rho": rho}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for r_ in rows:
        print(f"G {r_['g_bin']}: transits DR3 {r_['dr3_mean_transits']:.2f} mock {r_['mock_mean_transits']:.2f} -> kappa {r_['kappa']:.4f};"
              f" rows/tr DR3 {r_['dr3_rows_per_transit']:.3f} mock {r_['mock_rows_per_transit']:.3f} -> rho {r_['rho']:.4f}")
    if a.write_config:
        f = lambda xs: "[" + ", ".join(f"{v:.4f}" for v in xs) + "]"
        k = "[" + ", ".join(f"{v:.3f}" for v in knots) + "]"
        block = f"""    # #581 (EPOCH_MODEL_SPEC §8.14; Ryan 2026-10-27): transit keep scale kappa_T(G) on p_keep and an independent CCD-row
    # keep rho(G) after gaiamock's prediction, calibrated on the looks-single single stars (same rows, DR3 quality snapshot
    # 20261027T054741Z): mean matched transits and n_good_obs_al per matched transit per G bin
    # (scripts/epoch_transit_recal_581.py calibrate). Adds "+tr1" to the noise/epoch version. OFF until Ryan adopts it.
    transit_recal:
      enabled: false
      version: tr1
      transit_keep_scale_knots_g: {k}
      transit_keep_scale: {f(kap)}
      ccd_row_keep_knots_g: {k}
      ccd_row_keep: {f(rho)}
      provenance: "{a.provenance}"
"""
        cp = Path("config/config.yaml")
        c = cp.read_text()
        c = re.sub(r"    # #581 \(EPOCH_MODEL_SPEC §8\.14.*?      provenance: \"[^\n]*\"\n", "", c, flags=re.S)
        anchor = "    # El-Badry et al. (2024) Sect. 3.3.1 bright-star excess noise; validation only.\n    excess_noise:\n"
        assert c.count(anchor) == 1
        cp.write_text(c.replace(anchor, block + anchor))
        print(block)


def stats(r: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    return {"median": float(np.median(r)), "mean": float(np.mean(r)), "p10": float(np.percentile(r, 10)),
            "p90": float(np.percentile(r, 90)), "p99": float(np.percentile(r, 99))}


def cmd_validate(a: argparse.Namespace) -> None:
    res: dict = {"singles": {}}
    gb = [(6, 12), (12, 13), (13, 15), (15, 17), (17, 19), (6, 19)]
    for spec in a.runs:
        lab, path = spec.split("=", 1)
        m, meta, x = load(Path(path))
        ok = x["ok"]
        out = {"version": meta["noise_model_version"]}
        for nm, num, den in (("transits", m["n_transits"], x["mt"]), ("ccd_obs", m["n_obs"], x["ng"]), ("n_vis", m["n_vis"], x["nv"])):
            out[nm] = {}
            for lo, hi in gb:
                b = ok & (x["g"] >= lo) & (x["g"] < hi)
                _, first = np.unique(x["row"][b], return_index=True)
                out[nm][f"G{lo}-{hi}"] = {"paired": stats(num[b] / den[b]), "ratio_of_means": float(np.mean(num[b]) / np.mean(den[b][first])),
                                         "marginal_q": {"mock": np.percentile(num[b], [5, 50, 95, 99]).tolist(),
                                                        "dr3": np.percentile(den[b][first], [5, 50, 95, 99]).tolist()}}
        b = ok
        out["rows_per_transit"] = {"mock": float(np.sum(m["n_obs"][b]) / np.sum(m["n_transits"][b])),
                                   "dr3_good": float(np.nansum(x["ng"][b]) / np.nansum(x["mt"][b]))}
        res["singles"][lab] = out
        a_ = out["transits"]["G6-19"]
        print(f"{lab} ({out['version']}): transits paired med {a_['paired']['median']:.4f} mean {a_['paired']['mean']:.4f}"
              f" p99 {a_['paired']['p99']:.3f} ratio-of-means {a_['ratio_of_means']:.4f} | ccd paired med"
              f" {out['ccd_obs']['G6-19']['paired']['median']:.4f} mean {out['ccd_obs']['G6-19']['paired']['mean']:.4f}"
              f" ratio-of-means {out['ccd_obs']['G6-19']['ratio_of_means']:.4f} | nvis mean {out['n_vis']['G6-19']['paired']['mean']:.4f}"
              f" | rows/tr {out['rows_per_transit']['mock']:.3f} vs {out['rows_per_transit']['dr3_good']:.3f}")
        for lo, hi in gb[:-1]:
            t = out["transits"][f"G{lo}-{hi}"]
            print(f"   G{lo}-{hi}: transits RoM {t['ratio_of_means']:.4f}, ccd RoM {out['ccd_obs'][f'G{lo}-{hi}']['ratio_of_means']:.4f},"
                  f" q99 mock/DR3 {t['marginal_q']['mock'][3]:.0f}/{t['marginal_q']['dr3'][3]:.0f}")
    if a.reinjections:
        import h5py

        with h5py.File(a.gate390, "r") as f:
            T = f["systems/truth"]
            mt = T["published_astrometric_matched_transits"][()]
            nv = T["published_visibility_periods_used"][()]
            ix = {int(s): i for i, s in enumerate(f["systems/source_id"][()])}
        res["reinjection_390"] = {}
        for spec in a.reinjections:
            lab, path = spec.split("=", 1)
            rows = [json.loads(z) for z in (Path(path) / "inject.jsonl").read_text().splitlines()]
            rows = [r for r in rows if r.get("sim_n_obs")]
            i = np.array([ix[r["source_id"]] for r in rows])
            tr = np.array([r.get("sim_n_transits", np.nan) for r in rows], float)
            vis = np.array([r["sim_n_visibility_periods"] for r in rows], float)
            res["reinjection_390"][lab] = {"transits": stats(tr / mt[i]) if np.isfinite(tr).any() else None, "n_vis": stats(vis / nv[i]),
                                           "transits_ratio_of_means": float(np.nanmean(tr) / np.mean(mt[i])) if np.isfinite(tr).any() else None}
            print(lab, "#390:", json.dumps(res["reinjection_390"][lab]))
    Path(a.out).write_text(json.dumps(res, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--write-config", action="store_true")
    p.add_argument("--provenance", default="")
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("validate")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--reinjections", nargs="*", default=[])
    p.add_argument("--gate390", default="/Users/rfoley/darkhunter/pop/dark-hunter_pop/output/gate390/injection_test_full.h5")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_validate)
    a = ap.parse_args(argv)
    a.func(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
