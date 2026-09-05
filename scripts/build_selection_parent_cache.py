"""Build selection-parent cache: uncut photometry snapshot + NSS enrichment + MC.

Run after ``scripts/fetch_nss_enrichment.py`` has written
``data/dr3/gaia_snapshots/nss_enrichment/{query.ecsv,meta.yaml}``.

Example::

    .venv/bin/python scripts/build_selection_parent_cache.py
    .venv/bin/python scripts/build_selection_parent_cache.py --no-mc   # enrich only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from darkhunter_pop.config_loader import repo_root
from darkhunter_pop.sample_selection import load_selection_rows_from_uncut_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-mc",
        action="store_true",
        help="Skip Monte Carlo p_m2_above attachment (enrichment merge only)",
    )
    parser.add_argument(
        "--no-cache-read",
        action="store_true",
        help="Force rebuild even if a cache file already exists",
    )
    args = parser.parse_args(argv)

    photo_meta = (
        repo_root()
        / "data/dr3/gaia_snapshots/20260826T234425Z_3d3f740b080c/meta.yaml"
    )
    enrich_meta = repo_root() / "data/dr3/gaia_snapshots/nss_enrichment/meta.yaml"
    if not photo_meta.is_file():
        print(f"missing photometry snapshot meta: {photo_meta}", file=sys.stderr)
        return 1
    if not enrich_meta.is_file():
        print(
            f"missing enrichment meta: {enrich_meta}\n"
            "Run scripts/fetch_nss_enrichment.py --poll-job <JOBID> first.",
            file=sys.stderr,
        )
        return 1

    attach_mc = not args.no_mc
    print(
        f"build_selection_parent_cache: photo={photo_meta.name} "
        f"enrich={enrich_meta.parent.name} attach_mc={attach_mc}",
        flush=True,
    )
    t0 = time.time()
    rows = load_selection_rows_from_uncut_snapshot(
        photo_meta,
        enrichment_meta_path=enrich_meta,
        use_cache=not args.no_cache_read,
        attach_mc=attach_mc,
    )
    n_p = sum(1 for r in rows if r.get("p_m2_above") is not None)
    n_k1 = sum(1 for r in rows if r.get("k1_kms") is not None)
    n_sig = sum(1 for r in rows if r.get("k1_significance") is not None)
    print(
        f"build_selection_parent_cache: n_rows={len(rows)} "
        f"with_p_m2={n_p} with_k1={n_k1} with_k1_sig={n_sig} "
        f"elapsed_s={time.time() - t0:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
