#!/usr/bin/env python3
"""#628: snapshot a MIST bolometric-correction table set (default: GALEX FUV/NUV) — approved by Ryan 2026-11-06.

Downloads ``https://mist.science/BC_tables/v1/<SET>.txz`` (the v1 BC tables that go with the MIST v1.2 isochrones the
pipeline uses; the local ``~/.isochrones/BC/mist`` copy has only UBVRIplus and WISE) into a new timestamped directory
``<data-root>/reference/mist_bc/<UTC stamp>_<SET>/``: the archive, its members extracted with ``tarfile``'s ``data``
filter (no links, no absolute paths), and ``meta.yaml`` (URL, HTTP Last-Modified, size, sha256 of the archive and of
every member). Never overwrites a snapshot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path

import yaml

BASE = "https://mist.science/BC_tables/v1/"
MAX_BYTES = 1 << 30  # report anything above 1 GB instead of fetching it


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="GALEX")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    a = ap.parse_args(argv)
    url = f"{BASE}{a.set}.txz"
    head = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
    size = int(head.headers.get("Content-Length", "0"))
    if size > MAX_BYTES:
        raise SystemExit(f"{url} is {size / 1e9:.2f} GB > 1 GB: ask before fetching")
    date = dt.datetime.now(dt.timezone.utc)
    out = a.data_root / "reference" / "mist_bc" / f"{date.strftime('%Y%m%dT%H%M%SZ')}_{a.set}"
    out.mkdir(parents=True, exist_ok=False)
    arc = out / f"{a.set}.txz"
    with urllib.request.urlopen(url, timeout=300) as r, arc.open("wb") as fh:
        fh.write(r.read())
    ext = out / "extracted"
    ext.mkdir()
    with tarfile.open(arc, "r:xz") as tf:
        tf.extractall(ext, filter="data")
    members = {str(p.relative_to(ext)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ext.rglob("*")) if p.is_file()}
    (out / "meta.yaml").write_text(yaml.safe_dump({
        "issue": 628, "approved": "Ryan 2026-11-06", "source": "MIST v1 bolometric-correction tables (Choi et al. 2016; Dotter 2016)",
        "url": url, "last_modified": head.headers.get("Last-Modified"), "query_date": date.isoformat(),
        "size_bytes": arc.stat().st_size, "archive_sha256": hashlib.sha256(arc.read_bytes()).hexdigest(),
        "members_sha256": members,
    }, sort_keys=False))
    print(f"wrote {arc} ({arc.stat().st_size} bytes), {len(members)} members, to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
