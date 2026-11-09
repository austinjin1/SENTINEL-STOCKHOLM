#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Promote the 2026-11-08 approval-status refresh of the SENTINEL-Sonde daily and hourly tiers
(lead-approved procedure, 2026-11-08).

For each tier: refuse unless its identity check (``compare_vs_<old>.json`` written by
scripts/data/compare_sonde_daily_snapshots.py) reports ``identical_until`` = true for every date
<= 2023-09-30. Then rename the old directory to its dated name, and immediately create a symlink at
a temporary name and ``mv -T`` it over the loader path (atomic switch). Write a new manifest under
the loader name (a copy of the refresh manifest, ``supersedes`` = SHA-256 of the old manifest, which
is also kept under the dated name), and append dated notes to the datasheet and RESULTS.md.

    python -B scripts/data/promote_sonde_snapshots.py --dry-run
    python -B scripts/data/promote_sonde_snapshots.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from sentinel.core import paths

TIERS = {  # loader name -> (refreshed dir, dated name for the old dir)
    "sonde_daily": ("sonde_daily_refresh_20261108", "sonde_daily_20261022"),
    "sonde_hourly": ("sonde_hourly_refresh_20261108", "sonde_hourly_20261024"),
}


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    P, M = paths.PROCESSED, paths.MANIFESTS
    checks = {}
    for name, (new, dated) in TIERS.items():
        cur = P / name
        if cur.is_symlink():
            raise SystemExit(f"{cur} is already a symlink (promoted?)")
        cmp_ = P / new / f"compare_vs_{name}.json"
        if not cmp_.is_file():
            raise SystemExit(f"identity check missing: {cmp_}")
        c = json.loads(cmp_.read_text())
        if not c.get("identical_until") or c.get("only_in_old"):
            raise SystemExit(f"{name}: identity check failed ({c.get('n_sites_differing_until')} sites differ)")
        if (P / dated).exists():
            raise SystemExit(f"{P / dated} exists")
        checks[name] = c
    print("identity checks passed:", {k: (v["n_sites_compared"], v["until"]) for k, v in checks.items()})
    if a.dry_run:
        return 0
    now = _dt.datetime.now().isoformat(timespec="seconds")
    record = {}
    for name, (new, dated) in TIERS.items():
        old_man = M / f"{name}.json"
        old_sha = sha(old_man)
        shutil.copy2(old_man, M / f"{dated}.json")          # old manifest kept under the dated name
        tmp = P / f".{name}.tmp_symlink"
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(new, tmp)                                  # relative target inside processed/
        os.rename(P / name, P / dated)                        # old dir -> dated name ...
        subprocess.run(["mv", "-T", str(tmp), str(P / name)], check=True)   # ... immediately atomic switch
        man = json.loads((M / f"{new}.json").read_text())
        man.update({"name": name, "created": now, "supersedes": {"manifest": f"{dated}.json", "sha256": old_sha},
                    "symlink_target": new,
                    "description": man.get("description", "") + " Approval-status refresh of WY2024-26 "
                    "(USGS re-pull 2026-11-08); rows dated <= 2023-09-30 identical to the superseded snapshot."})
        (M / f"{name}.json").write_text(json.dumps(man, indent=1, default=str))
        record[name] = {"old_dir": dated, "new_target": new, "superseded_manifest_sha256": old_sha,
                        "new_manifest_sha256": sha(M / f"{name}.json"),
                        "identity_check": str(P / new / f"compare_vs_{name}.json"),
                        "n_sites_compared": checks[name]["n_sites_compared"]}
    (P / "sonde_promotion_20261108.json").write_text(json.dumps({"promoted": now, "tiers": record}, indent=1))
    d = checks["sonde_daily"]["n_sites_compared"]
    h = checks["sonde_hourly"]["n_sites_compared"]
    note = (f"\n## Approval-status refresh (promoted {now[:10]})\n"
            f"WY2024-26 re-pulled from USGS (daily values {checks['sonde_daily']['until']} < date <= 2026-11-08; "
            "15-min for the core set) before the legacy WaterServices slow-down, built with refresh precedence "
            "inside the window only and pre-window rows copied verbatim. `processed/sonde_daily` -> "
            "`sonde_daily_refresh_20261108`, `processed/sonde_hourly` -> `sonde_hourly_refresh_20261108` "
            "(symlinks; old snapshots kept as `sonde_daily_20261022`, `sonde_hourly_20261024`; manifests list the "
            f"superseded manifest hash). Identity check: {d} daily and {h} hourly site tables identical on every "
            "date <= 2023-09-30 (`compare_vs_sonde_daily.json`, `compare_vs_sonde_hourly.json`), so no train/val "
            "value changes.\n")
    ds = paths.REPO_ROOT / "docs" / "datasheets" / "sonde_daily.md"
    ds.write_text(ds.read_text() + note)
    res = paths.REPO_ROOT / "RESULTS.md"
    res.write_text(res.read_text().rstrip("\n") + "\n\n" +
                   f"- **{now[:10]} — SENTINEL-Sonde data refresh (no train/val change).** The daily and hourly tiers "
                   "were rebuilt from a 2026-11-08 USGS re-pull of WY2024–26 (approval status for the approved-only "
                   f"EW-v2/M1 test scoring) and promoted. Identity check: {d} daily and {h} hourly site tables are "
                   "identical on every date ≤ 2023-09-30 (`processed/sonde_daily_refresh_20261108/compare_vs_sonde_daily.json`, "
                   "`processed/sonde_hourly_refresh_20261108/compare_vs_sonde_hourly.json`); every train/val number "
                   "in the registry is unaffected.\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
