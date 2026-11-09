#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Promote the rebuilt EW-v2 H2 derived-chloride directory (review 2026-11-08 C1: the H2 series must
come from the promoted sonde_daily snapshot, not from the superseded 2026-10-22 pull).

Same procedure as scripts/data/promote_sonde_snapshots.py: refuse unless the identity check
``compare_vs_chloride_derived.json`` (written by scripts/data/build_chloride_derived.py --compare-with)
reports identical regressions and identical derived series on every date <= 2023-09-30; then rename
the old directory to its dated name and atomically switch ``processed/chloride_derived`` to a symlink
to the rebuilt directory; write the manifest under the loader name (``supersedes`` = old manifest sha,
old manifest kept under the dated name) and append a dated note to the datasheet.

    python -B scripts/data/promote_chloride_derived.py --dry-run
    python -B scripts/data/promote_chloride_derived.py
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

NAME, NEW, DATED = "chloride_derived", "chloride_derived_refresh_20261108", "chloride_derived_20261025"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    P, M = paths.PROCESSED, paths.MANIFESTS
    cur = P / NAME
    if cur.is_symlink():
        raise SystemExit(f"{cur} is already a symlink (promoted?)")
    c = json.loads((P / NEW / f"compare_vs_{NAME}.json").read_text())
    if not (c.get("identical_until") and c.get("regressions_identical") and c.get("until") == "2023-09-30"):
        raise SystemExit(f"identity check failed: {c}")
    if (P / DATED).exists():
        raise SystemExit(f"{P / DATED} exists")
    man = json.loads((M / f"{NEW}.json").read_text())
    sd = (man.get("stats") or {}).get("sonde_daily_dir")
    if sd != str((P / "sonde_daily").resolve()):
        raise SystemExit(f"{NEW} was built from {sd}, not from the loader's sonde_daily")
    print("identity check passed:", {k: c[k] for k in ("n_sites_compared", "n_regressions", "n_eligible", "until")})
    if a.dry_run:
        return 0
    now = _dt.datetime.now().isoformat(timespec="seconds")
    old_sha = sha(M / f"{NAME}.json")
    shutil.copy2(M / f"{NAME}.json", M / f"{DATED}.json")
    tmp = P / f".{NAME}.tmp_symlink"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(NEW, tmp)
    os.rename(P / NAME, P / DATED)
    subprocess.run(["mv", "-T", str(tmp), str(P / NAME)], check=True)
    man.update({"name": NAME, "created": now, "supersedes": {"manifest": f"{DATED}.json", "sha256": old_sha},
                "symlink_target": NEW,
                "description": man.get("description", "") + " Rebuilt 2026-11-08 from the promoted sonde_daily "
                "(USGS re-pull of WY2024-26); regressions and series <= 2023-09-30 identical to the superseded build."})
    (M / f"{NAME}.json").write_text(json.dumps(man, indent=1, default=str))
    rec = {"promoted": now, "old_dir": DATED, "new_target": NEW, "superseded_manifest_sha256": old_sha,
           "new_manifest_sha256": sha(M / f"{NAME}.json"), "refresh_manifest_sha256": sha(M / f"{NEW}.json"),
           "identity_check": str(P / NEW / f"compare_vs_{NAME}.json")}
    (P / "chloride_derived_promotion_20261108.json").write_text(json.dumps(rec, indent=1))
    ds = paths.REPO_ROOT / "docs" / "datasheets" / "chloride_derived.md"
    ds.write_text(ds.read_text().rstrip("\n") + "\n\n"
                  f"## Rebuild from the promoted daily tier (promoted {now[:10]}; EW-v2 pre-test re-review C1)\n"
                  "The 2026-10-25 build read `sonde_daily` when it was the 2026-10-22 USGS pull. After the 2026-11-08 "
                  "approval-status refresh was promoted, the series were rebuilt from the promoted daily tier "
                  f"(`build_chloride_derived.py --out-name {NEW} --compare-with {NAME}`). `processed/{NAME}` -> `{NEW}` "
                  f"(symlink; old build kept as `{DATED}`). Identity check (`compare_vs_{NAME}.json`): all "
                  f"{c['n_regressions']} candidate-site regressions identical (a, b, R², n_pairs, eligibility; "
                  f"{c['n_eligible']} eligible) and all {c['n_sites_compared']} derived site series identical on every "
                  "date <= 2023-09-30 (max |Δ| = 0, NaN pattern identical), so no train/val value changed. "
                  f"Pairs: {(man.get('stats') or {}).get('pairs')} (fit pairs unchanged; only post-2021 'later' pairs "
                  "can differ, they are unused). Manifest now hashes every file.\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
