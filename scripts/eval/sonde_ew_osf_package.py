#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the EW-v2 OSF archive (review O8): every hashed scoring-code file, the frozen calibration
file, all model artifacts it references, the split, the protocol, the freeze candidate/record and
the preregistration text, as one tar.gz plus a SHA-256 manifest. Verifies every hash against the
freeze candidate before packing. No data values are included.

    python -B scripts/eval/sonde_ew_osf_package.py --candidate docs/preregistration/ew_v2_freeze_candidate.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from sentinel.core import paths
from sentinel.models.sonde import guard


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", default="docs/preregistration/ew_v2_freeze_candidate.json")
    ap.add_argument("--out", default=str(paths.RUNS_ROOT / "sonde" / "osf_ew_v2"))
    a = ap.parse_args()
    root = paths.REPO_ROOT
    cand = json.loads((root / a.candidate).read_text())
    fr = cand["freeze_record_draft"]
    if guard.scoring_code_sha256(files=fr["scoring_code_files"]) != fr["scoring_code_sha256"]:
        raise SystemExit("scoring code changed since the candidate was written; regenerate it first")
    cf = Path(fr["thresholds_file"])
    if sha(cf) != fr["thresholds_sha256"]:
        raise SystemExit("calibration file hash mismatch")
    cal = json.loads(cf.read_text())
    items = {f"code/{rel}": root / rel for rel in fr["scoring_code_files"]}
    items["calibration/event_calibration.json"] = cf
    for key, ent in cal["entries"].items():
        for name, art in ent["artifacts"].items():
            f = cf.parent / art["file"]
            if sha(f) != art["sha256"] or fr["weights_sha256"].get(f"{key}|{name}") != art["sha256"]:
                raise SystemExit(f"artifact hash mismatch: {f}")
            items[f"calibration/{art['file']}"] = f
    for rel in ("docs/preregistration/sonde_split.json", "docs/preregistration/ew_v2_protocol.md",
                "docs/preregistration/ew_v2_osf_preregistration.md", a.candidate,
                "scripts/eval/sonde_ew_freeze_candidate.py", "scripts/eval/sonde_onset_counts.py",
                "scripts/eval/sonde_ew_osf_package.py", "tests/test_ew_controls.py"):
        if (root / rel).is_file():
            items[f"docs/{Path(rel).name}" if rel.startswith("docs/") else f"code/{rel}"] = root / rel
    # review C1: the H2 label definition (per-site Cl = a + b·SC regressions) + its identity check
    cl = paths.PROCESSED / "chloride_derived"
    pin = (fr.get("data_snapshots") or {}).get("chloride_derived") or {}
    items["data/chloride_derived/regressions.parquet"] = cl / "regressions.parquet"
    if (cl / "compare_vs_chloride_derived.json").is_file():
        items["data/chloride_derived/compare_vs_chloride_derived.json"] = cl / "compare_vs_chloride_derived.json"
    if pin.get("manifest") and (paths.MANIFESTS / f"{pin['manifest']}.json").is_file():
        mf = paths.MANIFESTS / f"{pin['manifest']}.json"
        if sha(mf) != pin.get("manifest_sha256"):
            raise SystemExit("chloride_derived manifest changed since the candidate was written")
        items[f"data/manifests/{mf.name}"] = mf
    for extra in ("scripts/data/build_chloride_derived.py", "scripts/data/promote_chloride_derived.py"):
        items[f"code/{extra}"] = root / extra
    ff = guard.FREEZE_FILE_EW
    if ff.is_file():
        items["docs/FREEZE_ew.json"] = ff
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {k: {"sha256": sha(v), "bytes": v.stat().st_size} for k, v in sorted(items.items())}
    (out / "MANIFEST.sha256.json").write_text(json.dumps(manifest, indent=1))
    tar = out / "ew_v2_osf.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for k, v in sorted(items.items()):
            t.add(v, arcname=f"ew_v2/{k}")
        t.add(out / "MANIFEST.sha256.json", arcname="ew_v2/MANIFEST.sha256.json")
    (out / "ew_v2_osf.tar.gz.sha256").write_text(sha(tar) + "  ew_v2_osf.tar.gz\n")
    print(f"{len(items)} files -> {tar} sha256 {sha(tar)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
