# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the SENTINEL-Lite national-track labels and split (protocol §12; RESEARCH_PLAN §15.5 items 5–7).

Reads the national NAIP matchups (``processed/naip_matchups/national/matchups.parquet``, moved there after
``build_naip_matchups.py --national``) and the frozen 4-state split/labels (unchanged). Writes
``processed/lite_labels/labels_national.parquet`` (manifest ``lite_labels_national``),
``docs/preregistration/lite_split_national.json`` and ``lite_split_national_report.json`` (with the power
check: label and water-body counts per role and target — counts only, no model output). Refuses to overwrite a
national split with a different hash.

    python -B scripts/train/lite_split_national.py
"""

from __future__ import annotations

import argparse
import json
import sys

from sentinel.models.lite import runner as R

R.cpu_threads(6)

from sentinel.core.manifest import write_manifest  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402
from sentinel.models.lite import national as N  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    a = ap.parse_args(argv)
    frozen = G.load_split()
    lab, rep = N.build_labels(frozen)
    split = N.build_split(lab, frozen)
    sha = N.split_sha256(split)
    srep = N.split_report(split, lab)
    srep["label_report"] = rep
    print(json.dumps({"sha256": sha, "by_role": {r: {k: v[k] for k in ("chips", "water_bodies", "labels")}
                                                for r, v in srep["by_role"].items()}}, indent=1))
    if a.dry_run:
        return 0
    if N.NATIONAL_SPLIT_FILE.exists():
        old = json.loads(N.NATIONAL_SPLIT_FILE.read_text())
        if N.split_sha256(old) != sha:
            print(f"refusing: existing national split {N.split_sha256(old)[:12]} differs from {sha[:12]}", file=sys.stderr)
            return 2
    out = N.labels_path()
    lab.to_parquet(out, compression="zstd", index=False)
    write_manifest("lite_labels_national", files=[out], sources=[str(N.national_matchups_path()), "manifest:lite_labels",
                                                                 "manifest:hydrolakes"],
                   description="SENTINEL-Lite national-track per-chip labels: frozen 4-state rows (unchanged) + new "
                               "chips from the national NAIP x WQP matchups, same label and water-body rules",
                   license="public (WQP)", synthetic=False, stats={"labels_sha256": split["labels_sha256"], **rep})
    R.write_json(N.NATIONAL_SPLIT_FILE, split)
    R.write_json(N.NATIONAL_SPLIT_FILE.with_name("lite_split_national_report.json"), srep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
