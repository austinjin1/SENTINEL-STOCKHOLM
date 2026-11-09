# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the SENTINEL-Lite label table and the frozen water-body split (protocol §2–§3).

Writes ``processed/lite_labels/labels.parquet`` (manifest ``lite_labels``),
``docs/preregistration/lite_split.json`` and ``lite_split_report.json``. Refuses to overwrite an
existing split whose hash differs unless ``--force`` (the split is frozen before model selection).

    python -B scripts/train/lite_split.py
"""

from __future__ import annotations

import argparse
import json
import sys

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.models.lite import data as D
from sentinel.models.lite import guard as G
from sentinel.models.lite import splits as S


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    lab, rep = D.build_label_table(radius_km=S.DEFAULTS["water_body_radius_km"],
                                   name_km=S.DEFAULTS["water_body_name_km"], use_hydrolakes=True)
    split = S.build_split(lab)
    sha = S.split_sha256(split)
    if G.SPLIT_FILE.exists() and not a.force:
        old = json.loads(G.SPLIT_FILE.read_text())
        if S.split_sha256(old) != sha:
            print(f"refusing: existing split {S.split_sha256(old)[:12]} differs from rebuilt {sha[:12]}",
                  file=sys.stderr)
            return 2
    # labels are written only after the split check passed (review W6); their content hash is in the split
    out = paths.processed("lite_labels") / "labels.parquet"
    lab.to_parquet(out, compression="zstd", index=False)
    rep["labels_sha256"] = split["labels_sha256"]
    write_manifest("lite_labels", files=[out], sources=[str(D.matchups_path()), "manifest:hydrolakes"],
                   description="SENTINEL-Lite per-chip labels (log10 Secchi m, turbidity NTU, TSS mg/L, chl-a ug/L) "
                               "from naip_matchups; water-body proxy groups (2-km components merged by name "
                               "within 40 km, by HUC8 for flowing/estuarine sites, and by HydroLAKES lake id)",
                   license="public (WQP)", synthetic=False, stats=rep)
    D.write_json(G.SPLIT_FILE, split)
    srep = S.split_report(split, lab)
    srep["label_report"] = rep
    D.write_json(G.SPLIT_FILE.with_name("lite_split_report.json"), srep)
    print(json.dumps({"sha256": sha, "n_chips": srep["n_chips"], "n_labels": srep["n_labels"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
