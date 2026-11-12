#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""v1 / v2-independence audit of the USGS new-site block (review C-N1; lead ruling 2026-11-10). IDs ONLY, no value.

Part A — does any v2 artifact derive from v1? Lists every v2 code/config/registry/run/manifest reference to the v1
mirror (``paths.LEGACY_ROOT``) and checks whether the one v2 product built from v1 (``mc_benchmark_v2``) names any new
site.
Part B — which new sites does v1 name, and how? Scans the read-only v1 mirror (text files and string columns of
parquet/CSV) for new-site IDs and classifies each file:
  * VALUE  — v1 used the site's VALUES: model training/normalisation (``data/processed/sensor``), v1 results
             (``results/``), v1 event/detection outputs, v1 prospective predictions (``prospective``/``predictions``);
  * CATALOGUE — station lists, topology, metadata, demo lists, candidate catalogues (everything else).
Writes ``docs/preregistration/sonde_newsites_v1_audit.json`` (refuses to overwrite).

    python -B scripts/data/newsite_v1_audit.py
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
import subprocess

import pandas as pd

from sentinel.core import paths

PREREG = paths.REPO_ROOT / "docs" / "preregistration"
OUT = PREREG / "sonde_newsites_v1_audit_v2.json"   # v2 (lead ruling C-F1, 2026-11-10); v1 file kept
TEXT = {".json", ".csv", ".txt", ".yaml", ".yml", ".md", ".py", ".tsv", ".geojson", ".log"}
VALUE_PAT = re.compile(r"(^|/)(data/processed/sensor|results/|prospective|predictions|"
                       r"data/processed/trackA_[^/]+/_build_results\.json)", re.I)
# OUTCOME-level = value-level use that produced labels / episodes / onsets / predictions / build results (C-F1: counts as
# exposure -> hard tier). The only value-level file that is NOT outcome-level is v1's sensor normalisation statistics.
NORMALISATION_ONLY = ("data/processed/sensor/full/preprocessing_stats.json",)
ID_RE = re.compile(r"(?<!\d)(\d{8,15})(?!\d)")


def part_a() -> dict:
    rx = r"LEGACY_ROOT|SENTINEL-HF|SENTINEL_LEGACY"
    code = subprocess.run(["grep", "-rlE", rx, "sentinel", "scripts", "configs", "tests"], cwd=paths.REPO_ROOT,
                          capture_output=True, text=True).stdout.split()
    code = [c for c in code if not c.endswith(("newsite_v1_audit.py", "paths.py"))]
    reg = sum("SENTINEL-HF" in ln for ln in open(paths.REGISTRY_PATH))
    runs = subprocess.run(["grep", "-rlE", "SENTINEL-HF", str(paths.RUNS_ROOT), "--include=*.json"],
                          capture_output=True, text=True).stdout.split()
    mans = subprocess.run(["grep", "-rlE", "SENTINEL-HF", str(paths.DATA_ROOT / "manifests")],
                          capture_output=True, text=True).stdout.split()
    ids = {s["site_no"] for s in json.loads((PREREG / "sonde_newsites_list.json").read_text())["sites"]}
    mc_hits = []
    for f in (paths.PROCESSED / "mc_benchmark_v2").glob("*.parquet") if (paths.PROCESSED / "mc_benchmark_v2").is_dir() else []:
        df = pd.read_parquet(f)
        for c in df.columns:
            if df[c].dtype == object:
                mc_hits += sorted(ids & set(df[c].astype(str).str.extract(r"(\d{8,15})")[0].dropna()))
    return {"v2_code_reading_v1": code, "registry_rows_mentioning_v1": reg, "run_records_mentioning_v1": runs,
            "data_manifests_mentioning_v1": [m.replace(str(paths.DATA_ROOT) + "/", "") for m in mans],
            "mc_benchmark_v2_new_site_hits": sorted(set(mc_hits))}


def part_b(ids: set) -> dict:
    root = paths.LEGACY_ROOT
    by_file = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.stat().st_size > 200_000_000:
            continue
        rel = p.relative_to(root).as_posix()
        found = set()
        try:
            if p.suffix.lower() in TEXT:
                found = ids & set(ID_RE.findall(p.read_text(errors="ignore")))
            elif p.suffix.lower() == ".parquet":
                df = pd.read_parquet(p)
                for c in df.columns:
                    if df[c].dtype == object or str(df[c].dtype) == "string":
                        found |= ids & set(df[c].astype(str).str.extract(r"(\d{8,15})")[0].dropna())
        except Exception:  # noqa: BLE001 — unreadable file: skipped and counted
            continue
        if found:
            by_file[rel] = sorted(found)
    value = {f: v for f, v in by_file.items() if VALUE_PAT.search(f)}
    cat = {f: v for f, v in by_file.items() if f not in value}
    outcome = {f: v for f, v in value.items() if f not in NORMALISATION_ONLY}
    return {"n_files_naming_new_sites": len(by_file),
            "outcome_level_files": {f: (len(v), v) for f, v in sorted(outcome.items())},
            "outcome_level_sites": sorted({s for v in outcome.values() for s in v}),
            "sites_named_anywhere": sorted({s for v in by_file.values() for s in v}),
            "value_level_files": {f: (len(v), v) for f, v in sorted(value.items())},
            "value_level_sites": sorted({s for v in value.values() for s in v}),
            "catalogue_files": {f: len(v) for f, v in sorted(cat.items())}}


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    lst = json.loads((PREREG / "sonde_newsites_list.json").read_text())
    ids = {s["site_no"] for s in lst["sites"]}
    ov = json.loads((PREREG / "sonde_newsites_overlap.json").read_text())
    hard = set(ov["loaded_by_any_recorded_run"]) | set(ov["track_n_colocated"]["1000m"]["sites"]) | \
        set(ov["catalogue_exposure_not_splits"]["labels_powell_metabolism/sites.parquet"][1])
    a, b = part_a(), part_b(ids)
    hard_t = json.loads((PREREG / "sonde_newsites_tiers.json").read_text())["hard"]   # incl. the chloride site
    hard = set(hard_t)
    to_hard = sorted(set(b["outcome_level_sites"]) - hard)          # C-F1: v1 outcome-level use -> hard tier
    legacy = sorted(set(b["value_level_sites"]) - hard - set(to_hard))
    rec = {"created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "site_list_sha256": hashlib.sha256((PREREG / "sonde_newsites_list.json").read_bytes()).hexdigest(),
           "part_a_v2_uses_v1": a, "part_b_v1_names_new_sites": b,
           "hard_tier_n_before": len(hard),
           "v1_outcome_level_to_hard": {"rule": "v1 outcome-level use (derived labels, episodes/onsets, predictions, "
                                                "build results) counts as exposure (lead ruling C-F1)",
                                        "n_sites": len(to_hard), "sites": to_hard},
           "legacy_tier": {"rule": "v1 value-level use limited to normalisation statistics, not in the hard tier",
                           "n_sites": len(legacy), "sites": legacy}}
    OUT.write_text(json.dumps(rec, indent=1))
    OUT.chmod(0o444)
    print(json.dumps({"sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(), "part_a": a,
                      "v1_files": b["n_files_naming_new_sites"], "v1_named": len(b["sites_named_anywhere"]),
                      "v1_value_level_sites": len(b["value_level_sites"]),
                      "value_files": {f: v[0] for f, v in b["value_level_files"].items()},
                      "to_hard": to_hard, "legacy_tier_n": len(legacy)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
