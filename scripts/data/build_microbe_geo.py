#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build ``processed/microbe_wq_geo``: GEO-8 target tables and the QMP input block (amendment §5, §7.1–7.2).

COUNTS ONLY are printed. For the protected blocks (2024 holdout, pre-2017 reserve) only row and key counts are
computed, printed, stored and compared (events, chemistry parents, sites, domains, qPCR input coverage): no
per-variable, censoring or missing-label count (lead ruling I-04, 2026-11-10). No label value is printed or
written anywhere but the two target tables:

* ``dev_targets.parquet``       development events + ``.001`` 2017–18 training events;
* ``protected_labels.parquet``  2024 holdout + pre-2017 reserve events (scorer only, under the open claim);
* ``ids.json``, ``counts.json``, ``qmp_features.parquet``, ``qmp_meta.parquet``.

The counts must reproduce Appendix A.1 / A.3 of the amendment (``docs/preregistration/microbe_arch_v2_evidence/
geo_counts.json``). On any difference nothing is written and the exit status is 2 (amendment §7.7 kill rule).

    PYTHONPATH=. python scripts/data/build_microbe_geo.py [--check-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from sentinel.core import paths
from sentinel.core.manifest import write_manifest
from sentinel.data import microbe_geo as G

EVIDENCE = "docs/preregistration/microbe_arch_v2_evidence/geo_counts.json"
SAMPLE_COLS = ["sample_id", "site_id", "domain_id", "collect_date", "chem_parent", "chem_match", "dna_sample_ids"]


def compare(counts: dict, evidence: dict) -> list[str]:
    """Differences between the build's counts and the evidence file (the variables the build carries)."""
    diff = []
    for blk, key in G.EVIDENCE_BLOCK.items():
        mine, ref = counts[blk], evidence["blocks"][key]
        for k in ("n_events", "n_with_chem_parent", "sites", "domains", "n_qpcr_16s", "n_qpcr_16s_and_volume",
                  "n_filtered_volume"):
            if mine[k] != ref[k]:
                diff.append(f"{blk}.{k}: build {mine[k]} != evidence {ref[k]}")
        for col, row in mine.get("vars", {}).items():       # absent for the protected blocks (keys only)
            for k, v in row.items():
                if ref["vars"][col][k] != v:
                    diff.append(f"{blk}.{col}.{k}: build {v} != evidence {ref['vars'][col][k]}")
    return diff


def main(argv=None) -> int:
    from sentinel.models.microbe.design import load_holdout_rule, load_release_holdout_rule

    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="compute and compare the counts, write nothing")
    a = ap.parse_args(argv)
    P = paths.PROCESSED
    rule, rrule = load_holdout_rule(), load_release_holdout_rule()

    chem_wide = pd.read_parquet(P / "neon_chem" / "chem_event_wide.parquet")
    chem_long = pd.read_parquet(P / "neon_chem" / "chem_long.parquet",
                                columns=["parentSampleID", "var", "detection_limit"])
    qpcr = pd.read_parquet(P / "neon_microbe_qpcr" / "mga_swGroupAbundances.parquet")
    fg = pd.read_parquet(P / "neon_microbe_seqmeta" / "amc_fieldGenetic.parquet",
                         columns=["geneticSampleID", "geneticFilteredSampleVolume"])

    tabs, qmps, metas, events = [], [], [], []
    for rel, name in (("002", "microbe_wq"), ("001", "microbe_wq_v1")):
        s = pd.read_parquet(P / name / "samples.parquet", columns=SAMPLE_COLS)
        s["sample_id"] = s["sample_id"].astype(str)
        s["block"] = G.event_blocks(rel, s["collect_date"], rule["date"], rrule["date"])
        s["release"] = rel
        t = G.build_targets(s, chem_wide, chem_long)
        f, m = G.build_qmp(s, qpcr, fg)
        assert (t["sample_id"].to_numpy() == s["sample_id"].to_numpy()).all()
        t.insert(1, "block", s["block"].to_numpy())
        t.insert(2, "release", rel)
        t.insert(3, "chem_match", s["chem_match"].to_numpy())
        tabs.append(t), qmps.append(f), metas.append(m), events.append(s)
    tab, qmp, meta, ev = (pd.concat(x, ignore_index=True) for x in (tabs, qmps, metas, events))
    if tab["sample_id"].duplicated().any():
        raise RuntimeError("sample ids collide between releases")

    counts = {}
    for blk in G.BLOCKS:
        m = (tab["block"] == blk).to_numpy()
        protected = blk in G.PROTECTED_BLOCKS
        c = G.target_counts(tab[m].reset_index(drop=True), ev[m].reset_index(drop=True), keys_only=protected)
        has_q, has_v = qmp.loc[m, "qmp_log_copies"].notna(), qmp.loc[m, "qmp_log_volume"].notna()
        c.update(n_qpcr_16s=int(has_q.sum()), n_qpcr_16s_and_volume=int((has_q & has_v).sum()),
                 n_filtered_volume=int(has_v.sum()),
                 qmp_missing={f: int(qmp.loc[m, f].isna().sum()) for f in G.QMP_FEATURES})
        counts[blk] = c
        print(f"\n== {blk}", {k: v for k, v in c.items() if k != "vars"})
        if protected:
            print("   protected block: row and key counts only")
        else:
            print(pd.DataFrame(c["vars"]).T.to_string())

    evidence = json.loads((paths.REPO_ROOT / EVIDENCE).read_text())
    diff = compare(counts, evidence)
    if diff:
        print("\nCOUNTS DO NOT REPRODUCE Appendix A.1 / A.3 — nothing written:", *diff, sep="\n  ")
        return 2
    print("\ncounts reproduce", EVIDENCE)
    if a.check_only:
        return 0

    out = paths.processed(G.DATASET)
    prot = tab["block"].isin(G.PROTECTED_BLOCKS).to_numpy()
    files = []
    for fn, df in (("dev_targets", tab[~prot]), ("protected_labels", tab[prot]), ("qmp_features", qmp),
                   ("qmp_meta", meta)):
        p = out / f"{fn}.parquet"
        df.reset_index(drop=True).to_parquet(p, compression="zstd", index=False)
        files.append(p)
    ids = {blk: sorted(tab.loc[tab["block"] == blk, "sample_id"]) for blk in G.BLOCKS}
    (out / "ids.json").write_text(json.dumps(ids, indent=0, sort_keys=True))
    (out / "counts.json").write_text(json.dumps(counts, indent=1, sort_keys=True))
    files += [out / "ids.json", out / "counts.json"]
    write_manifest(G.DATASET, files=files,
                   sources=["manifest:microbe_wq", "manifest:microbe_wq_v1", "manifest:neon_chem",
                            "manifest:neon_microbe_qpcr", "manifest:neon_microbe_seqmeta"],
                   description="GEO-8 targets (same-visit NEON grab chemistry and field meter) for the SENTINEL-Microbe "
                               "events, split into a development table and a protected table, plus the 16S qPCR input "
                               "block (amendment microbe_arch_v2 §5, §7).",
                   license="CC BY 4.0 (NEON)", synthetic=False,
                   stats={"blocks": {b: counts[b]["n_events"] for b in G.BLOCKS},
                          "targets": list(G.GEO_TARGETS), "auxiliary": list(G.AUXILIARY),
                          "descriptive": list(G.DESCRIPTIVE)})
    print("wrote", out, [Path(f).name for f in files])
    return 0


if __name__ == "__main__":
    sys.exit(main())
