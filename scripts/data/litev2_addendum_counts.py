#!/usr/bin/env python
"""Counts for the label addendum of architecture round v2 that are not in labels2b_counts.json / stamp_counts.json.

COUNTS ONLY. Reads the national-2b label table as present/absent flags (no label value is kept, compared or printed),
the chip_role mapping of the pin file and the reserved new-site list. Test roles are not counted here.

  python scripts/data/litev2_addendum_counts.py [--register]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from sentinel.core import paths
from sentinel.data.reserved import exclude_reserved

LABELS = ("secchi", "turbidity", "tss", "chla", "turb_fnu", "turb_ntru", "ssc")
CENS = ("secchi", "turbidity", "tss", "chla", "ssc")
S9_MIN = 20          # amendment l.276: "Run for a target only if it has >= 20 censored training rows (count)"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kish(n: pd.Series) -> float:
    return round(float(n.sum() ** 2 / (n ** 2).sum()), 2) if len(n) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    lab_p = paths.processed("lite_labels") / "labels_national2b.parquet"
    pin_p = paths.REPO_ROOT / "docs/preregistration/lite_split_national2b.json"
    pin = json.loads(pin_p.read_text())
    t = pq.read_table(lab_p, columns=["naip_chip_id", "site_id", "water_body", "version_source",
                                      *[f"y_{x}" for x in LABELS], *[f"cens_{x}" for x in CENS]])
    df = pd.DataFrame({"naip_chip_id": t["naip_chip_id"].to_pandas().astype(str),
                       "site_id": t["site_id"].to_pandas().astype(str),
                       "water_body": t["water_body"].to_pandas().astype(str),
                       "origin": t["version_source"].to_pandas().astype(str)})
    for x in LABELS:                                     # present/absent only
        df[x] = pc.is_valid(t[f"y_{x}"]).to_pandas().to_numpy()
    for x in CENS:
        df[f"censored_{x}"] = pc.not_equal(t[f"cens_{x}"], 0).to_pandas().fillna(False).to_numpy()
    del t
    df["role"] = df["naip_chip_id"].map(pin["chip_role"])
    free = set(exclude_reserved(df["site_id"].unique().tolist()))
    df["reserved"] = ~df["site_id"].isin(free)
    flags = [*LABELS, *[f"censored_{x}" for x in CENS]]
    tr = df[df["role"] == "train"]
    out = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "reads": "present/absent flags of label columns, chip roles, water-body ids; no label value",
           "labels_file_sha256": sha(lab_p), "pin_file_sha256": sha(pin_p),
           "train": {
               "chips": int(len(tr)), "chips_with_a_label": int(tr[flags].any(axis=1).sum()),
               "chips_at_reserved_newsite_block_sites": int(tr["reserved"].sum()),
               "labels_on_reserved_site_chips": {k: int(v) for k, v in tr.loc[tr["reserved"], flags].sum().items()},
               "chips_with_a_label_at_load": int((tr[flags].any(axis=1) & ~tr["reserved"]).sum()),
               "water_bodies_at_load": int(tr.loc[tr[flags].any(axis=1) & ~tr["reserved"], "water_body"].nunique()),
               "label_rows_in_file": {k: int(v) for k, v in tr[flags].sum().items()},
               "label_rows_at_load": {k: int(v) for k, v in tr.loc[~tr["reserved"], flags].sum().items()},
               "water_bodies_at_load_by_label": {k: int(tr.loc[tr[k] & ~tr["reserved"], "water_body"].nunique())
                                                 for k in flags},
               "label_rows_by_origin": {o: {k: int(v) for k, v in g[flags].sum().items()}
                                        for o, g in tr.groupby("origin")},
               "added_train_chips": int((tr["origin"] == "added").sum()),
               "added_train_water_bodies": int(tr.loc[tr["origin"] == "added", "water_body"].nunique())},
           "validation": {}, "reserved_chips_by_role": {r: int(g["reserved"].sum()) for r, g in df.groupby("role")}}
    for name, roles in (("val", ["val"]), ("val_n", ["val_n"]), ("val+val_n", ["val", "val_n"])):
        v = df[df["role"].isin(roles)]
        out["validation"][name] = {"chips": int(len(v)), "chips_with_a_label": int(v[flags].any(axis=1).sum()),
                                   **{k: {"chips": int(v[k].sum()),
                                          "water_bodies": int(v.loc[v[k], "water_body"].nunique()),
                                          "effective_water_bodies": kish(v[v[k]].groupby("water_body").size())}
                                      for k in LABELS}}
    s9 = {x: int((tr[f"censored_{x}"] & ~tr["reserved"]).sum()) for x in CENS}
    out["s9"] = {"minimum_censored_training_rows": S9_MIN, "source": "lite_arch_v2_amendment.md l.276 (S9 row)",
                 "censored_training_rows": s9, "targets": [x for x in CENS if s9[x] >= S9_MIN]}
    dst = paths.RUNS_ROOT / "lite" / "counts_arch_v2" / "addendum_counts.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("wrote", dst, sha(dst))
    if a.register:
        from sentinel.core.registry import log_result
        r = log_result(workstream="M7-lite", experiment="lite/s9_targets", model="national-2b", metric="n_s9_targets",
                       value=len(out["s9"]["targets"]), split="train", n=sum(s9.values()), artifacts=[str(dst)],
                       notes="S9 (GBM-parity-AFT) target list fixed from censored training-row counts under the label "
                             "rule, before any validation number (ruling DS-11); counts only",
                       extra={"written_by": "M7", "s9": out["s9"], "counts_file_sha256": sha(dst),
                              "national2b_split_sha256": "29f0dd009bb69a5e36a7829d3bf5b8c7c4301f683a87b6a1b6ddd2fd1ca08092"})
        print("registry row", r["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
