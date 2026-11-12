# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Counts for the label addendum of the SENTINEL-Lite round-v2 amendment (lead order 2026-11-10 15:05 EST).

Keys and counts only. No label value of a validation, test or lock row is printed, summarised or stored; the only
value comparison is an equal / not-equal count between the first P2 index build and the rebuild on ``train_p2`` and
``val_p2`` rows (``test_p2`` rows hold no value in either build).

* F7: ``train_p2`` removals (``p2_index.f7_removals``) on the rebuilt index;
* effective number of water bodies per P2 role (Kish, (sum n)^2 / sum n^2 over water bodies, usable rows);
* first build -> rebuild of the P2 index: rows kept / dropped / changed per role;
* ``test_p2`` stratum whose HUC8 and 25-km block hold no v2 training row (verifier, missed[3]);
* chips at sites of the reserved new-site block per role (national-2b, P2 lock, P2 index);
* M2 Test-S mask counts of national-2b (from ``tests_mask_national2b/meta.json``).

    python scripts/data/litev2_stamp_counts.py [--register]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sentinel.core import paths, registry
from sentinel.data.reserved import exclude_reserved
from sentinel.eval.splits import spatial_blocks
from sentinel.models.lite import national as N
from sentinel.models.litev2 import labels2b as L
from sentinel.models.litev2 import p2 as P2
from sentinel.models.litev2 import p2_index as PI

OUT = paths.RUNS_ROOT / "lite" / "counts_arch_v2" / "stamp_counts.json"
FIRST_BUILD = "index_firstbuild_68688fac.parquet"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _kish(n: pd.Series) -> float:
    n = n.to_numpy(float)
    return float(n.sum() ** 2 / (n ** 2).sum()) if len(n) else 0.0


def _usable(ix: pd.DataFrame) -> np.ndarray:
    return (ix["eligible"] & (ix["has_label"] | (ix["cens"] != 0))).to_numpy()


def _n_reserved(site_ids) -> int:
    ids = pd.Series(site_ids).astype(str)
    free = set(exclude_reserved(ids.unique().tolist()))
    return int((~ids.isin(free)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    out: dict = {"built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "reads": "keys, roles and present/absent flags; equal/not-equal count of train_p2 and val_p2 values "
                          "between the two index builds; no value printed or stored"}

    # ---------------------------------------------------------------- P2 index, F7, effective water bodies
    if PI.index_sha256() != PI.INDEX_SHA256:
        raise SystemExit("index.parquet is not the pinned P2 index")
    ix = pd.read_parquet(PI.index_path())
    PI.assert_no_excluded(ix["site_id"], "stamp counts")
    usable = _usable(ix)
    removed, f7 = PI.f7_removals(ix)
    out["p2_index_sha256"] = PI.INDEX_SHA256
    out["f7"] = f7
    eff = {}
    for role in ("train_p2", "val_p2", "test_p2"):
        k = usable & (ix["role"] == role).to_numpy()
        if role == "train_p2":
            eff["train_p2_before_f7"] = {"rows": int(k.sum()), "water_bodies": int(ix.loc[k, "water_body"].nunique()),
                                         "effective_water_bodies": round(_kish(ix.loc[k, "water_body"].value_counts()), 2)}
            k = k & ~removed
        g = ix.loc[k]
        eff[role] = {"rows": int(k.sum()), "water_bodies": int(g["water_body"].nunique()),
                     "sites": int(g["site_id"].nunique()),
                     "effective_water_bodies": round(_kish(g["water_body"].value_counts()), 2),
                     "effective_sites": round(_kish(g["site_id"].value_counts()), 2),
                     "largest_water_body_rows": int(g["water_body"].value_counts().max()) if len(g) else 0,
                     "by_label_src": {str(s): int(v) for s, v in g["label_src"].value_counts().items()},
                     "by_site_class": {str(s): int(v) for s, v in g["site_class"].value_counts().items()},
                     "naip_years": {str(int(y)): int(v) for y, v in g["naip_year"].value_counts().sort_index().items()}}
    out["p2_usable_rows"] = eff

    # ---------------------------------------------------------------- first build -> rebuild (label rule)
    old = pd.read_parquet(PI.out_dir() / FIRST_BUILD)
    j = old.merge(ix, on="naip_chip_id", how="outer", suffixes=("_old", "_new"), indicator=True)
    ch = {"first_build_sha256": _sha(PI.out_dir() / FIRST_BUILD), "rows_first_build": int(len(old)),
          "rows_rebuild": int(len(ix)), "by_role": {}}
    j["role"] = j["role_new"].fillna(j["role_old"])
    for role, g in j.groupby("role"):
        ho, hn = g["has_label_old"].fillna(False).astype(bool), g["has_label_new"].fillna(False).astype(bool)
        r = {"chips_first_build_only": int((g["_merge"] == "left_only").sum()),
             "chips_rebuild_only": int((g["_merge"] == "right_only").sum()),
             "label_present_first_build": int(ho.sum()), "label_present_rebuild": int(hn.sum()),
             "label_lost": int((ho & ~hn).sum()), "label_gained": int((~ho & hn).sum())}
        if role != "test_p2":
            both = ho & hn
            same = np.isclose(g.loc[both, "y_turb_fnu_old"].to_numpy(float), g.loc[both, "y_turb_fnu_new"].to_numpy(float),
                              rtol=1e-9, atol=0.0)
            r["label_value_changed"] = int((~same).sum())
            r["label_source_changed"] = int((g.loc[both, "label_src_old"] != g.loc[both, "label_src_new"]).sum())
        r["eligible_first_build"] = int(g["eligible_old"].fillna(False).astype(bool).sum())
        r["eligible_rebuild"] = int(g["eligible_new"].fillna(False).astype(bool).sum())
        ch["by_role"][str(role)] = r
    out["p2_first_build_vs_rebuild"] = ch

    # ---------------------------------------------------------------- test_p2 stratum without a v2 training row
    split = L.load_split()
    tr = L.role_frame(split, ["train"])                                  # national-2b training rows (v2 view)
    p = N.PARAMS
    tr_blocks = set(pd.Series(spatial_blocks(tr["lat"], tr["lon"], p["block_km"])).astype(str))
    tr_huc = set(tr["huc8"].dropna().astype(str).str.zfill(8))
    p2tr = ix[usable & ~removed & (ix["role"] == "train_p2").to_numpy()]
    p2_blocks = set(pd.Series(spatial_blocks(p2tr["lat"], p2tr["lon"], p["block_km"])).astype(str))
    p2_huc = set(p2tr["huc8"].dropna().astype(str).str.zfill(8))
    te = ix[usable & (ix["role"] == "test_p2").to_numpy()].copy()
    te_block = pd.Series(spatial_blocks(te["lat"], te["lon"], p["block_km"]), index=te.index).astype(str)
    te_huc = te["huc8"].astype("string").str.zfill(8)
    blk = te_block.isin(tr_blocks | p2_blocks).to_numpy()
    huc = te_huc.isin(tr_huc | p2_huc).fillna(False).to_numpy()
    no_huc = te["huc8"].isna().to_numpy()

    def cell(k):
        g = te[k]
        return {"chips": int(k.sum()), "water_bodies": int(g["water_body"].nunique()), "sites": int(g["site_id"].nunique()),
                "effective_water_bodies": round(_kish(g["water_body"].value_counts()), 2)}

    out["test_p2_stratum"] = {
        "definition": "usable test_p2 chips; v2 training rows = national-2b train rows with at least one label of "
                      "the version (reserved-site chips excluded) + usable train_p2 rows after F7; blocks = 25-km "
                      "blocks of the national split rule; HUC8 as stored (a chip without a HUC8 counts as 'HUC8 holds "
                      "no training row' only if its block holds none)",
        "v2_training_rows": {"national2b_train": int(len(tr)), "national2b_train_without_huc8": int(tr["huc8"].isna().sum()),
                             "train_p2_after_f7": int(len(p2tr)), "blocks": len(tr_blocks | p2_blocks),
                             "huc8": len(tr_huc | p2_huc)},
        "all": cell(np.ones(len(te), bool)),
        "test_p2_chips_without_huc8": int(no_huc.sum()),
        "huc8_and_block_hold_no_v2_training_row": cell(~blk & ~huc),
        "block_holds_none_huc8_holds_some": cell(~blk & huc),
        "huc8_holds_none_block_holds_some": cell(blk & ~huc),
        "huc8_and_block_both_hold_training_rows": cell(blk & huc),
        "by_national2b_only": {"huc8_and_block_hold_no_national2b_training_row":
                               cell(~te_block.isin(tr_blocks).to_numpy() & ~te_huc.isin(tr_huc).fillna(False).to_numpy())},
    }

    # ---------------------------------------------------------------- reserved new-site block, per role
    cr = pd.Series(split["chip_role"])
    lab = pq.read_table(L.labels_path(), columns=["naip_chip_id", "site_id"]).to_pandas()
    lab["role"] = lab["naip_chip_id"].map(cr)
    lock_sites = P2.site_roles(P2.load_lock())
    out["reserved_newsite_block"] = {
        "national2b_chips_by_role": {str(r): _n_reserved(g["site_id"]) for r, g in lab.groupby("role")},
        "p2_lock_sites_by_role": {str(r): _n_reserved(g["site_id"]) for r, g in lock_sites.groupby("role")},
        "p2_index_rows_by_role": {str(r): _n_reserved(g["site_id"]) for r, g in ix.groupby("role")},
        "p2_lock_sites_excluded_at_load": {str(r): int(v) for r, v in lock_sites[lock_sites["site_id"].isin(
            PI.EXCLUDED_AT_LOAD)]["role"].value_counts().items()},
    }

    # ---------------------------------------------------------------- Test-S mask (built by litev2_build_tests_mask)
    mm = paths.RUNS_ROOT / "lite" / "tests_mask_national2b" / "meta.json"
    meta = json.loads(mm.read_text())
    if meta["split_sha256"] != L.FROZEN_SHA256:
        raise SystemExit("the Test-S mask was not built for the pinned national-2b split")
    out["tests_mask"] = {"meta_file": str(mm), "meta_sha256": _sha(mm), "flags_sha256": meta["flags_sha256"],
                         "counts_by_role": meta["counts_by_role"], "kd_masked_total": meta["kd_masked_total"],
                         "kd_masked_in_m2_unit": meta["kd_masked_in_m2_unit"]}

    OUT.write_text(json.dumps(out, indent=1, default=str))
    sha = _sha(OUT)
    print(json.dumps(out, indent=1, default=str))
    print("counts file", OUT, sha)
    if a.register:
        art = [str(OUT)]
        common = {"counts_file_sha256": sha, "national2b_split_sha256": L.FROZEN_SHA256, "p2_index_sha256": PI.INDEX_SHA256,
                  "p2_lock_sha256": P2.LOCK_SHA256}
        rows = [
            registry.log_result(workstream="M7-lite", experiment="lite/p2_f7", model="p2_index", metric="n_train_p2_rows_left",
                                value=f7["usable_rows_left"], split="train_p2", n=f7["train_p2_rows_usable"], artifacts=art,
                                notes="F7 ruling: train_p2 water bodies with a row in a val/test-role 25-km block removed "
                                      "whole; rows in blocks without a Lite role removed unless the salted block rule "
                                      "draws train. Counts only.",
                                extra={**common, "f7": f7, "p2_usable_rows": eff}),
            registry.log_result(workstream="M7-lite", experiment="lite/tests_mask", model="national-2b",
                                metric="n_train_pairings_kd_masked", value=meta["counts_by_role"]["train"]["kd_masked"],
                                split="train", n=meta["counts_by_role"]["train"]["chips"], artifacts=[str(mm)],
                                notes="F4 ruling: teacher pairings on M2 Test-S site-dates carry KD weight 0 and no "
                                      "teacher-derived feature. M2's 242 is the same rule counted as distinct "
                                      "Sentinel-2 chips of the 4-state teacher store over all roles. Keys only.",
                                extra={**common, **out["tests_mask"]}),
            registry.log_result(workstream="M7-lite", experiment="lite/p2_stratum", model="p2_index",
                                metric="n_test_p2_chips_no_v2_training_row_in_huc8_and_block",
                                value=out["test_p2_stratum"]["huc8_and_block_hold_no_v2_training_row"]["chips"],
                                split="test_p2", n=out["test_p2_stratum"]["all"]["chips"], artifacts=art,
                                notes="Verifier missed[3]: stratum of test_p2 chips whose HUC8 and 25-km block hold no v2 "
                                      "training row, registered before the look. Keys only; no test_p2 value read.",
                                extra={**common, "test_p2_stratum": out["test_p2_stratum"],
                                       "reserved_newsite_block": out["reserved_newsite_block"]}),
        ]
        print("registry rows", [r.get("id") or r.get("run_id") for r in rows])


if __name__ == "__main__":
    main()
