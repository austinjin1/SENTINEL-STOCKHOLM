# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the M2 Test-S flag table for the SENTINEL-Lite national-2 / national-2b chips (conflict C2).

Keys only: chip id, coordinates, NAIP date, sample date, teacher chip id and the teacher scene date. No label
column and no teacher value is read. Output (``$SENTINEL_RUNS/lite/tests_mask_<version>/``; ``--version national2b`` is the
table every round-v2 arm, comparator and re-run uses — it also covers the chips added by the label version):

* ``flags.parquet`` — ``naip_chip_id`` + ``tests_mask.FLAG_COLS`` for every chip of the label table;
* ``meta.json``     — rule text, input hashes, counts by role, counts in M2's unit (distinct teacher-store chips),
  sha256 of the flag table.

It writes no registry row; the count is registered with the stamp of the round-v2 amendment.

    python scripts/data/litev2_build_tests_mask.py [--version national2|national2b] [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from sentinel.core import paths
from sentinel.models.lite import national2 as N2
from sentinel.models.litev2 import tests_mask as TM

KEYS = ["naip_chip_id", "lat", "lon", "naip_date", "sample_date", "s2_chip_id"]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="national2", choices=("national2", "national2b"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else paths.RUNS_ROOT / "lite" / f"tests_mask_{a.version}"
    out.mkdir(parents=True, exist_ok=True)

    if a.version == "national2b":
        from sentinel.models.litev2 import labels2b as L

        lab_path, split, split_sha = L.labels_path(), L.load_split(), L.FROZEN_SHA256
    else:
        lab_path, split, split_sha = (paths.PROCESSED / "lite_labels" / "labels_national2.parquet", N2.load_split(),
                                      N2.FROZEN_SHA256)
    chips = pq.read_table(lab_path, columns=KEYS).to_pandas()
    s2 = []
    for store, p in (("4state", paths.PROCESSED / "naip_matchups" / "matchups.parquet"),
                     ("national_pm7", paths.PROCESSED / "naip_matchups" / "national_pm7" / "matchups.parquet")):
        s2.append(pq.read_table(p, columns=["s2_chip_id", "s2_sensing_time_utc"]).to_pandas().assign(store=store))
    s2 = pd.concat(s2).dropna(subset=["s2_chip_id"]).drop_duplicates("s2_chip_id").set_index("s2_chip_id")
    s2_date = pd.to_datetime(s2["s2_sensing_time_utc"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    chips["s2_date"] = chips["s2_chip_id"].map(s2_date)

    flags = TM.flags(chips)
    roles = pd.Series(split["chip_role"])
    counts = TM.counts(flags, roles)
    # the same rule in M2's unit: distinct teacher-store (Sentinel-2) chips, per store that first holds the chip
    km = flags["kd_masked"].to_numpy()
    s2_masked = chips.loc[km, "s2_chip_id"]
    role_of = chips["naip_chip_id"].map(roles)
    m2_unit = {"distinct_s2_chips_all_roles": int(s2_masked.nunique()),
               "distinct_s2_chips_by_store": {str(k): int(v) for k, v in
                                              s2_masked.drop_duplicates().map(s2["store"]).fillna("unknown")
                                              .value_counts().items()},
               "distinct_s2_chips_by_role": {str(r): int(chips.loc[km & (role_of == r).to_numpy(), "s2_chip_id"]
                                                         .nunique()) for r in sorted(set(role_of.dropna()))}}
    fp = out / "flags.parquet"
    flags.to_parquet(fp, index=False)
    meta = {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": "M2 Test-S: in-lake, non-buffer point of an orbit split v3 test-role water body, site-date within "
                f"{TM.ARCHIVE_FROM.date()}..{TM.ARCHIVE_TO.date()} (NAIP date, sample date or teacher scene date); "
                "kd_masked = on_test_s and the chip has a teacher pairing",
        "inputs": {str(lab_path): _sha(lab_path)},
        "version": a.version,
        "split_sha256": split_sha,
        "n_chips": int(len(flags)),
        "n_pairings_without_scene_date": int((chips["s2_chip_id"].notna() & chips["s2_date"].isna()).sum()),
        "counts_by_role": counts,
        "kd_masked_total": int(km.sum()),
        "kd_masked_in_m2_unit": m2_unit,
        "flags_sha256": _sha(fp),
        "reads": "keys only (no label column, no teacher value)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("n_chips", "counts_by_role", "kd_masked_total", "kd_masked_in_m2_unit",
                                             "flags_sha256")}, indent=1))


if __name__ == "__main__":
    main()
