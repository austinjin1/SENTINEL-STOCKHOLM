# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Track P2 builds (lite_arch_v2_amendment.md §8).

  lock    build the pool from identifiers / coordinates / site types / dates only, write the lock list
          (docs/preregistration/lite_p2_lock.json, exclusive create, read-only) and log its SHA-256 in a
          ``lite/p2_lock`` registry row. No label value is read.
  index   (after the lock row exists) NAIP match of the locked site-dates; label values of ``test_p2`` are never read
          beyond a present/absent flag (see sentinel/models/litev2/p2_index.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel.core import registry  # noqa: E402
from sentinel.models.litev2 import p2  # noqa: E402


def cmd_lock(_):
    if p2.LOCK_FILE.exists():
        raise SystemExit(f"{p2.LOCK_FILE} exists: the lock list is written once and never rebuilt")
    rec = p2.build_lock()
    sha = p2.write_lock(rec)
    p2.load_lock(expect_sha256=sha)
    n_wb = len(rec["water_bodies"])
    registry.log_result(workstream="lite", experiment="lite/p2_lock", model="p2_lock", metric="n_water_bodies",
                        value=float(n_wb), split="prereg", n=n_wb,
                        notes="P2 lock list (metadata only; no label value read). Roles by salted hash of the "
                              "smallest site id per water body.",
                        extra={"path": "docs/preregistration/lite_p2_lock.json", "sha256": sha, "salt": p2.SALT,
                               "role_counts": rec["role_counts"], "pool_counts": rec["pool_counts"]})
    print(json.dumps({"sha256": sha, "pool_counts": rec["pool_counts"], "role_counts": rec["role_counts"]}, indent=1))


def cmd_index(a):
    import logging

    from sentinel.models.litev2 import p2_index as PI

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(PI.out_dir() / "build.log"), logging.StreamHandler()])
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    log = logging.getLogger("lite_p2").info
    if a.plan_only:
        rows, _, counts = PI.plan(log)
        counts["labelled_site_dates_by_role"] = rows.groupby("role").size().to_dict()
        counts["water_bodies_by_role"] = rows.groupby("role")["water_body"].nunique().to_dict()
        counts["by_role_label_src"] = {f"{k[0]}|{k[1]}": int(v) for k, v in rows.groupby(["role", "label_src"]).size().items()}
        print(json.dumps(counts, indent=1, default=str))
        return
    counts = PI.build(log, workers=a.workers)
    t = counts["by_role"].get("test_p2", {})
    lock = p2.load_lock()
    registry.log_result(workstream="lite", experiment="lite/p2_index", model="p2_index",
                        metric="n_test_p2_chips_with_label", value=float(t.get("chips_eligible_with_label", 0)),
                        split="prereg", n=int(t.get("chips_eligible_with_label", 0)),
                        notes="P2 index counts (no label statistic), rebuilt under the project WQP label rule "
                              "(lead order 2026-11-10 15:05 EST item 3: only the label filter changes; lock and "
                              "Q6 constants unchanged). test_p2: presence flags only; no test value was read. "
                              "P2 is descriptive this round (verifier ruling F8).",
                        extra={"lock_sha256": lock["_sha256"], "counts": json.loads(json.dumps(counts, default=str)),
                               "index": str(PI.index_path()), "index_sha256": PI.index_sha256(),
                               "label_rule": counts.get("label_rule"),
                               "first_build_index_sha256":
                                   "68688fac1af2c4e147070562f3dde8a035eb150360dd49759619456baf23ff44",
                               "excluded_at_load": sorted(PI.EXCLUDED_AT_LOAD)})
    print(json.dumps(counts, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock").set_defaults(fn=cmd_lock)
    ix = sub.add_parser("index")
    ix.add_argument("--plan-only", action="store_true", help="NAIP match and label presence counts; no chip is read")
    ix.add_argument("--workers", type=int, default=4)
    ix.set_defaults(fn=cmd_index)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
