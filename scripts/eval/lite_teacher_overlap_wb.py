# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Teacher overlap at Lite water-body level (pre-test review W8). Coordinates only: ``distill.lite_sites`` reads no
label column.

For each Lite role, this counts water bodies and chips whose nearest Lite-safe teacher row lies 2–5 km and
5–10 km away (the Lite-safe exclusion removed every row ≤ 2 km). It also counts flowing-class water bodies
(rivers, estuaries, canals; merged by HUC8 in Lite) with a teacher row within 5 km and 10 km. Teacher rows carry
no HUC8 or name, so the HUC8 and name merge rules cannot be checked directly; distance is the proxy. Output:
``$SENTINEL_RUNS/lite/teacher_litesafe/overlap_wb_level.json``.

    python -B scripts/eval/lite_teacher_overlap_wb.py
"""

from __future__ import annotations

import json

from sentinel.models.lite import runner as R

R.cpu_threads(2)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sentinel.core import paths  # noqa: E402
from sentinel.eval.splits import nearest_distance_km  # noqa: E402
from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import distill as KD  # noqa: E402
from sentinel.models.lite import guard as G  # noqa: E402


def main() -> int:
    tdir = paths.RUNS_ROOT / "lite" / "teacher_litesafe"
    man = json.loads((tdir / "manifest.json").read_text())
    if G.artifact_sha256(man["seen_points"]) != man["seen_points_sha256"]:
        raise RuntimeError("seen_points sha mismatch")
    seen = pd.read_parquet(man["seen_points"]).drop_duplicates(["lat", "lon"])
    sites = KD.lite_sites(G.load_split())
    sites["teacher_km"] = nearest_distance_km(sites["lat"], sites["lon"], seen["lat"], seen["lon"])
    flowing = sites["site_class"].isin(D.FLOWING_CLASSES)
    out = {}
    for role, s in sites.groupby("role"):
        wb = s.groupby("water_body").agg(dmin=("teacher_km", "min"), n_chips=("n_chips", "sum"),
                                         flowing=("site_class", lambda x: bool(x.isin(D.FLOWING_CLASSES).any())))
        r = {"water_bodies": int(len(wb)), "chips": int(wb["n_chips"].sum()),
             "min_teacher_km_all_sites": float(s["teacher_km"].min())}
        for lo, hi in ((0, 2), (2, 5), (5, 10)):
            m = (wb["dmin"] > lo) & (wb["dmin"] <= hi) if lo else wb["dmin"] <= hi
            r[f"wb_{lo}_{hi}km"] = int(m.sum())
            r[f"chips_{lo}_{hi}km"] = int(wb.loc[m, "n_chips"].sum())
        for km in (5, 10):
            m = wb["flowing"] & (wb["dmin"] <= km)
            r[f"flowing_wb_within_{km}km"] = int(m.sum())
            r[f"flowing_chips_within_{km}km"] = int(wb.loc[m, "n_chips"].sum())
        r["flowing_wb"] = int(wb["flowing"].sum())
        r["sites_2_5km"] = int(((s["teacher_km"] > 2) & (s["teacher_km"] <= 5)).sum())
        r["sites"] = int(len(s))
        r["flowing_sites_within_5km"] = int((flowing.loc[s.index] & (s["teacher_km"] <= 5)).sum())
        out[str(role)] = r
    R.write_json(tdir / "overlap_wb_level.json", {"teacher": man["label"], "by_role": out,
                                                  "note": "coordinates only; distance proxy for HUC8/name rules"})
    for k, v in out.items():
        print(k, {kk: vv for kk, vv in v.items() if kk.startswith(("wb_", "flowing_wb", "water_bodies", "sites_2"))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
