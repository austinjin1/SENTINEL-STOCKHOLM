#!/usr/bin/env python3
# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Event-layer label feasibility [S2] for M2 Orbit, M7 Lite, M3 Microbe, M5 BioAssess (NRSA + UK), Fuse
(Tier D, Tier C) and the M6 HydroMet track N (plan §3.2).

Protected rows are removed from id / role / date columns before any label column is read
(``events_feasibility_common``): Orbit dev_v4 cache role/tblock ∈ {train, val}, not buffer-excluded, not
protected; Lite ``chip_role`` ∈ {train, val, val_n}; Microbe ``development_ids`` and dates < 2024-01-01;
NRSA development surveys via ``load_nrsa`` (refuses holdouts) and UK via ``load_uk(include_reserved=False)``;
Fuse Tier D / Tier C dev sites; track N train/val-role stations, dates ≤ 2023-09-30. Labels via
``sentinel.eval.events.event_state``.

    python -B scripts/eval/events_feasibility_multi.py [--register]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import events_feasibility_common as C

from sentinel.core import paths
from sentinel.eval import events as EV

REPO = paths.REPO_ROOT
PRE = REPO / "docs" / "preregistration"
DATA = paths.PROCESSED
TABLE, ROWS = [], []


def sha12(p: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def state(x, code, scale=1.0, log10=False):
    v = np.asarray(pd.to_numeric(pd.Series(np.asarray(x)), errors="coerce"), float)
    # log10 round trip: 10**log10(20) can be 20.000000000000004; labs report <= 6 decimals, so round back
    v = np.round(10 ** v, 6) if log10 else v * scale
    return pd.Series(EV.event_state(v, code))


def rec(model, track, cell, event, y: pd.Series, clusters: dict, split_label: str, note: str = "",
        val_cell: bool = False, primary_cluster: str | None = None):
    y = pd.Series(np.asarray(y, dtype=float))
    ok = y.notna().to_numpy()
    yy = y[ok].astype(bool).to_numpy()
    r = {"model": model, "track": track, "cell": cell, "event": event, "n_labelled": int(ok.sum()),
         "n_pos": int(yy.sum()), "prevalence": float(yy.mean()) if ok.any() else np.nan, "note": note}
    for name, c in clusters.items():
        cc = np.asarray(c, dtype=object)[ok]
        r[f"n_{name}"] = int(pd.Series(cc).nunique())
        r[f"n_{name}_with_pos"] = int(pd.Series(cc[yy]).nunique())
    if val_cell:
        pc = primary_cluster or next(iter(clusters))
        r["power_flag"] = EV.power_flag(r["n_pos"], r[f"n_{pc}_with_pos"])
        r["power_cluster_unit"] = pc
    TABLE.append(r)
    extra = {k: v for k, v in r.items() if k not in ("model", "event")}
    ROWS.append({"model": model, "target": event, "split": split_label, "metric": "n_pos", "value": r["n_pos"],
                 "n": r["n_labelled"], "extra": extra})


def orbit():
    f = paths.RUNS_ROOT / "orbit" / "cache" / "dev_v4" / "table.parquet"
    filt = [("role", "in", ["train", "val"]), ("tblock", "in", ["train", "val"]), ("buffer_excluded", "=", False),
            ("protected", "=", False)]
    ids = pq.read_table(f, columns=["role", "tblock", "buffer_excluded", "protected", "in_lake", "waterbody", "block"],
                        filters=filt).to_pandas()
    C.assert_allowed(ids["role"], {"train", "val"}, "orbit role")
    C.assert_allowed(ids["tblock"], {"train", "val"}, "orbit tblock")
    if ids["protected"].any() or ids["buffer_excluded"].any():
        raise C.ProtectedRowsError("orbit protected/buffer rows kept")
    lab = C.read_labels(f, ["chla", "tss", "turbidity"], np.ones(len(ids), bool), filters=filt)
    sp = f"orbit_split_v3_{sha12(PRE / 'orbit_split_v3.json')}"
    cells = {"fit_trainxtrain": (ids.role == "train") & (ids.tblock == "train"),
             "valxval_lake": (ids.role == "val") & (ids.tblock == "val") & ids.in_lake,
             "valxany_lake": (ids.role == "val") & ids.in_lake}
    ev = {"E4a": state(lab.chla, "E4a"), "E4a_t50": state(lab.chla, "E4a_t50"), "E4a_t12": state(lab.chla, "E4a_t12"),
          "E3a_tss": state(lab.tss, "E3a_tss"), "E3a_turb": state(lab.turbidity, "E3a_turb")}
    for cn, m in cells.items():
        m = m.to_numpy()
        for en, y in ev.items():
            rec("M2_orbit", "dev_v4", cn, en, y[m], {"waterbody": ids.waterbody[m], "block50": ids.block[m]},
                f"{cn}/{sp}", val_cell=cn.startswith("val"), primary_cluster="waterbody")


def lite():
    for name, split_file, roles in (("lite_split_v2", "lite_split.json", ("train", "val")),
                                    ("national", "lite_split_national.json", ("train", "val", "val_n"))):
        role = pd.Series(json.loads((PRE / split_file).read_text())["chip_role"])
        f = DATA / "lite_labels" / ("labels.parquet" if name == "lite_split_v2" else "labels_national.parquet")
        ids = pq.read_table(f, columns=["naip_chip_id", "water_body"]).to_pandas()
        ids["role"] = ids["naip_chip_id"].map(role)
        keep = ids["role"].isin(roles).to_numpy()
        ids = ids[keep].reset_index(drop=True)
        C.assert_allowed(ids["role"], set(roles), f"lite {name} chip_role")
        lab = C.read_labels(f, ["y_tss", "y_turbidity", "y_chla"], keep)
        ev = {"E4a": state(lab.y_chla, "E4a", log10=True), "E3a_tss": state(lab.y_tss, "E3a_tss", log10=True),
              "E3a_turb": state(lab.y_turbidity, "E3a_turb", log10=True)}
        sp = f"{split_file.removesuffix('.json')}_{sha12(PRE / split_file)}"
        for r_ in roles:
            m = (ids.role == r_).to_numpy()
            for en, y in ev.items():
                rec("M7_lite", name, r_, en, y[m], {"waterbody": ids.water_body[m]}, f"{r_}/{sp}",
                    val_cell=r_.startswith("val"))


def microbe():
    ms = json.loads((PRE / "microbe_split.json").read_text())
    dev, hold = set(map(str, ms["development_ids"])), set(map(str, ms["holdout_ids"]))
    f = DATA / "microbe_wq" / "samples.parquet"
    ids = pq.read_table(f, columns=["sample_id", "site_id", "domain_id", "collect_date"]).to_pandas()
    ids["sample_id"] = ids["sample_id"].astype(str)
    keep = (ids["sample_id"].isin(dev) & ~ids["sample_id"].isin(hold)
            & (pd.to_datetime(ids["collect_date"]) < pd.Timestamp("2024-01-01"))).to_numpy()
    ids = ids[keep].reset_index(drop=True)
    C.assert_dates_within(ids["collect_date"], hi="2023-12-31", what="microbe collect_date")
    if ids["sample_id"].isin(hold).any():
        raise C.ProtectedRowsError("microbe holdout ids kept")
    lab = C.read_labels(f, ["raw_do_mgl", "raw_chla_ugl", "raw_no3_mgl", "raw_tp_mgl", "raw_turb_fnu", "raw_sc_uscm"], keep)
    ev = {"E1a_grab": state(lab.raw_do_mgl, "E1a_grab"), "E4a": state(lab.raw_chla_ugl, "E4a"),
          "E5a": state(lab.raw_no3_mgl, "E5a"), "E5b": state(lab.raw_tp_mgl, "E5b"),
          "E3a_turb": state(lab.raw_turb_fnu, "E3a_turb"), "E2c": state(lab.raw_sc_uscm, "E2c")}
    sp = f"dev_lodo/microbe_split_{sha12(PRE / 'microbe_split.json')}"
    for en, y in ev.items():
        rec("M3_microbe", "lodo_dev", "dev_all_lodo_oof", en, y, {"domain": ids.domain_id, "site": ids.site_id}, sp,
            note="E1a_grab = sonde daily-mean DO < 3 at the sample; E3a_turb on FNU", val_cell=True)


def bioassess():
    from sentinel.models.bioassess import data as BD
    from sentinel.models.bioassess import uk as UK

    D = BD.load_nrsa()                         # refuses the holdout surveys
    BD.assert_dev_only(D.samples)
    C.assert_allowed(D.samples["survey"], set(BD.DEV_SURVEYS), "NRSA survey")
    keep_ids = set(D.samples.index)
    cw_ids = pq.read_table(DATA / "nars_streams" / "chemistry_wide.parquet", columns=["visit_id"]).to_pandas()
    keep = cw_ids["visit_id"].isin(keep_ids).to_numpy()
    cw = C.read_labels(DATA / "nars_streams" / "chemistry_wide.parquet",
                       ["visit_id", "chloride_mg_l", "cond_us_cm", "tp_ug_l", "no3no2_n_mg_l", "ph_lab_su", "turbidity_ntu"],
                       keep).set_index("visit_id").reindex(D.samples.index)
    cd_ids = pq.read_table(DATA / "nars_streams" / "condition_wide.parquet", columns=["visit_id"]).to_pandas()
    keep = cd_ids["visit_id"].isin(keep_ids).to_numpy()
    cd = C.read_labels(DATA / "nars_streams" / "condition_wide.parquet",
                       ["visit_id", "tp_cond", "tn_cond", "salinity_cond", "bent_mmi_cond"], keep
                       ).set_index("visit_id").reindex(D.samples.index)
    s = D.samples
    cl = {"location": s.location_id.to_numpy(), "ecoregion9": s.ag_eco9.to_numpy()}
    ev = {"E2a_lab": state(cw.chloride_mg_l, "E2a_lab"), "E2c": state(cw.cond_us_cm, "E2c"),
          "E5b": state(cw.tp_ug_l, "E5b", scale=1e-3), "E5a": state(cw.no3no2_n_mg_l, "E5a"),
          "E6_snap": state(cw.ph_lab_su, "E6_snap"), "E3a_turb": state(cw.turbidity_ntu, "E3a_turb")}
    for c in ("tp_cond", "tn_cond", "salinity_cond", "bent_mmi_cond"):
        v = cd[c].astype("string")
        ev[f"E5b_nrsa:{c}_poor"] = pd.Series(np.where(v.isin(["Poor", "Fair", "Good"]).fillna(False),
                                                      (v == "Poor").fillna(False).astype(float), np.nan))
    sp = "dev_loeo/nrsa0809+nrsa1314"
    for en, y in ev.items():
        rec("M5_bioassess", "nrsa_dev", "dev_all_loeo_oof", en, y, cl, sp, val_cell=True, primary_cluster="location")
    U = UK.load_uk()                           # include_reserved=False: reserved blocks dropped first
    C.assert_allowed(U.samples["role"], {"pool_fit", "pool_val"}, "UK role")
    Y = U.Y
    ev = {"E2a_lab": state(Y.uk_log10_chloride_mg_l, "E2a_lab", log10=True),
          "E6_snap": state(Y.uk_ph, "E6_snap"), "E2c": state(Y.uk_log10_cond25_us_cm, "E2c", log10=True)}
    ev["E5b"] = pd.Series(EV.event_state(np.round(10 ** Y.uk_log10_tp_ug_l.to_numpy(float), 6) * 1e-3, "E5b"))  # ug/L -> mg/L
    role = U.samples["role"].reindex(Y.index).to_numpy()
    site = U.samples["site_id"].reindex(Y.index).to_numpy()
    for r_ in ("pool_fit", "pool_val"):
        m = role == r_
        for en, y in ev.items():
            rec("M5_bioassess", "uk_pool", r_, en, y[m], {"site": site[m]}, f"{r_}/uk_ea_blocks50km_seed0",
                val_cell=r_ == "pool_val")


def fuse():
    fs = json.loads((PRE / "fuse_tier_d_split_v2.json").read_text())
    devs = sorted(s for s, r in fs["site_role"].items() if r == "dev")
    C.assert_allowed([fs["site_role"][s] for s in devs], {"dev"}, "fuse tier D role")
    f = DATA / "fuse_tier_d" / "daily.parquet"
    ids = pq.read_table(f, columns=["site", "domain"], filters=[("site", "in", devs)]).to_pandas()
    C.assert_allowed([fs["site_role"].get(s, "unknown") for s in ids["site"].unique()], {"dev"}, "fuse rows")
    cols = ["ev__H1_hypoxia__onset_h7", "ev__H1_hypoxia__any_h7", "ev__H5b_chl_bloom__onset_h7",
            "ev__H5b_chl_bloom__any_h7", "ev__H4_nitrate__any_h7"]
    lab = C.read_labels(f, cols, np.ones(len(ids), bool), filters=[("site", "in", devs)])
    sp = f"dev_domain_folds/fuse_tier_d_split_v2_{sha12(PRE / 'fuse_tier_d_split_v2.json')}"
    code = {"H1_hypoxia": "E1a", "H5b_chl_bloom": "E4a", "H4_nitrate": "E5a"}
    for c in cols:
        hz, kind = c.split("__")[1], c.split("__")[2]
        y = pd.to_numeric(lab[c], errors="coerce")
        rec("Fuse", "tier_d_v2", "dev", f"{code[hz]}:{kind}", (y > 0.5).astype(float).where(y.notna()),
            {"domain": ids.domain.to_numpy(), "site": ids.site.to_numpy()}, sp, note="issue-row labels (h = 7)",
            val_cell=True, primary_cluster="domain")
    fc = json.loads((PRE / "fuse_tier_c_split.json").read_text())
    devc = sorted(s for s, r in fc["site_role"].items() if r == "dev")
    f = DATA / "fuse_tier_c" / "rows.parquet"
    ids = pq.read_table(f, columns=["site_id", "huc4"], filters=[("site_id", "in", devc)]).to_pandas()
    C.assert_allowed([fc["site_role"].get(s, "unknown") for s in ids["site_id"].unique()], {"dev"}, "fuse C rows")
    y = pd.to_numeric(C.read_labels(f, ["y_gt8"], np.ones(len(ids), bool), filters=[("site_id", "in", devc)])["y_gt8"],
                      errors="coerce")
    rec("Fuse", "tier_c", "dev", "E4b", (y > 0.5).astype(float).where(y.notna()),
        {"huc4": ids.huc4.to_numpy(), "site": ids.site_id.to_numpy()},
        f"dev/fuse_tier_c_split_{sha12(PRE / 'fuse_tier_c_split.json')}", val_cell=True, primary_cluster="huc4")


def track_n():
    tn = json.loads((PRE / "hydromet_track_n_stations.json").read_text())
    roles = pd.Series(tn["role_of"])
    bg = pd.Series(tn["basin_group_of"])
    roles.index = roles.index.str.replace("^WQP:", "", regex=True)
    bg.index = bg.index.str.replace("^WQP:", "", regex=True)
    okst = sorted(roles[roles.isin(["train", "val"])].index)
    f = DATA / "hydromet_wqp_national" / "obs_daily.parquet"
    filt = [("site_id", "in", okst), ("target", "=", "do_mgl_mean")]
    ids = pq.read_table(f, columns=["site_id", "date"], filters=filt).to_pandas()
    keep = (pd.to_datetime(ids["date"]) <= C.FUTURE_CUTOFF).to_numpy()
    ids = ids[keep].reset_index(drop=True)
    ids["role"] = ids["site_id"].map(roles)
    C.assert_allowed(ids["role"], {"train", "val"}, "track N station role")
    C.assert_dates_within(ids["date"], what="track N dates")
    v = C.read_labels(f, ["value"], keep, filters=filt)["value"]
    y = state(v, "E1a_grab")
    sp = f"hydromet_track_n_{tn['sha256'][:12]}"
    for r_ in ("train", "val"):
        m = (ids.role == r_).to_numpy()
        rec("M6_hydromet", "track_N_wqp_grab_WY2022_23", f"{r_}_role_stations", "E1a_grab", y[m],
            {"basin_group": ids.site_id[m].map(bg).to_numpy(), "station": ids.site_id[m].to_numpy()},
            f"{r_}_stations_valblock/{sp}", val_cell=r_ == "val", primary_cluster="basin_group")


def main() -> int:
    a = C.parser(__doc__).parse_args()
    out = C.out_dir(a, "multi")
    for fn in (orbit, lite, microbe, bioassess, fuse, track_n):
        fn()
        print(f"[done] {fn.__name__}", flush=True)
    res = pd.DataFrame(TABLE)
    f = C.write(res, out, "multi_event_counts", {"events_version": EV.__version__, "created": C.stamp()})
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 500)
    print(res.drop(columns=["note"]).to_string(index=False))
    if a.register:
        n = C.register(ROWS, script=__file__, artifacts=[f],
                       data_manifests=["lite_labels", "microbe_wq", "nars_streams", "uk_ea_ecology", "fuse_tier_d",
                                       "fuse_tier_c", "hydromet_wqp_national"])
        print(f"registered {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
