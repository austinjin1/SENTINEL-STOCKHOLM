# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Build the SENTINEL-Lite label version national-2b (sentinel/models/litev2/labels2b.py; amendment §3; lead order
2026-11-10 15:05 EST, rulings 2a/2b).

Writes ``processed/lite_labels/labels_national2b.parquet``, the pin file
``docs/preregistration/lite_split_national2b.json`` and a counts file. Counts and presence flags only: no label
value of any row is printed, logged or written outside the label table itself.

    python scripts/data/litev2_build_labels2b.py [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sentinel.core import paths  # noqa: E402
from sentinel.data.reserved import assert_no_reserved, exclude_reserved  # noqa: E402
from sentinel.models.lite import data as D  # noqa: E402
from sentinel.models.lite import national as N  # noqa: E402
from sentinel.models.lite import national2 as N2  # noqa: E402
from sentinel.models.litev2 import labels2b as L  # noqa: E402
from sentinel.models.litev2 import wqp_rule as WR  # noqa: E402

ROLE_ORDER = ("train", "val", "val_n", "test", "test_n", "test_time", "test_time_n", "excluded", "none")


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def by(df: pd.DataFrame, cols, name="n") -> dict:
    if not len(df):
        return {}
    g = df.groupby(list(cols), dropna=False).size()
    return {"|".join(map(str, k if isinstance(k, tuple) else (k,))): int(n) for k, n in g.items()}


def processed_build(name: str, path: Path, role_of: dict) -> dict:
    """Rule applied to one processed matchup build. Returns as-built chip table, corrected rows and chip table,
    row-level counts by target and role."""
    m = pd.read_parquet(path)
    kept, dropped, rec = WR.apply_rule(m, name, log=L._quiet)
    as_built, _ = L.chip_table(m)
    corr, rep = L.chip_table(kept)
    k = WR.keyed(m, name)
    k["role"] = k["naip_chip_id"].astype(str).map(role_of).fillna("none")
    why = dropped.drop_duplicates("sample_key").set_index("sample_key")["drop_reason"]
    k["status"] = k["sample_key"].map(why).fillna("kept")
    side, _sha = WR.load_side(k["sample_key"])
    ref = side.drop_duplicates("sample_key").set_index("sample_key")
    fewer = (ref["n_results"].reindex(k["sample_key"]).to_numpy() < ref["n_results_all"].reindex(k["sample_key"]).to_numpy())
    k["status"] = np.where((k["status"] == "kept") & fewer, "kept_with_fewer_results", k["status"])
    rows = by(k, ["target", "role", "status"])
    return {"matchups": m, "kept": kept, "dropped": dropped, "as_built": as_built.set_index("naip_chip_id"),
            "corrected": corr.set_index("naip_chip_id"), "rows": rows, "harmonise_report": rep,
            "record": {k2: v for k2, v in rec.items() if k2 != "orbit_record"}, "keyed": k}


def chip_reason(chips, keyed: pd.DataFrame) -> pd.Series:
    """Drop reason of chips that lose every label: from the status of their as-built rows."""
    k = keyed[keyed["naip_chip_id"].astype(str).isin(set(chips))]
    st = k.assign(c=k["naip_chip_id"].astype(str)).groupby("c")["status"].agg(lambda s: set(s))

    def one(s):
        if s & {"kept", "kept_with_fewer_results"}:
            return "rejected_by_frozen_label_rules"
        return "no_surface_result" if "no_surface_result" in s else "no_qc_passed_result"

    return st.map(one)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build and count, write only the counts file")
    args = ap.parse_args()
    out_dir = paths.run_dir("lite", "counts_arch_v2")
    counts: dict = {"version": L.VERSION, "rule": WR.RULE}

    lab2 = pd.read_parquet(N2.labels_path())
    lab2["naip_chip_id"] = lab2["naip_chip_id"].astype(str)
    split2 = N2.load_split()
    if N.labels_sha256(lab2) != split2["labels_sha256"]:
        raise SystemExit("labels_national2.parquet is not the pinned as-built table")
    role_of = dict(split2["chip_role"])
    lab2["role"] = lab2["naip_chip_id"].map(role_of)
    src = np.where(lab2["source"] == "frozen", "frozen", np.where(lab2["source2"] == "national1", "national",
                                                                 "national_pm7"))
    lab2["version_source"] = src
    counts["national2"] = {"split_sha256": split2["_sha256"], "labels_sha256": split2["labels_sha256"],
                           "chips_by_source_role": by(lab2, ["version_source", "role"])}

    # ---------------------------------------------------------------- corrected rows, three sources
    nat = processed_build("national", N.national_matchups_path(), role_of)
    pm7 = processed_build("national_pm7", N2.matchups_path(), role_of)
    m4 = D.load_matchups()
    fz_ids = lab2.loc[lab2["version_source"] == "frozen", "naip_chip_id"]
    kept4, st4 = L.rederive_frozen(m4, fz_ids)
    corr4, rep4 = L.chip_table(kept4)
    corr4 = corr4.set_index("naip_chip_id")
    st4["role"] = st4["naip_chip_id"].map(role_of).fillna("none")
    # 4-state rows: number of member results as built (all units of the key) against the re-derived number
    a = m4.assign(naip_chip_id=m4["naip_chip_id"].astype(str), site_id=m4["site_id"].astype(str),
                  characteristic=m4["characteristic"].astype(str), date=pd.to_datetime(m4["date"]).dt.normalize())
    n_old = a.groupby(["naip_chip_id", *L.SITE_KEY])["n_results"].sum()
    n_new = kept4.groupby(["naip_chip_id", *L.SITE_KEY])["n_results"].sum()
    idx = pd.MultiIndex.from_frame(st4[["naip_chip_id", *L.SITE_KEY]])
    differs = (n_old.reindex(idx).to_numpy(float) != n_new.reindex(idx).to_numpy(float)) & (st4["status"] == "kept").to_numpy()
    st4["status"] = np.where(differs, "kept_with_different_n_results", st4["status"])
    counts["rows"] = {"national": nat["rows"], "national_pm7": pm7["rows"],
                      "four_state": by(st4, ["target", "role", "status"])}
    counts["records"] = {"national": nat["record"], "national_pm7": pm7["record"],
                         "four_state": {"n_row_keys": int(len(st4)), "key": list(L.SITE_KEY),
                                        "harmonise_report": rep4}}

    # ---------------------------------------------------------------- pipeline check on the as-built labels
    chk = lab2.copy()
    for name, b in (("national", nat), ("national_pm7", pm7)):
        m = (chk["version_source"] == name).to_numpy()
        ids = chk.loc[m, "naip_chip_id"]
        if not ids.isin(b["as_built"].index).all():
            raise SystemExit(f"{name}: as-built chips missing from the rebuilt as-built table")
        for t in L.ORIG:
            chk.loc[m, f"y_{t}"] = b["as_built"][f"y_{t}"].reindex(ids).to_numpy()
    if N.labels_sha256(chk) != split2["labels_sha256"]:
        raise SystemExit("pipeline check failed: the as-built labels are not reproduced from the matchup builds")
    counts["pipeline_check"] = "as-built labels_sha256 reproduced from the national and national_pm7 matchup builds"

    # ---------------------------------------------------------------- corrected original labels
    lab = lab2.drop(columns=["role"]).copy()
    old_has = {t: lab2[f"y_{t}"].notna().to_numpy() for t in L.ORIG}
    for name, tab in (("frozen", corr4), ("national", nat["corrected"]), ("national_pm7", pm7["corrected"])):
        m = (lab["version_source"] == name).to_numpy()
        ids = lab.loc[m, "naip_chip_id"]
        for t in L.ORIG:
            lab.loc[m, f"y_{t}"] = tab[f"y_{t}"].reindex(ids).to_numpy()
            lab.loc[m, f"n_{t}"] = tab[f"n_{t}"].reindex(ids).fillna(0).astype(int).to_numpy()
        lab.loc[m, "sample_date"] = tab["sample_date"].reindex(ids).to_numpy()
    lab["y_chla_class"] = D.trophic_class_log10(lab["y_chla"])
    new_has = {t: lab[f"y_{t}"].notna().to_numpy() for t in L.ORIG}
    none_left = ~np.column_stack([new_has[t] for t in L.ORIG]).any(axis=1)
    lab["label_drop_reason"] = ""
    for name, keyed in (("national", nat["keyed"]), ("national_pm7", pm7["keyed"])):
        m = none_left & (lab["version_source"] == name).to_numpy()
        lab.loc[m, "label_drop_reason"] = chip_reason(lab.loc[m, "naip_chip_id"], keyed).reindex(
            lab.loc[m, "naip_chip_id"]).to_numpy()
    m = none_left & (lab["version_source"] == "frozen").to_numpy()
    st = st4.assign(status=st4["status"].replace({"kept_with_different_n_results": "kept"}))
    lab.loc[m, "label_drop_reason"] = chip_reason(lab.loc[m, "naip_chip_id"], st).reindex(
        lab.loc[m, "naip_chip_id"]).to_numpy()
    role = lab2["role"].to_numpy()
    chips = {}
    for s in ("frozen", "national", "national_pm7"):
        for r in sorted(set(role)):
            m = (lab["version_source"] == s).to_numpy() & (role == r)
            if not m.any():
                continue
            c = {"chips": int(m.sum()), "chips_losing_every_label": int((none_left & m).sum()),
                 "reasons": lab.loc[none_left & m, "label_drop_reason"].value_counts().to_dict()}
            for t in L.ORIG:
                c[t] = {"as_built": int((old_has[t] & m).sum()), "kept": int((old_has[t] & new_has[t] & m).sum()),
                        "lost": int((old_has[t] & ~new_has[t] & m).sum()),
                        "gained": int((~old_has[t] & new_has[t] & m).sum())}
            chips[f"{s}|{r}"] = c
    counts["chips_original_targets"] = chips
    tot = {}
    for r in sorted(set(role)):
        m = role == r
        tot[r] = {"chips": int(m.sum()), "chips_losing_every_label": int((none_left & m).sum()),
                  **{t: {"as_built": int((old_has[t] & m).sum()), "corrected": int((new_has[t] & m).sum()),
                         "lost": int((old_has[t] & ~new_has[t] & m).sum()),
                         "gained": int((~old_has[t] & new_has[t] & m).sum())} for t in L.ORIG}}
    counts["chips_original_targets_by_role"] = tot

    # ---------------------------------------------------------------- recovered labels (±7 d build, rule applied)
    k7 = pm7["kept"]
    k7 = k7[k7["naip_chip_id"].notna()].copy()
    k7["naip_chip_id"] = k7["naip_chip_id"].astype(str)
    k7["role"] = k7["naip_chip_id"].map(role_of).fillna("none")
    unit = k7["unit"].fillna("").astype(str).str.strip()
    cens = k7["censored"].fillna(False).astype(bool)
    grp = np.full(len(k7), "", dtype=object)
    for name, (tgt, u, _b) in L.RECOVERED.items():
        grp[((k7["target"] == tgt) & (unit == u) & ~cens).to_numpy()] = name
    for t, (tgt, u) in L.CENSORED.items():
        grp[((k7["target"] == tgt) & (unit == u) & cens).to_numpy()] = f"censored_{t}"
    grp[((k7["target"] == "turbidity") & unit.isin(["FNU", "NTRU"]) & cens).to_numpy()] = "censored_turbidity_fnu_ntru"
    k7["group"] = grp
    role6 = k7["role"].replace({r: "test_roles" for r in L.TEST_ROLES}).replace({"val_n": "val", "val": "val"})
    counts["recovered_rows_under_rule"] = by(k7[k7["group"] != ""].assign(role6=role6), ["group", "role6"])
    k7["u"] = unit.to_numpy()
    counts["other_rows_under_rule"] = by(k7[(k7["group"] == "") & ~k7["target"].isin(L.ORIG)], ["target", "u"])
    counts["turbidity_units_under_rule"] = by(k7[k7["target"] == "turbidity"], ["u"])
    on_test = k7["role"].isin(L.TEST_ROLES)
    use = k7[~on_test.to_numpy()]
    rec_tab, rec_rep = L.recovered_uncensored(use)
    harm7 = L.harmonised_results(use.loc[use["censored"].fillna(False).astype(bool), "site_id"].astype(str).unique())
    dirs = L.censor_directions(harm7)
    cen_tab, cen_rep = L.censored_chips(use, dirs)
    counts["recovered_uncensored_report"] = rec_rep
    counts["censored_report"] = cen_rep
    # chips with an uncensored, in-bounds value of a target in the ±7 d rows (the 'no uncensored value' condition)
    h7, _ = D.harmonise(use)
    unc = {t: set(h7.loc[h7["target"] == t, "naip_chip_id"].astype(str)) for t in L.ORIG}
    unc["ssc"] = set(rec_tab.index[rec_tab["y_ssc"].notna()])

    # ---------------------------------------------------------------- added chips (no national-2 role)
    meta_cols = ["site_id", "lat", "lon", "state", "site_class", "naip_item_id", "naip_date", "naip_year", "gsd_m",
                 "chip_crs", "chip_left", "chip_top", "naip_shard", "organization", "huc8", "name", "s2_chip_id",
                 "dt_days_s2_naip"]
    none = use[use["role"] == "none"]
    with_rec = (set(rec_tab.index) | set(cen_tab["naip_chip_id"])) & set(none["naip_chip_id"])
    add_rows = none[none["naip_chip_id"].isin(with_rec)]
    added = add_rows.sort_values("naip_chip_id").groupby("naip_chip_id")[meta_cols].first()
    sd = add_rows.assign(_ad=add_rows["dt_days_naip"].abs()).sort_values(["naip_chip_id", "_ad"])
    added["sample_date"] = sd.groupby("naip_chip_id")["date"].first()
    added = added.reset_index()
    store_ok = set(D.naip_store()._ok.index.astype(str))
    in_store = added["naip_chip_id"].isin(store_ok)
    counts["added_chips"] = {"role_less_chips_in_build": int(none["naip_chip_id"].nunique()),
                             "with_a_recovered_label": int(len(added)),
                             "dropped_chip_missing_from_naip_store": int((~in_store).sum()),
                             "role_less_chips_with_an_original_label_under_rule_not_built":
                                 int(len(set(pm7["corrected"].index) - set(role_of)))}
    added = added[in_store].reset_index(drop=True)
    # reserved new-site block (docs/preregistration/sonde_newsites_list.json): never a Lite row. Counts only.
    free = set(exclude_reserved(added["site_id"].astype(str).tolist()))
    counts["added_chips"]["dropped_site_in_reserved_newsite_block"] = int((~added["site_id"].astype(str).isin(free)).sum())
    added = added[added["site_id"].astype(str).isin(free)].reset_index(drop=True)
    assert_no_reserved(added["site_id"].astype(str), where="national-2b added chips")
    free2 = set(exclude_reserved(lab["site_id"].astype(str).unique().tolist()))
    on_res = ~lab["site_id"].astype(str).isin(free2)
    counts["national2_chips_at_reserved_newsite_block_sites"] = {
        "by_role": by(pd.DataFrame({"role": lab2["role"].to_numpy()[on_res.to_numpy()]}), ["role"]),
        "by_source": by(lab2[on_res.to_numpy()], ["source", "source2"]),
        "n_sites": int(lab.loc[on_res, "site_id"].nunique()),
        "treatment": "role unchanged in the pin file; never returned by labels2b.role_frame (load-time exclusion)"}
    wb2b = L.water_bodies_2b(lab2, added)
    add_role, role_rep = L.added_chip_roles(added, lab2, split2, wb2b)
    block_role_unseen_2b = role_rep.pop("block_role_unseen_2b")
    counts["added_chip_roles"] = role_rep
    added["doy"] = pd.to_datetime(added["naip_date"]).dt.dayofyear.astype(int)
    added["y_chla_class"] = -1
    added["water_body"] = "N2b:" + added["site_id"].astype(str).map(wb2b).astype(str)
    added["source"], added["source2"], added["version_source"] = "new", "national2b_added", "added"
    added["label_drop_reason"] = ""
    for t in L.ORIG:
        added[f"y_{t}"], added[f"n_{t}"] = np.nan, 0
    lab["national2b_wb"] = lab["site_id"].astype(str).map(wb2b)
    added["national2b_wb"] = added["site_id"].astype(str).map(wb2b)
    full = pd.concat([lab, added[[c for c in lab.columns if c in added.columns]]], ignore_index=True)
    chip_role = {**role_of, **add_role.to_dict()}
    full["_role"] = full["naip_chip_id"].map(chip_role)
    if full["_role"].isna().any() or full["naip_chip_id"].duplicated().any():
        raise SystemExit("chips without a role or duplicated chips")

    # recovered columns by role: FNU / SSC on train, val, val_n; NTRU and censored on train only; never on test roles
    r = full["_role"].to_numpy()
    is_train, is_val = r == "train", np.isin(r, L.VAL_ROLES)
    ids = full["naip_chip_id"]
    for name, where in (("turb_fnu", is_train | is_val), ("ssc", is_train | is_val), ("turb_ntru", is_train)):
        y = rec_tab[f"y_{name}"].reindex(ids).to_numpy(float)
        n = rec_tab[f"n_{name}"].reindex(ids).fillna(0).to_numpy(int)
        full[f"y_{name}"] = np.where(where, y, np.nan)
        full[f"n_{name}"] = np.where(where & np.isfinite(y), n, 0)
    cen_counts = {}
    for t in L.CENSORED:
        c = cen_tab[cen_tab["target"] == t].set_index("naip_chip_id")
        d = c["direction"].reindex(ids).fillna(0).to_numpy(int)
        lim = c["log10_limit"].reindex(ids).to_numpy(float)
        has_unc = full[f"y_{t}"].notna().to_numpy() | ids.isin(unc[t]).to_numpy()
        ok = (d != 0) & is_train & ~has_unc
        full[f"cens_{t}"] = np.where(ok, d, 0).astype(int)
        full[f"lim_{t}"] = np.where(ok, lim, np.nan)
        cen_counts[t] = {"chips_with_usable_censored_row": int((d != 0).sum()),
                         "not_on_a_train_chip": int(((d != 0) & ~is_train).sum()),
                         "train_chip_has_uncensored_value": int(((d != 0) & is_train & has_unc).sum()),
                         "train_rows_left_censored": int((ok & (d < 0)).sum()),
                         "train_rows_right_censored": int((ok & (d > 0)).sum())}
    counts["censored_training_rows"] = cen_counts
    if (full.loc[np.isin(r, L.TEST_ROLES), [*[f"y_{n}" for n in L.RECOVERED], *L.LIM_COLS]].notna().any().any()
            or (full.loc[np.isin(r, L.TEST_ROLES), list(L.CENS_COLS)] != 0).any().any()):
        raise SystemExit("a recovered label was built on a test-role chip")
    lab_cols = {t: full[f"y_{t}"].notna().to_numpy() for t in (*L.TARGETS, "turb_ntru")}
    counts["label_rows_by_role"] = {
        rr: {"chips": int((r == rr).sum()),
             "chips_with_a_label": int(((r == rr) & (np.column_stack(list(lab_cols.values())).any(axis=1) |
                                                     (full[list(L.CENS_COLS)] != 0).any(axis=1).to_numpy())).sum()),
             **{t: int((v & (r == rr)).sum()) for t, v in lab_cols.items()},
             **{f"censored_{t}": int(((full[f"cens_{t}"] != 0).to_numpy() & (r == rr)).sum()) for t in L.CENSORED}}
        for rr in sorted(set(r))}
    counts["added_train_chips_by_label"] = {
        t: int((v & is_train & (full["version_source"] == "added").to_numpy()).sum()) for t, v in lab_cols.items()}
    counts["s9_censored_training_rows"] = {t: int(((full[f"cens_{t}"] != 0).to_numpy() & is_train).sum())
                                           for t in L.CENSORED}
    counts["s9_targets_ge20"] = [t for t, n in counts["s9_censored_training_rows"].items() if n >= 20]
    full = full.drop(columns=["_role"])

    # ---------------------------------------------------------------- split, hashes, pin
    split = {"version": L.VERSION,
             "params": {**N.PARAMS, "secchi_window_days": 7, "added_rows": "train-only",
                        "label_rule": WR.RULE, "bounds": {k: list(v) for k, v in L.BOUNDS.items()},
                        "recovered": {k: list(v) for k, v in L.RECOVERED.items()},
                        "censored": {k: list(v) for k, v in L.CENSORED.items()},
                        "right_censored_several_limits": "not used", "four_state_key": list(L.SITE_KEY)},
             "national2_split_sha256": split2["_sha256"],
             "labels_sha256": L.labels_sha256(full),
             "labels_original4_sha256": N.labels_sha256(full),
             "side_table_sha256": pm7["record"]["side_table_sha256"],
             "block_role_unseen_2b": block_role_unseen_2b,
             "chip_role": chip_role}
    n2 = L.assert_roles_unchanged(split, split2)
    wb_same = (full.set_index("naip_chip_id")["water_body"].reindex(lab2["naip_chip_id"]).to_numpy()
               == lab2["water_body"].to_numpy()).all()
    if not wb_same:
        raise SystemExit("a national-2 chip changed water body")
    sha = L.split_sha256(split)
    counts["pin"] = {"split_file": str(L.SPLIT_FILE.relative_to(REPO)), "canonical_split_sha256": sha,
                     "labels_sha256": split["labels_sha256"], "labels_original4_sha256": split["labels_original4_sha256"],
                     "n_chips": int(len(full)), "n_national2_chips_same_role": int(n2),
                     "water_body_ids_of_national2_chips_unchanged": True,
                     "roles": pd.Series(chip_role).value_counts().to_dict()}
    if not args.dry_run:
        L.labels_path().parent.mkdir(parents=True, exist_ok=True)
        full.to_parquet(L.labels_path(), index=False)
        L.SPLIT_FILE.write_text(json.dumps(split, indent=0, sort_keys=True, default=str) + "\n")
        counts["pin"]["split_file_sha256"] = sha_file(L.SPLIT_FILE)
        counts["pin"]["labels_file"] = str(L.labels_path())
        counts["pin"]["labels_file_sha256"] = sha_file(L.labels_path())
    name = "labels2b_counts_dryrun.json" if args.dry_run else "labels2b_counts.json"
    (out_dir / name).write_text(json.dumps(counts, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"pin": counts["pin"], "counts_file": str(out_dir / name)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
