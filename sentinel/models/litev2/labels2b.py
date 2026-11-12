# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""SENTINEL-Lite label version **national-2b** (lite_arch_v2_amendment.md §3; lead order 2026-11-10 15:05 EST).

What changes against national-2 (``labels_national2.parquet``, canonical split hash 976b5be7…, which stays the pin of
the as-built version and is used only to reproduce numbers already seen):

1. **WQP label rule** (F3 ruling). Every label comes from the harmonised, QC-passed ``wqp_discrete/variable=*/``
   tables; a surface result has ``depth_m`` <= 1.0 m or unknown and is not flagged "bottom"; Secchi is exempt; a
   sample without such a result is dropped and the reason recorded. The rule is applied by M2's reader through
   :mod:`.wqp_rule` (the unchanged ``matchups.wqp_surface_samples``).

   * chips of the national-1 build (``naip_matchups/national``) and of the ±7 d build (``national_pm7``): the whole
     matchup table goes through ``wqp_rule.apply_rule`` (sample key site | date | target | characteristic | unit);
   * the 5,998 chips of the 4-state track, whose matchup table was built from raw WQP files with raw units: every
     row is re-derived by the key (site id, which carries the organisation; activity start date; characteristic)
     from the harmonised tables read here with ``depth_m`` (:func:`rederive_frozen`). No as-built value is kept.

   The frozen functions ``lite.data.harmonise`` and ``lite.data.chip_labels`` then make the chip labels, unchanged.
   Chip roles and water-body ids of national-2 are NOT changed (:func:`assert_roles_unchanged` compares the
   ``chip_role`` mapping alone). A chip that loses every label keeps its row, with ``label_drop_reason``.

2. **Recovered labels** (§3.2–3.3), from the rule-conforming rows of ``national_pm7``: FNU turbidity (own target
   ``turb_fnu``), NTRU turbidity (``y_turb_ntru``, training rows only), SSC in mg/L (own target ``ssc``) and
   censored rows (``cens_<t>`` = -1 left / +1 right / 0, ``lim_<t>`` = log10 limit; training rows only; used only
   when the chip has no uncensored value of that target). None of them is built on a chip of the four test roles.

3. **Added chips** (§3.4 item 2): chips of the ±7 d build without a national-2 role that hold a recovered label get
   ``train`` or ``excluded`` by the four conditions of that item.

Protected rows (val, val_n, the four test roles): this module computes values mechanically and reports counts and
presence flags only. Nothing here prints, logs or summarises a label value.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data import matchups as M
from ...data.reserved import assert_no_reserved, exclude_reserved
from ...eval.splits import group_split, nearest_distance_km, spatial_blocks
from ..lite import data as D
from ..lite import national as N
from ..lite import national2 as N2
from ..orbit import wqp_surface as WS
from . import wqp_rule as WR

VERSION = "national-2b"
SPLIT_FILE = paths.REPO_ROOT / "docs" / "preregistration" / "lite_split_national2b.json"
# canonical split hash (split_sha256) of the pinned file; "PENDING" until the build has written it
FROZEN_SHA256 = "29f0dd009bb69a5e36a7829d3bf5b8c7c4301f683a87b6a1b6ddd2fd1ca08092"   # canonical split hash, built 2026-11-10
ORIG = tuple(D.TARGETS)                                             # secchi, turbidity, tss, chla
TARGETS = ("secchi", "turbidity", "turb_fnu", "tss", "ssc", "chla")
BOUNDS = {**D.BOUNDS, "turb_fnu": (0.05, 1000.0), "ssc": (0.1, 10000.0)}        # amendment §3.2
TARGET_UNIT = {**D.TARGET_UNIT, "turb_fnu": "FNU", "ssc": "mg/L"}
# recovered uncensored groups: label column -> (WQP target, harmonised unit, bounds key)
RECOVERED = {"turb_fnu": ("turbidity", "FNU", "turb_fnu"), "turb_ntru": ("turbidity", "NTRU", "turbidity"),
             "ssc": ("ssc", "mg/L", "ssc")}
# censored groups of §3.1: label target -> (WQP target, harmonised unit). Censored FNU / NTRU are not in the table.
CENSORED = {"secchi": ("secchi", "m"), "turbidity": ("turbidity", "NTU"), "tss": ("tss", "mg/L"),
            "chla": ("chla", "ug/L"), "ssc": ("ssc", "mg/L")}
TEST_ROLES = tuple(N.NATIONAL_PROTECTED)                            # test, test_time, test_n, test_time_n
VAL_ROLES = ("val", "val_n")
Y_COLS = tuple(f"y_{t}" for t in TARGETS) + ("y_turb_ntru",)
CENS_COLS = tuple(f"cens_{t}" for t in CENSORED)
LIM_COLS = tuple(f"lim_{t}" for t in CENSORED)
LABEL_HASH_COLS = ("naip_chip_id", "water_body", "site_id", "naip_year", *Y_COLS, *CENS_COLS, *LIM_COLS)
SITE_KEY = ("site_id", "date", "target", "characteristic")         # the deterministic key of ruling 2(b)
FROZEN_REASONS = (*WR.DROP_REASONS, "rejected_by_frozen_label_rules")


class LabelVersionError(RuntimeError):
    """The national-2b label version cannot be built or loaded as ruled."""


def labels_path() -> Path:
    return paths.processed("lite_labels") / "labels_national2b.parquet"


def _quiet(*_a, **_k) -> None:
    return None


# ------------------------------------------------------------------------------------------------ harmonised rows

def harmonised_results(site_ids, extra=("depth_m", "censor_type")) -> pd.DataFrame:
    """Rows of the harmonised, QC-passed result tables (``wqp_discrete/variable=*/``; never a sidecar file) at
    ``site_ids``, read as ``matchups.load_wqp`` reads them (same columns, same characteristic filter) plus ``extra``."""
    import pyarrow.parquet as pq

    root = paths.PROCESSED / "wqp_discrete"
    names = [n for t in M.WQP_TARGETS for n in M.WQP_TARGETS[t]]
    sites = sorted({str(s) for s in site_ids})
    out = []
    for f in sorted(root.glob("variable=*/*.parquet")):
        if f.name in WS.SIDECARS:
            raise LabelVersionError(f"sidecar file inside a variable directory: {f}")
        cols = set(pq.ParquetFile(f).schema_arrow.names)
        if not ({"site_id", "value", "characteristic"} <= cols):
            continue
        if "depth_m" not in cols:
            raise LabelVersionError(f"{f}: harmonised result table without a depth_m column")
        keep = [c for c in (*M.RESULT_KEEP, *extra) if c in cols]
        df = pq.read_table(f, columns=keep, filters=[("characteristic", "in", names),
                                                     ("site_id", "in", sites)]).to_pandas()
        if len(df):
            out.append(df)
    if not out:
        raise LabelVersionError("no harmonised result row at the requested sites")
    return pd.concat(out, ignore_index=True)


# ------------------------------------------------------------------------------------------------ 4-state chips

def rederive_frozen(m4: pd.DataFrame, chip_ids, harm: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rule-conforming sample rows for the 4-state chips, one status row per as-built row key).

    ``m4``: the 4-state matchup table (``naip_status == 'ok'`` rows). Every distinct (chip, site, date, target,
    characteristic) of it is looked up in the harmonised tables by :data:`SITE_KEY`; the as-built value, unit and
    counts are not carried over. Status: ``kept`` / ``no_qc_passed_result`` / ``no_surface_result``."""
    ids = {str(c) for c in chip_ids}
    m = m4[m4["naip_chip_id"].astype(str).isin(ids)].copy()
    m["naip_chip_id"] = m["naip_chip_id"].astype(str)
    m["site_id"] = m["site_id"].astype(str)
    m["characteristic"] = m["characteristic"].astype(str)
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    drop = [c for c in (*WR.KEY_COLS[4:], "sample_key", *WS.VALUE_COLS, "time_utc") if c in m.columns]
    rows = (m.drop(columns=drop).sort_values(["naip_chip_id", *SITE_KEY])
            .drop_duplicates(["naip_chip_id", *SITE_KEY]).reset_index(drop=True))
    if harm is None:
        harm = harmonised_results(rows["site_id"].unique())
    tab = WR.sample_table(harm)
    tab["date"] = pd.to_datetime(tab["date"]).dt.normalize()
    j = rows.merge(tab, on=list(SITE_KEY), how="left")
    has_qc = j["sample_key"].notna()
    surf = j["surface"].fillna(False).astype(bool)
    grp = [j["naip_chip_id"], *[j[c] for c in SITE_KEY]]
    st = pd.DataFrame({"naip_chip_id": j["naip_chip_id"], **{c: j[c] for c in SITE_KEY},
                       "_qc": has_qc.to_numpy(), "_surf": surf.to_numpy()})
    st = st.groupby(["naip_chip_id", *SITE_KEY], as_index=False).agg(_qc=("_qc", "any"), _surf=("_surf", "any"))
    st["status"] = np.where(st["_surf"], "kept", np.where(st["_qc"], "no_surface_result", "no_qc_passed_result"))
    del grp
    kept = j[surf.to_numpy()].drop(columns=["surface", "drop_reason"]).reset_index(drop=True)
    return kept, st.drop(columns=["_qc", "_surf"])


def chip_table(samples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Frozen label rules on rule-conforming sample rows: ``lite.data.harmonise`` then ``lite.data.chip_labels``."""
    s, rep = D.harmonise(samples)
    if not len(s):
        raise LabelVersionError("no sample row survives the frozen label rules")
    lab = D.chip_labels(s)
    lab["naip_chip_id"] = lab["naip_chip_id"].astype(str)
    return lab, rep


# ------------------------------------------------------------------------------------------------ recovered labels

def recovered_uncensored(k7: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Chip medians (log10) of the recovered uncensored groups of :data:`RECOVERED` from rule-conforming rows of the
    ±7 d build. Index = chip id; columns ``y_<name>``, ``n_<name>``. Bounds of §3.2; out-of-bounds rows counted."""
    d = k7[k7["naip_chip_id"].notna()].copy()
    d["naip_chip_id"] = d["naip_chip_id"].astype(str)
    unit = d["unit"].fillna("").astype(str).str.strip()
    cens = d["censored"].fillna(False).astype(bool)
    val = pd.to_numeric(d["value"], errors="coerce")
    out, rep = {}, {}
    for name, (tgt, u, bkey) in RECOVERED.items():
        m = (d["target"] == tgt) & (unit == u)
        lo, hi = BOUNDS[bkey]
        ok = m & ~cens & np.isfinite(val) & (val >= lo) & (val <= hi)
        rep[name] = {"rows": int(m.sum()), "censored": int((m & cens).sum()),
                     "missing_value": int((m & ~cens & ~np.isfinite(val)).sum()),
                     "out_of_bounds": int((m & ~cens & np.isfinite(val) & ~((val >= lo) & (val <= hi))).sum()),
                     "kept_rows": int(ok.sum())}
        g = val[ok].groupby(d.loc[ok, "naip_chip_id"])
        out[f"y_{name}"] = np.log10(g.median())
        out[f"n_{name}"] = g.size()
    tab = pd.DataFrame(out)
    for name in RECOVERED:
        tab[f"n_{name}"] = tab[f"n_{name}"].fillna(0).astype(int)
    return tab, rep


def censor_directions(harm: pd.DataFrame) -> pd.DataFrame:
    """Per sample key: number of QC-passed SURFACE censored results by direction and the range of their limits.

    The surface rule is applied by the unchanged ``wqp_surface_samples`` (through ``wqp_rule.surface_samples``) to
    subsets of the censored result rows with the detection limit in the ``value`` slot, so no rule is written here.
    Columns: ``n_left``, ``n_right``, ``n_nodir``, ``n_left_nolimit``, ``n_right_nolimit``, ``lim_left_max``,
    ``lim_right_min``, ``lim_right_max``."""
    c = harm[harm["censored"].fillna(False).astype(bool)].copy()
    ct = c["censor_type"].fillna("").astype(str).str.strip()
    lim = pd.to_numeric(c["detection_limit"], errors="coerce")
    parts = {"left": (ct == "<") & lim.notna(), "right": (ct == ">") & lim.notna(),
             "left_nolimit": (ct == "<") & lim.isna(), "right_nolimit": (ct == ">") & lim.isna(),
             "nodir": ~ct.isin(["<", ">"])}
    out = None
    for name, mask in parts.items():
        sub = c[mask.to_numpy()]
        if not len(sub):
            continue
        s = WR.surface_samples(sub.assign(value=pd.to_numeric(sub["detection_limit"], errors="coerce"),
                                          censored=False))
        t = pd.DataFrame({"sample_key": s["sample_key"], f"n_{name}": s["n_results"].astype(int)})
        if name == "left":
            t["lim_left_max"] = s["value_max"].to_numpy(float)
        if name == "right":
            t["lim_right_min"] = s["value_min"].to_numpy(float)
            t["lim_right_max"] = s["value_max"].to_numpy(float)
        out = t if out is None else out.merge(t, on="sample_key", how="outer")
    cols = ["n_left", "n_right", "n_nodir", "n_left_nolimit", "n_right_nolimit", "lim_left_max", "lim_right_min",
            "lim_right_max"]
    if out is None:
        return pd.DataFrame(columns=["sample_key", *cols])
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan if col.startswith("lim") else 0
        if col.startswith("n_"):
            out[col] = out[col].fillna(0).astype(int)
    return out[["sample_key", *cols]]


def censored_chips(k7: pd.DataFrame, directions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Chip-level censored rows of §3.3 for the groups of :data:`CENSORED` (one row per chip x target).

    Sample rule: every censored surface result of the sample has one direction and a limit (missing or mixed
    direction, or a missing limit -> dropped); the limit lies inside the target's bounds. Chip rule: one direction
    over the chip's samples; '<' takes the largest limit; a '>' chip with more than one distinct limit is not used
    (the stricter rule of ``p2_index``; M7_verify missed[4]). Columns: naip_chip_id, target, direction, log10_limit."""
    d = k7[k7["naip_chip_id"].notna() & k7["censored"].fillna(False).astype(bool)].copy()
    d["naip_chip_id"] = d["naip_chip_id"].astype(str)
    unit = d["unit"].fillna("").astype(str).str.strip()
    rep: dict = {}
    rows = []
    ref = directions.set_index("sample_key")
    for t, (tgt, u) in CENSORED.items():
        g = d[(d["target"] == tgt) & (unit == u)].copy()
        r = {"rows": int(len(g))}
        if not len(g):
            rep[t] = r
            continue
        x = ref.reindex(g["sample_key"])
        nl, nr = x["n_left"].fillna(0).to_numpy(int), x["n_right"].fillna(0).to_numpy(int)
        bad_dir = (x["n_nodir"].fillna(0).to_numpy(int) > 0) | ((nl > 0) & (nr > 0)) | ((nl == 0) & (nr == 0))
        no_lim = ~bad_dir & ((x["n_left_nolimit"].fillna(0).to_numpy(int) > 0) |
                             (x["n_right_nolimit"].fillna(0).to_numpy(int) > 0))
        g["direction"] = np.where(nl > 0, -1, 1)
        g["lim_lo"] = np.where(nl > 0, x["lim_left_max"].to_numpy(float), x["lim_right_min"].to_numpy(float))
        g["lim_hi"] = np.where(nl > 0, x["lim_left_max"].to_numpy(float), x["lim_right_max"].to_numpy(float))
        lo, hi = BOUNDS[t]
        with np.errstate(invalid="ignore"):
            oob = ~bad_dir & ~no_lim & ~((g["lim_lo"].to_numpy() >= lo) & (g["lim_hi"].to_numpy() <= hi))
        r.update(dropped_missing_or_mixed_direction=int(bad_dir.sum()), dropped_missing_limit=int(no_lim.sum()),
                 dropped_limit_out_of_bounds=int(oob.sum()))
        g = g[~(bad_dir | no_lim | oob)]
        r["rows_usable"] = int(len(g))
        if len(g):
            a = g.groupby("naip_chip_id").agg(dmin=("direction", "min"), dmax=("direction", "max"),
                                              lo=("lim_lo", "min"), hi=("lim_hi", "max"), n=("direction", "size"))
            mixed = a["dmin"] != a["dmax"]
            multi_right = ~mixed & (a["dmin"] == 1) & (a["lo"] != a["hi"])
            r.update(chips=int(len(a)), chips_dropped_mixed_direction=int(mixed.sum()),
                     chips_dropped_right_several_limits=int(multi_right.sum()))
            a = a[~(mixed | multi_right)]
            rows.append(pd.DataFrame({"naip_chip_id": a.index, "target": t, "direction": a["dmin"].to_numpy(int),
                                      "log10_limit": np.log10(a["hi"].to_numpy(float)),
                                      "n_censored_samples": a["n"].to_numpy(int)}))
        rep[t] = r
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["naip_chip_id", "target", "direction", "log10_limit", "n_censored_samples"])
    return out, rep


# ------------------------------------------------------------------------------------------------ roles (§3.4)

def added_chip_roles(added: pd.DataFrame, lab2: pd.DataFrame, split2: dict, wb2b: pd.Series) -> tuple[pd.Series, dict]:
    """Role (``train`` / ``excluded``) of every chip without a national-2 role, by §3.4 item 2.

    ``added``: one row per added chip (naip_chip_id, site_id, lat, lon, naip_year). ``lab2``: the national-2 label
    table. ``wb2b``: site_id -> water body of the national-2b grouping (union of national-2 and new sites).
    Mirrors ``national2.build_split`` rules 1–4; block roles known so far = 4-state, national-1 and national-2."""
    from ..lite.guard import load_split as load_original

    p = dict(N.PARAMS)
    n1 = N.load_split()
    orig = load_original()
    old = lab2.assign(role=lab2["naip_chip_id"].astype(str).map(split2["chip_role"]),
                      wb=lab2["site_id"].astype(str).map(wb2b))
    new = added.assign(wb=added["site_id"].astype(str).map(wb2b)).reset_index(drop=True)
    if old["role"].isna().any() or old["wb"].isna().any() or new["wb"].isna().any():
        raise LabelVersionError("a chip without a national-2 role or without a national-2b water body")
    roles_in_wb = old.groupby("wb")["role"].agg(lambda s: set(s))
    touched = new["wb"].isin(roles_in_wb.index).to_numpy()
    all_train = new["wb"].map(lambda w: roles_in_wb.get(w, set()) <= {"train"}).to_numpy(bool)
    ok_wb = np.where(touched, all_train, True)
    fresh = new[~touched]
    known = {**orig["block_role"], **n1.get("block_role_new", {}), **split2.get("block_role_unseen", {})}
    brole_unseen: dict = {}
    ok_block = np.ones(len(new), bool)
    if len(fresh):
        wb = fresh.groupby("wb").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("naip_chip_id", "size"))
        wb["block"] = spatial_blocks(wb["lat"], wb["lon"], p["block_km"])
        unseen = sorted(set(wb["block"]) - set(known))
        if unseen:
            bw = wb[wb["block"].isin(unseen)].groupby("block")["n"].sum().to_dict()
            bs = group_split(unseen, test_frac=p["test_frac"], val_frac=p["val_frac"], seed=p["seed"], weights=bw)
            brole_unseen = {b: r for r, bl in (("train", bs.train), ("val", bs.val), ("test", bs.test)) for b in bl}
        brole = {**brole_unseen, **known}
        wb_train = wb["block"].map(brole) == "train"
        ok_block[~touched] = fresh["wb"].map(wb_train).fillna(False).to_numpy(bool)
    ok_year = new["naip_year"].astype(int).to_numpy() <= p["last_train_year"]
    ev = old[old["role"].isin(N2.EVAL_ROLES)].drop_duplicates("site_id")
    dist = nearest_distance_km(new["lat"], new["lon"], ev["lat"], ev["lon"])
    ok_buf = np.asarray(dist) > p["buffer_km"]
    ok = ok_wb & ok_block & ok_year & ok_buf
    role = pd.Series(np.where(ok, "train", "excluded"), index=new["naip_chip_id"].astype(str).to_numpy())
    rep = {"n_added_chips": int(len(new)), "n_train": int(ok.sum()), "n_excluded": int((~ok).sum()),
           "in_water_body_with_national2_chip": int(touched.sum()),
           "fails_water_body_rule": int((~ok_wb).sum()), "fails_block_rule": int((~ok_block).sum()),
           "fails_year_rule": int((~ok_year).sum()), "fails_2km_rule": int((~ok_buf).sum()),
           "n_unseen_blocks": len(brole_unseen),
           "unseen_block_roles": pd.Series(brole_unseen, dtype=object).value_counts().to_dict()}
    return role, {**rep, "block_role_unseen_2b": brole_unseen}


def water_bodies_2b(lab2: pd.DataFrame, added: pd.DataFrame) -> pd.Series:
    """site_id -> water body over the union of national-2 sites and the sites of the added chips (frozen
    ``water_body_groups``, national-2 parameters, HydroLAKES ids as in the frozen builds)."""
    cols = ["site_id", "lat", "lon", "name", "site_class", "huc8"]
    sites = pd.concat([lab2[cols], added[cols]], ignore_index=True)
    sites["site_id"] = sites["site_id"].astype(str)
    sites = sites.drop_duplicates("site_id").reset_index(drop=True)
    hylak = D.hydrolakes_ids(sites)
    wb = D.water_body_groups(sites, N.PARAMS["water_body_radius_km"], N.PARAMS["water_body_name_km"], hylak)
    return wb


# ------------------------------------------------------------------------------------------------ hashes, pin, load

def labels_sha256(lab: pd.DataFrame) -> str:
    """Content hash of the national-2b label table: chip ids, groups, every label column (log10, rounded to 1e-9),
    censoring direction and limit. Same recipe as ``lite.splits.labels_sha256`` over more columns."""
    d = lab[list(LABEL_HASH_COLS)].sort_values("naip_chip_id").copy()
    for c in (*Y_COLS, *LIM_COLS):
        d[c] = pd.to_numeric(d[c], errors="coerce").round(9)
    for c in CENS_COLS:
        d[c] = d[c].astype(int)
    return hashlib.sha256(d.to_csv(index=False, na_rep="NA").encode()).hexdigest()


def split_sha256(split: dict) -> str:
    """Canonical split hash (the ``national2.split_sha256`` recipe with the national-2 pin in place of national-1)."""
    blob = json.dumps({k: split.get(k) for k in ("version", "params", "national2_split_sha256", "labels_sha256",
                                                 "chip_role")}, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def assert_roles_unchanged(split: dict, split2: dict | None = None) -> int:
    """Every national-2 chip has the same role in ``split`` (compared on the ``chip_role`` mapping alone); every
    other chip is ``train`` or ``excluded``. Returns the number of national-2 chips."""
    split2 = N2.load_split() if split2 is None else split2
    r2, r = split2["chip_role"], split["chip_role"]
    miss = [c for c in r2 if c not in r]
    if miss:
        raise LabelVersionError(f"{len(miss)} national-2 chips have no national-2b role")
    diff = sum(1 for c, v in r2.items() if r[c] != v)
    if diff:
        raise LabelVersionError(f"{diff} national-2 chips changed role")
    extra = {v for c, v in r.items() if c not in r2}
    if extra - {"train", "excluded"}:
        raise LabelVersionError(f"added chips with roles {sorted(extra - {'train', 'excluded'})}")
    return len(r2)


def load_split(path: Path | None = None, *, check_frozen: bool = True) -> dict:
    from ..lite.guard import ProtectedDataError

    path = SPLIT_FILE if path is None else Path(path)
    split = json.loads(path.read_text())
    sha = split_sha256(split)
    if check_frozen:
        if FROZEN_SHA256 == "PENDING":
            raise ProtectedDataError("national-2b split is not pinned yet (FROZEN_SHA256 is PENDING)")
        if sha != FROZEN_SHA256:
            raise ProtectedDataError(f"national-2b split sha256 {sha[:12]} != pinned {FROZEN_SHA256[:12]}")
    split.update(_sha256=sha, _path=str(path), _track="national2b")
    return split


def load_labels(split: dict) -> pd.DataFrame:
    """The whole label table, checked against the split's ``labels_sha256``. Internal: use :func:`role_frame`."""
    from ..lite.guard import ProtectedDataError

    lab = pd.read_parquet(labels_path())
    if labels_sha256(lab) != split["labels_sha256"]:
        raise ProtectedDataError("labels_national2b.parquet differs from the national-2b split's labels_sha256")
    lab["naip_chip_id"] = lab["naip_chip_id"].astype(str)
    lab["role"] = lab["naip_chip_id"].map(split["chip_role"])
    return lab


def role_frame(split: dict, roles, *, view: str = "v2", training: bool = True) -> pd.DataFrame:
    """National-2b label rows for ``roles``.

    * the four test roles are never returned;
    * ``val`` / ``val_n`` rows are returned only after the amendment has been stamped (``stamp.require_stamp``);
    * ``view='rerun'``: the national-2 chips with at least one corrected label of the four original targets, with
      the columns of the national-2 table (re-runs of configs 1–6 and the no-KD twins; no recovered column, no
      added chip). ``view='v2'``: every chip with at least one label of this version, all columns."""
    from ..lite.guard import ProtectedDataError
    from . import stamp

    roles = [str(r) for r in roles]
    bad = sorted(set(roles) & set(TEST_ROLES))
    if bad:
        raise ProtectedDataError(f"national-2b: protected roles {bad} are not loaded in this phase")
    if set(roles) & set(VAL_ROLES):
        stamp.require_stamp(f"national-2b rows of roles {sorted(set(roles) & set(VAL_ROLES))}")
    lab = load_labels(split)
    lab = lab[lab["role"].isin(roles)]
    # reserved new-site block (docs/preregistration/sonde_newsites_list.json): a chip at such a site keeps its
    # national-2 role in the pin file but is never returned (deviations file, entry on the reserved block)
    free = set(exclude_reserved(lab["site_id"].astype(str).unique().tolist()))
    lab = lab[lab["site_id"].astype(str).isin(free)]
    assert_no_reserved(lab["site_id"].astype(str), where="national-2b role_frame")
    has_orig = lab[[f"y_{t}" for t in ORIG]].notna().any(axis=1)
    if view == "rerun":
        lab = lab[(lab["version_source"] != "added") & has_orig]
        drop = [c for c in lab.columns if c in (*[f"y_{t}" for t in ("turb_fnu", "ssc", "turb_ntru")],
                                                *[f"n_{t}" for t in ("turb_fnu", "ssc", "turb_ntru")],
                                                *CENS_COLS, *LIM_COLS, "label_drop_reason", "version_source",
                                                "national2b_wb")]
        lab = lab.drop(columns=drop)
    elif view == "v2":
        any_lab = lab[list(Y_COLS)].notna().any(axis=1) | (lab[list(CENS_COLS)] != 0).any(axis=1)
        lab = lab[any_lab]
    else:
        raise ValueError(f"unknown view {view!r}")
    return lab.reset_index(drop=True)


def activate_rerun() -> None:
    """Point the national-track entry points at the corrected labels of the national-2 chips, for this process only
    (the counterpart of ``national2.activate``). The hashed ``lite_*`` scripts then re-run configs 1–6 and the no-KD
    twins unchanged; registry experiment ``lite_val_national2b``. Validation rows need the stamp."""
    orig_tvf = N.train_val_frames
    N.load_split = lambda *a, **k: load_split()
    N.role_frame = lambda split, roles, *, training=True: role_frame(split, roles, view="rerun", training=training)
    N.labels_path = labels_path
    N.national_scene_stats_path = N2.scene_stats_path

    def tvf(track: str):
        out = orig_tvf(track)
        if track == "national":
            return (*out[:4], "lite_val_national2b")
        return out

    N.train_val_frames = tvf


__all__ = ["VERSION", "SPLIT_FILE", "FROZEN_SHA256", "TARGETS", "ORIG", "BOUNDS", "RECOVERED", "CENSORED",
           "LABEL_HASH_COLS", "LabelVersionError", "labels_path", "harmonised_results", "rederive_frozen",
           "chip_table", "recovered_uncensored", "censor_directions", "censored_chips", "added_chip_roles",
           "water_bodies_2b", "labels_sha256", "split_sha256", "assert_roles_unchanged", "load_split",
           "load_labels", "role_frame", "activate_rerun"]
