# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Track P2 index (lite_arch_v2_amendment.md §8.5–§8.6): NAIP match, label presence, chips, water eligibility.

Runs only after the lock list exists (``p2.load_lock`` checks the pinned SHA-256). Label handling by role:

- ``train_p2`` / ``val_p2``: FNU values are read (after the lock row) and stored as log10 labels.
- ``test_p2``: **no label value is read**. The Arrow scanner evaluates "value is valid" and returns only that
  boolean; the value and detection-limit columns are never projected for test sites. The index stores the flag
  ``has_label`` and NaN labels. Values are joined to the sealed predictions inside the P2 look.

Labels (§8.5): sonde daily FNU on the NAIP date (mean; median where the site has no mean series); otherwise the
within-day median of WQP FNU surface results on the nearest day within ±1 d (ties: the earlier day). WQP results
pass the project label rule (:mod:`.wqp_rule`, i.e. M2's reader: harmonised QC-passed rows, depth <= 1 m or unknown,
not "bottom"); lead order 2026-11-10 15:05 EST item 3 — only this label filter differs from the first build. Sonde takes
precedence. Chips go to their own store (``processed/naip_chips/lite_p2``), so no national store or pin changes.
The national build's JRC site filter is not applied; the dark-NIR eligibility rule replaces it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...core import paths
from ...data.reserved import _bare, assert_no_reserved
from . import p2
from . import wqp_rule as WR

WQP_WINDOW_D = 1
TURB_FNU_BOUNDS = (0.05, 1000.0)       # amendment §3.2
ELIG_RADIUS_M, ELIG_MIN_WATER_M2, ELIG_MAX_NODATA = 30.0, 50.0, 0.01
LAST_TRAIN_YEAR = 2021
CHIP_KB = 140.0                         # mean NAIP chip size in the national stores (N5)
DISK_CAP_GB = 0.5
WQP_SAFE_COLS = ["site_id", "activity_start_date", "characteristic", "unit", "censored", "status_class",
                 "activity_relative_depth", "state", "depth_m"]   # never a value or a limit
WQP_VALUE_COLS = ["value", "detection_limit", "censor_type"]
# Lock sites that are never trained on and never scored (lead order 2026-11-10 15:05 EST items 5 and F10; the lock
# file is not changed; deviations file lite_arch_v2_deviations.md). None has an index row.
EXCLUDED_AT_LOAD = {
    "MDNR-3256/1.8": "val_p2; within 100 m of a site of the reserved new-site block held under another id",
    "USGS-383848122215501": "test_p2; within 100 m of a site of the reserved new-site block held under another id",
    "USGS-374955122313301": "test_p2; within 100 m of a site of the reserved new-site block held under another id",
    "CBP_WQX-01608500": "train_p2; same station (25 m) as an M1 test-role sonde site held under a USGS id",
}
# sha256 of index.parquet as rebuilt under the label rule; "PENDING" until the rebuild
INDEX_SHA256 = "5c404be9251a22ae1b1d082446879e1466fa1c3fd41658f1f9649230be7f31b2"
F7_EVAL_BLOCK_ROLES = ("val", "test")


class P2IndexError(RuntimeError):
    """The P2 index breaks a rule of the lock, of the exclusion list or of its pin."""


def assert_no_excluded(site_ids, where: str) -> None:
    bad = sorted(set(map(str, site_ids)) & set(EXCLUDED_AT_LOAD))
    if bad:
        raise P2IndexError(f"{where}: {len(bad)} site(s) of the load-time exclusion list have a row")


def out_dir() -> Path:
    return paths.ensure(paths.processed("naip_matchups") / "lite_p2")


def chip_store() -> Path:
    return paths.processed("naip_chips") / "lite_p2"


def index_path() -> Path:
    return out_dir() / "index.parquet"


# --------------------------------------------------------------------------- NAIP catalogue
def naip_items_for_sites(sites: pd.DataFrame, cache: Path, chunk: int = 40, log=print) -> dict[str, dict]:
    """NAIP items whose footprint intersects any site (MultiPoint queries; cached item JSON). Same logic as
    ``scripts/data/build_naip_matchups_window.py``."""
    from ...data.sources import stac as S

    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    f, done_f = cache / "items.jsonl", cache / "queried_points.json"
    items: dict[str, dict] = {}
    if f.exists():
        for line in f.read_text().splitlines():
            it = json.loads(line)
            items[it["id"]] = it
    done = set(json.loads(done_f.read_text())) if done_f.exists() else set()
    todo = sites[~sites.point_key.isin(done)]
    with open(f, "a") as fh:
        for k in range(0, len(todo), chunk):
            g = todo.iloc[k:k + chunk]
            geom = {"type": "MultiPoint", "coordinates": [[float(a), float(b)] for a, b in zip(g.lon, g.lat)]}
            for it in S.search_naip(intersects=geom, start="2010-01-01", end="2023-12-31"):
                if it["id"] not in items:
                    items[it["id"]] = it
                    fh.write(json.dumps(it, separators=(",", ":")) + "\n")
            done |= set(g.point_key)
            if (k // chunk) % 10 == 0:
                done_f.write_text(json.dumps(sorted(done)))
                log(f"NAIP item search {k + len(g)}/{len(todo)} sites, {len(items)} items")
    done_f.write_text(json.dumps(sorted(done)))
    return items


# --------------------------------------------------------------------------- label presence / values
def sonde_days(site_ids: list[str], with_values: bool) -> pd.DataFrame:
    """Days with a valid daily FNU value per sonde site: site_id, date[, value]. ``with_values=False`` projects the
    date column only (the validity test runs inside the Arrow scanner)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    assert_no_reserved(site_ids, "lite P2 sonde labels")
    ix = pd.read_parquet(paths.processed("sonde_daily") / "index.parquet", columns=["site_no", "var"])
    has_mean = set(ix.loc[ix["var"] == "turb_fnu_mean", "site_no"].astype(str))
    out = []
    for sid in site_ids:
        bare = _bare(sid)
        var = "turb_fnu_mean" if bare in has_mean else "turb_fnu_median"
        f = paths.processed("sonde_daily") / "sites" / f"{bare}.parquet"
        if not f.is_file():
            continue
        d = ds.dataset(f)
        if var not in d.schema.names:
            continue
        flt = ds.field(var).is_valid() & ~pc.is_nan(ds.field(var))
        t = d.to_table(columns=["date", var] if with_values else ["date"], filter=flt).to_pandas()
        if "date" not in t.columns:                     # the per-site files keep the date as the pandas index
            t = t.reset_index()
        t["date"] = pd.to_datetime(t["date"]).dt.normalize()
        t = t[(t["date"] >= p2.DATE_MIN) & (t["date"] <= p2.DATE_MAX)]
        if with_values:
            t = t.rename(columns={var: "value"})
        t.insert(0, "site_id", sid)
        out.append(t)
    cols = ["site_id", "date"] + (["value"] if with_values else [])
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def wqp_days(site_ids: list[str], with_values: bool) -> pd.DataFrame:
    """WQP FNU surface site-days under the label rule (``wqp_rule.surface_samples``): site_id, date, value, censored,
    detection_limit, censor. With ``with_values=False`` the value column is replaced by a validity flag computed in
    the Arrow scanner (value 1.0 = an uncensored result exists), and no limit or censor direction is read."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    assert_no_reserved([s for s in site_ids if s.startswith("USGS-")], "lite P2 WQP labels")
    d = ds.dataset(paths.processed("wqp_discrete") / "variable=turbidity_fnu" / "data.parquet")
    flt = ds.field("site_id").isin(list(site_ids))
    if with_values:
        w = d.to_table(columns=WQP_SAFE_COLS + WQP_VALUE_COLS, filter=flt).to_pandas()
    else:
        proj = {c: ds.field(c) for c in WQP_SAFE_COLS}
        proj["value_present"] = ds.field("value").is_valid() & ~pc.is_nan(ds.field("value"))
        w = d.to_table(columns=proj, filter=flt).to_pandas()
        w["value"] = np.where(w.pop("value_present").fillna(False), 1.0, np.nan)
    w = w[w["unit"].astype(str).str.upper() == "FNU"]
    cols = ["site_id", "date", "value", "censored", "detection_limit", "censor", "n_censored"]
    if w.empty:
        return pd.DataFrame(columns=cols)
    s = WR.surface_samples(w, target="turbidity")
    s = s[(s["date"] >= p2.DATE_MIN) & (s["date"] <= p2.DATE_MAX)]
    s = s.groupby(["site_id", "date"], as_index=False).agg(
        value=("value", "median"), n_censored=("n_censored", "sum"), detection_limit=("detection_limit", "max"))
    s["censored"] = s["value"].isna() & (s["n_censored"] > 0)
    s["censor"] = ""
    if with_values:                                   # censor direction per site-day (train/val only)
        w["date"] = pd.to_datetime(w["activity_start_date"], errors="coerce").dt.normalize()
        c = w[w["censored"].fillna(False).astype(bool)].groupby(["site_id", "date"])["censor_type"].agg(
            lambda x: "|".join(sorted(set(x.dropna().astype(str))))).rename("censor").reset_index()
        s = s.drop(columns="censor").merge(c, on=["site_id", "date"], how="left")
        s["censor"] = s["censor"].fillna("")
    else:
        s["detection_limit"] = np.nan
    return s[cols]


def site_date_labels(pairs: pd.DataFrame, sites: pd.DataFrame, with_values: bool) -> pd.DataFrame:
    """Label (or presence) per (site, NAIP date) for the given sites. ``pairs``: site_id, naip_date."""
    out = pairs.copy()
    out["label_src"], out["value"] = None, np.nan
    sonde = sites.loc[sites["src"].str.startswith("sonde"), "site_id"].tolist()
    sd = sonde_days(sonde, with_values)
    if len(sd):
        sd = sd.rename(columns={"date": "naip_date"})
        if "value" not in sd:
            sd["value"] = 1.0                            # presence flag
        out = out.drop(columns=["label_src", "value"]).merge(sd.assign(label_src="sonde"),
                                                             on=["site_id", "naip_date"], how="left")
    for c in ("w_value", "w_limit", "w_dt"):
        out[c] = np.nan
    out["w_censored"], out["w_censor"], out["w_ncens"] = False, "", 0
    wq = wqp_days(sites["site_id"].tolist(), with_values)
    if len(wq):
        m = pairs.merge(wq, on="site_id")
        m["w_dt"] = (m["date"] - m["naip_date"]).dt.days
        m = m[m["w_dt"].abs() <= WQP_WINDOW_D].copy()
        m["absdt"] = m["w_dt"].abs()
        m["unc"] = m["value"].notna()
        # an uncensored day beats a censored one; then the nearest day; then the earlier day
        m = m.sort_values(["site_id", "naip_date", "unc", "absdt", "w_dt"], ascending=[True, True, False, True, True])
        m = m.drop_duplicates(["site_id", "naip_date"]).rename(columns={
            "value": "w_value", "censored": "w_censored", "detection_limit": "w_limit", "censor": "w_censor",
            "n_censored": "w_ncens"})
        keep = ["site_id", "naip_date", "w_value", "w_censored", "w_limit", "w_censor", "w_ncens", "w_dt"]
        out = out.drop(columns=keep[2:]).merge(m[keep], on=["site_id", "naip_date"], how="left")
    wc = out["w_censored"].astype("boolean").fillna(False).astype(bool)
    out["value"] = out["value"].astype("float64")
    out["label_src"] = out["label_src"].astype(object)
    use_w = out["label_src"].isna() & (out["w_value"].notna() | wc)
    out.loc[use_w, "label_src"] = "wqp"
    out.loc[use_w, "value"] = out.loc[use_w, "w_value"]
    out["censored"] = use_w & out["value"].isna()
    out["cens_limit"] = np.where(out["censored"], out["w_limit"], np.nan)
    out["cens_type"] = np.where(out["censored"], out["w_censor"].fillna(""), "")
    out["cens_n"] = np.where(out["censored"], out["w_ncens"].fillna(0), 0).astype(int)
    out["dt_days"] = np.where(out["label_src"] == "wqp", out["w_dt"], 0.0)
    return out[out["label_src"].notna()][["site_id", "naip_date", "label_src", "value", "censored", "cens_limit",
                                           "cens_type", "cens_n", "dt_days"]]


# --------------------------------------------------------------------------- eligibility
def water_eligibility(chips: np.ndarray, gsd_m) -> pd.DataFrame:
    """Per chip: dark-NIR water area (m²) within ``ELIG_RADIUS_M`` of the centre, the all-zero (outside-image)
    pixel fraction, and the eligibility flag. Label-free; the same for every model."""
    from ..lite import features as F

    n, _, H, W = chips.shape
    g = np.broadcast_to(np.asarray(gsd_m, float), (n,))
    wm = F.water_mask(chips)
    yy, xx = np.mgrid[:H, :W]
    r2 = (yy - H // 2) ** 2 + (xx - W // 2) ** 2
    area = np.array([(wm[i] & (r2 <= (ELIG_RADIUS_M / g[i]) ** 2)).sum() * g[i] ** 2 for i in range(n)])
    nodata = (chips.max(axis=1) == 0).reshape(n, -1).mean(axis=1)
    return pd.DataFrame({"water_m2_30m": area, "nodata_frac": nodata,
                         "eligible": (area >= ELIG_MIN_WATER_M2) & (nodata <= ELIG_MAX_NODATA)})


# --------------------------------------------------------------------------- build
def plan(log=print) -> tuple[pd.DataFrame, dict, dict]:
    """Site-dates with a label (or, for ``test_p2``, a present flag) and a NAIP image: (rows, items, counts)."""
    from ...data import matchups as M
    from ...data import naip as N
    from ...data.sources import stac as S

    rec = p2.load_lock()
    sites = p2.site_roles(rec)
    assert_no_reserved(sites["site_id"].tolist(), "lite P2 index")
    n_excl = sites[sites["site_id"].isin(EXCLUDED_AT_LOAD)].groupby("role").size().to_dict()
    sites = sites[~sites["site_id"].isin(EXCLUDED_AT_LOAD)].reset_index(drop=True)   # before any label is read
    sites["point_key"] = [M.point_key(lo, la) for lo, la in zip(sites["lon"], sites["lat"])]
    items = naip_items_for_sites(sites.drop_duplicates("point_key"), paths.interim("naip_catalog") / "lite_p2", log=log)
    cat = pd.DataFrame([{"item_id": i["id"], "date": S.naip_acquisition_date(i),
                         "year": int(i["properties"].get("naip:year", 0)), "gsd": i["properties"].get("gsd"),
                         "geometry": json.dumps(i["geometry"])} for i in items.values()])
    si = N.sites_in_items(sites.drop_duplicates("point_key"), cat)
    pairs = sites.merge(si, on="point_key")
    pairs["naip_date"] = pd.to_datetime(pairs["naip_date"]).dt.normalize()
    pairs = pairs[(pairs["naip_date"] >= p2.DATE_MIN) & (pairs["naip_date"] <= p2.DATE_MAX)]
    counts = {"sites": len(sites), "lock_sites_excluded_at_load": n_excl, "label_rule": WR.RULE,
              "naip_items": len(items), "site_naip_dates": len(pairs),
              "sites_with_naip": int(pairs["site_id"].nunique())}
    parts = []
    for role in ("train_p2", "val_p2", "test_p2"):
        s_r = sites[sites["role"] == role]
        p_r = pairs.loc[pairs["role"] == role, ["site_id", "naip_date"]].drop_duplicates()
        lab = site_date_labels(p_r, s_r, with_values=(role != "test_p2"))
        counts[f"censored_site_dates_{role}"] = int(lab["censored"].sum())
        if role != "train_p2":                         # censored rows are Tobit rows in train_p2 only (§8.5)
            lab = lab[~lab["censored"]]
        if role == "test_p2":                          # presence only: no value, limit or direction is kept
            lab = lab.assign(value=np.nan, cens_limit=np.nan, cens_type="", cens_n=0)
        parts.append(lab)
    lab = pd.concat(parts, ignore_index=True)
    rows = pairs.merge(lab, on=["site_id", "naip_date"])
    # one row per chip: two site ids at the same point share a chip; the sonde label wins, then the smaller site id
    rows = rows.assign(_s=(rows["label_src"] != "sonde")).sort_values(["_s", "site_id"]).drop_duplicates(
        ["item_id", "point_key"]).drop(columns="_s")
    rows = rows[(rows["role"] != "train_p2") | (rows["naip_year"] <= LAST_TRAIN_YEAR)]
    rows["request_key"] = rows["item_id"] + "|" + rows["point_key"]
    rows["thin"] = [p2.thin_order(s, d.strftime("%Y-%m-%d")) for s, d in zip(rows["site_id"], rows["naip_date"])]
    cap = int(DISK_CAP_GB * 1e6 / CHIP_KB)
    n_req = rows["request_key"].nunique()
    counts.update(labelled_site_dates=len(rows), chip_requests=n_req, chip_cap=cap)
    if n_req > cap:                                    # thin per site in the salted order, same rule in every role
        rank = rows.sort_values("thin").groupby("site_id").cumcount()
        k = int(rank.max()) + 1
        while k > 1 and rows.loc[rank < k, "request_key"].nunique() > cap:
            k -= 1
        rows = rows[rank < k]
        counts.update(thinned_to_per_site=k, chip_requests_after_thinning=int(rows["request_key"].nunique()))
    return rows.reset_index(drop=True), items, counts


def role_counts(ix: pd.DataFrame) -> dict:
    """Counts only (no label statistic): chips, water bodies and sites by role, label source and class."""
    out = {}
    for role, g in ix.groupby("role"):
        e = g[g["eligible"] & g["has_label"]]
        out[role] = {"chips_matched": int(len(g)), "chips_eligible_with_label": int(len(e)),
                     "water_bodies": int(e["water_body"].nunique()), "sites": int(e["site_id"].nunique()),
                     "by_label_src": e["label_src"].value_counts().sort_index().to_dict(),
                     "by_site_class": e["site_class"].value_counts().sort_index().to_dict(),
                     "censored_rows": int(g["cens"].ne(0).sum()),
                     "not_eligible": int((~g["eligible"]).sum())}
    return out


def build(log=print, workers: int = 4) -> dict:
    """Extract chips, compute eligibility and write the P2 index. Returns the counts record."""
    from ...data import chips as C
    from ...data import naip as N

    rows, items, counts = plan(log)
    log(f"P2 plan: {counts}")
    req = rows.drop_duplicates("request_key")[["request_key", "item_id", "point_key", "lon", "lat"]]
    nw = C.ChipWriter(chip_store(), layers=N.NAIP_LAYERS, dtype="uint8", semantics=N.NAIP_SEMANTICS)
    nx = N.NaipExtractor(nw, out_dir() / "naip_scene_stats.parquet", workers=workers, logger=log)
    counts["naip_extract"] = nx.run(req, items)
    nidx = C.consolidate(nw.root)
    keep = ["request_key", "chip_id", "status", "shard", "gsd_m", "chip_crs", "chip_left", "chip_top"]
    ix = rows.merge(nidx[keep].rename(columns={"chip_id": "naip_chip_id", "status": "naip_status",
                                               "shard": "naip_shard"}), on="request_key", how="left")
    ix = ix[ix["naip_status"] == "ok"].copy()
    store = C.ChipStore(nw.root)
    arr = np.stack([store.load(c) for c in ix["naip_chip_id"]])
    store.close()
    el = water_eligibility(arr, ix["gsd_m"].to_numpy(float))
    ix = pd.concat([ix.reset_index(drop=True), el], axis=1)
    lo, hi = TURB_FNU_BOUNDS
    v = ix["value"].to_numpy(float)
    is_test = (ix["role"] == "test_p2").to_numpy()
    inb = (v >= lo) & (v <= hi)
    lim = ix["cens_limit"].to_numpy(float)
    left = (ix["cens_type"] == "<").to_numpy()
    right = (ix["cens_type"] == ">").to_numpy() & (ix["cens_n"] == 1).to_numpy()   # one limit only (max is unsafe)
    cens_ok = ix["censored"].to_numpy(bool) & (left | right) & (lim >= lo) & (lim <= hi) \
        & (ix["role"] == "train_p2").to_numpy()
    ix["y_turb_fnu"] = np.where(~is_test & inb, np.log10(np.where(inb, v, 1.0)), np.nan)
    ix["cens"] = np.where(cens_ok, np.where(left, -1, 1), 0).astype(np.int8)
    ix["cens_log10_limit"] = np.where(cens_ok, np.log10(np.where(cens_ok, lim, 1.0)), np.nan)
    # has_label: an uncensored value in bounds (train/val), or the presence flag (test; bounds applied in the look)
    ix["has_label"] = np.where(is_test, ~ix["censored"].to_numpy(bool), np.isfinite(ix["y_turb_fnu"].to_numpy()))
    ix["doy"] = ix["naip_date"].dt.dayofyear
    ix = ix.rename(columns={"item_id": "naip_item_id"})
    cols = ["naip_chip_id", "site_id", "water_body", "role", "src", "site_class", "lat", "lon", "huc8", "hylak_id",
            "naip_item_id", "naip_date", "naip_year", "doy", "gsd_m", "naip_shard", "chip_crs", "chip_left",
            "chip_top", "label_src", "dt_days", "has_label", "y_turb_fnu", "cens", "cens_log10_limit",
            "water_m2_30m", "nodata_frac", "eligible"]
    ix = ix[cols].sort_values(["role", "site_id", "naip_date"]).reset_index(drop=True)
    usable = ix["has_label"] | (ix["cens"] != 0)
    counts["rows_without_usable_label_dropped"] = ix.loc[~usable].groupby("role").size().to_dict()
    ix = ix[usable].reset_index(drop=True)
    assert not ix.loc[ix["role"] == "test_p2", ["y_turb_fnu", "cens_log10_limit"]].notna().any().any()
    assert_no_reserved(ix["site_id"].tolist(), "lite P2 index (written)")
    assert_no_excluded(ix["site_id"], "lite P2 index (written)")
    ix.to_parquet(index_path(), index=False, compression="zstd")
    counts["by_role"] = role_counts(ix)
    (out_dir() / "counts.json").write_text(json.dumps(counts, indent=1, default=str))
    return counts


# --------------------------------------------------------------------------- load (F7, exclusions, stamp)
def known_block_roles() -> dict:
    """25-km block -> role over every Lite split so far: 4-state, national-1 new blocks, national-2 unseen blocks,
    national-2b unseen blocks."""
    from ..lite import national as N
    from ..lite import national2 as N2
    from ..lite.guard import load_split as load_original
    from . import labels2b as L

    return {**load_original()["block_role"], **N.load_split().get("block_role_new", {}),
            **N2.load_split().get("block_role_unseen", {}), **L.load_split()["block_role_unseen_2b"]}


def f7_removals(ix: pd.DataFrame, known: dict | None = None, sites: pd.DataFrame | None = None
                ) -> tuple[np.ndarray, dict]:
    """(mask over ``ix`` rows: a ``train_p2`` row removed from the v2 training rows, counts). Verifier ruling F7:

    * a ``train_p2`` water body with a chip (index row) or a site (lock site, with or without an index row) in a
      25-km block whose Lite block role is ``val`` or ``test`` is removed whole. ``sites``: water_body, role, lat,
      lon of the lock sites (default: the lock file);
    * ``train_p2`` rows in blocks without a Lite block role get the national salted block rule (amendment §3.4
      item 2: ``group_split`` over the sorted unseen blocks, national parameters and seed, weights = index rows per
      block) and are removed unless the block draws ``train``.

    Keys only (block of the chip centre, role, water body). Other roles are never removed."""
    from ...eval.splits import group_split, spatial_blocks
    from ..lite import national as N

    p = N.PARAMS
    known = known_block_roles() if known is None else known
    tr = (ix["role"] == "train_p2").to_numpy()
    block = pd.Series(spatial_blocks(ix["lat"], ix["lon"], p["block_km"]), index=ix.index).astype(str)
    brole = block.map(known)
    in_eval = tr & brole.isin(F7_EVAL_BLOCK_ROLES).to_numpy()
    wb_chip = set(ix.loc[in_eval, "water_body"])
    if sites is None:
        from . import p2 as P2

        sites = P2.site_roles(P2.load_lock())
    st = sites[sites["role"] == "train_p2"]
    sblock = pd.Series(spatial_blocks(st["lat"], st["lon"], p["block_km"]), index=st.index).astype(str)
    wb_site = set(st.loc[sblock.map(known).isin(F7_EVAL_BLOCK_ROLES).to_numpy(), "water_body"])
    wb_out = wb_chip | wb_site
    by_wb = tr & ix["water_body"].isin(wb_out).to_numpy()
    unseen_rows = tr & brole.isna().to_numpy()
    unseen = sorted(set(block[unseen_rows]))
    drawn: dict = {}
    if unseen:
        w = block[unseen_rows].value_counts().to_dict()
        bs = group_split(unseen, test_frac=p["test_frac"], val_frac=p["val_frac"], seed=p["seed"], weights=w)
        drawn = {b: r for r, bl in (("train", bs.train), ("val", bs.val), ("test", bs.test)) for b in bl}
    by_draw = unseen_rows & (block.map(drawn).fillna("train") != "train").to_numpy()
    out = by_wb | by_draw
    usable = (ix["eligible"] & (ix["has_label"] | (ix["cens"] != 0))).to_numpy()
    rep = {"train_p2_rows": int(tr.sum()), "train_p2_rows_usable": int((tr & usable).sum()),
           "rows_in_val_role_block": int((tr & (brole == "val").to_numpy()).sum()),
           "rows_in_test_role_block": int((tr & (brole == "test").to_numpy()).sum()),
           "usable_rows_in_val_role_block": int((tr & usable & (brole == "val").to_numpy()).sum()),
           "usable_rows_in_test_role_block": int((tr & usable & (brole == "test").to_numpy()).sum()),
           "water_bodies_with_chip_in_eval_block": len(wb_chip),
           "lock_water_bodies_with_site_in_eval_block": len(wb_site),
           "water_bodies_removed_eval_block": len(wb_out & set(ix.loc[tr, "water_body"])),
           "rows_removed_eval_block_water_body": int(by_wb.sum()),
           "usable_rows_removed_eval_block_water_body": int((by_wb & usable).sum()),
           "rows_in_blocks_without_a_lite_role": int(unseen_rows.sum()),
           "usable_rows_in_blocks_without_a_lite_role": int((unseen_rows & usable).sum()),
           "blocks_without_a_lite_role": len(unseen),
           "drawn_block_roles": pd.Series(drawn, dtype=object).value_counts().to_dict(),
           "rows_removed_by_draw": int((by_draw & ~by_wb).sum()),
           "usable_rows_removed_by_draw": int((by_draw & ~by_wb & usable).sum()),
           "rows_removed": int(out.sum()), "usable_rows_removed": int((out & usable).sum()),
           "usable_rows_left": int((tr & usable & ~out).sum()),
           "water_bodies_left": int(ix.loc[tr & usable & ~out, "water_body"].nunique()),
           "sites_left": int(ix.loc[tr & usable & ~out, "site_id"].nunique())}
    return out, rep


def index_sha256() -> str:
    import hashlib

    return hashlib.sha256(index_path().read_bytes()).hexdigest()


def load_index(roles, *, what: str = "lite P2 rows", check_pin: bool = True) -> pd.DataFrame:
    """Eligible P2 index rows with a usable label (or, for ``test_p2``, the present flag) for ``roles``.

    * the file must have the pinned hash (:data:`INDEX_SHA256`);
    * a row of a site of :data:`EXCLUDED_AT_LOAD` or of the reserved new-site block is a hard failure;
    * ``train_p2``: the F7 removals are applied (:func:`f7_removals`);
    * ``val_p2``: only after the lead's stamp (``stamp.require_stamp``);
    * ``test_p2``: keys and the present flag (the index stores no test value)."""
    from . import stamp

    roles = [str(r) for r in roles]
    bad = sorted(set(roles) - {"train_p2", "val_p2", "test_p2"})
    if bad:
        raise P2IndexError(f"unknown P2 roles {bad}")
    stamp.guard_roles(roles, what)
    if check_pin:
        if INDEX_SHA256 == "PENDING":
            raise P2IndexError("the P2 index is not pinned yet (INDEX_SHA256 is PENDING)")
        if index_sha256() != INDEX_SHA256:
            raise P2IndexError("index.parquet differs from the pinned P2 index")
    ix = pd.read_parquet(index_path())
    assert_no_excluded(ix["site_id"], "lite P2 index")
    assert_no_reserved(ix["site_id"].tolist(), "lite P2 index (load)")
    if ix.loc[ix["role"] == "test_p2", ["y_turb_fnu", "cens_log10_limit"]].notna().any().any():
        raise P2IndexError("the P2 index holds a test_p2 label value")
    removed, _ = f7_removals(ix)
    usable = (ix["eligible"] & (ix["has_label"] | (ix["cens"] != 0))).to_numpy()
    out = ix[usable & ~removed & ix["role"].isin(roles).to_numpy()]
    return out.reset_index(drop=True)
