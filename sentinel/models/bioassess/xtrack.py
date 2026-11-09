# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""M5-X — the larger / harder follow-on track (protocol ``docs/preregistration/bioassess_x_protocol.md``).

Builds the **frozen split table** from metadata: ids, survey/year, region, coordinates, site role and the
*availability* of each target (target values are loaded by the shared loaders but only ``notna()`` is used):

* **X-L (lakes):** test = NLA 2022 first visits with a benthic sample; fold = aggregated ecoregion (ag_eco9);
  leave-one-ecoregion-out x cycle-out. Training pool of fold g = NLA 2007/2012/2017 + NRSA 2008-24 rows
  outside g, + all UK rows that are not X-U test/excluded rows.
* **X-U (UK, England, EA):** test = invertebrate samples dated 2021-2026 at UK sites whose labels no
  workstream has scored (M5 role ``pool_fit``); fold = EA ``agency_area``; leave-one-area-out x time-out.
  Training pool of fold g = UK samples dated <= 2020 outside g + NRSA + NLA. Excluded (time-out rule): every UK
  sample dated 2021-2026 that is not an X-U test row, i.e. those at ``pool_val`` / ``reserved_test`` sites and
  ``pool_fit`` samples without a primary target or area.

``training_rows`` additionally removes rows of any frozen test block of any workstream (lead ruling on
RESEARCH_PLAN §15.5 item 7) and every row within 1 km of a fold-g test location (review C-MX-2).

The module never modifies ``data.py`` (D5 freeze) and only imports its helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from sentinel.core import paths
from sentinel.models.bioassess.data import ALL_SURVEYS, location_clusters

NLA_SURVEYS = ("nla07", "nla12", "nla17", "nla22")
NLA_TEST = "nla22"
UK_TEST_FROM_YEAR = 2021
NLA_TARGETS = {"log10_tp_ug_l": "tp_ug_l", "log10_tn_mg_l": "tn_mg_l", "log10_cond_us_cm": "cond_us_cm",
               "log10_chloride_mg_l": "chloride_mg_l", "log10_turbidity_ntu": "turbidity_ntu", "ph_lab": "ph_lab_su"}
UK_PRIMARY = ("uk_ph", "uk_log10_cond25_us_cm", "uk_log10_chloride_mg_l", "uk_log10_orthop_mg_l",
              "uk_log10_nh4n_mg_l", "uk_log10_nitrate_mg_l")
SPLIT_VERSION = "x_v1"
os.environ["SENTINEL_TRACK"] = "M5-X"   # any process importing this module is an M5-X process (C-MXR-2)


def nla_universe() -> pd.DataFrame:
    P = paths.processed("nars_lakes")
    v = pd.read_parquet(P / "visits.parquet")
    b = pd.read_parquet(P / "benthic_counts.parquet", columns=["visit_id"])
    v = v[(v["visit_no"].astype(str) == "1") & v["visit_id"].isin(set(b["visit_id"]))].copy()
    v = v.dropna(subset=["lat", "lon", "ag_eco9"])
    cw = pd.read_parquet(P / "chemistry_wide.parquet").set_index("visit_id")
    avail = {t: cw[c].reindex(v["visit_id"]).notna().to_numpy() for t, c in NLA_TARGETS.items()}
    out = pd.DataFrame({"source": "nla", "id": v["visit_id"].astype(str).to_numpy(), "survey": v["survey"].to_numpy(),
                        "year": pd.to_numeric(v["year"], errors="coerce").to_numpy(), "region": v["ag_eco9"].to_numpy(),
                        "lat": v["lat"].to_numpy(float), "lon": v["lon"].to_numpy(float)})
    out["cluster"] = ["nla_" + str(c) for c in location_clusters(out["lat"].to_numpy(), out["lon"].to_numpy())]
    out["n_targets"] = np.sum(list(avail.values()), axis=0)
    return out


def nrsa_universe() -> pd.DataFrame:
    P = paths.processed("nars_streams")
    v = pd.read_parquet(P / "visits.parquet")
    v = v[v["survey"].isin(ALL_SURVEYS) & (v["visit_no"].astype(str) == "1") & v["has_benthic"].fillna(False)]
    v = v.dropna(subset=["lat", "lon", "ag_eco9"])
    out = pd.DataFrame({"source": "nrsa", "id": v["visit_id"].astype(str).to_numpy(), "survey": v["survey"].to_numpy(),
                        "year": pd.to_numeric(v.get("year"), errors="coerce").to_numpy() if "year" in v else np.nan,
                        "region": v["ag_eco9"].to_numpy(), "lat": v["lat"].to_numpy(float),
                        "lon": v["lon"].to_numpy(float)})
    out["cluster"] = ["nrsa_" + str(c) for c in location_clusters(out["lat"].to_numpy(), out["lon"].to_numpy())]
    out["n_targets"] = np.nan          # not needed for the split (NRSA is training-only in track X)
    return out


def uk_universe() -> pd.DataFrame:
    from sentinel.models.bioassess.uk import load_uk

    uk = load_uk(include_reserved=True)
    S = uk.samples
    P = paths.processed("uk_ea_ecology")
    sites = pd.read_parquet(P / "sites.parquet", columns=["site_id", "assemblage", "agency_area"])
    sites = sites[sites["assemblage"] == "invertebrate"].assign(site_id=lambda d: d["site_id"].astype(str))
    area = sites.drop_duplicates("site_id").set_index("site_id")["agency_area"]
    smp = pd.read_parquet(P / "samples_invertebrate.parquet", columns=["analysis_id", "sample_date"])
    date = pd.to_datetime(smp.assign(analysis_id=smp["analysis_id"].astype(str)).set_index("analysis_id")["sample_date"])
    avail = uk.Y.reindex(S.index)[list(UK_PRIMARY)].notna()
    out = pd.DataFrame({"source": "uk", "id": S.index.astype(str), "survey": "uk_ea",
                        "year": date.reindex(S.index).dt.year.to_numpy(), "region": S["site_id"].map(area).to_numpy(),
                        "lat": S["lat"].to_numpy(float), "lon": S["lon"].to_numpy(float), "uk_role": S["role"].to_numpy(),
                        "site_id": S["site_id"].to_numpy()})
    out["cluster"] = ["uk_" + str(c) for c in location_clusters(out["lat"].to_numpy(), out["lon"].to_numpy())]
    out["n_targets"] = avail.sum(1).to_numpy()
    return out


def make_splits() -> pd.DataFrame:
    """One row per (block, sample): role in {test, train_pool, excluded}; ``fold`` = region for test rows."""
    nla, nrsa, uk = nla_universe(), nrsa_universe(), uk_universe()
    rows = []
    # ---- X-L
    t = nla[(nla["survey"] == NLA_TEST) & (nla["n_targets"] > 0)]
    rows.append(t.assign(block="X-L", role="test", fold=t["region"]))
    rows.append(nla[nla["survey"] != NLA_TEST].assign(block="X-L", role="train_pool", fold=None))
    rows.append(nla[(nla["survey"] == NLA_TEST) & (nla["n_targets"] == 0)].assign(block="X-L", role="excluded",
                                                                                     fold=None))
    rows.append(nrsa.assign(block="X-L", role="train_pool", fold=None))
    uk_test = (uk["year"] >= UK_TEST_FROM_YEAR) & (uk["uk_role"] == "pool_fit") & (uk["n_targets"] > 0) & \
        uk["region"].notna()
    uk_excl = (uk["year"] >= UK_TEST_FROM_YEAR) & ~uk_test
    rows.append(uk[~uk_test & ~uk_excl].assign(block="X-L", role="train_pool", fold=None))
    rows.append(uk[uk_test | uk_excl].assign(block="X-L", role="excluded", fold=None))
    # ---- X-U
    rows.append(uk[uk_test].assign(block="X-U", role="test", fold=uk.loc[uk_test, "region"]))
    rows.append(uk[uk_excl].assign(block="X-U", role="excluded", fold=None))
    rows.append(uk[~uk_test & ~uk_excl].assign(block="X-U", role="train_pool", fold=None))
    rows.append(nrsa.assign(block="X-U", role="train_pool", fold=None))
    rows.append(nla.assign(block="X-U", role="train_pool", fold=None))
    S = pd.concat(rows, ignore_index=True)
    S["previously_seen_at_test"] = (S["source"] == "nrsa") & S["survey"].isin(["nrsa1819", "nrsa2324"]) | \
        ((S["source"] == "uk") & (S.get("uk_role") == "reserved_test"))
    cols = ["block", "role", "fold", "source", "id", "survey", "year", "region", "cluster", "lat", "lon",
            "n_targets", "uk_role", "previously_seen_at_test"]
    S = S[cols].sort_values(["block", "role", "source", "id"], kind="stable").reset_index(drop=True)
    S["fold"] = S["fold"].astype(object)
    return S


# ---------------------------------------------------------------- training pools (review C-MX-2, item-7 ruling)
BUFFER_KM = 1.0
FROZEN_SPLIT_FILES = {   # other workstreams' frozen test blocks touching NRSA / NLA / UK EA (lead ruling 2026-11-09)
    "fuse_tier_c": "docs/preregistration/fuse_tier_c_split.json",
    "d2n_microcystin": "docs/preregistration/downstream_d2n_microcystin_split.json",
}
FUSE_TEST_ROLES = ("test_region", "test_spatial")
NRSA_FROZEN_TEST_SURVEYS = ("nrsa1819", "nrsa2324")   # M5 Tests A/B (2018-19), C (2023-24); D5 test (2023-24)


def frozen_test_keys() -> dict[tuple[str, str], str]:
    """(source, id) -> reason for every row of a frozen test block of any workstream that touches the M5-X
    sources (RESEARCH_PLAN §15.5 item 7 as ruled by the lead: no such row in any M5-X training pool)."""
    out: dict[tuple[str, str], str] = {}
    nrsa = nrsa_universe()
    for i in nrsa.loc[nrsa["survey"].isin(NRSA_FROZEN_TEST_SURVEYS), "id"]:
        out[("nrsa", i)] = "M5 Test A/B/C + D5 test cycle (NRSA 2018-19 / 2023-24)"
    from sentinel.models.bioassess.uk import load_uk
    uk = load_uk(include_reserved=True).samples
    for i in uk.index[uk["role"] == "reserved_test"]:
        out[("uk", str(i))] = "M5 T2 frozen test (UK reserved_test sites)"
    repo = paths.REPO_ROOT
    role = json.loads((repo / FROZEN_SPLIT_FILES["fuse_tier_c"]).read_text())["site_role"]
    rows = pd.read_parquet(paths.processed("fuse_tier_c") / "rows.parquet", columns=["site_id", "source", "visit_id"])
    rows = rows[(rows["source"] == "nla") & rows["site_id"].map(role).isin(FUSE_TEST_ROLES)]
    for i in rows["visit_id"].astype(str):
        out.setdefault(("nla", i), "Fuse tier C frozen test (Test-R / Test-S sites)")
    d2 = json.loads((repo / FROZEN_SPLIT_FILES["d2n_microcystin"]).read_text())["assignments"]
    test_huc4 = {h for h, r in d2["huc4_role"].items() if r == "test"}
    d2_nla = {s.replace("NARS_WQX-", "") for s in d2["site_cluster"] if s.startswith("NARS_WQX-NLA")}
    v = pd.read_parquet(paths.processed("nars_lakes") / "visits.parquet", columns=["visit_id", "unique_id", "huc8"])
    hit = v["unique_id"].isin(d2_nla) & v["huc8"].astype(str).str.zfill(8).str[:4].isin(test_huc4)
    for i in v.loc[hit, "visit_id"].astype(str):
        out.setdefault(("nla", i), "D2n frozen test (NLA site in a test HUC4)")
    for i in m2_test_nla_visits():
        out.setdefault(("nla", i), "M2-orbit test-role rows (same NLA sampling event)")
    return out


def m2_test_nla_visits() -> list[str]:
    """NLA visits that are the same sampling events (unique_id, date) as M2-orbit test-role rows in its national
    matchups (lead ruling 2026-11-09). Roles from M2's own ``split.assign`` (metadata only; no cache written)."""
    from sentinel.models.orbit import split as OS

    s = pd.read_parquet(paths.processed("s2_wqp_matchups_hist") / "samples.parquet",
                        columns=["site_id", "date", "lat", "lon"])
    s = s[s["site_id"].astype(str).str.startswith("NARS_WQX-NLA")].drop_duplicates(["site_id", "date"])
    lk = OS.assign_lakes(s["lon"].to_numpy(float), s["lat"].to_numpy(float), use_cache=False, log=lambda *a: None)
    r = OS.assign(s["lon"].to_numpy(float), s["lat"].to_numpy(float), pd.to_datetime(s["date"]), lakes=lk,
                  log=lambda *a: None)
    t = s[(r["role"] == "test").to_numpy()]
    ev = set(zip(t["site_id"].str.replace("NARS_WQX-", "", regex=False), pd.to_datetime(t["date"]).dt.date))
    v = pd.read_parquet(paths.processed("nars_lakes") / "visits.parquet", columns=["visit_id", "unique_id", "date"])
    v = v.dropna(subset=["unique_id"])
    hit = [(u, d) in ev for u, d in zip(v["unique_id"], pd.to_datetime(v["date"]).dt.date)]
    return sorted(v.loc[hit, "visit_id"].astype(str))


def _within_km(lat, lon, qlat, qlon, km: float) -> np.ndarray:
    """Boolean mask: point (lat, lon) lies within ``km`` of any query point (haversine BallTree)."""
    from sklearn.neighbors import BallTree

    if len(qlat) == 0 or len(lat) == 0:
        return np.zeros(len(lat), bool)
    tree = BallTree(np.radians(np.column_stack([qlat, qlon])), metric="haversine")
    cnt = tree.query_radius(np.radians(np.column_stack([lat, lon])), r=km / 6371.0088, count_only=True)
    return cnt > 0


def training_rows(S: pd.DataFrame, block: str, g, frozen: dict | None = None, buffer_km: float = BUFFER_KM,
                  report: dict | None = None) -> pd.DataFrame:
    """Training pool of test fold ``g``:
    1. pool rows of the block; same-continent sources (NLA+NRSA for X-L; UK for X-U) outside region g;
    2. minus every test/excluded row of either M5-X block (C-MXR-1) and every row of a frozen test block of any
       workstream (``frozen``; item-7 ruling);
    3. minus every row within ``buffer_km`` of a fold-g test location, any source and any region (C-MX-2:
       1-km location clusters may span regions; this also removes the same site under another id)."""
    P = S[(S["block"] == block) & (S["role"] == "train_pool")]
    same = ("nla", "nrsa") if block == "X-L" else ("uk",)
    P = P[~P["source"].isin(same) | (P["region"] != g)]
    prot = S[S["role"].isin(["test", "excluded"])]          # test/excluded rows of EITHER block (C-MXR-1)
    pk = set(zip(prot["source"], prot["id"].astype(str)))
    P = P[[k not in pk for k in zip(P["source"], P["id"].astype(str))]]
    n0 = len(P)
    if frozen is not None:
        keys = list(zip(P["source"], P["id"].astype(str)))
        P = P[[k not in frozen for k in keys]]
    n1 = len(P)
    T = S[(S["block"] == block) & (S["role"] == "test") & (S["fold"] == g)]
    near = _within_km(P["lat"].to_numpy(float), P["lon"].to_numpy(float), T["lat"].to_numpy(float),
                      T["lon"].to_numpy(float), buffer_km)
    near |= P["cluster"].isin(set(T["cluster"])).to_numpy()
    P = P[~near]
    if report is not None:
        report.update({"after_region": n0, "removed_frozen": n0 - n1, "removed_buffer": int(near.sum()), "final": len(P)})
    return P


def validate_training_pools(S: pd.DataFrame, frozen: dict, buffer_km: float = BUFFER_KM) -> tuple[list[str], list]:
    """Check every fold's training pool: no test id, no frozen-test row, nothing within buffer_km of a fold
    test location. Returns (problems, per-fold report)."""
    bad, rep = [], []
    for block in ("X-L", "X-U"):
        T_all = S[(S["block"] == block) & (S["role"] == "test")]
        for g in sorted(T_all["fold"].dropna().unique()):
            r = {"block": block, "fold": g}
            P = training_rows(S, block, g, frozen, buffer_km, report=r)
            T = T_all[T_all["fold"] == g]
            if set(zip(P["source"], P["id"])) & set(zip(T["source"], T["id"])):
                bad.append(f"{block}/{g}: test rows in training pool")
            if any((s, str(i)) in frozen for s, i in zip(P["source"], P["id"])):
                bad.append(f"{block}/{g}: frozen-test rows in training pool")
            other = S[(S["block"] != block) & S["role"].isin(["test", "excluded"])]
            if set(zip(P["source"], P["id"].astype(str))) & set(zip(other["source"], other["id"].astype(str))):
                bad.append(f"{block}/{g}: other block's test/excluded rows in training pool")
            if _within_km(P["lat"].to_numpy(float), P["lon"].to_numpy(float), T["lat"].to_numpy(float),
                          T["lon"].to_numpy(float), buffer_km).any():
                bad.append(f"{block}/{g}: training row within {buffer_km} km of a test location")
            r.update({f"n_{s}": int((P["source"] == s).sum()) for s in ("nla", "nrsa", "uk")})
            rep.append(r)
    return bad, rep


# ---------------------------------------------------------------- encoder guard (review C-MX-1)
FORBIDDEN_ENCODER_SHA256 = {
    # M5 UK-pretrained encoder (uk_pretrain_20261022T091937): pretrained with a chemistry loss on ALL pool_fit
    # samples of all years, i.e. including the X-U test labels -> never loadable in M5-X.
    "91453a229068ce5019717db25e66573b46acce3f23950dbccad5e03d313d0c31",
}


class EncoderLeakError(RuntimeError):
    pass


def write_train_ids_sidecar(checkpoint, train_ids) -> Path:
    """Written by the per-fold pretraining code from the ids it actually used: [(source, id), ...]."""
    side = Path(checkpoint).with_name(Path(checkpoint).name + ".train_ids.json")
    side.write_text(json.dumps({"train_ids": [[s, str(i)] for s, i in train_ids]}))
    return side


def guard_encoder(path, allowed_ids) -> None:
    """Refuse a pretrained checkpoint unless (a) it is not a forbidden M5 checkpoint, (b) a sidecar
    ``<checkpoint>.train_ids.json`` records its training ids, and (c) every recorded id lies in the fold's
    training pool (``allowed_ids``: (source, id) of ``training_rows(S, block, g, frozen)``) — so area-g, buffered,
    frozen-test and other-block test rows are all refused (C-MXR-2 b)."""
    from sentinel.core.registry import file_sha256

    path = Path(path)
    if file_sha256(path) in FORBIDDEN_ENCODER_SHA256:
        raise EncoderLeakError(f"{path} is a forbidden M5 checkpoint (trained on X-U test labels)")
    side = path.with_name(path.name + ".train_ids.json")
    if not side.exists():
        raise EncoderLeakError(f"{path} has no training-id record ({side.name}); M5-X loads only per-fold encoders")
    ids = {(s, str(i)) for s, i in json.loads(side.read_text())["train_ids"]}
    outside = ids - {(s, str(i)) for s, i in allowed_ids}
    if outside:
        raise EncoderLeakError(f"{path} was trained on {len(outside)} rows outside this fold's training pool")


def load_pretrained_encoder(spec, S: pd.DataFrame, block: str, g, frozen: dict):
    """THE M5-X checkpoint loader (single choke point). Refuses 'auto' / pointer files / missing paths, then
    guards against the fold's pool, then loads via ``cv.PretrainedEncoder``."""
    if spec is None or str(spec) == "auto" or str(spec).endswith("uk_pretrain_latest.json"):
        raise EncoderLeakError("M5-X never resolves 'auto' / M5 pointers; pass an explicit per-fold checkpoint")
    P = training_rows(S, block, g, frozen)
    guard_encoder(spec, zip(P["source"], P["id"].astype(str)))
    from sentinel.models.bioassess.cv import PretrainedEncoder

    enc = PretrainedEncoder(spec)
    enc.m5x_guard = (block, g)          # marks a guarded load; checked by require_guarded()
    return enc


def require_guarded(enc, block: str, g) -> None:
    """Every M5-X fit that passes a pretrained encoder calls this: the encoder must come from
    ``load_pretrained_encoder`` for the same (block, fold)."""
    if enc is not None and getattr(enc, "m5x_guard", None) != (block, g):
        raise EncoderLeakError("pretrained encoder was not loaded through xtrack.load_pretrained_encoder for this fold")


def assert_fit_ids(ids, S: pd.DataFrame, frozen: dict) -> None:
    """Loader guard (protocol §7): refuse any (source, id) that is a test/excluded row of either M5-X block or a
    frozen-test row of any workstream. Every M5-X fit must call this on its fit/validation ids."""
    prot = S[S["role"].isin(["test", "excluded"])]
    bad = {(s, str(i)) for s, i in ids} & (set(zip(prot["source"], prot["id"].astype(str))) | set(frozen))
    if bad:
        raise EncoderLeakError(f"refusing to fit on {len(bad)} protected rows (test/excluded/frozen-test)")


def write_splits(out: Path) -> dict:
    S = make_splits()
    out.parent.mkdir(parents=True, exist_ok=True)
    S.to_parquet(out, index=False)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    content = "\n".join(f"{r.block}|{r.role}|{r.fold}|{r.source}|{r.id}" for r in S.itertuples())
    hc = hashlib.sha256(content.encode()).hexdigest()       # library-independent content hash
    summ = (S.groupby(["block", "role", "source"]).size().rename("n").reset_index().to_dict("records"))
    meta = {"version": SPLIT_VERSION, "file": str(out), "sha256": h, "content_sha256": hc, "rows": len(S), "summary": summ}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def validate_splits(S: pd.DataFrame) -> list[str]:
    """Invariants of the frozen split table (empty list = OK)."""
    bad = []
    for b in ("X-L", "X-U"):
        B = S[S["block"] == b]
        test_ids = set(map(tuple, B[B["role"] == "test"][["source", "id"]].to_numpy()))
        pool_ids = set(map(tuple, B[B["role"] == "train_pool"][["source", "id"]].to_numpy()))
        if test_ids & pool_ids:
            bad.append(f"{b}: test rows in the training pool")
        if B[B["role"] == "test"]["fold"].isna().any():
            bad.append(f"{b}: test row without fold")
    T = S[(S["block"] == "X-L") & (S["role"] == "test")]
    if set(T["source"]) != {"nla"} or set(T["survey"]) != {NLA_TEST}:
        bad.append("X-L test is not NLA 2022 only")
    U = S[(S["block"] == "X-U") & (S["role"] == "test")]
    if set(U["source"]) != {"uk"} or set(U["uk_role"]) != {"pool_fit"} or U["year"].min() < UK_TEST_FROM_YEAR:
        bad.append("X-U test is not UK pool_fit sites dated >= 2021")
    XU_test = set(U["id"])
    XU_excl = set(S[(S["block"] == "X-U") & (S["role"] == "excluded")]["id"])
    XL_pool_uk = set(S[(S["block"] == "X-L") & (S["role"] == "train_pool") & (S["source"] == "uk")]["id"])
    if XL_pool_uk & (XU_test | XU_excl):
        bad.append("X-U test/excluded UK rows appear in the X-L training pool")
    P = S[(S["block"] == "X-U") & (S["role"] == "train_pool") & (S["source"] == "uk")]
    if P["year"].max() >= UK_TEST_FROM_YEAR:
        bad.append("X-U UK training pool contains samples dated >= 2021")
    return bad
