# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""AquaCorr-Global (Zenodo record 21927415, CC-BY-4.0) client: download, flag decoding, tidy table.

AquaCorr-Global v1.0 (Ghazi Zadeh Hashemi, Bolten, Kumar, Seegers; published 2026-08-14) pairs
harmonised in-situ water-quality records with same-calendar-day ACOLITE-DSF Rrs for Landsat 5/7/8/9
and Sentinel-2A/B (1984-2024). It is new and not peer-reviewed, so ``audit_matchups`` checks it
before use (duplicates, matchup windows, chl-a method classes, QC flags, provenance, geography).

Source flags (``<param>_srcflag``) are bit-packed (record description / metadata.json):
  bits 0-1 method_class (0 unknown, 1 extraction/lab, 2 in vivo/sensor, 3 curated)
  bits 2-3 quality_tier (0 n/a, 1 restrictive, 2 narrowed, 3 inclusive)
  bits 4-5 fraction     (0 n/a, 1 dissolved, 2 total, 3 unknown)
  bits 6-7 depth_type   (0 none, 1 discrete, 2 integrated, 3 mixed)
  bit  8   outlier      (1 flagged)
  bit  9   recommended  (1 yes)
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests

ZENODO_RECORD = "21927415"
ZENODO_DOI = "10.5281/zenodo.21927415"
CONCEPT_DOI = "10.5281/zenodo.21927414"
RECORD_API = f"https://zenodo.org/api/records/{ZENODO_RECORD}"
LICENSE = "CC-BY-4.0"
USER_AGENT = "SENTINEL-v2 research (public-data download)"

# Files we keep. The 505 MB csv.gz is a duplicate export of the parquet and is not downloaded.
WANTED_FILES = ("aquacorr_matchups.parquet", "aquacorr_metadata.json",
                "aquacorr_gloria_validation.parquet", "aquacorr_rrs_spectra.zarr.zip")

PARAMS = ("chla", "tss", "sdd", "turbidity", "acdom440", "doc", "toc")
SRCFLAG_PARAMS = ("chla", "sdd", "tss", "turbidity", "toc")
METHOD_CLASS = {0: "unknown", 1: "extraction_lab", 2: "in_vivo_sensor", 3: "curated"}
QUALITY_TIER = {0: "na", 1: "restrictive", 2: "narrowed", 3: "inclusive"}
FRACTION = {0: "na", 1: "dissolved", 2: "total", 3: "unknown"}
DEPTH_TYPE = {0: "none", 1: "discrete", 2: "integrated", 3: "mixed"}
S2_SENSORS = ("S2A", "S2B")

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def fetch_record(timeout: float = 120) -> dict:
    """Zenodo record JSON (files, sizes, md5 checksums, license, creators)."""
    for k in range(5):
        try:
            r = _session.get(RECORD_API, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 30)))
                continue
        except requests.RequestException:
            pass
        time.sleep(10 * (k + 1))
    raise RuntimeError(f"Zenodo record {ZENODO_RECORD} unavailable")


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()  # noqa: S324 — Zenodo publishes md5 checksums
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, md5: str | None = None, *, retries: int = 8,
                  chunk: int = 1 << 20, log=print) -> Path:
    """Resumable (HTTP Range) download with md5 verification; skips a verified existing file."""
    dest = Path(dest)
    if dest.exists() and md5 and _md5(dest) == md5:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for k in range(retries):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _session.get(url, headers=headers, stream=True, timeout=300) as r:
                if r.status_code == 416:  # already complete
                    break
                if r.status_code == 429:
                    time.sleep(float(r.headers.get("Retry-After", 60)))
                    continue
                if r.status_code not in (200, 206):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                mode = "ab" if (have and r.status_code == 206) else "wb"
                with open(part, mode) as fh:
                    for block in r.iter_content(chunk):
                        fh.write(block)
            break
        except (requests.RequestException, OSError) as e:
            log(f"download retry {k + 1}/{retries} for {dest.name}: {e!r}")
            time.sleep(15 * (k + 1))
    else:
        raise RuntimeError(f"download failed: {url}")
    if md5:
        got = _md5(part)
        if got != md5:
            part.unlink()
            raise RuntimeError(f"md5 mismatch for {dest.name}: {got} != {md5}")
    part.replace(dest)
    return dest


def decode_srcflag(flag: pd.Series | np.ndarray) -> pd.DataFrame:
    """Decode a bit-packed AquaCorr source flag. Missing flags give <NA> in every field."""
    s = pd.Series(flag)
    valid = s.notna().to_numpy()
    v = np.where(valid, pd.to_numeric(s, errors="coerce").fillna(0).to_numpy(), 0).astype(np.int64)
    out = pd.DataFrame({
        "method_class": pd.Categorical.from_codes(np.where(valid, v & 3, -1), list(METHOD_CLASS.values())),
        "quality_tier": pd.Categorical.from_codes(np.where(valid, (v >> 2) & 3, -1),
                                                  list(QUALITY_TIER.values())),
        "fraction": pd.Categorical.from_codes(np.where(valid, (v >> 4) & 3, -1), list(FRACTION.values())),
        "depth_type": pd.Categorical.from_codes(np.where(valid, (v >> 6) & 3, -1),
                                                list(DEPTH_TYPE.values())),
        "outlier": pd.array(np.where(valid, ((v >> 8) & 1) == 1, False), dtype="boolean"),
        "recommended": pd.array(np.where(valid, ((v >> 9) & 1) == 1, False), dtype="boolean"),
    }, index=s.index)
    out.loc[~valid, ["outlier", "recommended"]] = pd.NA
    return out


# ------------------------------------------------------------------ geography (coarse, documented)
# Bounding boxes are deliberately coarse; they only drive the US/Europe chip budget and audit counts.
REGION_BOXES = {
    "US_conus": (-125.0, 24.0, -66.5, 49.5),
    "US_alaska": (-170.0, 51.0, -129.0, 71.5),
    "US_hawaii": (-161.0, 18.5, -154.0, 22.5),
    "US_puerto_rico": (-67.5, 17.5, -65.0, 18.7),
    "Europe": (-25.0, 34.0, 45.0, 72.0),
}


def assign_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Coarse region label from bounding boxes (first match wins; Canada/Mexico border areas
    inside the CONUS box are labelled US_conus — counts are therefore upper bounds for the US)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    out = np.full(lat.shape, "other", dtype=object)
    done = np.zeros(lat.shape, bool)
    for name, (x0, y0, x1, y1) in REGION_BOXES.items():
        m = (~done) & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        out[m] = name
        done |= m
    out[~np.isfinite(lat) | ~np.isfinite(lon)] = "unknown"
    return out


# ------------------------------------------------------------------------------ tidy + audit

# Units are NOT stated in the published AquaCorr files (the column dictionary
# ``aquacorr_variables.xlsx`` named in metadata.json is not in the Zenodo record). These are the
# units of the harmonised source datasets (AquaMatch / WQP conventions) and are consistent with the
# value distributions (e.g. Secchi median 1.5, chl-a median 5.6). Stored as ``<param>_unit``.
ASSUMED_UNITS = {"chla": "ug/L", "tss": "mg/L", "sdd": "m", "turbidity": "NTU_or_FNU_unspecified",
                 "acdom440": "1/m", "doc": "mg/L", "toc": "mg/L"}
S2_SWIR_COLS = ("rrs_1610", "rrs_1614")
S2_BLUE_COLS = ("rrs_492",)


def scene_tile(scene_id: pd.Series) -> pd.Series:
    """MGRS tile from an ACOLITE S2 scene id 'S2B_MSI_2019_06_14_15_41_22_T19TCH_L2R'."""
    return scene_id.str.extract(r"_T(\d{2}[A-Z]{3})_")[0]


def scene_time(scene_id: pd.Series) -> pd.Series:
    """Granule sensing time (UTC) encoded in the ACOLITE S2 scene id."""
    s = scene_id.str.extract(r"_MSI_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_")
    txt = s[0] + "-" + s[1] + "-" + s[2] + "T" + s[3] + ":" + s[4] + ":" + s[5] + "Z"
    return pd.to_datetime(txt, utc=True, errors="coerce")


def first_present(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    have = [c for c in cols if c in df.columns]
    if not have:
        return pd.Series(np.nan, index=df.index)
    return df[have].bfill(axis=1).iloc[:, 0]


def tidy_matchups(df: pd.DataFrame) -> pd.DataFrame:
    """Typed, flag-decoded copy of the AquaCorr matchup table (drops the WKB geometry column)."""
    out = df.drop(columns=[c for c in ("geometry",) if c in df.columns]).copy()
    out["site_id"] = out["site_id"].astype(str)
    out["date_insitu"] = pd.to_datetime(out["date_insitu"]).dt.date
    out["date_satellite"] = pd.to_datetime(out["date_satellite"]).dt.date
    out["region"] = assign_region(out["lat"].to_numpy(), out["lon"].to_numpy())
    out["year"] = pd.to_datetime(out["date_insitu"]).dt.year.astype("int16")
    is_s2 = out["sensor"].isin(S2_SENSORS)
    out["mgrs_tile"] = np.where(is_s2, scene_tile(out["scene_id"].astype(str)), None)
    out["scene_sensing_time"] = scene_time(out["scene_id"].astype(str)).where(is_s2)
    for p in SRCFLAG_PARAMS:
        dec = decode_srcflag(out[f"{p}_srcflag"].where(out[f"{p}_value"].notna()))
        for c in dec.columns:
            out[f"{p}_{c}"] = dec[c]
    for p, u in ASSUMED_UNITS.items():
        out[f"{p}_unit"] = np.where(out[f"{p}_value"].notna(), u, None)
    # duplicate structure
    out["acq_group_n_scenes"] = out.groupby(["site_id", "date_insitu", "sensor"])["scene_id"].transform("nunique")
    key = [out["lat"].round(5), out["lon"].round(5), out["date_insitu"].astype(str), out["scene_id"].astype(str)]
    out["colocated_group_n_rows"] = out.groupby(key)["matchup_id"].transform("size")
    # spectral plausibility (water SWIR Rrs should be ~0; bright flat spectra indicate cloud/land)
    swir = first_present(out, S2_SWIR_COLS + ("rrs_1609", "rrs_1608", "rrs_1650", "rrs_1678"))
    blue = first_present(out, S2_BLUE_COLS + ("rrs_483", "rrs_482", "rrs_479", "rrs_486"))
    out["rrs_swir1"] = swir
    out["rrs_blue"] = blue
    out["flag_swir_gt_0p01"] = swir > 0.01
    out["flag_blue_gt_0p1"] = blue > 0.1
    out["matchup_window_days"] = 0  # same calendar day by construction (UTC date of the overpass)
    return out


def audit_matchups(t: pd.DataFrame, metadata: dict | None = None) -> dict[str, Any]:
    """Audit numbers for the datasheet (``t`` from ``tidy_matchups``)."""
    a: dict[str, Any] = {"n_rows": int(len(t)), "n_sites": int(t.site_id.nunique())}
    a["rows_by_sensor"] = t.sensor.value_counts().to_dict()
    a["rows_by_source"] = t.source.value_counts().to_dict()
    a["provenance_tier_by_source"] = pd.crosstab(t.source, t.quality_tier).to_dict(orient="index")
    a["rows_by_region"] = t.region.value_counts().to_dict()
    a["sites_by_region"] = t.groupby("region").site_id.nunique().to_dict()
    s2 = t[t.sensor.isin(S2_SENSORS)]
    a["s2_rows_by_region"] = s2.region.value_counts().to_dict()
    a["years"] = [int(t.year.min()), int(t.year.max())]
    a["s2_rows_by_year"] = s2.year.value_counts().sort_index().to_dict()
    # duplicates
    cols = [c for c in t.columns if c != "matchup_id"]
    a["dup_exact_rows"] = int(t.duplicated(cols).sum())
    a["dup_site_date_sensor_scene"] = int(t.duplicated(["site_id", "date_insitu", "sensor", "scene_id"]).sum())
    a["rows_in_multi_scene_acquisitions"] = int((t.acq_group_n_scenes > 1).sum())
    a["extra_rows_from_multi_scene"] = int(t.duplicated(["site_id", "date_insitu", "sensor"]).sum())
    a["rows_colocated_with_other_site_id"] = int((t.colocated_group_n_rows > 1).sum())
    a["colocated_rows_by_source"] = t[t.colocated_group_n_rows > 1].source.value_counts().to_dict()
    k5 = t[["date_insitu", "scene_id"]].assign(lat5=t.lat.round(5), lon5=t.lon.round(5))
    a["colocated_same_value_dups"] = {}
    for p in ("chla", "tss", "sdd", "turbidity"):
        m = t[f"{p}_value"].notna()
        a["colocated_same_value_dups"][p] = int(k5[m].assign(v=t.loc[m, f"{p}_value"]).duplicated().sum())
    # matchup windows
    ov_date = pd.to_datetime(t.overpass_utc, utc=True).dt.strftime("%Y-%m-%d")
    a["date_insitu_eq_date_satellite"] = float((t.date_insitu == t.date_satellite).mean())
    a["overpass_utc_date_eq_date_satellite"] = float((ov_date == t.date_satellite.astype(str)).mean())
    a["rows_lon_gt_140E"] = int((t.lon > 140).sum())  # overpass ~22-23 UTC: UTC date = local date - 1
    st = t.scene_sensing_time
    ok = st.notna() & t.overpass_utc.notna()
    a["s2_scene_id_time_vs_overpass_utc_max_abs_s"] = float(
        (pd.to_datetime(t.overpass_utc[ok], utc=True) - st[ok]).dt.total_seconds().abs().max()) if ok.any() else None
    # chl-a method / flags
    m = t.chla_value.notna()
    a["chla_with_value"] = int(m.sum())
    a["chla_method_class"] = t.loc[m, "chla_method_class"].value_counts().to_dict()
    a["chla_recommended"] = int(t.loc[m, "chla_recommended"].sum())
    a["chla_outlier"] = int(t.loc[m, "chla_outlier"].sum())
    a["chla_recommended_by_method"] = pd.crosstab(t.loc[m, "chla_method_class"],
                                                  t.loc[m, "chla_recommended"]).to_dict()
    a["chla_srcflag_nonzero"] = int((t.loc[m, "chla_srcflag"].fillna(0) != 0).sum())
    a["chla_daily_median_rows"] = int((t.loc[m, "chla_selection"] == "daily_median").sum())
    for p in ("tss", "sdd", "turbidity", "toc"):
        mm = t[f"{p}_value"].notna()
        a[f"{p}_method_class"] = t.loc[mm, f"{p}_method_class"].value_counts().to_dict()
        a[f"{p}_recommended"] = int(t.loc[mm, f"{p}_recommended"].sum())
    if metadata:
        par = metadata.get("parameters", {})
        a["metadata_claims_chla"] = par.get("chla")
        a["metadata_doi_field"] = metadata.get("doi")
        a["metadata_files_listed"] = sorted(metadata.get("files", {}))
    # QC flags
    for f in ("flag_negative_rrs", "flag_high_cv", "flag_insufficient_pixels", "flag_high_sza",
              "flag_acolite", "passed_qc"):
        a[f"rate_{f}"] = float(t[f].mean())
        a[f"s2_rate_{f}"] = float(s2[f].mean())
    q = s2[s2.passed_qc]
    a["s2_passed_qc_rows"] = int(len(q))
    a["s2_passed_qc_swir_gt_0p01"] = float(q.flag_swir_gt_0p01.mean())
    a["s2_passed_qc_blue_gt_0p1"] = float(q.flag_blue_gt_0p1.mean())
    a["s2_passed_qc_median_rrs"] = {c: float(q[c].median()) for c in
                                    ("rrs_492", "rrs_560", "rrs_665", "rrs_864", "rrs_swir1") if c in q}
    a["s2_passed_qc_swir_gt_0p01_by_source"] = q.groupby("source").flag_swir_gt_0p01.mean().round(4).to_dict()
    a["s2_window_valid_pixels_passed"] = q.window_valid_pixels.value_counts().sort_index().to_dict()
    # values
    vals = {}
    for p in PARAMS:
        x = t[f"{p}_value"].dropna()
        if len(x):
            vals[p] = {"n": int(len(x)), "min": float(x.min()), "p01": float(x.quantile(0.01)),
                       "median": float(x.median()), "p99": float(x.quantile(0.99)), "max": float(x.max()),
                       "n_le_0": int((x <= 0).sum())}
    a["values"] = vals
    a["coord_centroid_by_sensor"] = t.groupby("sensor").coord_centroid.sum().astype(int).to_dict()
    return a
