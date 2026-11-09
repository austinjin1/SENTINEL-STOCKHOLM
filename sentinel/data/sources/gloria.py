# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""GLORIA (Lehmann et al. 2023, Sci. Data 10:100; PANGAEA doi:10.1594/PANGAEA.948492, CC BY 4.0) and
spectral response functions for simulating sensor-band Rrs from its 1 nm hyperspectral spectra.

SRF sources (downloaded by scripts/data/download_gloria.py; official tables):
  * Sentinel-2A/2B MSI: ESA "Sentinel-2 Spectral Response Functions" COPE-GSEG-EOPG-TN-15-0007 v3.2
    (2023-10-10), sheets "Spectral Responses (S2A|S2B)", 1 nm, columns S2x_SR_AV_B1..B12.
  * Landsat 8 OLI: USGS spectral characteristics viewer L8_OLI_RSR.xlsx (band-average RSR, Ball).
  * Landsat 9 OLI-2: USGS L9_OLI2_RSR.xlsx (band-average RSR measured at GLAMR, Barsi et al. 2019).
  * Cameras (no measured SRFs exist for the target drone camera, docs/research/08 §b.3), modelled as
    top-hats from published band edges: ``naip_ads100`` (Leica ADS100 as listed in USDA NAIP FGDC
    metadata: B 435-495, G 525-585, R 619-651, NIR 808-882 nm) and ``dji_p4m_rgbn`` (DJI P4
    Multispectral spec sheet: 450+-16, 560+-16, 650+-16, 840+-26 nm). These are approximations.

Band Rrs = sum(Rrs(l) * SRF(l)) / sum(SRF(l)) on the 1 nm grid 350-900 nm. A band is simulated only
if >= 98 % of its SRF integral lies inside 350-900 nm, and per sample only if Rrs is finite at every
wavelength where SRF >= 1 % of its maximum.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .aquacorr import download_file  # shared resumable downloader

GLORIA_URL = "https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip"
GLORIA_DOI = "10.1594/PANGAEA.948492"
GLORIA_PAPER_DOI = "10.1038/s41597-023-01973-y"
LICENSE = "CC BY 4.0"
SRF_URLS = {
    "S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.2.xlsx":
        "https://sentinels.copernicus.eu/documents/247904/685211/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.2.xlsx",
    "L8_OLI_RSR.xlsx": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L8_OLI_RSR.xlsx",
    "L9_OLI2_RSR.xlsx": "https://landsat.usgs.gov/landsat/spectral_viewer/bands/L9_OLI2_RSR.xlsx",
}
WL = np.arange(350, 901)          # GLORIA Rrs grid (nm)
COVERAGE_MIN = 0.98
SRF_SUPPORT = 0.01

# Units from GLORIA_variables_and_methods.xlsx / Lehmann et al. 2023 Table 2.
UNITS = {"Chla": "mg/m3", "Chla_plus_phaeo": "mg/m3", "TSS": "g/m3", "aCDOM440": "1/m",
         "Secchi_depth": "m", "Turbidity": "NTU"}
TARGETS = tuple(UNITS)
# GLORIA_qc_flags.csv; ``Flagged`` is GLORIA's own composite (1,779 samples). ``Oxygen_signal`` (O2-A band
# artefact near 760 nm) is informational and not part of ``Flagged``.
QC_FLAG_COLS = ("Noisy_red", "Noisy_blue", "Baseline_shift", "Oxygen_signal", "Negative_uv_slope",
                "QWIP_fail", "Suspect", "Flagged")
WATER_BODY = {1: "lake", 2: "estuary", 3: "coastal", 4: "river", 5: "other"}

CAMERA_TOPHATS = {
    "naip_ads100": {"blue": (435, 495), "green": (525, 585), "red": (619, 651), "nir": (808, 882)},
    "dji_p4m_rgbn": {"blue": (434, 466), "green": (544, 576), "red": (634, 666), "nir": (814, 866)},
}

# ------------------------------------------------------------------------- minimal xlsx reader

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}


def _col_index(ref: str) -> int:
    n = 0
    for ch in re.match(r"([A-Z]+)", ref).group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx(path: str | Path | bytes) -> dict[str, list[list]]:
    """Sheets of an .xlsx as lists of rows (numbers as float, shared/inline strings as str).
    Minimal OOXML reader so no spreadsheet dependency is needed for the SRF tables."""
    z = zipfile.ZipFile(io.BytesIO(path) if isinstance(path, bytes) else path)
    names = set(z.namelist())
    shared = []
    if "xl/sharedStrings.xml" in names:
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
    target = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
    out = {}
    for sh in wb.find("m:sheets", _NS):
        t = target[sh.attrib[f"{{{_NS['r']}}}id"]].lstrip("/")
        t = t if t.startswith("xl/") else "xl/" + t
        rows = []
        for row in ET.fromstring(z.read(t)).iter(f"{{{_NS['m']}}}row"):
            vals = {}
            for c in row.findall("m:c", _NS):
                typ, v = c.attrib.get("t"), c.find("m:v", _NS)
                if typ == "s" and v is not None:
                    val = shared[int(v.text)]
                elif typ == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
                elif v is not None:
                    try:
                        val = float(v.text)
                    except ValueError:
                        val = v.text
                else:
                    val = None
                vals[_col_index(c.attrib["r"])] = val
            if vals:
                rows.append([vals.get(i) for i in range(max(vals) + 1)])
        out[sh.attrib["name"]] = rows
    return out


# ----------------------------------------------------------------------------------- SRFs


def _on_grid(wl: np.ndarray, resp: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """SRF interpolated onto ``grid`` (0 outside the measured range; negatives clipped to 0)."""
    o = np.argsort(wl)
    return np.clip(np.interp(grid, wl[o], resp[o], left=0.0, right=0.0), 0, None)


def load_s2_srf(path: str | Path) -> dict[str, pd.DataFrame]:
    """{'S2A': DataFrame(wavelength_nm, B1..B12), 'S2B': ...} from the ESA SRF workbook."""
    sheets = read_xlsx(path)
    out = {}
    for plat in ("S2A", "S2B"):
        rows = sheets[f"Spectral Responses ({plat})"]
        hdr = rows[0]
        body = [r for r in rows[1:] if r and isinstance(r[0], float)]
        cols = {}
        for j, h in enumerate(hdr):
            if isinstance(h, str) and h.startswith(f"{plat}_SR_AV_B"):
                cols["B" + h.split("_B")[-1].upper()] = j
        df = pd.DataFrame({"wavelength_nm": [r[0] for r in body]})
        for b, j in cols.items():
            df[b] = [float(r[j]) if j < len(r) and r[j] is not None else 0.0 for r in body]
        out[plat] = df
    return out


OLI_SHEETS = {"B1": "CoastalAerosol", "B2": "Blue", "B3": "Green", "B4": "Red", "B5": "NIR",
              "B6": "SWIR1", "B7": "SWIR2"}


def load_oli_rsr(path: str | Path) -> dict[str, pd.DataFrame]:
    """{'B1': DataFrame(wavelength_nm, rsr), ...} band-average RSRs (OLI or OLI-2 workbook)."""
    sheets = read_xlsx(path)
    out = {}
    for b, name in OLI_SHEETS.items():
        rows = [r for r in sheets[name] if r and isinstance(r[0], float) and len(r) > 1]
        out[b] = pd.DataFrame({"wavelength_nm": [r[0] for r in rows], "rsr": [float(r[1]) for r in rows]})
    return out


def srf_table(s2_xlsx: str | Path, l8_xlsx: str | Path, l9_xlsx: str | Path,
              grid: np.ndarray = WL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All sensor bands on the 1 nm GLORIA grid.

    Returns (srf[wavelength_nm, <sensor>_<band>...], info[sensor, band, coverage, centroid_nm,
    simulated]) where ``coverage`` is the fraction of the full SRF integral inside 350-900 nm.
    """
    cols: dict[str, np.ndarray] = {}
    info = []

    def add(sensor: str, band: str, wl: np.ndarray, resp: np.ndarray, source: str):
        full_grid = np.arange(int(np.floor(wl.min())), int(np.ceil(wl.max())) + 1)
        full = _on_grid(wl, resp, full_grid)
        on = _on_grid(wl, resp, grid)
        tot = full.sum()
        cov = float(on.sum() / tot) if tot > 0 else 0.0
        cen = float((full * full_grid).sum() / tot) if tot > 0 else float("nan")
        ok = cov >= COVERAGE_MIN
        info.append({"sensor": sensor, "band": band, "coverage_350_900": cov, "centroid_nm": cen,
                     "simulated": ok, "srf_source": source})
        if ok:
            cols[f"{sensor}_{band}"] = on / on.sum()

    for plat, df in load_s2_srf(s2_xlsx).items():
        for b in [c for c in df.columns if c != "wavelength_nm"]:
            add(plat, b, df.wavelength_nm.to_numpy(float), df[b].to_numpy(float), Path(s2_xlsx).name)
    for sensor, path in (("L8", l8_xlsx), ("L9", l9_xlsx)):
        for b, df in load_oli_rsr(path).items():
            add(sensor, b, df.wavelength_nm.to_numpy(float), df.rsr.to_numpy(float), Path(path).name)
    for cam, bands in CAMERA_TOPHATS.items():
        for b, (lo, hi) in bands.items():
            w = np.arange(lo - 1, hi + 2, dtype=float)
            r = ((w >= lo) & (w <= hi)).astype(float)
            add(cam, b, w, r, "top-hat from published band edges (approximation)")
    srf = pd.DataFrame({"wavelength_nm": grid, **cols})
    return srf, pd.DataFrame(info)


def simulate_bands(rrs: np.ndarray, srf: pd.DataFrame, grid: np.ndarray = WL,
                   support: float = SRF_SUPPORT) -> pd.DataFrame:
    """Band-averaged Rrs (N x bands) from 1 nm spectra (N x len(grid)); NaN where the spectrum is
    missing inside a band's support (SRF >= ``support`` x max)."""
    if not np.array_equal(srf.wavelength_nm.to_numpy(), grid):
        raise ValueError("SRF grid differs from spectrum grid")
    R = np.asarray(rrs, float)
    out = {}
    for c in srf.columns:
        if c == "wavelength_nm":
            continue
        w = srf[c].to_numpy(float)
        sup = w >= support * w.max()
        val = np.nansum(R[:, sup] * w[sup], axis=1) / w[sup].sum()
        bad = ~np.isfinite(R[:, sup]).all(axis=1)
        val[bad] = np.nan
        out[f"rrs_{c}"] = val
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------- GLORIA


def read_gloria(zip_path: str | Path) -> dict[str, pd.DataFrame]:
    """Tables from GLORIA-2022.zip: meta (metadata + lab), rrs (1 nm), rrs_std, qc, uncert."""
    z = zipfile.ZipFile(zip_path)
    names = {Path(n).name: n for n in z.namelist()}

    def csv(name):
        with z.open(names[name]) as fh:
            return pd.read_csv(fh, low_memory=False)
    out = {"meta": csv("GLORIA_meta_and_lab.csv"), "rrs": csv("GLORIA_Rrs.csv"),
           "qc": csv("GLORIA_qc_flags.csv")}
    for opt, key in (("GLORIA_Rrs_std.csv", "rrs_std"), ("GLORIA_waterqual_uncert.csv", "uncert"),
                     ("GLORIA_qc_ancillary.csv", "qc_ancillary")):
        if opt in names:
            out[key] = csv(opt)
    return out


def rrs_matrix(rrs: pd.DataFrame, grid: np.ndarray = WL) -> np.ndarray:
    cols = [f"Rrs_{w}" for w in grid]
    missing = [c for c in cols if c not in rrs.columns]
    if missing:
        raise KeyError(f"GLORIA Rrs columns missing: {missing[:5]}")
    return rrs[cols].to_numpy(float)


def tidy_samples(tables: dict[str, pd.DataFrame], sim: pd.DataFrame) -> pd.DataFrame:
    """One row per GLORIA_ID: identifiers, coordinates, time, water-body type, targets + units,
    QC flags (+ ``any_qc_flag``) and simulated band Rrs (``sim`` aligned with tables['rrs'])."""
    meta = tables["meta"].copy()
    qc = tables["qc"].dropna(subset=["GLORIA_ID"]).copy()  # the CSV ends with an empty row
    rrs_ids = tables["rrs"][["GLORIA_ID"]].reset_index(drop=True)
    sim = pd.concat([rrs_ids, sim.reset_index(drop=True)], axis=1)
    df = meta.merge(qc, on="GLORIA_ID", how="left").merge(sim, on="GLORIA_ID", how="left")
    df["GLORIA_ID"] = df["GLORIA_ID"].astype(str)
    df["Dataset_ID"] = df["Dataset_ID"].astype(str)
    flags = [c for c in QC_FLAG_COLS if c in df.columns]
    for c in flags:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int).astype(bool)
    df["any_qc_flag"] = df["Flagged"] if "Flagged" in df else df[flags].any(axis=1)
    df["time_utc"] = pd.to_datetime(df["Date_Time_UTC"], errors="coerce", utc=True)
    wbt = pd.to_numeric(df.get("Water_body_type"), errors="coerce")
    df["water_body_type_name"] = wbt.map(WATER_BODY)
    for t, u in UNITS.items():
        if t in df.columns:
            df[t] = pd.to_numeric(df[t], errors="coerce")
            df[f"{t}_unit"] = np.where(df[t].notna(), u, None)
    return df


def summarize(df: pd.DataFrame, info: pd.DataFrame) -> dict:
    s = {"n_samples": int(len(df)), "n_datasets": int(df.Dataset_ID.nunique()),
         "n_site_names": int(df.Site_name.nunique()) if "Site_name" in df else None,
         "any_qc_flag": int(df.any_qc_flag.sum()),
         "targets": {t: int(df[t].notna().sum()) for t in TARGETS if t in df},
         "targets_unflagged": {t: int((df[t].notna() & ~df.any_qc_flag).sum()) for t in TARGETS if t in df},
         "time_range": [str(df.time_utc.min()), str(df.time_utc.max())],
         "water_body_types": df.water_body_type_name.value_counts().to_dict(),
         "bands_simulated": info[info.simulated].groupby("sensor").band.apply(list).to_dict(),
         "bands_not_simulated": info[~info.simulated].groupby("sensor").band.apply(list).to_dict()}
    sim_cols = [c for c in df.columns if c.startswith("rrs_")]
    s["simulated_finite"] = {c: int(df[c].notna().sum()) for c in sim_cols}
    return s


def download_all(raw_dir: Path, srf_dir: Path, log=print) -> list[Path]:
    """GLORIA zip (no published checksum; sha256 recorded in the manifest) + SRF workbooks."""
    files = [download_file(GLORIA_URL, Path(raw_dir) / "GLORIA-2022.zip", None, log=log)]
    for name, url in SRF_URLS.items():
        files.append(download_file(url, Path(srf_dir) / name, None, log=log))
    return files


def iter_targets() -> Iterable[str]:
    return iter(TARGETS)
