# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""NOAA NCCOS western Lake Erie cyanoHAB bloom Severity Index (SI): every published vintage.

The SI is a basin-scale annual label (n ≈ 26 years) derived from satellite Cyanobacteria Index
(CI) composites: SI = 10·(log10 CI_avg30 − log10 CI_avg30[2005]) / (log10 CI_avg30[2011] −
log10 CI_avg30[2005]). **The series has been revised**: after the 2024 season NOAA reprocessed
the MERIS/MODIS/OLCI record (HAB_LE_TimeseriesUpdate_2025.pdf; recalibration note 2026-04-22),
so the same year has different values in different files. This client therefore keeps every
vintage side by side and never overwrites one with another:

* ``curated_2000to2025`` — NOAA_NCCOS_2000to2025_Curated_LE_Annual_Severity.xlsx (current);
* ``revised_2000to2024`` — NOAA_NCCOS_LE_Bloom_SeverityIndex_Revised2000to2024.xlsx (first
  reprocessed release, mid-2025);
* ``as_issued`` — the value stated in each season's final-assessment bulletin (pre-reprocessing
  values, 2018–2024 seasons, plus values for earlier years quoted in those bulletins);
* seasonal forecasts (point + likely range) from the July seasonal-forecast bulletins — the
  operational baseline for any seasonal-severity model.

Bulletin text is extracted with poppler ``pdftotext -layout``; old bulletins have kerning
artefacts ("sev erityind ex"), so all patterns are matched on whitespace-stripped lowercase text.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import events_common as C

SOURCE = "noaa_nccos_lake_erie_si"
PRODUCT_PAGE = "https://coastalscience.noaa.gov/science-areas/habs/hab-forecasts/lake-erie/"
_BLOB = "https://nccospublicstor.blob.core.windows.net/hab-data/bulletins/lake-erie"
_CDN = "https://cdn.coastalscience.noaa.gov/hab-data/bulletins/lake-erie"
WAYBACK = "https://web.archive.org/web/{ts}id_/{url}"

# Severity-index spreadsheets, newest first. ``label_vintage`` is the identifier stored with every row.
SEVERITY_FILES = [
    {"label_vintage": "curated_2000to2025",
     "url": f"{_BLOB}/2025/NOAA_NCCOS_2000to2025_Curated_LE_Annual_Severity.xlsx"},
    {"label_vintage": "revised_2000to2024",
     "url": f"{_CDN}/2025/NOAA_NCCOS_LE_Bloom_SeverityIndex_Revised2000to2024.xlsx"},
]

# Bulletins: kind ∈ {final_assessment, seasonal_forecast, timeseries_update, recalibration}.
# ``wayback`` = Internet Archive timestamp used only when the official URL no longer resolves.
BULLETINS = [
    {"season": 2018, "kind": "final_assessment", "url": f"{_CDN}/2018/bulletin_2018-final.pdf"},
    {"season": 2019, "kind": "final_assessment", "url": f"{_CDN}/2019/bulletin_2019-final.pdf"},
    {"season": 2020, "kind": "final_assessment", "url": f"{_CDN}/2020/finalAssessment_2020-09.pdf"},
    {"season": 2021, "kind": "final_assessment", "url": f"{_CDN}/2021/finalAssessment_2021-08.pdf"},
    {"season": 2022, "kind": "final_assessment", "url": f"{_CDN}/2022/FinalAssessment_2022_09.pdf"},
    {"season": 2023, "kind": "final_assessment", "url": f"{_CDN}/2023/FinalAssessment_2023_10.pdf"},
    {"season": 2024, "kind": "final_assessment", "url": f"{_CDN}/2024/FinalAssessment_2024_9.pdf"},
    {"season": 2025, "kind": "final_assessment", "url": f"{_CDN}/2025/SeasonalAssessment_2025_08.pdf",
     "wayback": "20251205085730"},
    {"season": 2018, "kind": "seasonal_forecast", "url": f"{_CDN}/2018/SeasonalForecast_2018-08.pdf"},
    {"season": 2020, "kind": "seasonal_forecast", "url": f"{_BLOB}/2020/seasonalForecast_2020-08.pdf"},
    {"season": 2021, "kind": "seasonal_forecast", "url": f"{_CDN}/2021/seasonalForecast_2021-07.pdf"},
    {"season": 2023, "kind": "seasonal_forecast", "url": f"{_CDN}/2023/SeasonalForecast_2023_08.pdf"},
    {"season": 2025, "kind": "seasonal_forecast", "url": f"{_BLOB}/2025/SeasonalForecast_2025_06.pdf",
     "wayback": "20250705130513"},
    {"season": 2026, "kind": "seasonal_forecast", "url": f"{_CDN}/2026/SeasonalForecast_20260625.pdf"},
    {"season": 2025, "kind": "timeseries_update", "url": f"{_CDN}/2025/HAB_LE_TimeseriesUpdate_2025.pdf"},
    {"season": 2026, "kind": "recalibration",
     "url": f"{_CDN}/2026/NOAA_NCCOS_2026_LakeErieSeasonal_ModelRecalibration.pdf"},
]

SI_CATEGORY_NOTE = ("NOAA 2025+ categorical descriptors (none/mild/moderate/severe/very severe) are shown "
                    "in figures only; numeric SI is stored, categories are not invented here.")

_NUM = r"(\d+(?:\.\d+)?)"
SI_FLOOR = 0.3


# ----------------------------------------------------------------------------- spreadsheets

def parse_severity_xlsx(src: str | Path | bytes, label_vintage: str) -> tuple[pd.DataFrame, dict]:
    """One SI spreadsheet → (tidy rows, metadata-sheet dict). Column names kept as published."""
    names = C.xlsx_sheet_names(src)
    data_sheet = next(n for n in names if n.lower() != "metadata")
    df = C.read_xlsx(src, data_sheet)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({"year": pd.to_numeric(df["year"], errors="coerce").astype("Int64")})
    for col in ("total_CI_max", "total_area_km2_max", "total_CI_avg_30d", "total_area_km2_avg_30d",
                "severity_index"):
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    if "Date_max" in df:
        out["date_ci_max"], out["date_ci_max_flag"] = C.clean_dates(df["Date_max"], min_date="2000-01-01")
    else:
        out["date_ci_max"], out["date_ci_max_flag"] = pd.NaT, "absent_in_vintage"
    out = out[out["year"].notna()].reset_index(drop=True)
    # NOAA floors SI at 0.3 ("If the calculated SI<0.3, then the SI was set to 0.3"): left-censored.
    out["si_floor_censored"] = out["severity_index"].le(SI_FLOOR + 1e-9)
    out.insert(0, "label_vintage", label_vintage)
    meta: dict[str, str] = {}
    if any(n.lower() == "metadata" for n in names):
        m = C.read_xlsx(src, next(n for n in names if n.lower() == "metadata"), header=None)
        key = ""
        for row in m.itertuples(index=False):
            cells = [str(x).strip() for x in row]
            if cells[0]:
                key = cells[0].rstrip(":")
            val = " | ".join(x for x in cells[1:] if x)
            meta[key] = (meta[key] + " || " + val) if key in meta else val
    return out, meta


# ----------------------------------------------------------------------------- bulletin text

def pdf_to_text(pdf: str | Path) -> str:
    """Text of a PDF via poppler ``pdftotext -layout`` (raises if poppler is missing)."""
    exe = shutil.which("pdftotext")
    if exe is None:
        raise RuntimeError("pdftotext (poppler-utils) not found; needed to read NOAA bulletins")
    res = subprocess.run([exe, "-layout", str(pdf), "-"], capture_output=True, timeout=120, check=True)
    return res.stdout.decode("utf-8", errors="replace")


def squash(text: str) -> str:
    """Lowercase, typographic quotes normalised, all whitespace removed (kerning-proof matching)."""
    t = text.replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", "", t.lower())


_PRIMARY = [
    re.compile(r"bloomin((?:19|20)\d{2})hadaseverityindex\(si\)of" + _NUM),
    re.compile(r"((?:19|20)\d{2})(?:western)?lakeeriecyanobacterialbloomhadaseverityindex\(si\)of" + _NUM),
    re.compile(r"((?:19|20)\d{2})(?:western)?lakeerie(?:cyanobacterial|cyanohab|hab)?bloom"
               r"(?:had|has)a(?:final)?severity(?:index)?(?:\(si\))?of" + _NUM),
]
_MENTION = [
    re.compile(r"((?:19|20)\d{2})hadaseverity(?:index)?of" + _NUM),
    re.compile(r"((?:19|20)\d{2})(?:was|were)" + _NUM + r"(?![\d.]*%)"),
    re.compile(r"than(?:the\w{0,12}bloomof|in)?((?:19|20)\d{2})\((?:si=|siof)?" + _NUM),
    re.compile(r"((?:19|20)\d{2})\((?:si=|siof)" + _NUM),
    re.compile(r"((?:19|20)\d{2}),whichhadaseverityof" + _NUM),
]
_LARGEST = re.compile(r"thelargestblooms,((?:19|20)\d{2})and((?:19|20)\d{2}),were" + _NUM + r"and" + _NUM)


def _clean_num(v: str) -> float:
    return float(v.rstrip("."))


def parse_final_assessment(text: str, season: int) -> pd.DataFrame:
    """As-issued SI statements in a season-end bulletin.

    ``statement_type='primary'`` = the bulletin's own season; ``'mentioned'`` = values it quotes
    for other years (these reveal the pre-reprocessing scale, e.g. 2015 = 10.5).
    """
    s = squash(text)
    rows = []
    for pat in _PRIMARY:
        for m in pat.finditer(s):
            rows.append({"year": int(m.group(1)), "si": _clean_num(m.group(2)), "statement_type": "primary",
                         "context": s[max(0, m.start() - 20): m.end() + 30]})
    for pat in _MENTION:
        for m in pat.finditer(s):
            yr, v = int(m.group(1)), _clean_num(m.group(2))
            if 0 <= v <= 15:
                rows.append({"year": yr, "si": v, "statement_type": "mentioned",
                             "context": s[max(0, m.start() - 30): m.end() + 20]})
    for m in _LARGEST.finditer(s):
        for yr, v in ((m.group(1), m.group(3)), (m.group(2), m.group(4))):
            rows.append({"year": int(yr), "si": _clean_num(v), "statement_type": "mentioned",
                         "context": m.group(0)})
    df = pd.DataFrame(rows, columns=["year", "si", "statement_type", "context"])
    if df.empty:
        return df.assign(season=pd.Series(dtype="Int64"))
    # a primary statement about the bulletin's own season wins over a mention of the same value
    df.loc[(df["statement_type"] == "mentioned") & (df["year"] == season)
           & df["si"].isin(df.loc[df["statement_type"] == "primary", "si"]), "statement_type"] = "dup"
    df = df[df["statement_type"] != "dup"]
    df = df.drop_duplicates(subset=["year", "si", "statement_type"]).reset_index(drop=True)
    df.loc[(df["statement_type"] == "primary") & (df["year"] != season), "statement_type"] = "mentioned"
    df.insert(0, "season", season)
    return df


_FORECAST = re.compile(
    r"(?:severity(?:index)?of(?:about)?~?|measure)" + _NUM + r"(?:ontheseverityindex)?,?"
    r"(?:butthiscould|butcould|withapotential|andapotential)range(?:between|of)"
    + _NUM + r"(?:-|and|to)" + _NUM)
_ISO_DATE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_LONG_DATE = re.compile(r"(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|"
                        r"November|December)\s+20\d{2})")


def parse_seasonal_forecast(text: str, season: int) -> dict:
    """Seasonal-forecast bulletin → issue date, forecast SI point and likely range."""
    head = text[:600]
    m_iso, m_long = _ISO_DATE.search(head), _LONG_DATE.search(head)
    issued = pd.NaT
    if m_iso:
        issued = pd.to_datetime(m_iso.group(1), errors="coerce")
    elif m_long:
        issued = pd.to_datetime(m_long.group(1), format="%d %B %Y", errors="coerce")
    m = _FORECAST.search(squash(text))
    return {
        "season": season, "issue_date": issued,
        "forecast_si": _clean_num(m.group(1)) if m else np.nan,
        "forecast_si_lo": _clean_num(m.group(2)) if m else np.nan,
        "forecast_si_hi": _clean_num(m.group(3)) if m else np.nan,
        "parse_ok": bool(m),
        "context": m.group(0) if m else None,
    }


# ----------------------------------------------------------------------------- assembly

def severity_long(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Stack vintages (one row per vintage × year)."""
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "source", SOURCE)
    return df.sort_values(["label_vintage", "year"]).reset_index(drop=True)


def vintage_comparison(long: pd.DataFrame, as_issued: pd.DataFrame | None = None) -> pd.DataFrame:
    """Year × vintage SI table (wide) with the as-issued primary value when available."""
    wide = long.pivot_table(index="year", columns="label_vintage", values="severity_index", aggfunc="first")
    wide.columns = [f"si_{c}" for c in wide.columns]
    wide = wide.reset_index()
    if as_issued is not None and not as_issued.empty:
        prim = (as_issued[as_issued["statement_type"] == "primary"]
                .drop_duplicates("year")[["year", "si"]].rename(columns={"si": "si_as_issued"}))
        wide = wide.merge(prim, on="year", how="left")
    wide["year"] = wide["year"].astype("Int64")
    return wide.sort_values("year").reset_index(drop=True)
