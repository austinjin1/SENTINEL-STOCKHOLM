# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Dreissenid mussel (zebra / quagga) labels for MusselGuard (RESEARCH_PLAN §8, D6).

Two public sources, both parsed into tidy tables:

* **USGS Nonindigenous Aquatic Species (NAS) database**, API v2
  (``https://nas.er.usgs.gov/api/v2/occurrence/search``). Presence-only occurrence records
  with a *current* population status (established, collected, failed, extirpated, ...),
  record type, coordinates with an accuracy class, and WBD HUC8/10/12 codes. The API does
  not return the record's temporal-accuracy class, so it is recovered by querying each
  ``yrAcc`` class separately (Actual / Estimated / Publication year) and tagging keys.
  Species IDs (verified via ``/species/{id}``): 5 = *Dreissena polymorpha* (zebra mussel),
  95 = *Dreissena bugensis* (quagga mussel; NAS keeps this name for
  *D. rostriformis bugensis*), 3118 = *Dreissena* sp. (unidentified dreissenid).
* **Minnesota DNR Infested Waters List** (``infested-waters.xlsx``): one row per water body
  × invasive species, with the year the water body was listed and either the year the
  species was first confirmed or a note that it was listed because it is *connected* to an
  infested water (an administrative listing, not a detection).

Parsing functions are pure (text/bytes in, DataFrame out) so they can be tested on recorded
fixtures; only ``nas_fetch_pages`` / ``nas_year_accuracy_keys`` touch the network.
"""

from __future__ import annotations

import calendar
import datetime as _dt
import html as _html
import re
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from . import events_common as C

# ----------------------------------------------------------------------------- USGS NAS

NAS_API = "https://nas.er.usgs.gov/api/v2"
NAS_SEARCH_URL = f"{NAS_API}/occurrence/search"
NAS_DOC_URL = "https://nas.er.usgs.gov/api/documentation.aspx"
NAS_DISCLAIMER_URL = "https://nas.er.usgs.gov/disclaimer.aspx"
NAS_DBCHAR_URL = "https://nas.er.usgs.gov/about/DBCharacterization.aspx"
NAS_SPECIES = {
    5: ("Dreissena polymorpha", "zebra mussel"),
    95: ("Dreissena bugensis", "quagga mussel"),
    3118: ("Dreissena sp.", "a dreissenid mussel"),
}
YR_ACC_CLASSES = ("Actual", "Estimated", "Publication year")
ESTABLISHED_STATUSES = {"established", "locally established"}
NAS_CITATION = ("U.S. Geological Survey. [{year}]. Nonindigenous Aquatic Species Database. "
                "Gainesville, Florida. Accessed [{accessed}].")
# North America incl. Alaska and Canadian provinces (W, S, E, N)
NAS_BBOX = (-170.0, 15.0, -50.0, 75.0)

US_STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "Puerto Rico": "PR", "Guam": "GU", "U.S. Virgin Islands": "VI", "Virgin Islands": "VI",
    "American Samoa": "AS", "Northern Mariana Islands": "MP",
}
CA_PROVINCES = {
    "ON": "ON", "ONT": "ON", "Ontario": "ON", "QC": "QC", "QUE": "QC", "PQ": "QC", "Quebec": "QC",
    "MB": "MB", "MAN": "MB", "Manitoba": "MB", "SK": "SK", "Saskatchewan": "SK", "AB": "AB",
    "Alberta": "AB", "BC": "BC", "British Columbia": "BC", "NB": "NB", "New Brunswick": "NB",
    "NS": "NS", "Nova Scotia": "NS", "PE": "PE", "Prince Edward Island": "PE", "NL": "NL",
    "Newfoundland and Labrador": "NL", "YT": "YT", "NT": "NT", "NU": "NU",
}


def normalize_state(value: str | None) -> tuple[str | None, str | None]:
    """NAS ``state`` (US full names, Canadian codes such as 'ON'/'ONT') → (code, country)."""
    v = (value or "").strip()
    if not v:
        return None, None
    if v in US_STATES:
        return US_STATES[v], "US"
    if v.upper() in set(US_STATES.values()) and v.upper() not in CA_PROVINCES:
        return v.upper(), "US"
    if v in CA_PROVINCES or v.upper() in CA_PROVINCES:
        return CA_PROVINCES.get(v, CA_PROVINCES.get(v.upper())), "CA"
    if v.lower() == "mexico":  # NAS uses the country name for records in Mexico
        return None, "MX"
    return None, None


def nas_fetch_pages(species_id: int, *, page_size: int = 5000, sleep: float = 1.0,
                    params: dict | None = None, max_pages: int = 500) -> list[dict]:
    """All result pages (raw API envelopes) of an occurrence search for one species.

    The API documents results as "ordered by species ID and occurrence record ID", so offset
    paging is stable. ``endOfRecords`` is the string 'true'/'false'; ``count`` is the page
    length, not the total. The loop advances by the number of results actually returned.
    """
    pages: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        q = {"species_ID": species_id, "offset": offset, "limit": page_size}
        if params:
            q.update(params)
        d = C.get_json(NAS_SEARCH_URL, q, timeout=600)
        res = d.get("results") or []
        pages.append(d)
        offset += len(res)
        if str(d.get("endOfRecords")).strip().lower() == "true" or not res:
            return pages
        time.sleep(sleep)
    raise RuntimeError(f"NAS paging did not terminate after {max_pages} pages (species {species_id})")


def records_from_pages(pages: Iterable[dict]) -> tuple[list[dict], dict]:
    """Concatenate page results; drop exact duplicate keys; return (records, paging report)."""
    pages = list(pages)
    recs: list[dict] = []
    seen: dict[int, int] = {}
    n_raw = 0
    for p in pages:
        for r in p.get("results") or []:
            n_raw += 1
            k = r.get("key")
            if k in seen:
                seen[k] += 1
                continue
            seen[k] = 1
            recs.append(r)
    dups = {k: v for k, v in seen.items() if v > 1}
    report = {"n_pages": len(pages), "n_results_raw": n_raw, "n_unique_keys": len(recs),
              "n_duplicate_keys": len(dups),
              "last_end_of_records": str((pages[-1] if pages else {}).get("endOfRecords"))}
    return recs, report


def nas_year_accuracy_keys(species_id: int, *, page_size: int = 5000, sleep: float = 1.0
                           ) -> dict[str, list[int]]:
    """Keys of the records in each NAS temporal-accuracy class (``yrAcc`` query filter)."""
    out = {}
    for cls in YR_ACC_CLASSES:
        recs, _ = records_from_pages(nas_fetch_pages(species_id, page_size=page_size, sleep=sleep,
                                                     params={"yrAcc": cls}))
        out[cls] = sorted(int(r["key"]) for r in recs)
        time.sleep(sleep)
    return out


def year_accuracy_map(classes: dict[str, Iterable[int]]) -> tuple[dict[int, str], list[int]]:
    """{class: keys} → ({key: class}, conflicting keys). Conflicts get the label 'conflict'."""
    m: dict[int, str] = {}
    conflicts = []
    for cls, keys in classes.items():
        for k in keys:
            k = int(k)
            if k in m and m[k] != cls:
                conflicts.append(k)
                m[k] = "conflict"
            else:
                m.setdefault(k, cls)
    return m, sorted(set(conflicts))


def _blank_to_none(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    return s or None


def _int_or_none(v):
    v = _blank_to_none(v)
    if v is None:
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def _add_flag(flag: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    """Append ``label`` (';'-separated) to ``flag`` where ``mask`` is True."""
    flag = flag.copy()
    flag[mask] = [f"{f};{label}" if f else label for f in flag[mask]]
    return flag


def nas_date_fields(year, month, day) -> dict:
    """Year/month/day → exact date (day precision only), an uncertainty interval and precision.

    No day is invented: month-precision records get ``date`` = NaT and
    ``date_start``/``date_end`` = first/last day of that month; year-only records span the year.
    Impossible calendar values (month 13, day 31 in a 30-day month) degrade the precision.
    """
    y, m, d = _int_or_none(year), _int_or_none(month), _int_or_none(day)
    out = {"date": pd.NaT, "date_start": pd.NaT, "date_end": pd.NaT, "date_precision": "none"}
    if y is None or not (1700 <= y <= _dt.date.today().year + 1):
        return out
    if m is not None and 1 <= m <= 12:
        last = calendar.monthrange(y, m)[1]
        if d is not None and 1 <= d <= last:
            ts = pd.Timestamp(y, m, d)
            return {"date": ts, "date_start": ts, "date_end": ts, "date_precision": "day"}
        return {"date": pd.NaT, "date_start": pd.Timestamp(y, m, 1),
                "date_end": pd.Timestamp(y, m, last), "date_precision": "month"}
    return {"date": pd.NaT, "date_start": pd.Timestamp(y, 1, 1), "date_end": pd.Timestamp(y, 12, 31),
            "date_precision": "year"}


NAS_COLUMNS = [
    "source", "record_id", "nas_key", "species_id", "scientific_name", "common_name",
    "date", "date_start", "date_end", "date_precision", "year", "month", "day", "date_raw",
    "year_accuracy", "status", "status_raw", "is_established", "record_type",
    "latitude", "longitude", "coord_flag", "latlong_source", "latlong_accuracy", "centroid_type",
    "state", "state_raw", "country", "county", "locality",
    "huc8", "huc8_name", "huc10", "huc10_name", "huc12", "huc12_name", "huc_flag",
    "jan1_date_suspect", "fresh_marine_intro", "n_references", "reference_keys", "reference_year_min",
    "comments", "museum_cat_number", "disposal", "uuid", "group", "family", "genus", "species",
]


def parse_nas_records(records: Iterable[dict], year_accuracy: dict[int, str] | None = None
                      ) -> pd.DataFrame:
    """NAS occurrence JSON records → tidy table (one row per occurrence key)."""
    rows = []
    for r in records:
        key = int(r["key"])
        state_raw = _blank_to_none(r.get("state"))
        code, country = normalize_state(state_raw)
        status_raw = r.get("status")
        status = (_blank_to_none(status_raw) or "").lower() or None
        refs = r.get("references") or []
        ref_keys = [str(x.get("key")) for x in refs if x.get("key") is not None]
        ref_years = [x.get("year") for x in refs if isinstance(x.get("year"), int)]
        row = {
            "source": "usgs_nas", "record_id": str(key), "nas_key": str(key),
            "species_id": str(r.get("speciesID")),
            "scientific_name": _blank_to_none(r.get("scientificName")),
            "common_name": _blank_to_none(r.get("commonName")),
            "year": _int_or_none(r.get("year")), "month": _int_or_none(r.get("month")),
            "day": _int_or_none(r.get("day")), "date_raw": _blank_to_none(r.get("date")),
            "year_accuracy": (year_accuracy or {}).get(key, "untagged") if year_accuracy is not None
            else None,
            "status": status, "status_raw": status_raw if status_raw is not None else None,
            "is_established": status in ESTABLISHED_STATUSES,
            "record_type": _blank_to_none(r.get("recordType")),
            "lat_raw": r.get("decimalLatitude"), "lon_raw": r.get("decimalLongitude"),
            "latlong_source": _blank_to_none(r.get("latLongSource")),
            "latlong_accuracy": _blank_to_none(r.get("latLongAccuracy")),
            "centroid_type": _blank_to_none(r.get("Centroid Type")),
            "state": code, "state_raw": state_raw, "country": country,
            "county": _blank_to_none(r.get("county")), "locality": _blank_to_none(r.get("locality")),
            "huc8": r.get("huc8"), "huc8_name": _blank_to_none(r.get("huc8Name")),
            "huc10": r.get("huc10"), "huc10_name": _blank_to_none(r.get("huc10Name")),
            "huc12": r.get("huc12"), "huc12_name": _blank_to_none(r.get("huc12Name")),
            "fresh_marine_intro": _blank_to_none(r.get("freshMarineIntro")),
            "n_references": len(refs), "reference_keys": ";".join(ref_keys) or None,
            "reference_year_min": min(ref_years) if ref_years else None,
            "comments": _blank_to_none(r.get("comments")),
            "museum_cat_number": _blank_to_none(r.get("museumCatNumber")),
            "disposal": _blank_to_none(r.get("disposal")), "uuid": _blank_to_none(r.get("UUID")),
            "group": _blank_to_none(r.get("group")), "family": _blank_to_none(r.get("family")),
            "genus": _blank_to_none(r.get("genus")), "species": _blank_to_none(r.get("species")),
        }
        row.update(nas_date_fields(r.get("year"), r.get("month"), r.get("day")))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=NAS_COLUMNS)
    if df["nas_key"].duplicated().any():
        raise ValueError("duplicate NAS keys in input records")
    lat, lon, flag = C.clean_latlon(df.pop("lat_raw"), df.pop("lon_raw"), bbox=NAS_BBOX)
    df["latitude"], df["longitude"], df["coord_flag"] = lat, lon, flag
    huc_flag = pd.Series("", index=df.index, dtype="object")
    for col, width in (("huc8", 8), ("huc10", 10), ("huc12", 12)):
        s = C.str_id(df[col], width=width)
        zero = s.str.fullmatch(r"0+").fillna(False).astype(bool)  # NAS placeholder for 'no WBD unit'
        bad = s.notna() & ~s.str.fullmatch(rf"\d{{{width}}}").fillna(False).astype(bool)
        huc_flag = _add_flag(huc_flag, zero, f"{col}_zero_placeholder")
        huc_flag = _add_flag(huc_flag, bad, f"{col}_malformed")
        df[col] = s.mask(zero | bad)
    df["huc_flag"] = huc_flag
    # 1 January dates are mostly literature records that give only a year (see datasheet)
    df["jan1_date_suspect"] = (df["month"] == 1) & (df["day"] == 1)
    for col in ("year", "month", "day", "n_references", "reference_year_min"):
        df[col] = df[col].astype("Int64")
    for col in ("date", "date_start", "date_end"):
        df[col] = pd.to_datetime(df[col])
    order = np.lexsort((df["nas_key"].astype(int).to_numpy(), df["species_id"].astype(int).to_numpy()))
    return df[NAS_COLUMNS].iloc[order].reset_index(drop=True)


def nas_references(records: Iterable[dict]) -> pd.DataFrame:
    """Long table of the literature/database references cited by each occurrence."""
    rows = []
    for r in records:
        for x in r.get("references") or []:
            rows.append({"nas_key": str(r["key"]), "species_id": str(r.get("speciesID")),
                         "ref_key": None if x.get("key") is None else str(x.get("key")),
                         "ref_type": _blank_to_none(x.get("refType")),
                         "ref_year": _int_or_none(x.get("year")),
                         "author": _blank_to_none(x.get("author")), "title": _blank_to_none(x.get("title")),
                         "publisher": _blank_to_none(x.get("publisher")),
                         "publisher_location": _blank_to_none(x.get("publisherLocation"))})
    df = pd.DataFrame(rows, columns=["nas_key", "species_id", "ref_key", "ref_type", "ref_year", "author",
                                     "title", "publisher", "publisher_location"])
    df["ref_year"] = df["ref_year"].astype("Int64")
    return df


SPECIES_GROUPS = {"zebra": ("5",), "quagga": ("95",), "dreissena_any": ("5", "95", "3118")}


def first_detection_by_huc12(occ: pd.DataFrame) -> pd.DataFrame:
    """Earliest NAS record year per (species group, HUC12) for rolling-origin forecast tests.

    ``first_year_any``: earliest record of any status; ``first_year_actual``: earliest record
    whose year accuracy is 'Actual'; ``first_year_established``: earliest record whose
    *current* NAS status is established/locally established. Status is the status at download
    time, not at the record date, so ``first_year_established`` is not a historical
    establishment date. Ties on the earliest year are broken by the smallest NAS key.
    """
    out = []
    base = occ[occ["huc12"].notna() & occ["year"].notna()].copy()
    base["_key"] = base["nas_key"].astype(int)
    for g, ids in SPECIES_GROUPS.items():
        sub = base[base["species_id"].isin(ids)].sort_values(["huc12", "year", "_key"])
        if sub.empty:
            continue
        grp = sub.groupby("huc12", sort=True)
        first = grp.head(1).set_index("huc12")
        res = pd.DataFrame({
            "species_group": g,
            "first_year_any": grp["year"].min(),
            "first_key_any": first["nas_key"],
            "first_status_any": first["status"],
            "n_records": grp.size(),
            "n_established": grp["is_established"].sum().astype(int),
        })
        act = sub[sub["year_accuracy"] == "Actual"] if "year_accuracy" in sub else sub.iloc[0:0]
        res["first_year_actual"] = act.groupby("huc12")["year"].min()
        est = sub[sub["is_established"]]
        res["first_year_established"] = est.groupby("huc12")["year"].min()
        res["first_key_established"] = est.groupby("huc12").head(1).set_index("huc12")["nas_key"]
        res["huc8"] = res.index.str[:8]
        res["states"] = grp["state"].agg(lambda s: ";".join(sorted({x for x in s.dropna()})) or None)
        res["huc12_name"] = first["huc12_name"]
        out.append(res.reset_index())
    cols = ["species_group", "huc12", "huc12_name", "huc8", "states", "first_year_any", "first_key_any",
            "first_status_any", "first_year_actual", "first_year_established", "first_key_established",
            "n_records", "n_established"]
    if not out:
        return pd.DataFrame(columns=cols)
    df = pd.concat(out, ignore_index=True)[cols]
    for c in ("first_year_any", "first_year_actual", "first_year_established", "n_records", "n_established"):
        df[c] = df[c].astype("Int64")
    for c in ("huc12", "huc8", "first_key_any", "first_key_established"):
        df[c] = df[c].astype("string")
    return df


# ----------------------------------------------------------------------------- Minnesota DNR

MN_XLSX_URL = "https://files.dnr.state.mn.us/eco/invasives/infested-waters.xlsx"
MN_PAGE_URL = "https://www.dnr.state.mn.us/invasives/ais/infested.html"
MN_TERMS_URL = "https://www.dnr.state.mn.us/aboutdnr/disclaimers_and_policies.html"
MN_COLS = {
    "Water body name": "waterbody_name",
    "County or counties": "county_raw",
    "Listed for aquatic invasive species": "species_raw",
    "Year listed as infested": "year_listed_raw",
    "Year species was first confirmed, or connected water body": "confirmed_or_connected_raw",
    "DOW number": "dow_raw",
}
MN_PLACEHOLDERS = {"no data"}
MN_SPECIES = {  # common name on the list → scientific name (taxonomy as commonly used by MN DNR)
    "zebra mussel": "Dreissena polymorpha",
    "eurasian watermilfoil": "Myriophyllum spicatum",
    "spiny waterflea": "Bythotrephes longimanus",
    "faucet snail": "Bithynia tentaculata",
    "flowering rush": "Butomus umbellatus",
    "bighead carp": "Hypophthalmichthys nobilis",
    "silver carp": "Hypophthalmichthys molitrix",
    "starry stonewort": "Nitellopsis obtusa",
    "grass carp": "Ctenopharyngodon idella",
    "brittle naiad": "Najas minor",
    "ruffe": "Gymnocephalus cernua",
    "white perch": "Morone americana",
    "round goby": "Neogobius melanostomus",
    "viral hemorrhagic septicemia (vhs)": "Piscine novirhabdovirus (VHSV)",
    "new zealand mud snail": "Potamopyrgus antipodarum",
    "red swamp crayfish": "Procambarus clarkii",
    "brazilian waterweed": "Egeria densa",
}
_MN_SPECIES_ALIASES = {"vhs": "viral hemorrhagic septicemia (vhs)",
                       "viral hemorrhagic septicemia": "viral hemorrhagic septicemia (vhs)",
                       "zebra mussels": "zebra mussel"}
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def mn_list_version(title: str) -> str | None:
    """'... List of Infested Waters -  October 21, 2026' → '2026-10-21'."""
    m = re.search(rf"({_MONTHS})\s+(\d{{1,2}}),\s*(\d{{4}})", title or "")
    if not m:
        return None
    return _dt.datetime.strptime(f"{m[1]} {m[2]} {m[3]}", "%B %d %Y").date().isoformat()


def normalize_mn_species(text: str | None) -> tuple[str | None, str | None]:
    """List species text → (normalised common name, scientific name)."""
    s = re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip().lower()
    if not s or s in MN_PLACEHOLDERS:
        return None, None
    s = _MN_SPECIES_ALIASES.get(s, s)
    return s, MN_SPECIES.get(s)


def parse_mn_dow(text: str | None) -> tuple[str | None, str | None, str]:
    """DOW number → (dow 'CC-LLLL[-BB]', 8-digit dow8 'CCLLLLBB', dow_class).

    The list uses 'CC-LLLL' (whole lake) or 'CC-LLLL-BB' (basin); MN DNR datasets use the
    8-digit form (basin '00' = whole lake), e.g. 'Villard (61006700)' in the list itself.
    Classes: 'dow', 'river_na' ('NA': rivers/streams on the public waters inventory),
    'none' (ponds/wetlands not on the inventory), 'missing', 'unparsed'.
    """
    s = re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()
    if not s or s.lower() in MN_PLACEHOLDERS:
        return None, None, "missing"
    m = re.fullmatch(r"(\d{1,2})-(\d{1,4})(?:-(\d{1,2}))?", s)
    if m:
        cc, ll, bb = m[1].zfill(2), m[2].zfill(4), m[3]
        dow = f"{cc}-{ll}" + (f"-{bb.zfill(2)}" if bb else "")
        return dow, f"{cc}{ll}{(bb or '0').zfill(2)}", "dow"
    if re.fullmatch(r"\d{8}", s):
        return f"{s[:2]}-{s[2:6]}" + ("" if s[6:] == "00" else f"-{s[6:]}"), s, "dow"
    low = s.lower()
    if low == "na":
        return None, None, "river_na"
    if low.startswith("none"):
        return None, None, "none"
    return None, None, "unparsed"


_YEAR = re.compile(r"(?<![\d-])(1[89]\d\d|20\d\d)(?!\d)")


def parse_mn_confirmed(text: str | None) -> dict:
    """'Year species was first confirmed, or connected water body' → listing basis and years.

    Rules (from the workbook's column-description sheet): the cell holds either the year the
    species was first confirmed in the water body, or 'connected …' when the water body was
    listed because it is connected to a confirmed water. Implemented as:
      * text starting 'connected' / 'conected' (typo in the list) or 'part of … chain' →
        basis 'connected' (no year parsed; DOW numbers in the text are not years);
      * 'year unknown' → basis 'confirmed', year missing;
      * otherwise any 4-digit year → basis 'confirmed'; first year = earliest year in the
        cell ('1977; 2008; 2013; 2025' → 1977; '2017 (dead); 2018 (live)' → 2017);
        'veliger' in the text → ``veliger_note`` True (larvae, not adults, found);
      * blank / 'NA' → basis 'unknown'.
    """
    s = re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()
    low = s.lower()
    out = {"listing_basis": "unknown", "year_confirmed": None, "years_confirmed_all": None,
           "veliger_note": "veliger" in low, "confirmed_flag": ""}
    if not s or low in MN_PLACEHOLDERS or low == "na":
        out["confirmed_flag"] = "missing"
        return out
    if low.startswith(("connected", "conected")) or re.match(r"part of .*chain", low):
        out["listing_basis"] = "connected"
        return out
    if "year unknown" in low:
        out.update(listing_basis="confirmed", confirmed_flag="year_unknown")
        return out
    years = [int(y) for y in _YEAR.findall(s)]
    if years:
        out.update(listing_basis="confirmed", year_confirmed=min(years),
                   years_confirmed_all=";".join(str(y) for y in years),
                   confirmed_flag="" if re.fullmatch(r"\d{4}", s) else "annotated")
        return out
    out["confirmed_flag"] = "unparsed"
    return out


def mn_column_descriptions(src: str | Path | bytes) -> list[str]:
    """Text lines of the workbook's 'Column descriptions' sheet (definitions of each field)."""
    names = C.xlsx_sheet_names(src)
    sheet = next((n for n in names if "description" in n.lower()), None)
    if sheet is None:
        return []
    d = C.read_xlsx(src, sheet, header=None)
    return [re.sub(r"\s+", " ", " ".join(x for x in row if x)).strip() for row in d.values.tolist()
            if any(x.strip() for x in row)]


MN_OUT_COLUMNS = [
    "source", "record_id", "list_version", "waterbody_name", "county", "county_multiple",
    "species_common", "species_scientific", "species_raw", "year_listed", "year_listed_raw",
    "listing_basis", "year_confirmed", "years_confirmed_all", "veliger_note", "confirmed_flag",
    "confirmed_or_connected_raw", "listing_lag_years", "dow", "dow8", "dow_class", "dow_raw",
    "row_number", "extra_cell",
]


def parse_mn_infested(src: str | Path | bytes) -> tuple[pd.DataFrame, dict]:
    """Infested-waters workbook → (tidy table, metadata).

    The first worksheet has a title row (with the list date), a header row, data rows and a
    trailing 'End of Worksheet' row; 'no data' placeholders become missing values.
    """
    names = C.xlsx_sheet_names(src)
    # keep blank rows so that the frame index equals the Excel row number minus one
    grid = C.read_xlsx(src, names[0], header=None, skip_blank_rows=False)
    nonblank = [i for i in range(len(grid)) if any(str(x).strip() for x in grid.iloc[i])]
    title = str(grid.iloc[nonblank[0], 0]) if nonblank else ""
    hdr_idx = next(i for i in nonblank[:10]
                   if any(str(x).strip() == "Water body name" for x in grid.iloc[i]))
    header = [str(x).strip() for x in grid.iloc[hdr_idx]]
    body = grid.iloc[[i for i in nonblank if i > hdr_idx]].copy()
    body.columns = [h or f"col_{i}" for i, h in enumerate(header)]
    body["row_number"] = body.index + 1  # Excel row number (1-based)
    missing = [k for k in MN_COLS if k not in body.columns]
    if missing:
        raise ValueError(f"MN infested-waters workbook is missing columns {missing}")
    extra_cols = [c for c in body.columns if c not in MN_COLS and c != "row_number"]
    df = body.rename(columns=MN_COLS)
    df["extra_cell"] = (df[extra_cols].apply(lambda r: ";".join(x for x in r if str(x).strip()), axis=1)
                        if extra_cols else "")
    df["extra_cell"] = df["extra_cell"].replace("", None)
    for c in MN_COLS.values():
        df[c] = df[c].map(lambda v: None if str(v).strip().lower() in MN_PLACEHOLDERS | {""} else
                          re.sub(r"\s+", " ", str(v).replace("\xa0", " ")).strip())
    end = df["waterbody_name"].fillna("").str.lower().str.startswith("end of worksheet")
    df = df[~end & df["waterbody_name"].notna()].copy()
    sp = df["species_raw"].map(normalize_mn_species)
    df["species_common"] = sp.map(lambda t: t[0])
    df["species_scientific"] = sp.map(lambda t: t[1])
    df["year_listed"] = pd.to_numeric(df["year_listed_raw"].where(
        df["year_listed_raw"].fillna("").str.fullmatch(r"\d{4}")), errors="coerce").astype("Int64")
    conf = pd.DataFrame(df["confirmed_or_connected_raw"].map(parse_mn_confirmed).tolist(), index=df.index)
    df = df.join(conf)
    df["year_confirmed"] = df["year_confirmed"].astype("Int64")
    dow = df["dow_raw"].map(parse_mn_dow)
    df["dow"] = dow.map(lambda t: t[0])
    df["dow8"] = dow.map(lambda t: t[1])
    df["dow_class"] = dow.map(lambda t: t[2])
    df["county"] = df["county_raw"]
    df["county_multiple"] = df["county"].fillna("").str.lower().str.startswith("multiple")
    df["listing_lag_years"] = (df["year_listed"] - df["year_confirmed"]).astype("Int64")
    version = mn_list_version(title)
    df["list_version"] = version
    df["source"] = "mn_dnr_infested_waters"
    df["record_id"] = [f"mn_iw_{version or 'unknown'}_{int(n):05d}" for n in df["row_number"]]
    for c in ("dow", "dow8"):
        df[c] = df[c].astype("string")
    meta = {"title": title, "list_version": version, "sheets": names, "header_row": hdr_idx,
            "n_data_rows": len(df), "extra_columns": extra_cols}
    return df[MN_OUT_COLUMNS].reset_index(drop=True), meta


def mn_page_counts(html_text: str) -> dict:
    """Headline zebra-mussel counts stated on the DNR infested-waters web page."""
    t = re.sub(r"(?s)<[^>]+>", " ", html_text or "")
    t = re.sub(r"\s+", " ", _html.unescape(t)).replace("\u2019", "'")
    out: dict = {}
    m = re.search(r"As of ([A-Z][a-z]+ \d{4}), we have confirmed zebra mussels in ([\d,]+) lakes "
                  r"and wetlands", t)
    if m:
        out["as_of"] = m[1]
        out["zm_confirmed_lakes_wetlands"] = int(m[2].replace(",", ""))
    m = re.search(r"We have listed ([\d,]+) bodies of water as infested with zebra mussels because they are "
                  r"closely connected", t)
    if m:
        out["zm_connected_waters"] = int(m[1].replace(",", ""))
    m = re.search(r"About (\d+)% of Minnesota's ([\d,]+) lakes", t)
    if m:
        out["pct_lakes_zm"] = int(m[1])
        out["mn_lakes_total"] = int(m[2].replace(",", ""))
    return out
