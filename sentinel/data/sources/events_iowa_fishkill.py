# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Iowa DNR Fish Kill Database client (D3 KillAlert labels; EW-v2 event context).

Sources (no login; public API documented at https://programs.iowadnr.gov/fishkill/docs/api/index):

* ``GET /fishkill/api/v1/events?f=strict`` — every event with release status "Final":
  id, date, waterbody, county, waterbody type, cause / cause origin / cause mode, magnitude
  class + estimated fish killed, lat/lon (+ estimated-location flag), animal-waste details,
  impacted 305(b) assessment segments.
* ``/fishkill/events/{id}`` HTML detail pages — fields the API omits: kill length (miles),
  valuation and investigation expenses (USD), fisheries office, created / last-update
  timestamps, location description and the investigators' onsite comments (free text,
  sometimes with a SPECIES / NUMBER KILLED / MONETARY VALUE table).
* ``/adbnet/api/v1/segments/{id}`` — Iowa ADBNet 305(b) assessment segments (ADB code,
  HUC8, waterbody type, size) for the segments listed in ``impactedSegs``.
* Optional: NLDI ``comid/position`` (point-in-catchment NHDPlus V2 COMID) for each event point.

The event time in the API (``T12:00:00``) is a placeholder; only the local calendar date is
meaningful. Locations are the (approximate) point of origin, not the kill extent; ``estLoc``
marks estimated locations (DNR Codex "Data Disclaimer").
"""

from __future__ import annotations

import html as _html
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import events_common as C

SOURCE = "iowa_dnr_fishkill"
BASE = "https://programs.iowadnr.gov/fishkill"
API_EVENTS = f"{BASE}/api/v1/events"
CSV_EXPORT = f"{BASE}/Events/Download/Events"
EVENT_PAGE = f"{BASE}/events/{{id}}"
SWAGGER = f"{BASE}/swagger/docs/V1"
DOC_PAGES = {
    "user_guide": f"{BASE}/Docs",
    "codex_index": f"{BASE}/Docs/Codex",
    "codex_causes": f"{BASE}/Docs/Codex/Fish%20Kill%20Causes",
    "api_guide": f"{BASE}/Docs/API",
    "api_changelog": f"{BASE}/Docs/Codex/API%20Changelog",
}
ADBNET_SEGMENT = "https://programs.iowadnr.gov/adbnet/api/v1/segments/{id}"
NLDI_POSITION = "https://api.water.usgs.gov/nldi/linked-data/comid/position"
# EPA ATTAINS geospatial layers (no key). Iowa AU id = "IA " + ADBNet ADB code (verified 2026-10-22:
# ADB 01-YEL-447 ↔ AU "IA 01-YEL-447", Hecker Creek, reporting cycle 2024).
_ATTAINS = "https://gispub.epa.gov/arcgis/rest/services/OW/ATTAINS_Assessment/MapServer"
ATTAINS_LAYERS = {"line": f"{_ATTAINS}/1", "area": f"{_ATTAINS}/2", "point": f"{_ATTAINS}/0"}
ATTAINS_FIELDS = ("assessmentunitidentifier,assessmentunitname,reportingcycle,ircategory,overallstatus,"
                  "cause_unknown_fish_kills")

# Iowa extent (W, S, E, N) plus ~0.2° so border-river (Mississippi / Missouri / Big Sioux) points survive.
IOWA_BBOX = (-96.85, 40.17, -89.94, 43.71)

# Causes whose mechanism is a water-quality / physical condition a WQ forecast could anticipate
# (DNR Codex "Fish Kill Causes": Environmental = temperature extremes, summer/winter kill,
# stagnation, low DO, general algal blooms; Cyanobacteria Bloom = cyanotoxins).
ENV_WQ_CAUSES = ("Environmental", "Cyanobacteria Bloom")

_MAG_RANGE = re.compile(r"^\s*([\d,]+)\s*-\s*([\d,]+)\s*$")
_MAG_OPEN = re.compile(r"^\s*([\d,]+)\s*or more\s*$", re.I)


# ----------------------------------------------------------------------------- network fetchers

def fetch_events(**kw) -> dict:
    """The full released-event list (``{"total": n, "rows": [...]}``)."""
    return C.get_json(API_EVENTS, {"f": "strict"}, **kw)


def fetch_event_page(event_id: int | str, **kw) -> str:
    return C.get_text(EVENT_PAGE.format(id=event_id), allow_404=True, **kw)


def fetch_segment(seg_id: int | str, **kw) -> dict | None:
    return C.get_json(ADBNET_SEGMENT.format(id=seg_id), allow_404=True, **kw)


def fetch_nldi_comid(lon: float, lat: float, **kw) -> dict | None:
    """NLDI point → containing NHDPlus V2 catchment/flowline (GeoJSON FeatureCollection).

    Short timeout + ``Connection: close``: reused keep-alive connections to api.water.usgs.gov
    were observed to hang until the read timeout (2026-10-22), while fresh requests took < 1 s.
    """
    opts = {"timeout": 30.0, "retries": 5, "backoff": 3.0, "headers": {"Connection": "close"}}
    opts.update(kw)
    return C.get_json(NLDI_POSITION, {"f": "json", "coords": f"POINT({lon:.6f} {lat:.6f})"},
                      allow_404=True, **opts)


# ----------------------------------------------------------------------------- API parsing

def _rows(payload: dict | list) -> list[dict]:
    if isinstance(payload, dict):
        return list(payload.get("rows") or [])
    return list(payload)


def magnitude_bounds(desc: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'1 - 100' → (1, 100); '1,000,000 or more' → (1e6, inf); 'Unknown/Unspecified' → NaN."""
    lo, hi = [], []
    for d in desc.fillna(""):
        m, o = _MAG_RANGE.match(d), _MAG_OPEN.match(d)
        if m:
            lo.append(float(m.group(1).replace(",", "")))
            hi.append(float(m.group(2).replace(",", "")))
        elif o:
            lo.append(float(o.group(1).replace(",", "")))
            hi.append(np.inf)
        else:
            lo.append(np.nan)
            hi.append(np.nan)
    return pd.Series(lo, index=desc.index), pd.Series(hi, index=desc.index)


def parse_events(payload: dict | list) -> pd.DataFrame:
    """API event list → one tidy row per event (published fields + documented derived labels)."""
    rows = _rows(payload)
    recs = []
    for r in rows:
        mag = r.get("magnitude") or {}
        aw = r.get("awEvent") or {}
        segs = r.get("impactedSegs") or []
        origin = [str(s.get("id")) for s in segs if s.get("isOriginSeg")]
        recs.append({
            "event_id": r.get("id"),
            "date_raw": r.get("date"),
            "waterbody_name": r.get("originWaterbody"),
            "waterbody_type": r.get("wbType"),
            "county": r.get("county"),
            "field_office": r.get("fo"),
            "cause": r.get("cause"),
            "cause_origin": r.get("causeOrigin"),
            "cause_mode": r.get("causeMode"),
            "magnitude_id": mag.get("magnitudeID"),
            "magnitude_desc": mag.get("magnitudeDesc"),
            "est_fish_killed_raw": mag.get("estFishKilled"),
            "latitude_raw": r.get("lat"),
            "longitude_raw": r.get("lon"),
            "est_location": r.get("estLoc"),
            "aw_source": aw.get("source"),
            "aw_animal": aw.get("animal"),
            "aw_approx_head": aw.get("approxHead"),
            "origin_segment_ids": ";".join(origin) or None,
            "impacted_segment_ids": ";".join(str(s.get("id")) for s in segs) or None,
            "n_impacted_segments": len(segs),
            "html_url": r.get("html_url"),
        })
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df.insert(0, "source", SOURCE)
    df["event_id"] = C.str_id(df["event_id"])
    # "YYYY-MM-DDT12:00:00" — the noon time is a placeholder: keep the local calendar date only.
    date_part = df["date_raw"].astype("string").str.slice(0, 10)
    df["date"], df["date_flag"] = C.clean_dates(date_part, min_date="1950-01-01")
    df["date"] = df["date"].dt.normalize()
    df["year"] = df["date"].dt.year.astype("Int64")
    df["month"] = df["date"].dt.month.astype("Int64")
    df["latitude"], df["longitude"], df["coord_flag"] = C.clean_latlon(
        df["latitude_raw"], df["longitude_raw"], bbox=IOWA_BBOX)
    df["est_location"] = df["est_location"].astype("boolean")
    df["state"] = "IA"
    df["field_office"] = C.str_id(df["field_office"])
    df["magnitude_id"] = pd.to_numeric(df["magnitude_id"], errors="coerce").astype("Int64")
    df["magnitude_lo"], df["magnitude_hi"] = magnitude_bounds(df["magnitude_desc"])
    # estFishKilled is "if enumerated"; null → NaN (not 0). Keep the raw value as text.
    df["est_fish_killed"], df["est_fish_killed_qualifier"] = C.to_number(df["est_fish_killed_raw"],
                                                                         missing_codes=())
    df["aw_approx_head"], _ = C.to_number(df["aw_approx_head"], missing_codes=())
    # Derived labels (definitions in docs/datasheets/labels_iowa_fishkill.md).
    df["is_natural"] = df["cause_origin"].eq("Natural")
    df["is_anthropogenic"] = df["cause_origin"].eq("Anthropogenic")
    df["is_unknown_origin"] = df["cause_origin"].eq("Unknown")
    df["is_env_wq"] = df["cause"].isin(ENV_WQ_CAUSES)
    df["is_natural_suspected"] = df["cause"].eq("Unknown (Natural Suspected)")
    df["est_fish_killed_raw"] = df["est_fish_killed_raw"].map(lambda v: None if v is None else str(v))
    for c in ("latitude_raw", "longitude_raw"):
        df[c] = df[c].map(lambda v: None if v is None else str(v)).astype("string")
    df["date_raw"] = df["date_raw"].astype("string")
    for c in ("origin_segment_ids", "impacted_segment_ids"):
        df[c] = df[c].astype("string")
    df["n_impacted_segments"] = df["n_impacted_segments"].astype("int16")
    return df.sort_values(["date", "event_id"]).reset_index(drop=True)


def parse_event_segments(payload: dict | list) -> pd.DataFrame:
    """Event × impacted-assessment-segment link table."""
    recs = []
    for r in _rows(payload):
        for s in r.get("impactedSegs") or []:
            recs.append({"event_id": r.get("id"), "segment_id": s.get("id"),
                         "is_origin_segment": bool(s.get("isOriginSeg"))})
    df = pd.DataFrame(recs, columns=["event_id", "segment_id", "is_origin_segment"])
    df["event_id"] = C.str_id(df["event_id"])
    df["segment_id"] = C.str_id(df["segment_id"])
    return df


def parse_segment(js: dict) -> dict:
    """ADBNet segment JSON → flat record (IDs as strings, HUC8 zero-padded)."""
    huc = js.get("HUC8")
    return {
        "segment_id": None if js.get("SegID") is None else str(js.get("SegID")),
        "adb_code": js.get("ADBCode"),
        "segment_name": js.get("Name"),
        "segment_wb_type": js.get("WBType"),
        "segment_size": js.get("Size"),
        "segment_size_unit": js.get("SizeUnit"),
        "huc8": None if huc in (None, "") else str(huc).strip().zfill(8),
        "location_desc": js.get("LocationDesc"),
        "assigned_classes": ";".join(js.get("AssignedClasses") or []) or None,
        "is_spol": js.get("IsSPOL"), "is_oiw": js.get("IsOIW"), "is_onrw": js.get("IsONRW"),
        "segment_url": js.get("html_url"),
    }


def parse_segments(items: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame([parse_segment(j) for j in items if j])
    if df.empty:
        return df
    df["segment_id"] = C.str_id(df["segment_id"])
    df["huc8"] = df["huc8"].astype("string")
    df["segment_size"] = pd.to_numeric(df["segment_size"], errors="coerce")
    return df.sort_values("segment_id").reset_index(drop=True)


# ----------------------------------------------------------------------------- HTML detail pages

def _strip_tags(fragment: str) -> str:
    txt = re.sub(r"<script.*?</script>|<style.*?</style>", " ", fragment, flags=re.S | re.I)
    txt = re.sub(r"<br\s*/?>|</p>|</tr>|</h\d>|</li>", "\n", txt, flags=re.I)
    txt = re.sub(r"</t[dh]>", " | ", txt, flags=re.I)
    txt = _html.unescape(re.sub(r"<[^>]+>", " ", txt))
    txt = re.sub(r"[ \t\r\f\v\xa0]+", " ", txt)
    txt = re.sub(r" *\n[ \n]*", "\n", txt)
    return txt.strip()


def _money(text: str | None) -> float:
    if not text:
        return np.nan
    m = re.search(r"-?\$?\s*([\d,]+(?:\.\d+)?)", text)
    return float(m.group(1).replace(",", "")) if m else np.nan


def parse_event_page(page: str) -> dict[str, Any]:
    """Detail page → dict of the <dl> fields, location subtitle, comments (HTML and text)."""
    main_a, main_b = page.find("<main"), page.find("</main>")
    main = page[main_a:main_b] if main_a >= 0 and main_b > main_a else page
    fields: dict[str, str] = {}
    for dt, dd in re.findall(r"<dt[^>]*>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", main, flags=re.S):
        key = _strip_tags(dt)
        fields[key] = re.sub(r"\s+", " ", _strip_tags(dd)).strip()
    sub = re.search(r'<p class="event-location-subtitle[^"]*">(.*?)</p>', main, flags=re.S)
    title = re.search(r"<h1>(.*?)</h1>", main, flags=re.S)
    k = main.find("Onsite Comments</div>")
    comments_html = ""
    if k >= 0:
        frag = main[k + len("Onsite Comments</div>"):]
        frag = re.sub(r'^\s*<div class="card-body">', "", frag)
        frag = re.sub(r"(\s*</p>)?(\s*</div>){2,}\s*$", "", frag)
        comments_html = re.sub(r'^\s*<p class="card-text">', "", frag).strip()
    kl = fields.get("Kill Length", "")
    m = re.search(r"([\d,]+(?:\.\d+)?)\s*miles?", kl, flags=re.I)
    return {
        "event_id": fields.get("Event ID"),
        "page_title": _strip_tags(title.group(1)) if title else None,
        "location_desc": _strip_tags(sub.group(1)) if sub else None,
        "page_date_of_kill": fields.get("Date of Kill"),
        "kill_length_mi": float(m.group(1).replace(",", "")) if m else np.nan,
        "valuation_usd": _money(fields.get("Valuation")),
        "investigation_expenses_usd": _money(fields.get("Investigation Expenses")),
        "fisheries_office": fields.get("Fisheries Office") or None,
        "page_created": fields.get("Created Date"),
        "page_last_update": fields.get("Last Update"),
        "comments_text": _strip_tags(comments_html) or None,
        "comments_html": comments_html or None,
        "page_fields": ";".join(fields),
    }


def parse_event_pages(pages: dict[str, str]) -> pd.DataFrame:
    """{event_id: html} → one row per page (ID checked against the page's own Event ID)."""
    recs = []
    for eid, page in pages.items():
        if not page:
            continue
        rec = parse_event_page(page)
        rec["page_id_mismatch"] = rec["event_id"] not in (None, str(eid))
        rec["event_id"] = str(eid)
        recs.append(rec)
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["event_id"] = C.str_id(df["event_id"])
    for c in ("page_created", "page_last_update"):
        df[c], _ = C.clean_dates(df[c], fmt="%m/%d/%Y %I:%M:%S %p", min_date="1990-01-01")
    df["page_date_of_kill"], _ = C.clean_dates(df["page_date_of_kill"], fmt="%m/%d/%Y", min_date="1950-01-01")
    return df.sort_values("event_id").reset_index(drop=True)


_SPECIES_HDR = re.compile(r"species", re.I)
# number-column header patterns in priority order ("FISH KILLED" captions must lose to "Number Killed")
_NUMBER_HDRS = [re.compile(p, re.I) for p in (r"number", r"total\s*#|^#$|no\.\s*killed", r"count|total fish",
                                              r"killed")]
_VALUE_HDR = re.compile(r"value|\$|dollar", re.I)


def parse_species_tables(comments_html: str | None) -> pd.DataFrame:
    """SPECIES / NUMBER KILLED / MONETARY VALUE tables embedded in onsite comments.

    Best-effort: only HTML tables whose header row contains "species" are parsed. Narrative
    species mentions are left in ``comments_text``.
    """
    cols = ["table_index", "row_index", "species_text", "number_killed", "number_raw",
            "value_usd", "is_total_row"]
    if not comments_html or "<table" not in comments_html.lower():
        return pd.DataFrame(columns=cols)
    out = []
    for ti, rows in enumerate(C.html_tables(comments_html, expand_colspan=True)):
        hdr = next((i for i, r in enumerate(rows) if any(_SPECIES_HDR.search(c or "") for c in r)), None)
        if hdr is None:
            continue
        h = rows[hdr]
        sp = next(i for i, c in enumerate(h) if _SPECIES_HDR.search(c or ""))
        num = next((i for pat in _NUMBER_HDRS for i, c in enumerate(h) if i != sp and pat.search(c or "")),
                   None)
        val = next((i for i, c in enumerate(h) if i not in (sp, num) and _VALUE_HDR.search(c or "")), None)
        for ri, r in enumerate(rows[hdr + 1:], start=1):
            name = (r[sp] if sp < len(r) else "").strip()
            nraw = (r[num] if num is not None and num < len(r) else "").strip()
            vraw = (r[val] if val is not None and val < len(r) else "").strip()
            if not name and not nraw and not vraw:
                continue
            n = pd.to_numeric(nraw.replace(",", ""), errors="coerce")
            # an unnamed row with a number in a species table is the (unlabelled) total row
            is_total = bool(re.match(r"^\s*totals?\b", name, re.I)) or (not name and pd.notna(n))
            out.append({"table_index": ti, "row_index": ri, "species_text": name or None,
                        "number_killed": float(n) if pd.notna(n) else np.nan, "number_raw": nraw or None,
                        "value_usd": _money(vraw), "is_total_row": is_total})
    return pd.DataFrame(out, columns=cols)


def species_counts(pages: pd.DataFrame) -> pd.DataFrame:
    """Species tables for all parsed pages (needs ``event_id`` and ``comments_html``)."""
    frames = []
    for eid, html_ in zip(pages["event_id"], pages["comments_html"], strict=True):
        t = parse_species_tables(html_)
        if not t.empty:
            t.insert(0, "event_id", eid)
            frames.append(t)
    if not frames:
        return pd.DataFrame(columns=["event_id", "table_index", "row_index", "species_text", "number_killed",
                                     "number_raw", "value_usd", "is_total_row"])
    df = pd.concat(frames, ignore_index=True)
    df["event_id"] = df["event_id"].astype("string")
    return df


# ----------------------------------------------------------------------------- ATTAINS join key

def attains_au_id(adb_code: str | None) -> str | None:
    """Iowa ADBNet ADB code → EPA ATTAINS assessment-unit identifier ("IA " + code)."""
    if adb_code is None or (isinstance(adb_code, float) and np.isnan(adb_code)) or not str(adb_code).strip():
        return None
    return f"IA {str(adb_code).strip()}"


def fetch_attains(au_ids: list[str], *, batch: int = 100) -> list[dict]:
    """ATTAINS point/line/area records for the given AU ids (attributes only, all cycles listed)."""
    out: list[dict] = []
    for layer, url in ATTAINS_LAYERS.items():
        for k in range(0, len(au_ids), batch):
            quoted = ",".join("'" + i.replace("'", "''") + "'" for i in au_ids[k:k + batch])
            where = f"state='IA' AND assessmentunitidentifier IN ({quoted})"
            for f in C.arcgis_query(url, where=where, out_fields=ATTAINS_FIELDS, return_geometry=False):
                rec = dict(f.get("attributes") or {})
                rec["attains_layer"] = layer
                out.append(rec)
    return out


def parse_attains(records: Iterable[dict]) -> pd.DataFrame:
    """One row per AU id (line > area > point; latest reporting cycle first)."""
    df = pd.DataFrame(list(records))
    cols = ["attains_au_id", "attains_au_name", "attains_reporting_cycle", "attains_ir_category",
            "attains_overall_status", "attains_cause_unknown_fish_kills", "attains_layer"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={"assessmentunitidentifier": "attains_au_id",
                            "assessmentunitname": "attains_au_name",
                            "reportingcycle": "attains_reporting_cycle", "ircategory": "attains_ir_category",
                            "overallstatus": "attains_overall_status",
                            "cause_unknown_fish_kills": "attains_cause_unknown_fish_kills"})
    rank = {"line": 0, "area": 1, "point": 2}
    df["_r"] = df["attains_layer"].map(rank)
    df = df.sort_values(["attains_au_id", "attains_reporting_cycle", "_r"], ascending=[True, False, True])
    df = df.drop_duplicates("attains_au_id").drop(columns="_r")
    for c in cols:
        if c not in df:
            df[c] = None
    cycle = pd.to_numeric(df["attains_reporting_cycle"], errors="coerce")
    df["attains_reporting_cycle"] = cycle.astype("Int64")
    df["attains_au_id"] = df["attains_au_id"].astype("string")
    return df[cols].reset_index(drop=True)


def add_attains(segments: pd.DataFrame, attains: pd.DataFrame) -> pd.DataFrame:
    s = segments.copy()
    s["attains_au_id"] = s["adb_code"].map(attains_au_id).astype("string")
    s = s.merge(attains, on="attains_au_id", how="left")
    s["in_attains"] = s["attains_layer"].notna()
    return s


# ----------------------------------------------------------------------------- NLDI join key

def parse_nldi_position(js: dict | None) -> str | None:
    """NLDI comid/position response → COMID string (None if no catchment)."""
    if not js:
        return None
    feats = js.get("features") or []
    if not feats:
        return None
    props = feats[0].get("properties") or {}
    v = props.get("comid") or props.get("identifier")
    return None if v in (None, "") else str(v)


# ----------------------------------------------------------------------------- assembly

EVENT_COLUMNS = [
    "source", "event_id", "date", "year", "month", "date_flag", "date_raw",
    "waterbody_name", "waterbody_type", "county", "state", "field_office",
    "latitude", "longitude", "coord_flag", "est_location", "latitude_raw", "longitude_raw",
    "cause", "cause_origin", "cause_mode",
    "is_natural", "is_anthropogenic", "is_unknown_origin", "is_env_wq", "is_natural_suspected",
    "magnitude_id", "magnitude_desc", "magnitude_lo", "magnitude_hi",
    "est_fish_killed", "est_fish_killed_raw", "est_fish_killed_qualifier",
    "aw_source", "aw_animal", "aw_approx_head",
    "origin_segment_ids", "impacted_segment_ids", "n_impacted_segments", "html_url",
]
PAGE_COLUMNS = ["kill_length_mi", "valuation_usd", "investigation_expenses_usd", "fisheries_office",
                "location_desc", "page_created", "page_last_update", "comments_text", "page_date_of_kill",
                "page_id_mismatch"]


def species_check(events: pd.DataFrame, species: pd.DataFrame) -> pd.DataFrame:
    """Per event: species-table total (labelled total row, else sum of rows) vs API estFishKilled.

    ``species_table_check``: 'match' (within 1 % or 1 fish), 'mismatch', 'no_api_count',
    'no_table_count' (table without parseable numbers).
    """
    cols = ["event_id", "species_table_total", "species_table_check"]
    if species is None or species.empty:
        return pd.DataFrame(columns=cols)
    is_tot = species["is_total_row"].astype(bool)
    body = species[~is_tot].groupby("event_id")["number_killed"].sum(min_count=1)
    tot = species[is_tot].groupby("event_id")["number_killed"].max()
    ref = tot.reindex(body.index.union(tot.index)).fillna(body)
    out = pd.DataFrame({"event_id": ref.index.astype(str), "species_table_total": ref.to_numpy()})
    api = events.set_index("event_id")["est_fish_killed"].reindex(out["event_id"]).to_numpy()
    t = out["species_table_total"].to_numpy()
    close = np.abs(t - api) <= np.maximum(1.0, 0.01 * np.abs(api))
    out["species_table_check"] = np.where(np.isnan(t), "no_table_count",
                                          np.where(np.isnan(api), "no_api_count",
                                                   np.where(close, "match", "mismatch")))
    out["event_id"] = out["event_id"].astype("string")
    return out[cols]


def assemble_events(events: pd.DataFrame, pages: pd.DataFrame | None = None,
                    comids: pd.DataFrame | None = None, species: pd.DataFrame | None = None) -> pd.DataFrame:
    """Join API events with detail-page fields, species-table check and NLDI COMIDs on ``event_id``."""
    df = events[EVENT_COLUMNS].copy()
    if pages is not None and not pages.empty:
        df = df.merge(pages[["event_id"] + PAGE_COLUMNS], on="event_id", how="left")
        df["page_date_mismatch"] = (df["page_date_of_kill"].notna() & df["date"].notna()
                                    & (df["page_date_of_kill"] != df["date"]))
    if species is not None:
        df = df.merge(species_check(events, species), on="event_id", how="left")
    if comids is not None and not comids.empty:
        df = df.merge(comids[["event_id", "comid", "comid_method"]], on="event_id", how="left")
        # point-in-catchment COMID is the kill reach only for streams; for lakes/ponds it is merely
        # the catchment containing the waterbody (use NHD waterbody / LAGOS ids instead)
        ctx = np.where(df["waterbody_type"].eq("River/Stream"), "stream_reach",
                       np.where(df["waterbody_type"].isin(["Lake", "Pond", "Quarry/Pit", "Wetland"]),
                                "catchment_of_standing_water", "unknown"))
        df["comid_context"] = pd.Series(ctx, index=df.index).where(df["comid"].notna())
    return df
