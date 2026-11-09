# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""Delaware DNREC non-tidal fish-kill reports (data.delaware.gov Socrata dataset ``8pfi-yck7``).

Public-domain table (licence field "Public Domain", 290 rows on 2026-10-25; one row per kill × species
group, so rows are de-duplicated to events on (location, date, lat, long)). Free-text
``probable_cause`` is harmonised to the Iowa classes used by D3 KillAlert with explicit, auditable
keyword rules (:func:`harmonise_cause`); the rule hit for every event is stored. The tidal companion
dataset ``ycvg-4ys9`` has no coordinates and is not used.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

DATASET = "8pfi-yck7"
RESOURCE = f"https://data.delaware.gov/resource/{DATASET}.json"
METADATA = f"https://data.delaware.gov/api/views/{DATASET}.json"
USER_AGENT = "SENTINEL-v2 research (public-data download)"

# Order matters: anthropogenic evidence wins over environmental wording ("low DO caused by manure runoff").
ANTHRO = r"pesticid|herbicid|insecticid|fungicid|manure|sewage|septic|spill|discharg|chemical|chlorin|bleach|" \
         r"fertili[sz]|pollut|petroleum|oil\b|fuel|construction|dredg|runoff of|lawn additive|treat(ed|ment) .*vegetation|" \
         r"wastewater|toxic substance|copper sulfate|rotenone|draw ?down|drained|pump|treat|algaecid|water main"
DISEASE = r"disease|epizoot|parasit|columnaris|bacteri|virus|gill"
ENV = r"low d\.?o\b|dissolved oxygen|\bdo\b|d\.o\.|oxygen|winter ?kill|ice cover|algal|algae|bloom|temperature|" \
      r"\bheat|hot\b|turnover|low water|drought|stagnat|low flow|not flowing|spawning stress|cold|\bice\b|ice[- ]out"


def harmonise_cause(text) -> tuple[str, str, str]:
    """Free text -> (cause_origin ∈ Natural/Anthropogenic/Unknown, cause_class, rule)."""
    t = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text).lower()
    if not t.strip():
        return "Unknown", "Unknown", "blank"
    if re.search(ANTHRO, t):
        return "Anthropogenic", "Anthropogenic (keyword)", "anthro_keyword"
    if re.search(DISEASE, t):
        return "Natural", "Disease/Parasite/Stress", "disease_keyword"
    if re.search(ENV, t):
        hedged = bool(re.search(r"\bunknown\b|possibl|likely|probabl|potential", t))
        return "Natural", "Environmental", "env_keyword_hedged" if hedged else "env_keyword"
    if "unknown" in t:
        return "Unknown", "Unknown", "unknown"
    return "Unknown", "Other/unclassified", "no_rule"


def waterbody_type(location) -> str:
    t = str(location or "").lower()
    if re.search(r"pond|lake|millpond|basin|reservoir|spillpool|marsh", t):
        return "Lake" if "lake" in t else "Pond"
    if re.search(r"creek|river|branch|run\b|ditch|stream|tax ditch|gut", t):
        return "River/Stream"
    return "Unspecified"


def parse_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for c in ("location", "date", "lat", "long", "probable_cause", "species", "no_killed", "county", "watershed", "basin"):
        if c not in df:
            df[c] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["long"], errors="coerce")
    df.loc[df["longitude"] > 0, "longitude"] *= -1  # guard against unsigned west longitudes
    bad = ~(df["latitude"].between(38.3, 40.0) & df["longitude"].between(-76.0, -74.9))
    df["coord_flag"] = np.where(df["latitude"].isna(), "missing", np.where(bad, "outside_de", ""))
    df.loc[bad, ["latitude", "longitude"]] = np.nan
    df["n_killed"] = pd.to_numeric(df["no_killed"], errors="coerce")
    return df


def to_events(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (location, date, lat, long) — species-group rows merged."""
    key = ["location", "date", "latitude", "longitude"]
    g = df.groupby(key, dropna=False)
    ev = g.agg(probable_cause=("probable_cause", "first"), county=("county", "first"), watershed=("watershed", "first"),
               basin=("basin", "first"), n_species_rows=("species", "size"),
               species=("species", lambda s: "; ".join(sorted({str(x) for x in s if x}))),
               est_fish_killed=("n_killed", lambda s: s.sum(min_count=1)), coord_flag=("coord_flag", "first")).reset_index()
    h = (ev["location"].astype(str) + "|" + ev["date"].astype(str) + "|" + ev["latitude"].astype(str) + "|"
         + ev["longitude"].astype(str)).map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:10])
    ev.insert(0, "event_id", "DE-" + h)
    hc = ev["probable_cause"].map(harmonise_cause)
    ev["cause_origin"] = hc.str[0]
    ev["cause"] = hc.str[1]
    ev["cause_rule"] = hc.str[2]
    ev["is_env_wq"] = ev["cause"].eq("Environmental")
    ev["is_natural"] = ev["cause_origin"].eq("Natural")
    ev["is_anthropogenic"] = ev["cause_origin"].eq("Anthropogenic")
    ev["is_natural_suspected"] = False
    ev["waterbody_type"] = ev["location"].map(waterbody_type)
    ev["est_location"] = False
    ev["state"] = "DE"
    ev["source"] = "de_dnrec_nontidal"
    ev["year"] = ev["date"].dt.year
    if ev["event_id"].duplicated().any():
        raise AssertionError("duplicate Delaware event ids")
    return ev.sort_values(["date", "event_id"]).reset_index(drop=True)
