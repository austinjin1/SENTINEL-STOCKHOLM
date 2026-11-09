# MIT License — Bryan Cheng, 2026
# Part of SENTINEL v2
"""US Census 2010 county gazetteer (internal point, land area, 2010 population and housing units).

Single public file ``Gaz_counties_national.zip`` (tab-separated, 2010 Census; US Government work, public
domain). Used by D6 MusselGuard for a population-pressure proxy (boater origin / road access) in the
Kanankege-2018 incumbent re-implementation and challenger.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_counties_national.zip"
USER_AGENT = "SENTINEL-v2 research (public-data download)"


def parse_gazetteer(zip_bytes: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        name = [n for n in z.namelist() if n.endswith(".txt")][0]
        df = pd.read_csv(z.open(name), sep="\t", dtype={"GEOID": str, "ANSICODE": str}, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame({"geoid": df["GEOID"].str.zfill(5), "state": df["USPS"].str.strip(), "name": df["NAME"],
                        "pop2010": pd.to_numeric(df["POP10"], errors="coerce"),
                        "housing2010": pd.to_numeric(df["HU10"], errors="coerce"),
                        "aland_km2": pd.to_numeric(df["ALAND"], errors="coerce") / 1e6,
                        "lat": pd.to_numeric(df["INTPTLAT"], errors="coerce"),
                        "lon": pd.to_numeric(df["INTPTLONG"], errors="coerce")})
    if out["geoid"].duplicated().any() or out[["lat", "lon", "pop2010"]].isna().any().any():
        raise AssertionError("unexpected duplicates / missing values in the county gazetteer")
    return out
