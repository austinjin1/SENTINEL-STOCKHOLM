# Datasheet — `census_county` (2010 Census county gazetteer)

| | |
|---|---|
| **Use** | D6 MusselGuard population-pressure proxy (boater origin / road access) for the Kanankege-2018 incumbent re-implementation and the challenger |
| **Source** | US Census Bureau, 2010 Gazetteer county file, `https://www2.census.gov/geo/docs/maps-data/data/gazetteer/Gaz_counties_national.zip` (one 167-kB request) |
| **Licence** | US Government work, public domain |
| **Retrieved** | 2026-10-25 |
| **Files** | raw `raw/census_county/Gaz_counties_national_2010.zip`; processed `processed/census_county/counties_2010.parquet` (3,221 counties incl. PR); manifest `census_county` |
| **Code** | `sentinel/data/sources/census_county.py` · `scripts/data/download_census_county.py` |
| **Columns** | `geoid` (5-digit string), `state`, `name`, `pop2010`, `housing2010`, `aland_km2`, `lat`/`lon` (internal point) |
| **Known issues** | County internal points are a coarse representation of where people live; population is the 2010 snapshot (after the 2006/2009 origins, before 2012/2015) — a stated limitation, analogous to Kanankege's 2013 boater survey used for a 2010 origin |
