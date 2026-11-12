# Datasheet — `d7_mast` (EPA StreamCat modelled stream temperature for NRSA COMIDs) — **D7-ONLY, RESTRICTED USE**

**Use restriction (lead decision 2026-11-09):** this dataset may be read **only by D7 TroutRefuge**, and only as the
**score of the published thermal-threshold incumbent** (Hill et al. 2013). It must never enter any model's feature
matrix (enforced by `troutrefuge.assert_no_modelled_temperature`, unit-tested). **No other workstream may read
`d7_mast`.** Reason: modelled stream temperature overlaps SENTINEL temperature targets, which is why the shared
StreamCat client excludes these metrics (`docs/datasheets/streamcat.md`); the shared client was not modified.

| Item | Value |
|---|---|
| Source | EPA StreamCat API `https://api.epa.gov/StreamCat/streams/metrics` (POST, `aoi=other`), metrics `mast`, `msst`, `mwst` × 2008, 2009, 2013, 2014 |
| Model | Hill, Hawkins & Carlisle 2013, *Freshwater Science* 32(1):39–55 — random forest of **MAST = mean annual**, **MSST = mean summer (Jul–Aug)** and **MWST = mean winter (Jan–Feb)** stream temperature from landscape and climate, trained on 2008–2014 temperature records. (Corrected 2026-11-10: an earlier version of this sheet and D7 protocol §3 called MAST a "mean August" temperature — wrong.) |
| Units | 5,504 NRSA site COMIDs (from `downstream_catchment/unit_links.parquet`, unit_source `nrsa`) |
| Coverage | MAST 4,973–5,004 COMIDs per year; MSST ~5,190; MWST 4,688–5,027 |
| Fetch | `scripts/data/fetch_mast_d7.py` (dedicated; builds the POST form itself, reuses only the client's transport/cache), 2026-11-09 |
| Licence | US public domain (EPA StreamCat) |
| File | `processed/d7_mast/mast_comid.parquet` (comid + 12 metric columns); manifest `manifests/d7_mast.json` |

Known limitations: model output (not measurements); years 2008–2014 only — D7 maps 2008 → NRSA 2008-09, 2013 → 2013-14,
2014 → 2018-19 and 2023-24 (disclosed); its training data may overlap NRSA reaches (favours the incumbent).

Column summaries (label-free, °C): MAST median ≈ 11.8 (range 5.7–23.4); MSST median ≈ 21 (9.0–28.1); MWST median ≈ 3.2 (0.5–16.7).
D7 comparators: `mast_rule` (−MAST; pre-registered) and `msst_rule` (−MSST and MSST ≤ 20 °C; added 2026-11-10, protocol §8) — scores only.
