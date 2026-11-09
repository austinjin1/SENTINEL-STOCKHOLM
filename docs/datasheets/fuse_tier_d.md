# Datasheet — `fuse_tier_d` (SENTINEL-Fuse Tier D: NEON aligned multimodal table)

| | |
|---|---|
| **Location** | `$SENTINEL_DATA/processed/fuse_tier_d/` · manifest `manifests/fuse_tier_d.json` |
| **Built** | v1 2026-10-25; **v2 2026-11-08** (adds hydro, macroinv, periphyton) by `scripts/data/build_fuse_tiers.py d` (`sentinel/data/fuse_tiers.py::build_tier_d`) |
| **Inputs** | manifests `neon_sites`, `neon_sonde_daily`, `neon_nitrate`, `neon_tsw`, `neon_tsd`, `neon_chem`, `microbe_wq`, `forcings_daily`, `streamcat`, `sites`, `neon_eos`, `neon_discharge`, `neon_macroinv`, `neon_periphyton` |
| **License** | CC BY 4.0 (NEON; attribution: NSF NEON, RELEASE-2026, https://data.neonscience.org); gridMET and StreamCat public domain; ERA5 via Open-Meteo CC BY 4.0 |
| **Synthetic data** | none |

## Files
| File | Content |
|---|---|
| `daily.parquet` | one row per (site, local issue date): `<modality>__present`, engineered features `<modality>__…` (model space), `micro__sample_id` pointer, targets `y__<var>__h{1,3,7,14}`, lab targets `ylab__<analyte>` (+ `__cens`, `__dl`), hazard labels `ev__<hazard>__{any,onset}_h7`, `is_chem_event`, `is_micro_event` |
| `chem_events.parquet` | site × local date chemistry events (value, `__cens`, `__dl` per analyte) |
| `micro_events.parquet`, `micro_genus.parquet` | 16S events and their genus read-count matrix (label-free; CLR/PCA fit per fold in the model) |
| `columns.json` | modality → feature columns |
| `availability_all.csv`, `availability_by_target.csv` | availability-pattern counts (rows, sites), overall and for rows with each P1 / lab target observed |

## Coverage — v2 (2026-11-08, current; manifest stats)
* 105,117 rows, 34 sites, 2016-01-01 … 2025-06-30. Rows with modality present: sonde 67,874; nitrate
  54,825; temp 68,011; **hydro 74,949**; chem 88,226; micro 56,961; **macroinv 77,431**; **periphyton
  67,042**; wx 105,089 (ERA5 now covers the AK/PR sites); static 105,117.
* Rows by number of non-hub modalities present: 0: 236 · 1: 4,674 · 2: 8,042 · 3: 14,150 · 4: 10,797 ·
  5: 11,447 · 6: 18,107 · 7: 18,266 · **8: 19,398** (all eight modalities, 31 sites).
* New modalities: `hydro` = surface-water elevation (DP1.20016.001; one HOR/VER per site, the one with
  most valid non-provisional days; value minus trailing 365-d median, past-only) + log(Q + 1 L/s)
  (DP4.00130.001, RELEASE ends WY2021); `macroinv` = per-event log10 density (m⁻²), family richness,
  Shannon, density fractions EPT / Chironomidae / Oligochaeta / non-insect / Gastropoda / Amphipoda+Isopoda
  (830 events, 12–32 per site); `periphyton` = per-event medians of log10 extract chl-a and C, chl-a
  fraction, log C:N, N:P, C:chl-a, δ13C, δ15N, δ34S, separately benthic vs water column (718 events).
  Biology: latest event ≤ d, age ≤ 120 d; non-provisional only. Periphyton external-lab rows flagged
  "legacyData|did not meet quality audit" are kept (flag not used; listed as a known issue).
* Positional row order changed versus v1 (more rows kept), so Stage-0 runs `stage0_d_p1_v2` (v1 table)
  are not row-aligned with v2 runs; they remain in the registry as v1-table results.
* Every test-site row still carries all eight modalities at all 8 test sites (P3 feasibility, protocol §8.2).

## Coverage — v1 (2026-10-25, superseded by v2)
* 99,568 rows, 34 sites, 2016-01-01 … 2025-06-30 (RELEASE-2026 window; provisional values dropped).
  Rows kept: at least one non-hub modality present or a target observed.
* Rows with modality present: sonde 67,874; nitrate 54,825; temp 68,011; chem 88,226; micro 56,961;
  wx 86,428 (CARI, OKSR, TOOK in Alaska and CUPE, GUIL in Puerto Rico lack ERA5 forcings so far —
  the ERA5 job of `forcings_daily` is still filling these); static 99,568; hydro / macroinv / periphyton
  0 (NEON P2 products not yet processed; `raw/neon/_logs/builds_done.flag` absent on 2026-10-25).
* Rows by number of non-hub modalities present: 0: 616 · 1: 16,062 · 2: 14,440 · 3: 14,888 · 4: 21,519 ·
  **5: 32,043** (pattern sonde+nitrate+temp+chem+micro at 32 sites).
* Chemistry events 5,550; 16S events 1,429 (all co-dated with a chemistry event). MAYF has no 16S event.
* Targets with h = 7 observed: DO 58,189; SC 61,570; pH 60,329; turbidity 54,541; fDOM 42,016;
  chl-a (µg/L) 32,858; nitrate 48,579; temperature 64,029. Lab targets on event days: TN 5,286, TP 5,429,
  DOC 5,279 (+ censored rows), 11 analytes total.
* Hazards (valid rows / positives / sites with a positive): H1 hypoxia any-day-in-7 55,556 / 4,444 / 20;
  H1 onset 50,631 / 307 / 16; H5b chl bloom any 31,150 / 5,200 / 32; H5b onset 26,328 / 392 / 25;
  H4 nitrate 0 positives (not evaluated).

### Availability among rows with a P1 / lab target (top patterns; rows / sites)
| Target | rows | top patterns | rows with 5 modalities |
|---|---|---|---|
| DO h7 | 58,189 | 5-mod 27,437/32 · sonde+nitrate+temp+chem 8,072/30 · sonde+temp+chem+micro 5,991/33 | 27,437 |
| turbidity h7 | 54,541 | 5-mod 27,239/32 · sonde+nitrate+temp+chem 6,967/30 · sonde+temp+chem+micro 5,611/33 | 27,239 |
| nitrate h7 | 48,579 | 5-mod 27,359/31 · sonde+nitrate+temp+chem 7,575/30 · sonde+nitrate+temp 4,313/30 | 27,359 |
| chl-a h7 | 32,858 | 5-mod 12,810/32 · sonde+nitrate+temp+chem 5,179/30 · sonde+temp+chem+micro 4,224/33 | 12,810 |
| TN (lab) | 5,403 | 5-mod 2,042/32 · chem only 737/33 · sonde+nitrate+temp+chem 507/29 | 2,042 |
| TP (lab) | 5,437 | 5-mod 2,014/32 · chem only 752/33 | 2,014 |
| DOC (lab) | 5,280 | 5-mod 1,961/32 · chem only 743/33 | 1,961 |

Power note: daily rows are strongly autocorrelated; the effective sample is closer to the number of
site-months (≈ 2,600) and the CI unit is the site (26 development, 8 test sites). P1/P3 claims at the
test sites therefore need large effects; development CV uses 26 sites.

## Rules (pre-registered, `docs/preregistration/fuse_protocol.md` §2.1)
* Sensor station: HOR 102 → 112 (wadeable streams), 103 (lakes/rivers), shallowest VER; per (date,
  variable) the first available. Daily values are NEON-QC'd daily aggregates (`neon_sonde_daily`).
* Model space: log(x + offset) with SC 1.0, turbidity 0.1, fDOM 0.1, chl-a 0.1, nitrate 0.01 (as M1);
  lab analytes log(x + analyte offset) (`fuse_tiers.LAB_LOG_OFFSET`).
* Freshness: sensors present if observed on d−6…d; chemistry latest event strictly before d (≤ 35 d);
  16S latest event on or before d (≤ 65 d); biology ≤ 120 d; weather windows end at d.
* Censoring: lab `__cens` from NEON's below-detection flag; censored lab values keep the published value,
  else the detection limit as a bound — never DL/2.
* µg/L duplicates of mg/L analytes converted with the exact factor 10⁻³ (a handful of rows).

## Known issues
* Hub forcings missing for 3 Alaska and 2 Puerto Rico sites (see above) — those rows have `wx__present = False`.
* `chl_ugl` exists at all 34 sites but sensor chlorophyll is noisy (H5b any-day prevalence 16.7 %);
  chl-a RFU (10 sites, later years) is not used.
* fDOM is often QF-failed for long periods; early 2016 months are empty at several sites.
* Lab chemistry changed labs over time; censoring limits change with lab.
* No satellite modality at NEON yet (no S2 extraction at NEON sites).

## Rebuild
```bash
python -B scripts/data/build_fuse_tiers.py d
```
