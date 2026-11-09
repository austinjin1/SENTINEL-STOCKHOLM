# Datasheet — GLORIA with simulated sensor bands (`gloria`)

| | |
|---|---|
| **Source** | Lehmann, M.K. et al. (2023) *GLORIA — A globally representative hyperspectral in situ dataset for optical sensing of water quality*, Sci. Data 10:100 (doi:10.1038/s41597-023-01973-y). Data: PANGAEA doi:10.1594/PANGAEA.948492, file `GLORIA-2022.zip` (58,956,647 B; last-modified 2022-12-20) |
| **URL** | https://download.pangaea.de/dataset/948492/files/GLORIA-2022.zip |
| **Licence** | CC BY 4.0 (PANGAEA record) |
| **SRF tables** | ESA *Sentinel-2 Spectral Response Functions* COPE-GSEG-EOPG-TN-15-0007 **v3.2** (2023-10-10; S2A/S2B, 1 nm; https://sentinels.copernicus.eu/documents/247904/685211/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.2.xlsx); USGS spectral characteristics viewer **L8_OLI_RSR.xlsx** (Ball band-average RSR) and **L9_OLI2_RSR.xlsx** (band-average RSR measured at GLAMR; Barsi et al. 2019, doi:10.1117/12.2529776), https://landsat.usgs.gov/landsat/spectral_viewer/bands/ |
| **Retrieved** | 2026-10-22 (manifests `gloria_raw`, `srf_tables_raw`, `gloria`) |
| **Local paths** | raw: `$SENTINEL_DATA/raw/gloria/GLORIA-2022.zip`, `$SENTINEL_DATA/raw/srf/*.xlsx`; processed: `$SENTINEL_DATA/processed/gloria/` (26 MB) |
| **Code** | `sentinel/data/sources/gloria.py` (minimal xlsx reader, SRF loading, band simulation, tidy), `scripts/data/download_gloria.py`, `scripts/data/build_gloria.py` |

## Contents
* `gloria_samples.parquet` — 7,572 samples × (metadata, targets, methods, QC flags, simulated band Rrs):
  `GLORIA_ID`, `Dataset_ID` (54 datasets; the leave-one-dataset-out grouping), `Site_name` (486 names; *not* a lake ID —
  e.g. "Lake Erie" spans US and Canada), `Latitude/Longitude`, `Date_Time_UTC` → `time_utc` (1990-08-28 → 2022-10-05),
  `Water_body_type` → `water_body_type_name` (lake 4,510; coastal 2,426; estuary 311; river 249; other 76), all
  investigator metadata columns (radiometry method, Chl/TSS/aCDOM method, sample depth …).
* Targets with units (`<target>_unit`, from `GLORIA_variables_and_methods.xlsx`): `Chla` mg m⁻³ (5,132; 3,746 not
  flagged), `Chla_plus_phaeo` mg m⁻³ (1,316), `TSS` g m⁻³ (4,623; 3,460), `aCDOM440` m⁻¹ (4,393; 3,344),
  `Secchi_depth` m (3,948; 3,099), `Turbidity` NTU (1,252; 895). Counts in brackets: not `Flagged`.
* QC flags (GLORIA_qc_flags.csv): `Noisy_red` 40, `Noisy_blue` 15, `Baseline_shift` 164, `Negative_uv_slope` 139,
  `QWIP_fail` 278, `Suspect` 226, composite `Flagged` 1,779 (= `any_qc_flag`); `Oxygen_signal` 1,311 is
  informational (O₂-A artefact) and not part of `Flagged`.
* Simulated band Rrs (sr⁻¹) `rrs_<sensor>_<band>`: S2A and S2B B1–B8A, L8 and L9 B1–B5, `naip_ads100_{blue,green,red,nir}`,
  `dji_p4m_rgbn_{blue,green,red,nir}`.
* `gloria_rrs_1nm.parquet` — `Rrs_350…Rrs_900` and `Rrs_std_*` (float32) per `GLORIA_ID`.
* `srf_1nm.parquet` (normalised SRFs on 350–900 nm), `srf_info.parquet` (band coverage, centroid, simulated?).

## Band simulation
`Rrs_band = Σ Rrs(λ)·SRF(λ) / Σ SRF(λ)` on the 1 nm grid 350–900 nm (SRF linearly interpolated, negative RSR values from
measurement noise clipped to 0). A band is simulated only if ≥ 98 % of its full SRF integral lies within 350–900 nm
(S2 B8: 99.5 %; B9–B12, OLI B6/B7 not simulated), and per sample only if Rrs is finite at every wavelength where the
SRF ≥ 1 % of its maximum (no gap filling). SRF centroids reproduce the official values (e.g. S2A B2 492.7 nm,
S2B B3 558.9 nm, OLI B2 482.6 nm).

**Camera bands (approximations, documented).** No measured spectral response exists for the proposed drone camera
(docs/research/08 §b.3). Two band models are provided as top-hats from published band edges: `naip_ads100` — Leica
ADS100 bands as listed in USDA NAIP FGDC metadata (B 435–495, G 525–585, R 619–651, NIR 808–882 nm; the sensor of most
2015+ NAIP state-years, not all); `dji_p4m_rgbn` — DJI P4 Multispectral spec sheet (450 ± 16, 560 ± 16, 650 ± 16,
840 ± 26 nm). Replace with measured curves when available.

**Cross-check (computed):** for the 143 GLORIA samples that AquaCorr-Global also convolved (`aquacorr_gloria_validation`,
column `insitu_Rrs_*`), our S2A/S2B/L8 band values agree with theirs to within a few per cent (median ratio ours/AquaCorr:
S2A blue 0.976, green 1.021, red 0.992; S2B blue 0.965, green 1.018, red 0.984; L8 coastal 0.995, blue 0.974, green
1.050, red 1.002; NIR (B8A/B5, n = 4) 0.995–1.000). The residual differences reflect SRF versions/definitions, not a
units or scaling error.

## Known issues
* **NIR is often missing:** many spectra end below 900 nm (e.g. 350–750 nm), so NIR bands are finite for only
  ~2,450 samples (S2 B8/B8A 2,452/2,453; OLI B5 2,452; camera NIR 2,453–2,454) vs 7,572 for blue–red. RGB+NIR
  experiments on GLORIA are limited to that subset.
* Geographic skew (US 3,339 samples; then CN, FR, NL, JP, CH, IT); the median site has very few samples while Lake
  Erie has hundreds. Group splits must use `Dataset_ID` (and site/water body), never random rows.
* Overlaps: GLORIA contains SeaBASS and LIMNADES samples; AquaCorr-Global includes 1,535 GLORIA matchups; the MDN
  development set shares contributors with GLORIA (docs/research/02 §b.2).
* `GLORIA_qc_flags.csv` ends with an empty row (dropped).

## Rebuild
```bash
python -B scripts/data/download_gloria.py   # GLORIA zip + 3 SRF workbooks -> manifests gloria_raw, srf_tables_raw
python -B scripts/data/build_gloria.py      # tidy samples + simulated bands -> manifest gloria
```
