# Datasheet — SENTINEL-Orbit benchmark (`orbit_dev_v3`)

| | |
|---|---|
| **What** | Sentinel-2 L2A same-day matchups with in-situ chl-a, TSS, turbidity and Secchi, with frozen split roles (unseen lakes × future years) |
| **Inputs** | `s2_chips_aquacorr` (AquaCorr-Global S2 rows with SENTINEL L2A chips; in-situ values from AquaCorr), `s2_wqp_matchups` (WQP WQX3, MN 2024–2025 pilot), `hydrolakes` |
| **Code** | `sentinel/models/orbit/{data,split,guard}.py`, `scripts/train/orbit_build_split.py` |
| **Protocol** | `docs/preregistration/orbit_protocol.md`. Split rule v3 `docs/preregistration/orbit_split_v3.json` (sha256 `ffe288f2…5e32`). |
| **Outputs** | `$SENTINEL_RUNS/orbit/cache/dev_v3/{table.parquet, pixels.npz, info.json}` (manifest `orbit_dev_v3`, ≈ 33 MB) and `docs/preregistration/orbit_split_report_dev_v3.json` |
| **Licences** | AquaCorr CC BY 4.0; WQP public; Copernicus Sentinel data free, full and open; JRC GSW; HydroLAKES CC BY 4.0 |

## Build: dev_v3 (2026-10-24 22:40, split v3; AquaCorr chip extraction about 57% complete)
- Superseded snapshots dev_v1 and dev_v2 were used for smoke tests only. They were built under split v1/v2.
- **Rows with an extracted chip:** 70,973 (AquaCorr 68,102; WQP same-day 2,871).
- **Image QC:** 26,929 pass; de-duplication is distance-based (orbit_protocol.md §2.1).
- **Protected rows** (test role or 2024+): **7,655, dropped before any label use.** The WQP rows are all
  2024–25, so the development table is AquaCorr-only.
- **Development table:** 18,552 rows; 832 buffer-excluded.

| Cell (dev_v3) | chl-a | TSS | turbidity | Secchi | role blocks | water bodies |
|---|---|---|---|---|---|---|
| fit (train role × ≤ 2021, minus inner and buffer) | 5,586 | 715 | 334 | 6,161 | 322 | 1,651 |
| inner (early stopping / baseline tuning) | 1,146 | 183 | 169 | 1,429 | 58 | 375 |
| **valxval_lake (model selection)** | 812 | 25 | 11 | 789 | 55 | 225 |
| valxany_lake (val role × ≤ 2023, in-lake) | 2,381 | 51 | 36 | 2,593 | 110 | 467 |
| valxval_nonlake | 554 | 12 | 13 | 513 | 11 | 140 |
| trainxval_lake (seen lakes, val years) | 1,269 | 19 | 24 | 1,390 | 147 | 592 |

- **Sources:** almost all rows are AquaMatch (WQP-derived); the rest are AquaSat_v1, ChemLotUS, GLORIA and
  SeaBASS.
- **Lake share:** 63.5% of rows are in a HydroLAKES lake (2,089 lakes, 500 m capture).
- **Distances** (`orbit_split_report_dev_v3.json`): about 35% of in-lake val points have a train-role
  point within 10 km.
- **Value ranges:** unchanged from earlier builds (chl-a median ≈ 5.5 µg/L; Secchi median ≈ 1.4 m).
- **Pixels:** up to 64 nearest SCL-water pixels per chip in the centre 15 × 15.

## Known issues
- TSS and turbidity have very few rows in the selection cells, so their CIs are wide.
- The test cell held only 392 rows (117 water bodies) at the dev_v3 build. The national WQP 2024–25
  rebuild started 2026-10-25 00:32 and runs overnight; the frozen rule assigns its roles automatically.
- AquaCorr units are unstated (assumed).
- Lakes and blocks are held out whole, but points just across a 50-km block edge can be close. The
  >10-km isolation subset is a pre-registered sensitivity analysis.

## Rebuild
```bash
python -B scripts/train/orbit_build_split.py --workers 6      # split v3 rule file already frozen; never re-run --freeze-rule
```
