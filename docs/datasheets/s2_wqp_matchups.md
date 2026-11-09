# Datasheet — Sentinel-2 L2A × Water Quality Portal matchups (`s2_wqp_matchups`)

| | |
|---|---|
| **In situ** | Water Quality Portal WQX 3.0 results from the WQP workstream: `processed('wqp_discrete')` (datasheet `wqp_discrete.md`), or — while that table is not built — its raw parse output `raw('wqp')/results/group={optical,cyanotoxins}` + `raw('wqp')/locations` (`--source raw`) |
| **Imagery** | S2 L2A chips (`s2_chips.md`): Earth Search `sentinel-2-c1-l2a`, Planetary Computer `sentinel-2-l2a` fallback; JRC GSW occurrence |
| **Licences** | WQP data public (cite providers / doi:10.5066/P9QRKUVJ); Copernicus Sentinel data free, full and open; JRC GSW without restriction |
| **Code** | `sentinel/data/matchups.py` (`load_wqp`, `wqp_surface_samples`, `site_jrc`, planning/screening/extraction), `scripts/data/build_s2_wqp_matchups.py` |
| **Outputs** | `$SENTINEL_DATA/processed/s2_wqp_matchups/{samples,sites,matchups}.parquet`, `coverage.json`; chips `$SENTINEL_DATA/processed/s2_chips/wqp/` |
| **Manifest** | `s2_wqp_matchups` |

## Sample rules (`wqp_surface_samples`)
* Targets (exact WQX `CharacteristicName`, probe/RFU names never included — they are not requested upstream):
  `chla` (Chlorophyll a; …corrected/uncorrected for pheophytin; …free of pheophytin; - Phytoplankton (suspended);
  Chlorophyll A; Chlorophyll), `tss` (Total suspended solids; Total Suspended Particulate Matter), `ssc` (Suspended
  Sediment Concentration, kept separate from TSS), `turbidity` (Turbidity; Turbidity Field — unit NTU/FNU kept),
  `secchi` (Depth, Secchi disk depth; Secchi depth; Water transparency, Secchi disc), `microcystin` (total
  microcystins/nodularins and congeners; the exact name is kept in `characteristic`), `phycocyanin` (lab names only).
* QC: activities classed as QC (`qc_class` ≠ '') or data loggers dropped; `status_class == 'rejected'` dropped.
* **Surface only:** sample depth (result depth, else activity depth, else top depth; ft/in/cm converted) ≤ 1 m or
  unspecified; "bottom" relative depths dropped; Secchi exempt (optical depth, not a sampling depth).
  `depth_rule` ∈ {<=1m, unspecified, exempt}.
* Replicates at the same site, local date, characteristic and unit → median (`value`), with `value_min/max`,
  `n_results`, `n_censored`; censored results stay NaN with `detection_limit` (no DL/2 substitution).
* Site class from the WQX location type (fallback: location name): `lake_reservoir`, `river_stream`,
  `estuary_coastal`, `wetland`, `canal`, `other`, `unknown`.
* **Water test:** JRC GSW occurrence ≥ 50 % at the site or at a pixel within 60 m (`jrc_water_ok`); only those sites
  are matched (others stay in `samples.parquet` with the flag).

## Matchup rules
* Every S2 acquisition covering the site within ±3 days of the sample's local date (one STAC query per UTC date and
  region; Earth Search C1 first, MPC for uncovered sites; MPC reprocessing duplicates → highest baseline).
* SCL pre-screen: usable if the centre pixel has data and the centre 7 × 7 (70 m) has ≥ 5 SCL-valid pixels; usable
  acquisitions get a full chip (priority microcystin > phycocyanin > chl-a > Secchi > TSS > SSC > turbidity,
  nearest |Δt| first, optional `--max-chips`).
* One row per (sample, acquisition): `dt_days` (image UTC date − sample date), `abs_dt_days`, `win_same_day`,
  `win_pm1`, `win_pm3`; screening counts; chip id + all chip metadata (`s2_chips.md`); `qc_water_c3` (≥ 5 SCL-water
  pixels in the centre 3 × 3) and `qc_pass` (+ no radiometric sanity flag).

## Coverage — pilot run 2026-10-24 (interim)
`processed('wqp_discrete')` (and its datasheet) did not exist; the run used the WQP workstream's complete raw WQX3
download (`raw('wqp')`, manifest `wqp_raw`, results + embedded locations), **Minnesota, 2024-01-01 → 2025-12-31**
(the future-years test block beyond AquaCorr), chip cap 4,000:
* 76,578 surface samples at 4,007 sites (Secchi 30,615; TSS 20,582; chl-a 20,332; turbidity 4,858; SSC 97;
  microcystin 94); 52,775 samples at 2,846 sites pass the JRC water test (lakes/reservoirs 2,733 sites, rivers 1,235).
* 109,807 (site, acquisition) candidates in ±3 d (5,962 scenes); 88,142 sample × acquisition rows usable after SCL
  screening (40,240 samples); 4,000 chips extracted (cap; tier 0 = all, since every row is 2024+), 148 MB.
* Samples with a `qc_pass` chip: same day — chl-a 2,447, Secchi 2,160, TSS 291, turbidity 263, microcystin 8;
  ±1 d — 2,476 / 2,326 / 296 / 264 / 8; ±3 d — 2,518 / 2,490 / 301 / 265 / 8.
* Remaining: national run and the chip cap (`--max-chips 0`), and a rerun on `--source processed` once
  `wqp_discrete` (with Station/search coordinates for the ~25 % of rows lacking embedded locations) exists.

## Known issues
* S2 overpasses are ~10:30 local solar time; WQP sample times are often missing, so `dt_days` is a calendar-day
  difference (UTC image date vs local sample date — equal dates for the Americas).
* In raw mode ~25 % of WQX3 result rows lack embedded location fields (no coordinates/type); those sites are dropped
  until the WQP workstream's Station/search table is available.
* Lake samples at shore stations can fall on land pixels; the JRC test admits sites within 60 m of water, so the
  centre 3 × 3 may still be land — use `qc_water_c3` / water-masked medians.

## Rebuild
```bash
python -B scripts/data/build_s2_wqp_matchups.py --source raw --states MN --start 2024-01-01 --end 2025-12-31 --max-chips 4000 --read-workers 4  # pilot as run
python -B scripts/data/build_s2_wqp_matchups.py --source processed --start 2015-06-23 --end 2025-12-31   # national (pending)
```
