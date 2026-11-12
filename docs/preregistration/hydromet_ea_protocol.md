# SENTINEL-Hydro (M6) — track T-E, EA England new geography: data, split and test-block protocol

**Status:** v1.0 (2026-11-10 15:09 EST, lever engineer) — **submitted for the lead's stamp; the split is not frozen.** Written
**before any model was fitted, tuned or scored on EA data**, before any EA reading was QC-summarised or aggregated (the
`daily` step has not run) and before any EA value dated ≥ 2023-10-01 was requested. What has been read: the station and
measure catalogue (metadata), geometry, and — at download time only — each monthly readings file was parsed to check
that no row is dated ≥ 2023-10-01 and its row and measure counts were logged; no value entered any statistic. Every
count in §4–§6 comes from catalogue metadata and geometry only. Order fixed by the lead: stamped T-E entry (sha256 of
this file, of `hydromet_ea_split.json`, S, per-endpoint G / G_eff) → the lead runs the split writer once → M1's draw
(`sonde_ea_split.json`, exists even if empty) → first M6 England fit. Not part of the frozen M6 code hash.

Scope split (lead rulings 2026-11-10, `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md`): this file fixes the
**data, eligibility, input schema, groups, roles, blocks, power, strata and the single-look requirements** of T-E. The
**arms, claim-model rule and Holm family** of T-E are fixed in `hydromet_arch_v2_amendment.md` (architecture-v2
engineer), which refers to this file. Reservation: `docs/plans/ea_england_allocation.md` (unchanged).

## 1. Practical justification (fixed before any result)
A deployable estimator for reaches without a sensor has to transfer to another country's river network, forcing
product and regulator. England has thousands of WFD river water bodies without continuous sondes; the EA's own sonde
network (≈ 1,960 stations, 2012 → present) is deployed at impacted waters where temperature and oxygen matter most.
T-E is the earliest genuinely unseen M6 block: its test values have never been downloaded.

## 2. Data (`processed/ea_wq_sonde`, shared, OGL v3; build `scripts/data/sota_hydromet_ea_build.py`)
- Source: EA Hydrology API, observed properties `temperature` and `dissolved-oxygen` (mg/L; % saturation kept for QC
  only), 15-min (some 30/60-min) instantaneous readings, 2012-01-01 … 2023-09-30 only (`max-date` exclusive; every row
  checked). Values from 2023-10-01 are requested only by the single test look after the freeze.
- Daily statistics per measure-day from QC'd readings (`sentinel/hydromet_sota/ea.py`): Suspect/Missing and non-finite
  readings dropped; range −2…35 °C and 0…25 mg/L; spikes > 5 °C / 5 mg/L from the centred 5-reading median removed;
  flat lines ≥ 12 h removed; a day is valid with ≥ 80 % of the expected readings (modal interval) spanning ≥ 20 h.
  Nearly all EA water-quality readings carry quality "Unchecked"; this is disclosed.
- Targets (M6 names): `temp_c_mean`, `temp_c_7dadm` (trailing 7-day mean of the daily max, all 7 days valid),
  `do_mgl_mean`, `do_mgl_min`.

## 3. Common input schema (one schema for every model, US and EA)
- **Forcing:** the 11 M6 site-cell channels (`precip_mm, tmax_c, tmin_c, srad_wm2, wind_ms, rhmax_pct, rhmin_pct,
  sph_kgkg, vpd_kpa, pet_mm, tmean_c`) by a fixed variable mapping: gridMET for US sites (as all M6 models), ERA5-Land
  (Open-Meteo `era5_seamless`, land cell, harmonised by `sentinel.data.sources.openmeteo.era5_to_harmonised`) for EA
  stations. The forcing-source shift is part of the track and is disclosed. Upstream-basin (`ws_`) channels are **not**
  in the schema (their EA cost would exceed the Open-Meteo allowance).
- **Statics:** a fixed compact RiverATLAS v1.0 (HydroATLAS, CC BY 4.0) set computed identically in both domains from the
  nearest RiverATLAS reach (US sites snapped in EPSG:5070, EA stations in EPSG:27700): lat, lon; `ele_mt_cav/uav`,
  `slp_dg_cav/uav`, `sgr_dk_rav`, `UPLAND_SKM`, `CATCH_SKM`, `ORD_STRA`; `tmp_dc_cyr/uyr/cmx/cmn`, `pre_mm_cyr/uyr`,
  `pet_mm_uyr`, `ari_ix_uav`, `cmi_ix_uyr`, `snw_pc_uyr`; `dis_m3_pyr`, `run_mm_cyr`, `gwt_cm_cav`, `lka_pc_use`,
  `dor_pc_pva`, `rev_mc_usu`; `for_pc_cse/use`, `crp_pc_use`, `pst_pc_use`, `urb_pc_cse/use`, `wet_pc_ug1`;
  `cly_pc_uav`, `snd_pc_uav`, `soc_th_uav`, `kar_pc_use`; `ppd_pk_cav/uav`, `hft_ix_u09`, `rdd_mk_uav`, `nli_ix_uav`
  (40 RiverATLAS attributes plus lat/lon, `sentinel.hydromet_sota.ea_data.COMMON_STATIC`; `elev_m` = `ele_mt_cav` in both domains; categories mirror protocol `COMPACT_STATIC`). StreamCat attributes are not available in England, so
  StreamCat-name matching would mix definitions across domains; HydroATLAS regional subsets are used instead
  (permitted by the lead's ruling "HydroATLAS only if the zero-shot schema needs it, regional subsets").
  `DIS_AV_CMS`/`dis_m3_*` are WaterGAP model outputs (no gauge read at prediction), disclosed like StreamCat runoff.
- **Reach-distance rule (PROPOSED 2026-11-10 14:31 EST; metadata only, written before any T-E fit and before any consumer
  loaded the US common schema):** a location enters the common schema only if its nearest RiverATLAS reach lies within
  1 km — the EA eligibility rule of §4 applied to US locations as well. RiverATLAS holds only reaches with ≥ 10 km²
  upland area or ≥ 0.1 m³/s, so a small-stream location farther away would receive another river's attributes.
  Counts (snap distances only): US FINAL eligible sites **1,625 of 1,712 train-role and 279 of 300 val-role** kept
  (2 have no reach within 5 km, 106 lie 1–5 km away); new USGS lever sites 429 of 445; WQP stations 6,483 of 7,609.
  `sentinel.hydromet_sota.ea_data.load_us_common(max_snap_m=1000)` (default) applies it; subsets built by
  `sota_hydromet_ea_build.py atlas`: `processed/ea_wq_sonde/riveratlas_gb.parquet` (18,143 GB reaches × 296 columns)
  and `processed/hydromet_sota/riveratlas_us_sites.parquet` (10,168 US train/val-role locations, 10,066 with a reach
  within 5 km, median distance 154 m; no test-role or M1-reserved location is in the file).
- **Calendar:** DOY harmonics. No in-situ input of any kind (as M6 §3).

## 4. Eligibility, groups, roles and blocks
Rule ruled by the lead on 2026-11-10 (`hydromet_arch_v2_lead_rulings.md` §2.1 Q4, §2.4): the eligible-only rule with
**S = 0.50**, the 15 % guard and the hash order; no per-group cap. One implementation: `sentinel/hydromet_sota/ea_split.py`
(rule text `RULE_TEXT`, stored in the split file and hashed). Only catalogue metadata and geometry enter the split.
- **Eligible station (three conditions, all from metadata):** (i) the nearest HydroRIVERS v1.0 reach (rivers with
  ≥ 10 km² upland area) lies within 1 km of the station point (EPSG:27700); (ii) **the station point lies inside a
  HydroBASINS level-7 unit (`hybas_l7` not null)**; (iii) the catalogue open span (`dateOpened`…`dateClosed`, open
  stations closed at 2026-09-30, clipped to 2012-01-01…2026-09-30) is ≥ 60 days. **1,100 of 1,964** catalogue stations
  are eligible; 864 are not (277 have no reach within 1 km — 3 of them none within 5 km; 587 have a reach within 1 km
  but an open span < 60 days; 0 fail condition (ii)). 309 eligible stations are open ≥ 60 days after 2023-11-12
  ("test-capable").
- **Basin groups (eligible stations only):** the level-7 unit of the station point, merged (union-find) with every
  unit holding an eligible station directly flow-connected to one of its eligible stations (same reach, or one
  station's reach on the other's HydroRIVERS `NEXT_DOWN` chain, any river distance — the US split's NLDI rule
  transposed). **55 groups** made of 77 level-7 units (at most 9 units in a group); the largest group has 130 eligible
  stations (12 %). Merging over all 1,964 stations instead chains the Thames, Trent and Severn systems through poorly
  snapped stations into one group of 727 stations — rejected in v0.1 and not used.
- **Group-id rule:** the id of a group is the **smallest level-7 `HYBAS_ID` among the units merged into it**, written
  as a decimal string (all ids have 10 digits, so string and numeric order agree; `ea_geo._UF.union` keeps the smaller
  root). The split file lists the level-7 units of every group (`group_units`); a unit lies in at most one group.
- **Roles:** groups are ordered by the hex digest of sha256(`"M6-EA-2026-11-10:"` + group id). A group holding more than
  15 % of the eligible stations is train-role (guard; no group reaches it). Otherwise the role follows the cumulative
  share of eligible stations **before** the group in that order: test while the share is < S = 0.50, val while it is
  < S + 0.15, train after. Result: **28 test / 10 val / 17 train groups; 583 / 136 / 381 eligible stations** (the last
  test group carries the share past S: 53 %). Test-capable stations: 141 in 21 test groups (123 with a DO mg/L
  measure), 42 in 7 val groups, 126 in 14 train groups.
- **Every catalogue station has a status** (`stations` in the split file: `eligible`, `status`, `basis`, `groups`):
  an eligible station has its group's role (`basis` = eligible). An ineligible station takes the role of the eligible
  station(s) it is flow-connected to — in either direction, through its nearest reach within 5 km (`flow_connected`);
  otherwise the role of the group that holds its level-7 unit (`l7_unit`); otherwise **"reserved"** (`none`).
  Tie-break (lever engineer's reading of the ruling, **for the lead to confirm with the stamp**): if the flow-connected
  eligible stations lie in groups of different roles (possible only at a confluence below two groups), the most closed
  role is taken — test before val before train. It applies to **0** stations of this catalogue (no ineligible station
  is flow-connected to more than one group). Counts: test 1,088 (583 eligible + 409 flow-connected + 96 by unit),
  val 232 (136 + 54 + 42), train 640 (381 + 235 + 24), reserved 4.
  **Ineligible stations never supply M6 rows** in any role (not for fitting, selection, scoring or merging groups);
  M6 rows come only from `station_role` (eligible stations). **The 1,092 stations with status test or reserved
  (`closed_stations`) are closed to M1 and to every other workstream for all years**; the time reservation (every
  station from 2023-10-01) is unchanged.
- **One canonical core.** `ea_split.CORE_KEYS` = version, rule, salt, share_test, share_val, big_group_frac,
  eligibility parameters, role_of_group, group_units, station_group, station_role, stations, closed_stations and
  inputs_sha256. The writer and every reader compute the file's sha256 with the single function
  `ea_split.core_sha256` (canonical JSON of exactly these keys) and open the file only through `ea_split.verify` /
  `ea_split.load`, which also re-draws the roles from the stated rule and checks every status, and requires the ruled
  salt, S, val share and guard. **The check is never loosened to make a frozen file load**; a file that fails is
  refused. Test on synthetic splits: `tests/test_hydromet_sota_ea.py` (round trip; a change of any core key, salt /
  share_test / share_val included, is refused; the writer of the build script and the reader `ea_data.load_split`
  accept exactly the same file).
- **Pinned inputs (`inputs_sha256`, part of the core):** `raw/hydrosheds/HydroRIVERS_v10_eu_shp.zip`
  (500da7d3…68e5) and `raw/hydrosheds/hybas_eu_lev01-12_v1c.zip` (26e11e89…7be8), every shapefile member read from
  them (HydroRIVERS_v10_eu and hybas_eu_lev06/lev07 `.shp/.shx/.dbf/.prj`; the extracted copies are checked equal to
  the archive members), the UWWTR table `raw/ea_uwwtd/UWWTR_Art15_24thOct2024.ods` (ba928ea7…e54f) and the four
  metadata tables `stations.parquet` (74a94bb1…6c05), `measures.parquet` (1624efa0…d498), `stations_geo.parquet`
  (5456b0b0…fea2), `stations_strata.parquet` (f0a96f72…2801).
- **Split writer (run once, by the lead, after the T-E entry is stamped):**
  `python -B scripts/data/sota_hydromet_ea_build.py split --i-have-the-lead-ruling` writes
  `docs/preregistration/hydromet_ea_split.json` (0444), reads it back through `ea_split.load`, accepts only S = 0.50
  and refuses to replace an existing split with another sha256. `… preview` prints the same body without writing
  (default S = 0.50). With the inputs pinned above the core sha256 is
  **1395957e471284ff09dce3e26de576b50369a08c572e5a60afe60ad3419b0246** (preview of 2026-11-10; if the written file
  has another sha256, stop and report).
- **Loader (`ea_data.load_ea`)**: refuses unless **both** `hydromet_ea_split.json` and M1's
  `docs/preregistration/sonde_ea_split.json` exist and verify (`ea_split.load_pair`; lead ruling Q8), loads only
  eligible train/val-role stations, **drops every M1-test group for all years**, and refuses the EA-test block and
  test-role or reserved stations unconditionally. Contract checked on M1's file (M1 owns its draw rule): it carries
  `core_keys` (including `hydromet_ea_split_sha256` and `m1_test_groups`) and `sha256` = sha256 of the canonical JSON
  of those keys; its groups are distinct M6 train/val-role groups, at most 20 % of the 27 non-test groups (≤ 5); the
  list may be empty. `ea_split.seal_m1` writes that form.
- **Blocks:** EA-train = 2012-01-01 … 2021-09-30; EA-val = 2021-10-01 … 2023-09-30 (42-day embargo); **EA-test =
  2023-10-01 … 2026-09-30 (42-day embargo, reserved, never downloaded before the look)**.
- **Cells.** Zero-shot: models fitted on US data only. Fine-tuned: EA train-role groups × EA-train + EA-val blocks for
  fitting, EA val-role groups for early stopping / selection; every comparator fine-tuned by the same rule (GBM: boosting
  continued from the US model on the same EA rows with early stopping on the same val rows; physics: EA train stations
  added as air2stream donors and to the DO-ratio pool). **Primary test cell (both modes): EA test-role groups × EA-test.**
  Secondary (descriptive): test-role groups × pre-2023-10 days (spatial only); train-role groups × EA-test (temporal only).
- **Validation cells (lead ruling §2.4).** *Zero-shot go/no-go:* scored once, before any fine-tuning, on all England
  groups that are neither M6 test-role nor M1-test, in the EA-val block (scored days 2021-11-12 … 2023-09-30).
  *Fine-tuned go/no-go:* val-role groups × EA-val; declared low-information by the lead. M1-test groups are in no M6
  fit and no M6 validation cell.

## 5. Endpoints and strata (claim family per the arch-v2 amendment)
- Median per-site KGE of the 4 targets (≥ 60 valid days), paired basin-group bootstrap (B = 2,000), every-seed rule.
- T-H heatwave windows (forcing-defined, ERA5-Land Tmax at the station cell > its EA-train-block P95 for ≥ 3 days, + 3
  following days): RMSE of temperature mean and 7-DADMax on window days.
- **Near-discharge stratum** (public metadata only; defined before any score): a station is "near discharge" if a
  discharge point of the Defra/EA *Wastewater Treatment in England* UWWTR Article 15 table (as at 24 Oct 2024,
  `T_DischargePoints`, `dcpLatitude/dcpLongitude`; CC BY; `raw/ea_uwwtd/UWWTR_Art15_24thOct2024.ods`, sha256 pinned in
  the split file) lies (a) within 1 km of the station, or (b) snapped (≤ 1 km) to a HydroRIVERS reach on the station's
  upstream network with a river distance ≤ 5 km (`DIST_DN_KM` difference). The table covers agglomerations ≥ 2,000
  population equivalent only (storm overflows and small works are not in it) — disclosed. The EA "Consented discharges"
  register was not used (EA Conditional Licence, not OGL).
- **Minimum groups** (`hydromet_arch_v2_amendment.md` §9; lead ruling Q5): for an endpoint's paired cell, G = number
  of basin groups and G_eff = (Σ n_g)² / Σ n_g² (n_g = scored sites in group g). An endpoint is a claim endpoint only if
  **G ≥ 15 and G_eff ≥ 8**; otherwise it is descriptive and stays in its family with p = 1. Applied mechanically by
  one function, `ea_split.min_groups` (thresholds are module constants, never lowered): on metadata before the split
  freeze (§6) and on the realised paired cell inside the look before any score is computed. S, the thresholds and the
  list of 6 endpoints per family (`ea_split.ENDPOINTS`) do not change after the split freeze.

## 6. Power: per-endpoint G and G_eff from metadata (counts only; `sota_hydromet_ea_build.py preview`, 2026-11-10 15:09 EST)
Expected stations of a cell = eligible stations of the cell's role(s) whose catalogue open span inside the cell's scored
window is ≥ 60 days and whose catalogue lists the endpoint's measure (a temperature measure for the two temperature and
the two heatwave endpoints, a DO mg/L measure for the two DO endpoints). No reading is used. The realised cells will
be smaller (valid days ≥ 60, gaps; heatwave endpoints need ≥ 10 scored window days per site, which metadata cannot
show — their rows are upper bounds).

| Cell | Endpoint | Stations | G | G_eff | Status by metadata |
|---|---|---|---|---|---|
| **Test** (test-role groups × EA-test, 2023-11-12 … 2026-09-30) | KGE `temp_c_mean` | 141 | 21 | 11.15 | claim |
| | KGE `temp_c_7dadm` | 141 | 21 | 11.15 | claim |
| | KGE `do_mgl_mean` | 123 | 21 | 12.07 | claim |
| | KGE `do_mgl_min` | 123 | 21 | 12.07 | claim |
| | heatwave RMSE `temp_c_mean` | ≤ 141 | ≤ 21 | 11.15 (upper bound) | claim by the upper bound; decided on the realised cell |
| | heatwave RMSE `temp_c_7dadm` | ≤ 141 | ≤ 21 | 11.15 (upper bound) | claim by the upper bound; decided on the realised cell |
| **Zero-shot go/no-go** (train- and val-role groups × EA-val, 2021-11-12 … 2023-09-30; before M1's draw) | 2 temperature KGE + 2 heatwave endpoints | 141 | 20 | 6.53 | below the minimum (G_eff < 8) |
| | DO endpoints (2) | 131 | 18 | 6.18 | below the minimum (G_eff < 8) |
| **Fine-tuned go/no-go** (val-role groups × EA-val; before M1's draw) | 2 temperature KGE + 2 heatwave endpoints | 46 | 8 | 3.15 | below the minimum |
| | DO endpoints (2) | 44 | 7 | 3.09 | below the minimum |

- The same test cell serves the zero-shot and the fine-tuned family (6 endpoints each). All 12 test endpoints pass
  the minimum **by metadata**; the margin is small for G_eff (11–12 against 8) and the realised check inside the look
  decides.
- **Stated plainly:** both validation cells are below the minimum on metadata. The zero-shot go/no-go cell has enough
  groups (20 / 18) but G_eff ≈ 6.2–6.5, because the stations open in the EA-val block are concentrated (temperature: the largest
  group holds 44 of the 141 stations and the three largest 80); M1's draw (≤ 5 groups) can only lower it. The fine-tuned go/no-go cell (8 / 7 groups, G_eff ≈ 3.1) was
  already declared low-information by the lead. Neither is a claim cell; how the go/no-go gate G-D1b reads a cell
  below the minimum is the lead's ruling, to be written in the stamped T-E entry.
- **MDE restated with these G_eff** (amendment §9.1: MDE ≈ MDE₀ · √(14 / G_eff), MDE₀ = 0.014 / 0.022 / 0.07 / 0.06
  KGE for temperature mean / 7-DADMax / DO mean / DO min; best case, one comparator, US-like between-group variance —
  unverified and probably optimistic for zero-shot; **not a power claim**): test cell 0.016 / 0.025 / 0.075 / 0.065
  (× 1.24 for the worst Holm-6 step: 0.019 / 0.031 / 0.094 / 0.080); heatwave RMSE ≈ 0.5 °C (0.6 °C at G_eff = 8,
  scaled). Zero-shot go/no-go cell (G_eff 6.53 / 6.18): 0.020 / 0.032 / 0.105 / 0.090.
- Near-discharge stratum (descriptive): 53 of the 141 test-capable test-role stations are "near discharge" (§5).
- Superseded v0.1 figure: 399 stations open after 2023-11-12 in 41 groups counted all catalogue stations before
  eligibility.

## 7. Forcing pull (Open-Meteo, shared budget)
Size from catalogue metadata (2026-11-10): all 1,920 dated stations → 653 ERA5-Land 0.1° cells; deployment windows plus
365 warm-up days, 9 daily variables, calendar-year requests: ≈ 48,000 weighted calls; the EA-test window (2022-10-01 …
2026-09-30) of the 207 cells with stations open after 2023-11-12 ≈ 19,400 of them. At ≤ 40 % of the shared ledger cap
(8,000/day → 3,200/day) that is ≈ 6 days for the test windows and ≈ 15 days for everything; eligibility (§4) reduces it.
Order: test-window cells → val/train windows of eligible stations. Weather is not reserved (only EA sonde values are).
Actual request plan after the metadata eligibility of §4 (2026-11-10 14:31 EST, `forcing_requests.parquet`): 450 cells, 2,416
calendar-year requests, 43,011 weighted calls (16,342 for the test-window cells, pulled first) → ≈ 13.4 days at 3,200/day.

## 8. Single look (requirements for the T-E freeze)
Own FREEZE file written by the lead; full import-closure code hash; data pins (`ea_wq_sonde` content pins, split sha,
forcing and statics manifests); sealed registry claim row tied to pid/host; rescore refused after scoring; protected
access closed at sealing; every artifact verified before the claim; BaseException aborts; the test values are
downloaded inside the claimed look only. Look prefix `M6EA/test` (not `M6E`, which is the event track).

## 9. Deviations log
| Date | Change | Reason |
|---|---|---|
| 2026-11-10 | v0.1 draft | initial |
| 2026-11-10 14:31 EST | §3 reach-distance rule for US locations proposed (same ≤ 1 km rule as EA eligibility; counts from snap distances only); §7 actual request plan added; RiverATLAS subsets built. Still DRAFT: the split is on hold (lead), no EA value has been read for any statistic, the `daily` QC step has not run. | Symmetric schema; lead to rule with S |
| 2026-11-10 15:09 EST | **v1.0, submitted for the lead's stamp** (split not frozen; the writer is run once by the lead). §4: ruled rule S = 0.50 (was "the lead chooses S"); `hybas_l7`-not-null eligibility condition and group-id rule written out; a status for all 1,964 catalogue stations (replaces "ineligible stations get no group and no role"), with the L7 units of every group; one canonical core for writer and readers (`ea_split.py`; the v0.1 writer hashed 7 keys and the reader 4 — found by the verifier, fixed, test added); HydroRIVERS / HydroBASINS / UWWTR pins inside the hashed core; preview default S = 0.50; loader requires both split files and drops M1-test groups; validation cells of the lead's ruling added. §5: G ≥ 15 and G_eff ≥ 8 by `ea_split.min_groups`. §6: per-endpoint metadata G / G_eff and MDE restated. Visible when written: catalogue metadata and geometry only; no England reading QC-summarised, aggregated, fitted or scored; US lever numbers visible: 4 `M6SOTA/inner_v1` rows (C2a s0, C1b s0) and the C1a s0 training log — none of them enters any choice in this file. | Lead rulings `hydromet_arch_v2_lead_rulings.md` §2.1 Q4/Q5/Q8, §2.4 |
