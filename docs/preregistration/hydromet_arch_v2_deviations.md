# M6 SENTINEL-Hydro — architecture v2: deviations and required entries

Append-only file of dated, numbered entries (lead rulings §3 L2). It belongs to
`docs/preregistration/hydromet_arch_v2_amendment.md` (sha256
`a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656`, stamp row `b37a0319964c`,
2026-11-10T19:52:24Z) and `docs/preregistration/hydromet_arch_v2_lead_rulings.md` (sha256
`b144974a96c2140c933ccbe1e8d3247d9a1ffa0ad6663ee958237e33eb4179b0`). Neither file is edited. Each entry is checked
and stamped by the lead with its own registry row (`M6-hydromet`, `hydromet_arch_v2/prereg_snapshot`, carrying the
amendment sha256 unchanged and the sha256 of this file) before the run it affects. The engineer writes no stamp row
and does not change the mode of this file.

Code reads two kinds of line in this file, exactly as written, and only in the state the lead has stamped:
`PIN CHANGE: <label> sha256 <sha256>` (entry 1 §8.2) and `D2 RELEASED: sentinel/hydromet_sota/thermox/nets.py sha256
<sha256>` (entry 1 §8.3). Entry 1 contains neither line.

---

## Entry 1 — 2026-11-10, 19:12 EST

Written by the M6 architecture-v2 engineer (AI-assisted). Status: **submitted to the lead, not stamped**. Until the
lead's row exists, no non-smoke entry point runs (the guard of §8.1 refuses).

**What this entry asks to release when stamped** (rulings §4): the pin check; the references G1, G2, P, R-TD and
R-TD-W (both label forms each); then the D1 grid, selection, seeds, ablation (a), gate G-D1a, ablation (b) and
refits; the stacks when their members exist. Inner rows only (train-role inner groups × train block).
**Not released by this entry:** any validation-, test-block or England number; D2 grid runs and the DO stage (own
entries, L4); any Powell-derived row or prior; Tier 2; a freeze or a look.

### 1. (a) Shared constants module

- Path `sentinel/hydromet_sota/gso_mt/constants.py`, sha256 `fc668307b97a5299373d47bf3649b7255d2a1d6e0d5bdd127e6089f7eb4712a7` (a pinned file).
- `sentinel/hydromet_sota/gso_mt/solar.py` and `sentinel/hydromet_sota/thermox/physics.py` import every number from
  it and hold no numeric literal of their own; the unit test checks this.
- Unit tests: `tests/test_hydromet_sota_v2_constants.py` — 7 tests, all pass (2026-11-10 19:08 EST, pins in place). The tests compare each value with the
  printed text, recompute the worked examples of FAO-56 (Examples 2, 3, 8, 9, 14) and check the derived values.
- Status words: *verified* = read in the source itself on 2026-11-10; *secondary* = read in a paper that reprints the
  formula; *definition* = exact definition or unit identity; *nominal* = rounded textbook value, no page checked.

| Constant | Value | Source and place | Status | Used by |
|---|---|---|---|---|
| Solar constant G_sc | 0.0820 MJ m⁻² min⁻¹ | FAO-56 (Allen et al. 1998) ch. 3, eq. 21 | verified | 4 solar channels |
| Earth–Sun distance amplitude | 0.033 | FAO-56 eq. 23 | verified | solar channels |
| Declination amplitude, phase | 0.409 rad, 1.39 rad | FAO-56 eq. 24 | verified | solar channels |
| Days per year in eq. 23, 24 | 365 | FAO-56 eq. 23, 24 | verified | solar channels |
| Clear-sky coefficients | 0.75, 2·10⁻⁵ m⁻¹ | FAO-56 eq. 37 | verified | solar channels |
| Saturation vapour pressure | 0.6108 kPa, 17.27, 237.3 °C | FAO-56 eq. 11 | verified | G2 equilibrium temperature |
| Slope of the curve | 4098 | FAO-56 eq. 13 | verified | G2 equilibrium temperature |
| Psychrometric coefficient | 0.665·10⁻³ °C⁻¹ | FAO-56 eq. 8 | verified | G2 equilibrium temperature |
| Ratio of molecular weights ε | 0.622 | FAO-56 eq. 8 (symbols) | verified | vapour pressure from specific humidity |
| Wind profile 10 m → 2 m | 4.87, 67.8, 5.42 | FAO-56 eq. 47 | verified | G2 equilibrium temperature |
| Wind measurement height | 10 m | gridMET variable definition (Abatzoglou 2013) | definition | G2 equilibrium temperature |
| Penman wind function | 6.43 (1 + 0.536 U) MJ m⁻² d⁻¹ kPa⁻¹ | Shuttleworth 2007, HESS 11:210–244, eq. 1, p. 211 | verified | G2 equilibrium temperature |
| Clear-sky emissivity | 1.24 (e/T)^(1/7) | Brutsaert 1975, **via** Crawford & Duchon 1999, J. Appl. Meteorol. 38(4):474–480, eq. 16, p. 476 | secondary (original not read) | G2 equilibrium temperature |
| Emissivity of water | 0.97 | Jobson 1978, USGS OFR 78-1026, eq. 9 (PDF p. 34); McMahon et al. 2013, HESS 17:1331, Table 2, p. 1340 | verified | G2 equilibrium temperature |
| Albedo of open water | 0.08 | McMahon et al. 2013 supplement, Table S3 (PDF p. 143), Sect. S4 (PDF p. 27) | verified | G2 equilibrium temperature |
| Stefan–Boltzmann σ | 5.670374419·10⁻⁸ W m⁻² K⁻⁴ | NIST CODATA 2018 (exact) | verified | G2 equilibrium temperature |
| 1 atm, Celsius offset, hPa per kPa, s per day, degrees per hour | 101.325 kPa, 273.15 K, 10, 86 400, 15 | definitions | definition | unit conversions |
| ρ·c_p of water | 4.184·10⁶ J m⁻³ K⁻¹ | none (1000 kg m⁻³ × 4184 J kg⁻¹ K⁻¹) | **nominal, no page checked** | D2 only — not released |

Formulas (no new number): sunset hour angle, extraterrestrial radiation and daylight hours — FAO-56 eq. 25, 21, 34,
verified; cloud fraction and all-sky emissivity — Crawford & Duchon 1999 eq. 3 (p. 475) and eq. 11 (p. 476),
verified; vapour pressure from specific humidity — definition.

Stated limits:
- **Equilibrium temperature and exchange coefficient** (Edinger, Duttweiler & Geyer 1968, WRR 4(5):1137–1143): only
  the abstract was read. Status *secondary*. The G2 feature is built from the verified terms above.
- **DO saturation** and **barometric pressure**: the frozen routines `sentinel.models.hydromet.physics.do_sat_mgl`
  (Benson & Krause 1984; table test in `tests/test_hydromet.py`) and `pressure_atm_from_elevation` are imported, not
  copied. I did not re-read Benson & Krause on 2026-11-10. The frozen pressure routine uses a standard atmosphere at
  15 °C; FAO-56 eq. 7 uses 20 °C (81.5 against 81.8 kPa at 1800 m, 0.4 %). One formula is kept: the frozen one.
- **Shuttleworth 1993** (Handbook of Hydrology) was not read; the wind function is cited from Shuttleworth 2007.
- **Gas exchange.** Raymond et al. 2012 Table 2 eq. 1 is **NOT VERIFIED** (publisher HTTP 403). The Schmidt-number
  polynomial for O₂ is *secondary* (read in the LakeMetabolizer source, not in the paper). Neither is in the
  constants module and nothing released uses them (DO stage only; own entry, L4).
- The prior parameters of the G2 equilibrium-temperature input (shade 0.3, wind shelter 0.6; `gbm.yaml` `te_prior`)
  are implementation constants fixed before any fit, not physical constants. They have no literature source.

### 2. (b) Pins, licences, lever table

**2.1 `configs/hydromet_sota/v2/pins.json`** — sha256 `84958df5acf2e468a8ec926a5f59d779901f50a72375188b63723c71bff1c5d9`, mode 0444, written once at
2026-11-11T00:06:02Z (19:06 EST on 2026-11-10) by `scripts/data/sota_hydromet_v2_pins.py`; 38 files pinned by sha256. Record row (not a
stamp): registry `M6V2/pins_v1`, id `4a3779d5f227`.

It holds: the sha256 of every file returned by `gso_mt.data.pin_targets()` — the v2 pool tables and
`build_report.json`, `lever_sites.parquet`, the three upstream-forcing tables, the WQP national forcing, station and
StreamCat tables, `forcings_daily.parquet`, `streamcat_sites.parquet`, `sites.parquet`, the FINAL split
`sonde_split.json`, the track-N role table, `sonde_newsites_list.json`, the frozen reference's `data_info.json`,
`sites.py`, both reserved modules, `gso_mt/{grab,data,solar,constants}.py`, `thermox/physics.py`, the v2 build script
and every file in `configs/hydromet_sota/v2/` (D2, stack, GBM, P and `schema_variant.yaml` included); the values
pinned by content (`pin_values()`: the sha256 and the count, 42, of the `COMMON_STATIC` names); and the fingerprint and counts of the
pool in both label forms.

Pool as pinned (counts are binding, rulings §2.3):

| Count | With grab rows | No-grab form |
|---|---|---|
| Locations | 6,060 | 2,157 |
| USGS sonde sites (of them new) | 2,157 (445) | 2,157 (445) |
| Fit sites / inner sites / FINAL train-role fit sites | 1,901 / 256 / 1,456 | 1,901 / 256 / 1,456 |
| Grab stations, fit / inner | 3,545 / 358 | 0 / 0 |
| Hour-resolved rows, fit: T / DO | 256,636 / 231,089 | 0 / 0 |
| Hour-resolved rows, inner: T / DO | 25,299 / 19,503 | 0 / 0 |
| Sites with hourly sonde series | 421 | 421 |

Daily label cells (both forms): T mean 3,189,176; T max 3,471,588; T min 3,472,149; T 7-DADMax 3,370,246; DO mean
909,336; DO min 949,165; DO max 950,515; log Q 6,499,598; specific conductance 1,708,823; turbidity 469,869; nitrate
61,806. Dates 2005-10-01 .. 2021-09-30 (5,844 days), 22 forcing channels, 191 statics. Daily site tables: 2,157
files, 238,075,886 bytes, digest `1b8d15b5…bfd4`, tier `sonde_daily_refresh_20261108`. Pool fingerprint (11 component
hashes per label form) is in `pins.json` (`pool`, `pool_nograb`). A dry run at 2026-11-10T23:51:04Z produced the
fingerprint; the pin run reproduced it exactly. The pin script also wrote the data manifest `hydromet_sota_v2`
(sha256 `1c1fbfda60fc…`, recorded in `pins.json`).

- `upstream_forcing_sota.parquet` sha256 `925a21ecf6e3678eaaf824179e09e453851030673418aefa1285ddeea026984f` (the
  lever engineer's statement of 14:58 EST: complete and final). The pin script asserted this value.
- The pool is pinned with the 445 new USGS fit sites as built (lead, 15:00 EST). No reduced pool exists.
- **Two forcing-only keys.** 445 of the 447 keys of that table are fit sites. The other two (rule-1b
  `reserved_comid` sites) carry forcing rows only. The loader selects by fit-site key and stops if a forcing-only key
  reaches a frame (`N_FORCING_ONLY = 2` is asserted; `forcing_only_keys_excluded: 2` in the pool record; two unit
  tests in `tests/test_hydromet_sota_v2_guard.py`). The two site numbers are not written in any run record or in this file's
  pins.
- **Provisional gridMET values** (after 2026-09-09): the loader asserts that the last date used is not after the
  date filter. The pool ends 2021-09-30.
- **StreamPULSE:** the pool loader asserts zero StreamPULSE rows (`streampulse_rows: 0`). **Powell:** no arch-v2 code
  reads a Powell row. **England:** no arch-v2 code has read an England reading.
- **Track-N role table:** `pins.json` pins the file sha256 (`d4ee6dd2…f697d`). The lever pins record the content
  hash stored inside the file (`ffb41471…20e2d0`). Both are in `pins.json`; the pin script checked the content hash
  against `docs/preregistration/hydromet_sota_lever_pins.json`, together with `lever_sites.parquet`,
  `sonde_split.json` and `sonde_newsites_list.json`.
- **Sonde-tier site list:** enforced through the pool fingerprint (site keys, roles and label counts of the pool),
  which `verify_pins` callers compare at load time, and through the pinned `sites.parquet`.
- **Static inputs.** The pin script asserted 30 dynamic and 191 static inputs on the full pool before anything was
  written (scales from the 1,456 FINAL train-role fit sites; elevation missing at 0 locations). Smoke subsets gave 183 statics, because the frozen normaliser keeps a static column only if
  it is at most 50 % missing and not constant at the fit sites of the subset. Non-smoke neural runs assert 30 / 191.
- **Not pinned by file:** `runner.py`, `nets.py`, `trainer.py`, `criterion.py`, `gbm_ref.py`, `gates.py`, `stack/*`,
  the train and evaluation drivers and the pin script. Rulings §2.3 does not list them. Each run record stores the
  sha256 of its config; the look freeze records the import closure (rulings §2.3, look guards). If the lead wants
  these pinned as well, it needs a new pins file under a later entry — say so before the first fit.
- `ea_data.py` is the lever engineer's living file. `COMMON_STATIC` is pinned by content (sha256 of the 42 names); the file's
  sha256 is recorded for information only. The file pin follows in the T-E entry.

**2.2 Licence rows.** Appended to `docs/licenses.md` on 2026-11-10 (9 rows, from the row beginning "Water Quality Portal — M6 use" to the row beginning "Powell Center stream-metabolism estimates"): WQP with M6 use;
gridMET; ERA5 through Open-Meteo (existing rows of `forcings_daily` for 156 sites outside the gridMET domain; this
round makes no Open-Meteo call); EPA StreamCat; NHDPlus V2.1 (catchments, VAA, NLDI basins); EA sonde archive (not
read); HydroATLAS / RiverATLAS (column names only; no value read); UWWTR table (not read); Powell estimates (not
read). The USGS NWIS row already lists M6. **Limit:** the licence terms are those of the data manifests and of the
lever protocol. I did not re-check a provider page on 2026-11-10; each row says so. The licence statement of the
HydroShare copy of the NHDPlus VAA is marked "[verify]".

### 3. (c) Metadata G and G_eff of every T-B and N2 endpoint

Counts only; no label value, prediction or score was read. Rule (rulings Q5): claim endpoint needs G ≥ 15 groups and
G_eff = (Σ n_g)² / Σ n_g² ≥ 8, with n_g the number of sites (stations) of basin group g in the cell. The same function
(`gso_mt.gates.min_groups`) is used inside the look on the realised cell.

**T-B** (FINAL test-role sites, 564, block `test_b` 2025-10-01 .. 2026-09-30). "With the variable" = site-registry
flag `has_temp` / `has_do`. "Expected in WY2026" = also registry series end ≥ 2025-10-01 (377 sites).

| Endpoint | With the variable: G / G_eff / sites | Expected in WY2026: G / G_eff / sites | Metadata result |
|---|---|---|---|
| temp_c_mean | 41 / 14.32 / 554 | 39 / 13.06 / 373 | claim endpoint |
| temp_c_7dadm | 41 / 14.32 / 554 | 39 / 13.06 / 373 | claim endpoint |
| do_mgl_mean | 28 / 10.27 / 264 | 23 / 9.05 / 204 | claim endpoint (close to the floor) |
| do_mgl_min | 28 / 10.27 / 264 | 23 / 9.05 / 204 | claim endpoint (close to the floor) |
| heatwave RMSE, temp_c_mean | 41 / 14.32 / 554 | 39 / 13.06 / 373 | claim endpoint (upper bound) |
| heatwave RMSE, temp_c_7dadm | 41 / 14.32 / 554 | 39 / 13.06 / 373 | claim endpoint (upper bound) |

All 564 test-role sites: G 42, G_eff 14.08 (the amendment's 42 / 14.1).

**N2** (track-N test-role stations, 5,046). "With the variable" = any eligibility row of the variable. "Expected" =
at least 8 sample days of the variable in the role table's `test` block (WY2024–2025), used as a sign that the station
is still sampled.

| Endpoint | With the variable: G / G_eff / stations | Expected: G / G_eff / stations | Metadata result |
|---|---|---|---|
| T (hour-resolved) | 38 / 7.94 / 4,824 | 38 / 7.55 / 3,014 | **below the G_eff floor → descriptive, p = 1** |
| DO (hour-resolved) | 36 / 7.90 / 4,423 | 36 / 7.70 / 2,724 | **below the G_eff floor → descriptive, p = 1** |

**Result, stated plainly: both N2 endpoints fail G_eff ≥ 8 on metadata.** The number of groups is sufficient (36–38);
the stations are concentrated in a few groups. By the stamped rule (Q5, Q6, §2.2) the floor is never lowered, the
endpoints stay in the N2 family with p = 1, and N2 is reported as descriptive. The realised cell is checked again
inside the look. For the lead to decide: gate G-D1b says "N2: both endpoints"; I read it as still applying as a
go/no-go on the validation cell (it is not a claim), but the wording of an N2 result changes to "descriptive".

Limits of the metadata check:
- `wq_end` is one date per site, not per parameter. A site may have stopped one parameter earlier.
- 7-DADMax needs a daily maximum; heatwave endpoints need at least 10 window days. Neither is known from metadata.
  The counts for these endpoints are upper bounds.
- T-B DO endpoints have G_eff 9.05 on metadata. A few sites without WY2026 data can put the realised cell below 8.
- Sites that start in WY2026 are not counted (they are in no role).

Inputs (sha256): `sonde_split.json` `fd4bb14f…e06a80`; `sites.parquet` `0e280071…fd3a44`; track-N role table
`d4ee6dd2…f697d`; track-N `eligibility.parquet` `5fc2a3ee…f05aa4`; `stations.parquet` `2b8d3de2…d2ea21`.

### 4. (d) Joint look (Q7) and the lever models

- **Deviation recorded.** The v1 protocol planned a separate `M6/test_b` look for the v1 models. Under the stamped
  amendment (§9.2) and rulings Q7 there is **one joint look** after 2027-06-30: the v1 models and the arch-v2 claim
  model are frozen together. The v1 models are analysed exactly as frozen in `hydromet_protocol.md` (same models,
  Holm family, comparators, every-seed rule) and are reported as secondary with no new label. The arch-v2 families
  are additional. There is no earlier, separate v1 `test_b` freeze.
- **Lever models are SECONDARY** (lead relay 15:00 EST). No lever model is or will be a claim candidate (lever
  protocol sha256 `9848790ae242f8f6c5dfbc8e2e91a6270c1c5392e663e5054ff429be32fd3bd1`, snapshot row `71bdf660366f`,
  2026-11-10T19:57:20Z). Every lever configuration with 3 seeds at the joint freeze is co-scored by the same code
  and rows. The necessary condition stands: the claim model's point estimate must be better than that of every v1
  and lever model.
- The co-scoring of lever models is not written yet. It is done at the joint freeze on the lead's order.

### 5. (e) Lever inner rows and declarations

- Lever inner rows in the registry (`M6SOTA/inner_v1`), counted by experiment name only, values not read:
  **16** at 2026-11-11T00:06:02Z (19:06 EST on 2026-11-10) (the time `pins.json` was written). At 14:58 EST the lever engineer reported
  2 (`cc4ae2da6f60`, `9772f206ad93`, both C2a seed 0).
- Lever engineer's declaration (relayed by the lead): "I have read lever inner numbers today (C2a s0 scored; C1a/C1b
  log lines)".
- **My declaration:** I have read no lever inner number, before or after the v2 configs were pinned. I have not
  opened a lever run directory, a lever log or the value of an `M6SOTA/inner_v1` row.
- What I did see, for completeness: train-only smoke of my own arms on a small subset (grid point 1 of each arm,
  rule SMOKE-J). The neural smoke logs print the train-only criterion of the smoke subset per epoch. These numbers
  were not tabulated, not compared between arms and not used for any setting. No arch-v2 inner number exists.

### 6. (f) Corrections of the amendment text (L5)

- "[194 → …]" in §6.1 and §6.2 reads "[191 → …]". Statics = 188 of the frozen `ealstm_ws` reference + 3 NHDPlus VAA
  = **191**; dynamic inputs = 26 + 4 solar = **30**. Non-smoke neural runs assert 30 / 191.
- "39 RiverATLAS statics" in §3 reads "40". The T-E static set is `ea_data.COMMON_STATIC`: lat, lon and 40
  RiverATLAS attributes = 42 columns (`schema_variant.yaml`, status on hold).
- Amendment §5 gives R-TD "hidden 256". This is replaced by L11 below (hidden 100).

### 7. Lead ruling L11, verbatim

Received 2026-11-10T20:11:06Z (15:11 EST). The lead's note of 18:26 EST: the time "15:15 EST" written in the ruling
was approximate, the message was sent about 15:10 EST; the binding time is the stamp of this entry.

> Lead ruling on the R-TD recipe (15:15 EST, 2026-11-10): neither A alone nor B alone — BOTH published recipes are run as comparators. Thank you for flagging that A was in your own favour; that is the right call to escalate.
>
> Reason: Willard et al. 2024 does give hidden sizes and dropouts, so "where that paper gives none" does not strictly apply, and a 100-unit one-layer net alone would be a weaker comparator than the stamped text points at. A larger 4-layer net alone could also be the weaker one on our row count. I cannot know which is stronger without a validation number, and no comparator choice may depend on one. So the bar is raised, not chosen. This amends rulings §2.2 "No other value is run / one configuration" in the conservative direction only. Record it verbatim as lead ruling **L11** in entry 1 of `hydromet_arch_v2_deviations.md` (the stamped rulings file is not edited); I stamp it with entry 1.
>
> **L11 (binding).**
> 1. **R-TD** (the form described in amendment §5; unchanged name): one LSTM layer, hidden 100, dropout 0.5 — Rahmani, Lawson et al. 2021 (ERL 16 024025) Supplementary Table S2, and Rahmani, Shen et al. 2021 (Hydrol. Process. 35 e14400). Cite page and table in entry 1.
> 2. **R-TD-W** (new comparator, not a candidate): the configuration listed first in Willard et al. 2024 Table S6 — hidden size, number of layers and dropout from that row (as you read them: 570 / 4 / 2.2e-5), nothing else. The row is fixed now, before any fit; it is not chosen among the five on any number. Before `rtd.yaml`/`rtd_w.yaml` is pinned, the three values are checked visually against the arXiv PDF (not the flattened HTML) and the check is recorded in entry 1 with page and table. If the PDF shows different values for the first row, the PDF values are used and the difference is recorded. Whether Table S6 is rank-ordered is stated in entry 1 as read; the rule "first row listed" holds either way.
> 3. Everything else is identical for both recipes and is that of the amendment (§5 and §7): 365-day windows, inputs, heads, auxiliary targets, observation operator, label pool, AdamW with the common recipe (lr, weight decay, batch, steps, epochs, patience, clipping), inner rows, 3 seeds, grab share 0.5, both label forms (with grab rows and the no-grab published form). Willard's sequence length, batch size, learning rate and weight decay are NOT taken; say so in entry 1. Dropout in the multi-layer recipe is applied between LSTM layers; any further unstated detail is recorded in entry 1 before the first fit.
> 4. **Wherever the amendment or the rulings name R-TD as a comparator, both recipes are meant**: every IUT comparator set, every gate (the "≥ 0 on 3 of 4 endpoints" go/no-go applies against R-TD and against R-TD-W), every table, T-E, and the no-grab variant rule. The claim model must clear both.
> 5. **Stacks.** D3's member list is unchanged: its R-TD member is the one-layer R-TD of item 1. The mandatory control {R-TD, G1, G2} is unchanged. A second references-only stack {R-TD, R-TD-W, G1, G2} is also built by the same code and is in the comparator set whenever D3 is the claim model.
> 6. **Counting.** R-TD-W is a comparator and is counted outside the cap of 12, as one fixed configuration; no candidate slot is added or freed. Labels state: "R-TD was run in two published configurations (Rahmani 2021; Willard 2024, first row of Table S6); neither was tuned here."
> 7. No other R-TD value is run. Neither recipe is changed after its first fit. If an R-TD-W job cannot finish inside the 2 h job limit it continues with `--resume`; it is never shrunk. If it does not fit in GPU memory at batch 256, stop and report to me before changing anything.
> 8. R-TD and R-TD-W fits follow L8 (references first, with parity). Both gate on entry 1 (or a later stamped entry) like the R-TD transcription did.
>
> Noted for entry 1, no action from me: Raymond 2012 eq. 1 stays NOT VERIFIED and is used by nothing released (DO stage only); the Brutsaert 1975 secondary citation is acceptable if labelled as via Crawford and Duchon eq. 16.
>
> Continue with constants, pins code, G/G_eff and drivers; send me the entry-1 path and sha256 when ready.

**7.1 Transcription record (L11 items 1–3).**

- **R-TD** (`rtd.yaml`): hidden 100, one LSTM layer, dropout 0.5.
  - Rahmani, Lawson, Ouyang, Appling, Oliver & Shen 2021, Environ. Res. Lett. 16 024025, Supplementary Table S2
    (supplement `erlabd501supp1.pdf`, PDF p. 6 of 7), row "Final value": epochs 2000 / hidden 100 / rho 365 /
    batch 59 / dropout 0.5. Main text Sect. 3.3: "100 hidden units".
  - Rahmani, Shen, Oliver, Lawson & Appling 2021, Hydrol. Process. 35(11) e14400, Methods: hidden 100, rho 365,
    2000 epochs, batch = half the sites, dropout 0.5. Read in the **accepted manuscript**
    (`https://par.nsf.gov/servlets/purl/10299543`, PDF p. 9 of 55). The version of record was not read. The optimiser
    is not stated there.
- **R-TD-W** (`rtd_w.yaml`): hidden 570, 4 LSTM layers, dropout 2.216e-05.
  - Willard, Ciulla, Weierbach, Kumar & Varadharajan 2024, arXiv:2410.19865v1, Table S6 "Top 5 Hyperparameter
    Combinations for LSTM", **PDF p. 75**, checked visually on the rendered page on 2026-11-10 (PDF sha256
    `ad51e8d6cd12c5aaaf4679406d8b2a6bcceb35aab54c6a1548a3ded4d5cbc905`).
  - Columns: batch_size | n_hidden | num_layers | weight_decay | dropout. Rows as printed:
    496 | 570 | 4 | 4.128e-06 | 2.216e-05; 236 | 594 | 3 | 4.753e-07 | 0.0138; 521 | 699 | 5 | 2.974e-07 | 0.0883;
    201 | 760 | 3 | 9.684e-06 | 0.0518; 489 | 764 | 5 | 2.005e-04 | 0.0145.
  - The PDF shows the same values as the HTML reading. The first row is used: **570 / 4 / 2.216e-05** (the ruling's
    "2.2e-5" with the digits as printed).
  - **Rank order: not stated.** The caption says only "Top 5". The rows are in ascending order of n_hidden, so the
    table is probably sorted by hidden size and not by score. The rule "first row listed" is applied as ruled.
- **Not taken from Willard et al.** (their Sect. 3.2.3 and Table S6): sequence length 200 d, batch size 496, learning
  rate 0.001, weight decay 4.128e-06, patience 300 epochs. **Not taken from Rahmani et al.:** 2000 epochs, batch 59 /
  half the sites. Both recipes use the amendment's common recipe (`common.yaml`).
- Unstated details, fixed here before the first fit:
  - R-TD-W: dropout acts between the LSTM layers only (`torch.nn.LSTM(dropout=…)`), not on the last layer's output.
  - R-TD (one layer): dropout 0.5 acts on the LSTM output before the heads. The dropout placement of the authors'
    `hydroDL` code (inside the recurrent cell) is not reproduced.
  - Both: statics concatenated to the inputs at every step; independent heads for the daily targets; amplitude and
    phase heads feed the observation operator for grab and hourly rows only; `static_noise: 0.5` and
    `attr_dropout: 0.0`; the published form has `grab_share: 0` and no grab row.
- **Smoke (train-only, subset; SMOKE-J):** R-TD 0.069 s/step, 1.4 GB allocated, 129,244 parameters. R-TD-W
  0.661 s/step, 6.85 GB allocated (7.43 GB reserved), 9,619,364 parameters: it fits at batch 256. One epoch is about
  132 s of steps plus scoring; a full R-TD-W fit can pass the 2 h job limit and then continues with `--resume`
  (L11.7). It is not shrunk.

### 8. Guard, pin changes, releases

**8.1 Guard (L10).** Every non-smoke entry point (neural, GBM, P, stack, inner evaluation) calls
`gso_mt.runner.assert_released`, which requires: (i) `assert_stamped`; (ii) a lead-written
`M6-hydromet` `hydromet_arch_v2/prereg_snapshot` row that carries the amendment sha256 unchanged **and** the current
sha256 of this file, with this file read-only; rows written by a script whose name contains `sota_hydromet_v2` are
never accepted as a stamp; (iii) `verify_pins`; (iv) the sha256 of `pins.json` appears in the stamped text of this
file (it does: §2.1). Tests: `tests/test_hydromet_sota_v2_guard.py`.

**8.2 Later change of a pinned file — proposed route, for the lead to accept or reject.** `pins.json` is written
once. Without a route, the D2 entry (which may change `d2.yaml`) or an edit of `sites.py` by the lever engineer would
block every arch-v2 entry point for good. The code therefore accepts a pinned **code or config** file (label starting
with `sentinel/`, `scripts/` or `configs/`) whose sha256 differs from its pin **only if** this file, in the state the
lead has stamped, contains the exact line `PIN CHANGE: <label> sha256 <new sha256>`. Values pinned by content use the
label `configs/hydromet_sota/v2/pins.json#values`. Data tables, the split, the role tables and the pool fingerprint
have **no** change route. Every such line is a counted change under rulings §2.3 ("needs a stamped deviation") and is
listed in the run record (`changed_under_stamp`). If the lead rejects the route, I remove it before the first fit
(`data.py` is pinned, so that removal is itself the first use of a new pins file).

**8.3 D2.** `assert_arm_released` refuses every non-smoke D2 run until this file, as stamped, contains the line
`D2 RELEASED: sentinel/hydromet_sota/thermox/nets.py sha256 <sha256>`. That line belongs to the D2 entry (L4). D2's
torch code goes into the new file `thermox/nets.py`; the pinned `thermox/physics.py` is not edited.

### 9. Further details that the amendment and the rulings leave to the code (fixed before any fit)

1. **Cube reader.** The lever loader reads the daily site tables through the name
   `sentinel.models.hydromet.data.load_cube` (frozen code), which reads a site table whole and then re-indexes. To
   apply the date filter inside the file scan (rulings §2.3) without editing a frozen or lever **file**, the v2
   loader replaces that module attribute by its own `scan_cube` in memory for the duration of that one call and
   restores it afterwards. No file changes and the frozen code hash is unchanged, but it is a run-time substitution
   of a frozen routine and I state it as such. Tests: `scan_cube` equals the frozen reader on the train block and
   never materialises a later row; the name is replaced for one call only. Alternative, if the lead prefers: the
   lever engineer adds a `cube_loader=` parameter to the lever loader.
2. **P (physics-family reference).** One label form (P reads no grab row) and no seeds (deterministic). Member per
   primary endpoint = lowest standardised inner-sonde MSE among the site-cell and basin-mean members (ties: the
   earlier). Hour-resolved rows: per variable the member with the lowest term of J_grab; P's daily mean is its
   hour-resolved value ("daily-mean baseline"). The every-seed rule compares each claim-model seed with this one
   model.
3. **GBM selection** as in rulings §2.2; the hour-resolved model per variable is the grid point with the lowest term
   of J_grab (30-day station cap). Choices go to the registry (`M6V2/decisions_v1`) before any validation number.
4. **Stacks** (`stack.yaml`): weights per endpoint on the simplex, least squares on the members' 3-seed-mean
   early-stopped inner predictions; exact active-set solver. Hour-resolved endpoints use the J_grab row weights. In
   the no-grab label form the hour-resolved endpoints take the daily-mean weights. A member that does not exist is
   left out and reported (this can only be D2). A member is never averaged over fewer seeds than it has. An endpoint
   with fewer than 200 common inner rows is not stacked. The seed-k stack applies the one weight set to the seed-k
   refits. The cross-fit leaves one inner basin group out (one fold per inner group of the pinned pool).
5. **D2 grab share** 0.5 at every grid point; D2's physical parameters are in code (`thermox/physics.py`), not in
   YAML. Both are part of the D2 entry.
6. **G-D1a** is the plain comparison `relative reduction ≥ 0.03` on the 3-seed means of J_sonde in floating point.
   No tolerance is added.
7. **Registry experiments of this round:** `M6V2/inner_v1` (inner numbers), `M6V2/decisions_v1` (selections, gate
   outcomes, `variant_of`), `M6V2/pins_v1` (the pin record). None of them is a stamp. Smoke rows are refused by the
   inner-result logger.
8. **Not written yet:** gate G-DO; the validation-cell scoring script (never run without the lead); the D2 nets
   (`thermox/nets.py`) and the D2 kill-rule smoke; the co-scoring of lever models.

### 10. Tests and state at the time of writing

- `tests/test_hydromet_sota_v2_*.py` and `tests/test_reserved_loaders.py`: 105 collected, 104 passed, 1 skipped (the skipped test waits for `thermox/nets.py`), 2026-11-10 19:08 EST, with `pins.json` in place.
- Frozen M6 scoring code: `scoring_code_sha256()` = `5bd7d3292a29aa9beb46cd1da5e480ca58b3856eeb909dd7fd43b0e3561fd60b` (unchanged).
- No non-smoke entry point has been run. No validation-, test-block or England number exists for any arch-v2 model.

### 11. For the lead to decide with this entry

1. N2: both endpoints are below the G_eff floor on metadata (§3). Confirm "descriptive, p = 1", and whether gate
   G-D1b still requires both N2 endpoints on the validation cell.
2. The `PIN CHANGE` route (§8.2): accept or reject.
3. The `scan_cube` substitution (§9.1): accept, or ask the lever engineer for a parameter.
4. Whether driver, trainer, net and stack code must also be pinned by file (§2.1, last-but-one point).
