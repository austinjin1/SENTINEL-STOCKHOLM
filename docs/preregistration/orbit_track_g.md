# SENTINEL-Orbit (M2) — Track G: new-geography lakes (international matchups)

**Status:** DRAFT pre-registration, 2026-11-09. It was written before any label from these sources was downloaded
(no rows, no counts, no results exist). It becomes frozen, with a sha256 recorded in `orbit_protocol.md` §11.5,
once the licence checks in §2 are complete. That still happens **before any download**. Lead approval: 2026-11-09
(lever 3, sota_campaign.md).

## 1. Why this track exists (rule 2 of sota_campaign.md, written before any result)
- The headline cell is under-powered (MDE 10–14 MdSA points; §5a). All current data are US (AquaCorr is US-only so
  far; WQP is US).
- Out-of-country transfer is the practical use case: European and Canadian lakes have different optical types (humic
  boreal lakes, alpine lakes), different monitoring protocols and different S2 viewing geometry.
- It adds training data that the baselines also get (rule 4) and gives an **unseen test block from a new source and
  new geography**.

## 2. Sources (fixed list; one is dropped if its licence check fails, with a dated note; none is added later)
| # | Source | Variables | Licence (to verify before download) |
|---|---|---|---|
| G1 | EEA Waterbase – Water Quality ICM (2026 release), lake water bodies | chl-a, Secchi, TSS, turbidity | CC BY 4.0 (to confirm on the EEA record) |
| G2 | SYKE VESLA surface-water quality (OData), Finnish lakes | chl-a, Secchi, turbidity (FNU expected) | Syke open data CC BY 4.0 (to confirm) |
| G3 | SLU Miljödata MVM, Swedish lakes | chl-a, Secchi, turbidity | to confirm |
| G4 | Ontario Inland Waters Lakes – Water Chemistry; Lake Partner Program (Secchi) | chl-a, Secchi | OGL – Ontario |
| G5 | ECCC Freshwater Quality Monitoring, lake stations | chl-a, TSS, turbidity | OGL – Canada (to confirm) |

- **Duplicates.** G1 can contain national data that are also in G2/G3. A G1 row is dropped when the same station
  coordinates (≤ 100 m) and date exist in G2/G3; the national source is kept.
- **LAGOS-US** is used only to check that no US row is duplicated; no row is added from it.
- Licences go in `docs/licenses.md` (entries marked PLANNED until verified).

## 3. Matchup rule (identical to the WQP build, `build_s2_wqp_matchups.py`)
- Surface samples only (depth ≤ 1 m, or no depth with a surface/epilimnion code).
- Lab chl-a only (no in-vivo sensor).
- Units harmonised to µg/L, m and mg/L. Turbidity: NTU family only; FNU goes to `turbidity_fnu` (§2.1 unit rule;
  most European turbidity is expected to be FNU).
- Same-day S2 L2A acquisition (Earth Search `sentinel-2-c1-l2a`) with the identical chip extraction and image QC
  (§2.1).
- Valid ranges and de-duplication as §2.1.
- Dates 2015-06-23 to 2025-12-31.

## 4. Split (frozen v3 rule, unchanged; `orbit_split_v3.json` sha256 `ffe288f2…`)
- Roles come from the same salted hash of 50-km lake blocks / 200-km non-lake blocks. HydroLAKES `na_eu` covers
  Europe and Canada. Years as §2.2.
- **Track-G test block (unseen):** in-lake, non-buffer rows of **test-role** lakes **outside the US** × **2024–2025**.
  All are protected (test role); only counts are read before the track-G freeze.
- **Track-G val cell** (for power and calibration only; no model selection): non-US val-role lakes × 2022–2023.
- **Training additions:** non-US train-role × ≤ 2021 rows join the fit cell of a new data version (`dev_v6`). They
  are given to every model, comparators included.

## 5. Models, selection and caps
- **No selection on track G.** The configuration selected on the main val cell (dev_v5, §6/§6a) is refitted on
  dev_v6, 5 seeds. MDN_B, GBM imagery + context and location kNN are refitted on dev_v6 with their val-selected
  members. **Cap: 1 SatWQ configuration** (counted in `configs_tried`).
- **Endpoint (secondary, named "track G").** The H1 comparison on the track-G test block, same metrics, role-block
  bootstrap, ≥ 3 seeds; E4a event metrics as §12. Its own freeze file `FREEZE_orbit_trackG.json`, its own look
  `M2G/test` with the full single-look guard (§9b), scored once. Wins and losses are reported, and the original
  headline is always reported alongside.
- **Power check** before the freeze: the counts of the track-G test block (protected counts only) and the MDE from
  val pseudo-test sampling (§5a method on the track-G val cell). If under-powered, the track is descriptive.

## 6. Disk and compute
- Disk cap **8 GB** of new data (lead, 2026-11-09). Expected: labels < 0.5 GB, chips at about 38 KB each, so a cap of
  150,000 chips ≈ 5.7 GB.
- `df -h` is checked before every download/build; stop and report if free space would drop below 60 GB.
- GPU: about 6 h for the dev_v6 refits (one process ≤ 2 h per seed, lock helper on GPUs 0/1).
- ETA: 6–9 days (licence checks 1 d; downloads and harmonisation 2 d; chips 1–3 d; refits 1 d).
