# SENTINEL-Orbit (M2): Track G amendments (draft)

**Status: DRAFT. Not approved, not funded, no download.** Drafted 2026-11-10 (EST) by the M2 engineer on the lead's
instruction (`docs/plans/architecture_research/LEAD_DECISIONS_20261110.md`, M2 section: "Track G amendments (draft
them; no download yet)"). Source: `docs/plans/architecture_research/M2_architecture_plan.md` §7.3.

**What this file is.** Four changes to `docs/preregistration/orbit_track_g.md` (itself still a DRAFT), to be merged
into it, or stamped next to it, **before** its licence checks finish and before any Track-G download. That file is
not edited here.

**State when this was written.**
- No row, count or label of sources G1–G5 has been downloaded or read.
- No Track-G result of any kind exists.
- The numbers in item 1 are counts of development (non-protected) rows of dev_v4.

Everything in `orbit_track_g.md` that is not changed below stays as drafted (sources G1–G5, the matchup rule, split
v3, the 8 GB disk cap, the look `M2G/test` with its own freeze file and the full single-look guard).

---

## G-A1. Correct the premise, de-duplicate, and restate the unseen block

**Correction.** `orbit_track_g.md` §1 says "All current data are US". That is wrong. dev_v4 already holds **485
non-US rows**:

| Source | Rows | Notes |
|---|---|---|
| Caravan-Qual (through AquaCorr) | 422 | |
| GLORIA (through AquaCorr) | 63 | |
| **Total** | **485** | 151 in-lake; train role 277, val role 208 |

Test-role non-US rows may also exist in the AquaCorr chip store. They are protected and were dropped from every
development cache; they are not counted here.

**Changes.**
1. **Premise.** Replace the sentence by: "Nearly all current data are US; dev_v4 holds 485 non-US rows (Caravan-Qual
   and GLORIA)."
2. **De-duplication against existing rows.** A G1–G5 row is dropped when an existing AquaCorr or WQP row of **any
   role** lies within 100 m on the same date. The existing row is kept. This is done on keys only (coordinates and
   dates), so no protected label is read.
3. **Unseen block, restated.** The Track-G test block is: in-lake, non-buffer rows of **test-role** lakes outside
   the US, dated **2024–2025**, **from sources G1–G5 only**, after the de-duplication above.
   - Non-US rows that already sit in the AquaCorr chip store are not part of the Track-G block. Test-role 2024–25
     ones are scored (and spent) by the headline look; test-role 2015–2023 ones belong to Test-S.
4. **Training additions.** Unchanged in form: non-US train-role × ≤ 2021 rows from G1–G5, after the same
   de-duplication, join the fit cell of the Track-G data version for every model, comparators included.
5. **Test-S interaction.** G1–G5 rows of test-role lakes dated 2015–2023 fall under the Test-S definition
   (`testxarchive_lake`, reserved 2026-11-10). If the `M2S/test` look has already been sealed when Track G is
   built, they are not added to it and are not used for training or tuning; they are counted and left unused. If
   Track G is built first, they are protected rows of Test-S like any other.

## G-A2. Coordinate-free prediction on Track G

**Reason.** Latitude/longitude inputs are a known out-of-region failure mode: a model can learn a location prior
that does not transfer to a new country (Prasad et al. 2026, cited in the plan §2.1). Track G is an out-of-country
claim, so the primary SatWQ-Net output there must not use coordinates.

**Rule (fixed before any download).**
- If the configuration given the Track-G cap (G-A4) is **v1**: its **context-off output** is primary on Track G.
  That is the prediction with the whole context vector (DOY harmonics and location unit vector) switched off, which
  v1 already writes for every run (`preds_ctx_off`). The context-on output is reported next to it, descriptively.
- If it is **v2**: the primary output is the coordinate-free A1 configuration **c2** when v2 is an A1
  configuration; when v2 is A0, it is A0's context-off output. If the selected v2 is c1 (with coordinates), c2 is
  still the Track-G primary and c1 is reported descriptively.
- No model is selected on Track G. The rule above only fixes which already-defined output is primary.
- Comparators: the location kNN is by construction a coordinate model; it stays in the comparison as the
  "location only" reference. The MDN_B takes no coordinates. The GBM family is set in G-A3.

## G-A3. GBM comparator family on Track G

- The GBM comparator family on Track G is **{imagery only, imagery + season/location context}**, each with its
  validation-selected grid member (existing 16-configuration grid), refitted on the Track-G data version.
- The member with the lower MdSA on the **Track-G val cell** (non-US val-role lakes × 2022–2023) is the comparator.
- This is optimistic for the baseline. It is comparator selection only: no SENTINEL model is selected, tuned or
  calibrated on the Track-G val cell beyond what `orbit_track_g.md` §4 already allows (power and calibration).
- Ensembling parity as in `orbit_arch_v2_amendment.md` §3.5 if the capped configuration is a v2 arm (3-seed bags,
  9-seed bag for the selected member; MDN_B with 10 initialisations).

## G-A4. Which configuration gets the 1-configuration cap

- The single SatWQ configuration scored on Track G is **the most recently frozen M2 configuration at the time of
  the Track-G freeze**: v2 if `FREEZE_orbit_S.json` exists by then, otherwise v1 (`FREEZE_orbit.json`).
- It is refitted on the Track-G data version with the same recipe and the same number of seeds or seed units.
- This rule is fixed now, before any download, so the choice cannot follow a Track-G validation number.
- The cap stays at 1 and the configuration is counted in `configs_tried.jsonl`.

## G-A5. Power (unchanged; expectation recorded)

- As drafted: protected counts only, MDE by the protocol §5a method on the Track-G val cell; the track is
  **descriptive** if under-powered.
- Expectation, recorded before any count: EEA, SYKE and SLU lakes cluster in space, so the number of 50-km test
  blocks may be small and the MDE large. A descriptive outcome is likely and will be reported as such.

---

## Points for the lead
1. Is the Test-S interaction in G-A1 item 5 acceptable (archive-year G rows of test-role lakes left unused if the
   `M2S/test` look is already sealed)?
2. G-A2 for v2 = c1: the draft makes c2 the Track-G primary even though c2 was not the selected v2. The
   alternative is to restrict the cap to the selected configuration and use its context-off output.
3. The data version name: `orbit_track_g.md` calls it `dev_v6`. With dev_v5m in between, it should be named when
   Track G is funded.
