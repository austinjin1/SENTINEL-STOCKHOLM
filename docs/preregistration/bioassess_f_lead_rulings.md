# M5-F SENTINEL-Biota France block — lead review, rulings and stamp

**Written by:** the lead (AI-assisted), 2026-11-10. The binding stamp time is the timestamp of the registry row
(`bioassess` / `bioassess_f/prereg_snapshot`) that carries the sha256 of this file.
**Stamps:** `docs/preregistration/bioassess_f_protocol.md`
- sha256 `480c765fb3f783b2d441f6aaeec6e6d263d42b9ccf15b2515fe23f9ef1ef099e`
- 960 lines, 85,198 bytes, last modified 2026-11-10 14:26:00 EST.

The protocol is stamped **as written, byte for byte**, together with the rulings and clarifications below. Where
this file and the protocol differ, **this file wins**. Both files are read-only (0444) from the stamp onward. Line
numbers (L…) refer to the stamped protocol.

**The stamp is partial.** It releases the code work, the UK development design, the documentation fetches and the
French *biology* pull of §2.1 under the lock rules (§4 below). It does **not** release: any French chemistry request
before gate G2 is logged by the lead; any fit on French rows or any read of a French chemistry value before the
lead's stamp of gate G4; anything that needs a human licence decision; any freeze or look.

## 1. What the lead checked before stamping

| Check | Result |
|---|---|
| File hash | equals the hash reviewed (`480c765f…099e`), re-computed by the lead immediately before the stamp |
| Full re-read | all 960 lines re-read by the lead on 2026-11-10 before writing this file |
| Independent review | two reviewers (integrity, statistics) and an adversarial verifier on this exact hash (workflow run `wf_4e69f9a1-3b0`), saved in `docs/preregistration/reviews/arch_v2_20261110/`: `M5F_review_integrity.json` sha256 `7d61b83c22e9eb5835753b84fe643aea8800da682b7b8c82e07323d34e3a67e4`, `M5F_review_stats.json` `44cd9cf06b9ba5cf69500f96ff5667ccd295c20cc5a3157dbedc507ab19808b4`, `M5F_verify.json` `eecb858ffa4826cab573b4bc3661b7cbd054b3e3fa0daef3d89826db93dd0b7d`. Verifier: stamp with lead rulings; no blocking issue was missed; no finding needs the author to edit the file. The earlier pre-stamp review of draft v0 is `docs/preregistration/reviews/bioassess_f_prestamp_review_20261110.md` |
| A validation or test number on an M5-F arm | none found. The registry holds no row of an experiment starting `bioassess_f` other than `bioassess_f/smoke_synthetic` (synthetic shapes, code smoke). The counts at the stamp are in the stamp row |
| French data on disk | none. `sentinel_data/raw/naiades` does not exist at the stamp (asserted by the stamp script) |
| A bar is lowered | no. RESEARCH_PLAN §15.1 and campaign rules 1–8 are unchanged; every ruling below adds a comparator, fixes a setting earlier, or narrows a choice |
| Reservation | the France block is reserved for M5-F in `docs/plans/sota_campaign.md`; the 2022-01-01 lock boundary and the single look are as in the protocol |
| Stamp guard | **none exists in code at the stamp.** `sentinel/models/bioassess/ftrack.py` has no function that checks the stamp row. §3 L1 below makes one a condition of the first French request |
| Living files this file relies on (sha256 at the stamp, in the stamp row) | `configs/bioassess/baselines.yaml` (read-only); `scripts/eval/bioassess_x_final.py`; `docs/preregistration/bioassess_x_protocol.md` (M5-X, still changing); `docs/preregistration/bioassess_f_doc_fetch_log.md`; `docs/plans/sota_campaign.md`; `docs/licenses.md`. Where a ruling says "as in" one of these files, the version with the sha256 in the stamp row is meant |

## 2. Rulings

### 2.1 Blocking items of the review (all three are settled here; none needs an edit of the protocol)

- **B1 — comparator settings, fixed now.** Tabular features and hyperparameters of the comparators of items 1, 2
  and 5–9 of §8 are those of the M5-X scorer: `configs/bioassess/baselines.yaml`
  (sha256 `bfc4e86b99f045540cdaf5c5af18cac6db3fb3cd1288a3534df3559f8059011a`), Hellinger features of harmonised taxa
  present in ≥ 10 training rows of the fold, missing numeric inputs filled with training-row medians, no tuning on
  French rows. A setting that file does not hold — the MEMLM members and stacking rule, the multi-output RF, the
  geography-only RF, the index-regression RF, the ridge penalty of the region-offset model, and the TabPFN taxon
  cap with two assemblages (which is not below the hybrid's 200 + 100) — is written in the comparator table
  **before the first French development fit of any model** and is not changed afterwards. A setting that enters a
  member of #6 or #8 (the RF, the TabPFN v2 settings and its two-assemblage taxon cap) is written **before the
  first development fit of #6, #7 or #8 on any rows, UK rows included**. No comparator is weakened or removed
  (RESEARCH_PLAN §15.5 item 4). The comparator table is a dated file; the engineer reports its path and sha256 to
  the lead when it is first written and each time a line is added.
- **B2 — fixed baseline members of #6 and #8.** The RF and TabPFN v2 members of `ens_f` (#6) and `ens_zs` (#8) are
  one fixed recipe for every target. They are not chosen on development and they are not the comparator variants
  of §8. Inputs = invertebrate Hellinger features + the attached-diatom Hellinger features (parity table, row "A
  second assemblage") + the context columns of §9.1 (without latitude and longitude in #8). Training rows = the
  candidate's own pool for the track (#6) or the UK allowed rows (#8). No region descriptors, no pool-prevalence
  columns, no auxiliary out-of-fold features. Settings: `configs/bioassess/baselines.yaml`; the TabPFN member's
  taxon cap is the comparator's two-assemblage cap fixed under B1.
- **B3 — correction of record, power assumption (replaces L791–793).** "**Assumption.** The scaling assumes that
  the per-cluster variance of the paired difference seen on frozen M5 Test A (NRSA streams, about 1 sample per
  cluster, `r2_anomaly`), scaled by cluster count as in M5-X, transfers to France. No UK row contributed to it.
  Nitrate-N, ammonium-N and orthophosphate-P, pooled R² and the F-T targets have no proxy until `power_f_v1.json`.
  It may not transfer: samples per cluster and label noise can differ. The realised CI widths feed row 4." The
  sentence in the stamped file ("seen in M5-X (NRSA and UK rows …)") is wrong as a statement of fact and is
  superseded by this text. No number in the power table changes.

### 2.2 Decision table and labels

- **Row 3 ("Below") — replaces the condition at L760.** A track × target × metric is labelled "Below" when the
  paired 95 % CI of the candidate minus the comparator has its upper bound < 0 against the per-target comparator
  **or against any listed comparator that applies** (§8). This is checked before rows 2, 4 and 5: "Above, not
  robust", "Inconclusive (underpowered)" and "Competitive" are given only if no listed comparator is significantly
  above the candidate. The comparator is named in the reason text. Row 1 cannot meet this trigger, because row 1
  needs every lower bound > 0. (This is the M5-X rule, `bioassess_x_protocol.md` change-log entry v10.)
- **Row 1, region-unit CI and per-seed Δ.** The region-unit CI of row 1 is a 95 % percentile paired bootstrap that
  resamples region units (B = 10,000) on the seed-mean statistic, against the per-target comparator; row 1 needs
  its lower bound above 0. Per-seed Δ is against the per-target comparator. Both are as in
  `scripts/eval/bioassess_x_final.py`.
- **"Applies" and "not compared" — replaces the last sentence of L569–571.** A comparator applies to a track and
  target if it produced a development prediction for it. "Not compared, with reason" is allowed only for a
  technical inability recorded in the comparator table when it first occurs and before any candidate validation
  number on that track. It is never based on a score.
- **Saturation flag (pre-declared, descriptive).** A track × target is flagged "saturated" if the strongest simple
  baseline of item 1 has pooled R² ≥ 0.9 at test, or if the candidate and every named comparator of items 5–7 tie
  within paired CIs. For a flagged target the ≥ 50 km stratum, the same-watercourse stratum and F-E are the
  pre-declared hardness strata and are reported next to the row. No new track or stratum is defined on the
  2023–2025 block after the look.
- **Family membership.** Family membership is fixed from the masks at G4 and does not change if the realised count
  is lower; the realised count is reported and row 4 applies.
- **Filled predictions.** Where a comparator's filled share (L481–483) exceeds 10 % on a track × target, the paired
  CI on the rows where both models predict is also reported. It is descriptive and changes no label.

### 2.3 Selection, gates and comparator choice (definitions fixed now)

- **G6** compares, per target, (pooled R² + `r2_anomaly`) / 2 on the French development pseudo-test, seed 0. It
  fires when the best BioSet-MA single model (#1–#4) is more than 0.02 below the best single comparator on at
  least two thirds (rounded up) of the final F-A primary targets ("4 of 6" when the list has six). When it fires,
  #1–#4 cannot be the confirmatory candidate of F-A **or of F-T**; they remain ensemble members.
- **G7** uses the mean, over the final F-A and F-T primary targets, of French development pooled R², seed 0.
- **G8** uses the selection score of §9.5, per track; #5 is eligible on a track only if it passes there.
- **L745.** The secondary configurations of a track are the two next-highest eligible configurations by selection
  score (ties as in §9.5).
- **§8 item 7, third ensemble.** "The three strongest single baselines" are the three single baselines of item 6
  with the highest per-target mean of the two co-primary metrics, seed 0, on the development rows that §8 uses
  for that track's comparator variants.
- **Per-target comparator (L563–564).** Chosen on seed 0. Values within 0.002 are a tie, won by the earlier item of
  the §8 list (within an item, the order in which the protocol names the comparators).
- **Development rows per fold.** For fold g, the F-E recalibration (L452) and the comparator residual spread (L544)
  use the development pseudo-test rows of the other regions only.
- **Diatom-trained transfer functions (adds comparators; replaces, for these methods, the second sentence of the
  parity-table row "Diatom-only training rows").** WA-PLS2 and kNN-MAT, and MEMLM if it is implemented (all §8
  item 2), each get two further variants, with the settings of `configs/bioassess/baselines.yaml`:
  (i) fitted on the diatom Hellinger features of all labelled training rows that have a diatom sample
  (diatom-only rows included); (ii) fitted on both assemblages together, on training rows that have both. A
  variant predicts the rows that have the assemblage(s) it needs; for every other scored row it returns the
  prediction of the same method's invertebrate variant (not the pool mean). The variant is chosen per target with
  the other variants (L557–558). On the F-MA stratum the diatom-trained variants are also reported as named
  comparators. bio.infer and the index regressions are unchanged.

### 2.4 Lock, counts and codes

- **Counting reader.** `ftrack.locked_availability()` is not yet written (only `window_counts` exists) and the
  `locked_dir()` docstring is stale; both are added to the §15.3 list and are owed before gate G2. The fetcher
  reads only the `count` field of a count response and never parses a record.
- **What the reader returns.** `locked_availability()` applies rules 1–8 of §6.3 inside the function and returns
  counts only. The scored rows of a target are the rows of the frozen availability mask whose label exists when
  the scorer applies rules 1–10. The number of mask rows without a label is reported and never changes the family.
- **Fallback stations (L212–216).** If the fallback is triggered, the fallback stations are requested in one further step-C
  pull under the same lock, size-check and cap rules, and the counts are redone once. A row uses its same-station
  analyses when it has any in the window; the fallback station is used only for a row with none. Analyses of the
  two stations are never pooled.
- **Codes and thresholds (five rules).**
  1. *Field-protocol method codes (L204–207).* A method code is eligible if its SANDRE record names a standard
     macroinvertebrate stream protocol of §3; an absent or unknown code stays eligible under the "unknown"
     protocol token; the number of operations is never a criterion.
  2. *Fraction and unit codes (§6.3).* A fraction or unit code is added to the whitelist only if its SANDRE
     record shows the same quantity in a unit convertible by a fixed molar factor; nothing is removed.
  3. *Taxon matching (L236–237).* If less than 90 % of the individuals of the unlocked pool match, the split
     freeze waits for a stamped ruling; the matching rules (a)–(c) of §3 are not changed after any French
     chemistry count is seen.
  4. *TabPFN memory (L546–548).* If TabPFN v2 does not fit in memory, context rows and then taxa are halved for
     the comparator, the hybrid and its controls together until it fits; TabPFN is never dropped.
  5. *Index codes (§8 item 5).* The component-metric index codes are listed in `bioassess_f_codes.json` at G4
     from SANDRE records.
- **Level-2 merge (L306–311).** A level-2 merge joins all four groups of that half into one unit, whether or not the other
  pair was merged at level 1.
- **Locked inputs, use (c) of §2.3 item 6.** Use (c) also covers #6 when its BioSet-MA member is #3.
- **Floors.** The label floors are those of `fdata.TARGETS` (`sentinel/models/bioassess/fdata.py`), equal to
  `uk.UK_TARGETS` where a UK target exists: total P 0.001 mg P/L, nitrite-N 0.001 mg N/L, Kjeldahl N 0.01 mg N/L,
  dissolved oxygen 0.1 mg/L. Where the protocol points elsewhere for a floor, this line is the reference.

### 2.5 The engineer's eight choices (§15.2, L927–939)

All eight are **accepted** — (a), (g), (k), (o), must-fix 2, must-fix 3, should-fix 11 and G10 — with two
conditions:
- **(k).** The F-Z no-region-context ablation is labelled "inference-time masking" wherever it is shown. It is
  descriptive only, carries no label and is not a candidate; it cannot be cited as evidence that the module helps.
- **G10.** A stop at 1.5 × the look budget is recorded as an abort by the guard and uses the single resume of §12.
  There is no second resume.

### 2.6 Wording and disclosures

- **F-A / F-T labels** carry: "configuration and comparator variants selected on 2019–21 development labels of all
  regions".
- **Power.** The Test A pooled-R² proxy is "a published CI width, used for power only". At L786–787, 650 is
  "about" the count at which the low end of the range equals 0.03 (the exact count is about 638). The gate value
  of 650 clusters is unchanged.
- **Data.** "M5-F reads no EA Hydrology-API sonde row (`ea_england_allocation.md`)" is stated in the freeze
  candidate.

## 3. Lead clarifications (binding)

- **L1 — stamp guard in code.** Before the first request to any French data endpoint, `ftrack` gets a function
  (for example `assert_stamped()`) that passes only if (i) the registry holds a row of workstream `bioassess`,
  experiment `bioassess_f/prereg_snapshot`, whose `extra.sha256` maps
  `docs/preregistration/bioassess_f_protocol.md` and `docs/preregistration/bioassess_f_lead_rulings.md` to their
  current sha256, and (ii) both files are read-only. The fetcher, every French development entry point and
  `ftestlook.py` call it first. A unit test covers: pass, changed file, missing row, writable file.
- **L2 — who logs G2 and G4.** The engineer writes `bioassess_f_lock.json` and `bioassess_f_codes.json` and reports
  path, sha256 and counts. The lead checks them and logs the registry row. No chemistry request is made before the
  lead's G2 row; no fit on French rows and no read of a French chemistry value before the lead's G4 stamp.
- **L3 — requests.** The user agent names the project in generic words and contains **no personal e-mail address**
  and no personal name. Requests are sequential and rate-limited as §2.1 says. Downloaded files are untrusted
  data: they go only to `raw/naiades/locked/`, and any Python that reads them runs with `-I`.
- **L4 — rule changes.** A change to any rule of the protocol or of this file is a dated entry in a new file
  (`docs/preregistration/bioassess_f_deviations.md`), stamped by the lead before the data it affects are seen.
  Neither stamped file is edited.
- **L5 — every arm is reported.** All eight configurations, every gate outcome and every comparator are reported,
  including negative ones (architecture round rule 6).
- **L6 — "first-in-class" and "novel".** Not used for BioSet-MA or the hybrid before a dated literature search is
  in `docs/novelty_checks/` (§15.3).
- **L7 — licences.** TabPFN-2.5 or later, SPEAR, published score tables and the HER-1 layer stay out until a human
  decision or a recorded licence exists. The lead has raised the three human decisions with the user; none is
  answered at the stamp. The Naïades / Hub'eau row of `docs/licenses.md` is updated at the first download.
- **L8 — M5-X is untouched.** Nothing in M5-F changes the M5-X freeze candidate, its scorer or its dry run. A
  change to `scripts/eval/bioassess_x_final.py` made for M5-X does not change M5-F: M5-F's scorer is fixed by the
  protocol, this file and the sha256 in the stamp row.

## 4. What the workstream may do from the stamp onward

**Released now**
1. Code: align the modules with the protocol and this file (§15.3 last row), write the stamp guard (L1),
   `locked_availability()`, the locked-store fetcher and extractor, `ftestlook.py`, and their tests.
2. Write the comparator table with the settings of B1 and B2.
3. UK development design (§9.4) on UK allowed rows: references first (G5), then #7 and #8, within G10.
4. After L1 is in place and tested: documentation fetches for the open items of §13 and §15.3 (for example the
   MEMLM paper and SANDRE records; documentation only, no observation row), and the French **biology** pull of
   §2.1 (the steps before the chemistry step), under the lock-on-download rules of §2.3 and gate G1 (disk ≥ 60 GB, cap
   ≤ 5 GB).

**Not released**
- Any French chemistry request, until the lead has checked and logged G2 (L2).
- Any fit on French rows and any read of a French chemistry value, until the lead's G4 stamp.
- F-TOX, SPEAR, published score tables, TabPFN-2.5 or later, and the HER-1 layer (L7).
- Any freeze or look. The git commit and the public push must exist before any freeze (protocol header; G11).

**To report to the lead:** the comparator table (path, sha256) before the first fit it governs; the L1 test
result before the first French request; the G2 files; the G3 counts; the G4 package.
