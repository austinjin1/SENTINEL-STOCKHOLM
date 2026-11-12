# M3 SENTINEL-Microbe — architecture round v2: lead review, rulings and stamp

**Written by:** the lead (AI-assisted), 2026-11-10, 19:45 UTC (14:45 EST).
**Stamps:** `docs/preregistration/microbe_arch_v2_amendment.md`
- sha256 `022841f3e94498efd94b4258449730472f790c5c2ef800556350cfe55d82eb6a`
- 845 lines, 61,069 bytes, last modified 2026-11-10 12:48:21 EST.

The amendment is stamped **as written, byte for byte**, together with the rulings and clarifications below. Where
this file and the amendment differ, **this file wins**. Both files are read-only (0444) from the stamp onward.

**The stamp is partial.** It releases the NEON work of §2, §3, §5, §6 and §7 under the rulings below. It does
**not** release arm D2 (§4), the added EA family (§9), track H4 (§10), any EA label download, or the GAS-3
download and fits (§8). Section 4 of this file says what each of those still needs.

## 1. What the lead checked before stamping

| Check | Result |
|---|---|
| File hash equals the hash the reviewers and the verifier worked on | yes (`022841f3…eb6a`, checked at 14:38:51 EST) |
| The whole file was re-read by the lead before the stamp | yes, lines 1–845 |
| Independent review | two reviewers (integrity; design and statistics) and one adversarial verifier. Stored in `docs/preregistration/reviews/arch_v2_20261110/`: `M3_review_integrity.json` (sha256 `c184293a…7afd`), `M3_review_design.json` (`cbdcff75…0576`), `M3_verify.json` (`392fcbff…6bee`). Verifier recommendation: stamp with rulings. |
| A development or validation number exists on a new arm or target | none found. The registry (73,107 rows) has 0 rows of `microbe/prereg_snapshot` and 0 M3 rows for NicheInvert, V4-MR, stacking, WA, GEO, GAS, QMP or EA. |
| An EA label file or the dissolved-gas product is on disk | none found under the data root (search by file name for `water_chem`, `warren`, `20097`). |
| Governing-input hashes in the amendment header | six of seven match the files now. The seventh, `LEAD_DECISIONS_20261110.md`, has been appended to since; its first 193 lines still give the recorded hash `eedd50b0…866b`. Its hash at this stamp is recorded in the registry row. |
| Appendix B hashes | all six reproduce. |
| A bar is lowered | no, once the rulings below apply. Three proposals that would have loosened a frozen rule are refused: the cap of 10 (ruling A3), V4-MR as a selectable model for H-M3 (B12), and the reduced Warren search as "the published method" (OQ8). |
| Stamp guard in code | `sentinel/models/microbe/stamp.py::require_stamp` exists and matches this stamp's registry row format, but no entry point calls it yet (ruling M4 below). |

## 2. Rulings

Each ruling is binding. The ids are those of the review files.

### 2.1 Slots, cap and eligibility

- **A3 / OQ1 — the cap stays 8.** Protocol §10.8 is not raised. Slot #5 is retired and is not reassigned. Slots
  #6, #7 and #8 each run on `lodo` **only**. GEO entries G2, G4 and G6 are dropped. The sentence "the stamp of
  this file is taken as approval of 10" is void, and the author's fallback (#6 on both tracks) is refused.
- **Ablations and controls** are a closed list: A0, A1, A2, A3 and `xgb_nichez_context`. Each is counted in the
  "configurations tried" total, runs on `lodo` only, is never selectable and never carries a claim. Nothing is
  added to this list without a stamped deviations file.
- **B12 / OQ3 — eligibility.** `xgb_v4mr` and `xgb_v4mr_stack` are **not** eligible as M\* for the H-M3 family
  (protocol §6: DNA-encoded candidates, no taxonomy labels). They run in slots #7 and #8, are reported with their
  twins, and act as comparators for D1 and D2. In the GEO family they may be selected only with the wording
  "table model with taxonomy labels" and with no architecture label. NicheInvert-S and NicheGeo-Phys are eligible
  only under the protocol §10.4 +context bar, which includes beating `xgb_v4clr_context` and
  `xgb_nichez_context`.
- **OQ2 — the 17-dimension master vector** is accepted as a declared deviation from the plan, on condition of
  parity (B5). `field_temp_c` is an auxiliary training target only. It is never a prediction-time input and never
  a scored target.
- **B8 — spares.** All spare slots are struck: GAS-3 has at most 3 candidates, the EA model list at most 5, H4
  at most 4. G10 is chosen mechanically once, when every remaining G entry is finished, and the choice is logged
  with registry row ids before G10 is fitted.
- **Reruns (amendment §2.5).** A rerun after an abort is the same configuration only if the registry and the run
  log hold no development metric from the aborted run. Otherwise it is a new counted configuration, which the
  cap does not allow.

### 2.2 Comparators, the D1 gate and labels

- **B5 — parity for labels and auxiliary targets.** `xgb_v4clr_mo` is fitted on all 17 master dimensions with
  missing labels masked (not mean-filled). `xgb_clr_stack` (17-dimension out-of-fold stacking inputs) runs
  **unconditionally** on `lodo`. Both are comparators and are not counted against the cap.
- **B3 — every comparator must be beaten.** Any SOTA or architecture label needs the paired lower bound above 0
  against **every** applicable comparator at the sealed look, not against one development-chosen baseline. For
  D1 and D2 the comparator set includes `xgb_nichez_context`, `xgb_v4mr`, `xgb_v4mr_stack` and `xgb_clr_stack`
  (each if run), and XGB-chained-physics for D2.
- **Gate condition 2 (amendment §2.6).** `xgb_nichez_context` and `xgb_v4mr` enter the maximum that defines the
  strongest simple baseline for the gate. The words "not derived from NicheInvert" do not remove them.
- **A1 — safeguards before the gate.** Before the D1 gate is evaluated, two things are run and logged on `lodo`:
  ablation A0 (NicheInvert-S with no context prior, κ = 0 and no deshrinking clip), and the calibration
  diagnostic (within-domain slope and predicted/observed SD ratio per target on D1's development predictions).
  WA-tol stays a mandatory comparator. The prior probability of gate condition 1 is recorded as about 0.35.
- **B6 / OQ10 — who evaluates the gate.** The D1 gate is evaluated **once, by the lead**, from registry rows,
  only after every unconditional §6 comparator, `xgb_nichez_context`, `xgb_v4mr`, A0 and the calibration
  diagnostic are finished on `lodo`. It uses the 8 primary targets only. The evaluation is logged with the row
  ids and is not repeated. M3 sends the lead the row ids and does not announce a gate outcome itself.
- **B4 — seeds.** A stochastic candidate must exceed each comparator in each of seeds 0–2, not only in the seed
  mean. A deterministic candidate is compared with each seed of every stochastic comparator, and its label
  states that the candidate has no seed variation.
- **B9 — power.** Power statements use the holdout-scaled minimum detectable difference at the family's first
  Holm step (0.05/16 for H-M3, 0.05/8 for H-GEO). A target is labelled underpowered from that figure, fixed in
  the power file before the freeze.
- **OQ8 — the Warren recipe.** Table 1 hyperparameters with a page reference if they can be transcribed;
  otherwise the full published search space at the published budget. A reduced 60-draw search may be run and
  reported, but it is not "the published method" and cannot be the comparator behind a label.
- **I-09.** `SYNTHESIS.md` §3.4 item 1, C3 and X7 bind M3 as written in the rulings on the D1 gate, H4 and D2 in
  this file.

### 2.3 Families and looks

- **OQ4 — a separate M\*_GEO** is accepted. It is selected by the same mechanical rule among the remaining G
  entries. M\* and M\*_GEO are fixed in one freeze. Each claim is scoped to its own family. Every report states
  that the error rate across the two families can reach 0.10. A GEO result never supports or rescues an H-M3
  claim.
- **B11 — which look.** GEO is scored in `M3/test`. GAS-3 is scored in Test-B only. Both are fixed now. A
  family that misses a §7.7 condition is reported descriptively in its own look. No family changes look after
  any development number on it exists.
- **OQ12 — date.** The freeze candidate is due no later than 2026-11-30. Arms unfinished on that date are
  reported as not run and cannot join later.

### 2.4 Guards, protected rows and records

- **M4 — the stamp guard is wired before any number.** Every entry point that can write a development number
  for D1–D4, for a comparator added by the amendment, or on a GEO, GAS or EA target calls `require_stamp` first.
  A test asserts that each such entry point refuses without the registry row. The entry points for D2, the EA
  family and H4 additionally refuse unless a registry row of `microbe/prereg_snapshot` carries the sha256 of the
  stamped deviations file that releases them (section 4), with a test.
- **I-03 — protected loaders.** `load_protected_labels` (`sentinel/data/microbe_geo.py`) and
  `load_design(allow_holdout=True)` (`sentinel/models/microbe/design.py`) require the open claim for every root
  that holds real data, however the root was passed. Only a synthetic test root is exempt, and a test asserts
  both cases. `processed/microbe_wq_geo/protected_labels.parquet` is set read-only (0440 or stricter) and is
  pinned by sha256 in the freeze. **This is stricter than the verifier's timing: it is done before the first
  D1–D4 or GEO development run**, together with M4.
- **I-04 — disclosed exposure.** The "what had been seen" list of the amendment is corrected here: per-target
  censored-flag counts and value-present counts of the 2024 holdout and of the reserve were computed, stored and
  printed (Appendix A.1, §7.1). A censored flag is derived from a protected label. Every censoring and
  detection-limit rule stays exactly as written. No rule, selection, family membership, power figure or kill
  decision may use those counts. From now on, count scripts emit only row and key counts for protected blocks.
- **I-05 — look guard and freeze pins.** Before the freeze candidate, the look guard is extended and
  independently reviewed: claims are per look (`M3/test`, `M3/test_b`, and the M3E look when it exists). The
  freeze pins the amendment, this rulings file, the log, every stamped deviations file,
  `microbe_split_lodo_x001.json`, `microbe_ea_protocol.md`, both GEO tables, `sentinel/data/microbe_geo.py` and
  the GEO scorer code.
- **I-06 — licences.** A licence row for Zenodo record 18668846 (each file used) and for NEON DP1.20097.001 is
  added to `docs/licenses.md` before the respective download. The Warren CSV stays "research evaluation only"
  and is never a training label.
- **I-11 — disk.** The EA read stream is not restarted until its script stops at 60 GB free, with the check made
  before every run. All new M3 data of this round (EA windows, GAS-3, the Zenodo files, GEO tables) totals at
  most 5 GB, reported by `du` before each download.
- **OQ11 — log and deviations.** `microbe_arch_v2_log.md` holds execution records only (what ran, when, row ids,
  counts). Anything that fixes a design choice needs a stamped deviations file (L3 below).

### 2.5 EA rulings that apply now (the EA work itself is not released; see section 4)

- **I-01 / OQ5 — designs B and C are rejected.** EA-dev labels are never scored in any look. If H4 runs: design
  A, one look on EA-test, in which H3 (the registered TON analysis and the added determinands) and H4 are scored
  once, together, under one freeze. If H4 does not run: the registered M3E analysis on all 22 groups, unchanged.
  The choice is made once, in the EA ids freeze, before any EA label file is on disk. The sentence "H4 never
  delays `M3E/test`" is void under design A.
- **I-02.** The words "the registered TON analysis is unchanged" are struck. Scoring TON on fewer than the 22
  registered letter groups, any added comparator and any changed model list are deviations from M3E §2.5, §3
  and §4, and are declared as deviations in every report. The 22-group figure is not computed if H4 runs.
- **A2 / OQ6 — H4 is on hold.** After the header check: if the label file carries a site key, EA-dev rows that
  match a frozen M5-X X-U test key (site and date window) are removed by a key list, and a counts-only overlap
  report is filed before the H4 freeze. If it carries no site key, H4 is not run and is reported as not run.
- **I-07 — wording.** H4 is named "letter-group-blocked". The words "area", "unsampled areas" and any
  geographic claim are not used unless written documentation of the letter code is filed before the H4 freeze.
  Every H4 report states that site disjointness between EA-dev and EA-test is unverified.
- **B10 — H4 carries no §15.1 SOTA label.** The single pre-declared composite sign-flip test over the 7 EA-test
  groups is its only confirmatory result. Per-determinand results are descriptive and are shown with
  letter-group cluster intervals. The row bootstrap is not used for any claim.
- **I-08 — the count-only script.** One hashed script, run once after the ids freeze, with its output logged
  for the lead, may parse the label file. It emits only the number of linked ids and, per determinand, the
  number of non-empty cells per partition and letter group. It computes, prints and stores no value, range,
  mean, quantile or censoring share. Its output is pinned in the EA freeze.
- **OQ7 — pins and the header check.** md5 and byte size now; sha256 at download, pinned in the EA freeze, for
  `water_chem.csv` and the two ASV files. The header check is the **only** contact with an EA label file that
  this stamp allows, under L4 below.
- **Design-A thresholds.** Under design A the thresholds of amendment §10.6 (at least 300 linked labelled
  EA-test samples and at least 5 EA-test groups with 20 or more) apply to H3 and H4 alike. Power is restated
  label-free for 549 runs in 7 groups in `microbe_ea_ids.json`.
- **H4 label source.** H4 is a declared deviation from M3E §3. It trains only on the CC BY `water_chem.csv` of
  Zenodo 18668846, including for TON. The EA datasheet text in `scripts/data/download_ea_biofilm.py` is
  corrected before H4 starts.

## 3. Lead clarifications (binding)

- **L1 — what the stamp is.** The stamp is the sha256 above, the 0444 mode of both files and the
  `microbe/prereg_snapshot` registry row. The git commit named in the amendment header and in §12 rule 1 is
  **deferred**; it and a public push must exist before any freeze on M3. Reports describe the record as
  "pre-specified and frozen in a public GitHub commit before testing (not registered on OSF)".
- **L2 — governing rulings.** The registry row records the hash of `LEAD_DECISIONS_20261110.md` at the stamp.
  The later sections of that file bind M3 as well as the lines the amendment cites.
- **L3 — deviations after the stamp.** Neither stamped file is edited. A deviation or a deferred design choice
  goes in a new file, `docs/preregistration/microbe_arch_v2_deviations_NN.md` (NN = 01, 02, …), with its date
  and reason. Each such file is reviewed, then stamped by the lead with its own registry row, **before** the
  affected run produces a development number or the affected download starts. A stamped deviations file is
  never edited; a correction is the next numbered file.
- **L4 — the header check of `water_chem.csv`.** M3 writes one script for it and sends the lead its path and
  sha256. The lead reads the script and approves it in writing before it runs. The script makes one range
  request, keeps only the bytes before the first newline in memory, prints and stores the column names and
  nothing else, and is run once.
- **L5 — the dissolved-gas download (OQ9).** M3 does not run it and never reads a key file. The lead runs it
  through the project downloader, after the licence row, a size check against the 1 GB cap and `df` before and
  after. It also waits for the project owner's go-ahead on the use of the access token. GAS-3 is priority 3 and
  is dropped first if the schedule slips.
- **L6 — existing-track work is not affected.** The run of the registered secondary scheme
  `release_001_to_002` (registry experiment `neon_dev/release_001_to_002_cv`, started 13:30 EST) is
  existing-track work under protocol line 159. W6 and the logged comparisons continue as planned.
- **L7 — two records owed in the log.** (a) The 98 grouped EA runs dated on or after 2023-10-01 (50 in EA-dev,
  48 in EA-test) are recorded with a reference to lead ruling 4b: the biofilm chemistry is outside M6's sonde
  reservation. (b) Before the count-only script runs, the author states where the figure "1,620 of 1,649 linked
  by id" in `microbe_ea_protocol.md` came from.

## 4. What M3 may do from the stamp onward, and what stays closed

**Step 0, before any development number (code and tests only):** the guard wiring of M4, the loader hardening
and read-only mode of I-03, and the unit tests the amendment §13 lists as owed. M3 reports the test results to
the lead, then starts step 1 without waiting for a reply unless a test fails.

**Released by this stamp (step 1 onward), on `lodo` unless stated:**
- D1 NicheInvert-S (slot #6); ablations A0 and A1; `xgb_nichez_context`; the calibration diagnostic.
- D3a `xgb_v4mr` (slot #7) with its twins `xgb_v4clr_3000` and `xgb_gfo`.
- The §6 parity list for families P and G, including RF[V4-CLR], TabPFNv2[V4-CLR] (GPU through the lock
  helper), WA-tol, WA-PLS, `xgb_v4clr_mo` and `xgb_clr_stack` as ruled in B5.
- The GEO development runs of the remaining G entries (G1, G3, G7, G8, G9; G5 once slot #8 is decided), and the
  GEO power file.
- D4: the label-free confound check first, then the +QMP twins on the GEO track under the amendment's kill
  rules; G10 as ruled in B8.
- Slot #8: after the lead has evaluated the D1 gate. If the gate fails, D3b `xgb_v4mr_stack` on `lodo`. If it
  passes, slot #8 is D2 and waits for its deviations file.
- The header check under L4. The label-free Zenodo ASV files (md5-pinned) after their licence rows.
- The label-free EA read stream once I-11 is met.

**Not released. Each needs a reviewed and stamped deviations file first (L3):**
- **D2 NicheGeo-Phys (B7).** The file fixes the optimiser, the epochs, the three scalars with their priors and
  the latent layout. The decoder constants come from a unit-tested shared constants module. "May use D1's
  results" means the gate outcome only. The 4 GPU-h stop rule uses the primary mean alone; its GEO clause is
  removed.
- **The added EA family (§9) and H4 (§10).** The file restates §9.3–§9.6 and §10 for design A with the
  thresholds and power of section 2.5. In addition, an M3E single-look guard with the campaign rule-7 hardening
  must exist and be independently reviewed. Until both exist, no EA label file is downloaded or opened; the
  header check under L4 is the only exception.
- **GAS-3 (§8).** The download waits for L5. The three floor values and the verified field names are written in
  a stamped deviations file before any GAS fit.

Resources: at most 6 CPU threads and one heavy CPU job at a time; GPU jobs through the lock helper, at most 2 h
each; the 60 GB disk floor. Every run gets a registry row, failed runs included. M3 makes no git commit.
