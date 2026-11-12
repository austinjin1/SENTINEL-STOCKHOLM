# Architecture round v2: lead decisions (2026-11-10, 12:10 EST)

**Trigger.** User directive 2026-11-10: "deep research into these architectures ... build novel architectures and also
if simple ML is doing the best the task needs to be harder so more data more predictive targets".
**Inputs.** `M{1,2,3,4,5,7}_architecture_plan.md` in this directory (M6 and `SYNTHESIS.md` still being written).
**Written before** any result exists on any arm or track approved below. Nothing here lowers a bar: RESEARCH_PLAN
§15.1 and `sota_campaign.md` rules 1–8 are unchanged.

## Rules for this round (all workstreams)

1. **Amendment first.** Each workstream drafts ONE new file `docs/preregistration/<ws>_arch_v2_amendment.md`
   (never an edit to a frozen file). It holds, for every approved arm and track: the practical reason, the unseen
   test block, power from counts/metadata only, comparators with parity, configuration caps, the selection rule,
   and every gate or kill rule. The lead reviews and stamps it (sha256 + `<ws>/prereg_snapshot` registry row +
   git commit).
2. **Before the stamp** a workstream may write code and tests, build data (labels of protected rows: counts only),
   and run smoke tests on synthetic or train-only data. **No validation number on a new arm or track before the
   stamp.**
3. **References first.** The strongest published method (re-run here) and the strongest simple baseline, with the
   same inputs, labels, auxiliary targets and ensembling as the candidate, are built before or with the candidate.
4. **Nothing already queued is delayed.** Existing freezes and looks proceed; v1 results stay reported.
5. **Compute.** The plans assumed 2 shared GPUs. GPUs 0–9 are now allowed (lock helper, `--min-free-mb`, ≤ 2 h per
   job, never a GPU with another user's process). CPU ≤ 6 threads per job; load is ~40 of 48 cores, so prefer GPU
   work and keep CPU jobs sequential per workstream. Disk floor 60 GB free on `/`.
6. **Every arm and track is reported**, wins and losses, scoped to its track.

## M1 Sonde

- **Approved to build:** references R-NH (Nearing-style hindcast–forecast LSTM), GBM-NWP, PF-GBM, Chronos-2-cov
  zero-shot; candidates SONDE-HF (rank 1) and E2 with the matched baseline-ensemble control E2-B.
- **Gated:** PhiDA-Lite (gate C3) and Chronos-2-NWP, both by the step-1 information-use test (train-block inner CV
  only, decision rule as in plan §8).
- **Tracks:** H-1 (M1-N forecast mode with GEFS) approved subject to the HEAD-request size check; transfer only as
  stream-extract-delete, extractions < 1 GB, disk floor respected. H-2 (M1-H hourly DO / hypoxia hours) approved;
  the protocol must state that targets are hourly IV values themselves, so the failed IV→daily rule does not apply.
  H-3 forcing strata approved as secondary. H-5 StreamPULSE approved as secondary, lock-on-download, protocol
  before download. H-6 (EW-H1h) is **descriptive by default**; confirmatory only if the metadata-only power check
  written into the M1-H protocol passes, and always disclosed as defined after the IV-rule failure. H-4 waits for
  M6. H-7/H-8 unchanged.
- **Time-critical:** the amendment must be stamped before any NWP-based M1-N validation number exists.

## M2 Orbit

- **Approved:** Appendix A.1 (Test-S), A.2 (arms A0 and A1; caps A0 1, A1 3), A.3 (Track N: TP, TN, DOC), and the
  dev_v5m multi-target build. **Seed unit** = 3 inits (mixture average) + EMA is approved **only with parity**:
  every learned comparator gets the same 3-init averaging, and the GBM gets 3-seed bagging.
- **Gated / not funded yet:** A2 (gate as written), A3, A4, Track G amendments (draft them; no download yet).
- **The headline M2/test look with v1 is not delayed.**
- **Test-S is reserved from now** (see `sota_campaign.md`).

## M3 Microbe

- **Approved:** §8.1 items 1–3 and 5–6: GEO protocol (GEO-8) in the same sealed look as H-M3; reserve slots
  #6 NicheInvert (D1), #7 V4-MR (D3a), #8 NicheGeo-Phys (D2) only if the D1 gate passes, else stacking (D3b); the
  DNA-LM slot is dropped; the M3E multi-determinand amendment; GAS-3 amendment before its download; qPCR input
  amendment.
- **Ruling 4a (H4, EA-internal area-blocked):** approved as a separate conditional track. The EA-dev / EA-test
  partition is written to `microbe_ea_ids.json` before any EA label is downloaded. EA-dev rows are never used to
  train the NEON→England (H3) models. Any EA chemistry row that is a frozen M5-X test key is excluded from H4
  training.
- **Ruling 4b:** M6 is told about M3's EA use by the lead (no overlap: M6 reserves Hydrology-API sonde data).

## M4 ChemTox

- **On hold.** The M4 workstream session was stopped by the platform's safety system on 2026-11-10 and the user
  has not yet chosen how to resume. The plan is on file; nothing in it is dispatched. Jobs already queued run to
  completion unsupervised.

## M5 BioAssess

- **M5-X is left alone** (freeze candidate → dry run → pre-test review, as planned).
- **Approved:** the France block (M5-F; tracks F-A, F-Z, F-T in one look and one Holm family; F-E, F-MA secondary;
  F-TOX exploratory), BioSet-MA plus the TabPFN hybrid, cap ≤ 8 configurations. New-data cap for M5-F: 5 GB.
  UK diatoms and prior-365-day windows (on disk) may be used for development on allowed pool rows only.
- **Not funded:** full GeoNiche. TabPFN-2.5 weights need a human licence decision and are not used; the installed
  TabPFN v2 stays the comparator.
- **M5-F reservation** recorded in `sota_campaign.md`.

## M7 Lite

- **Approved:** label recovery (FNU, NTRU, SSC, censored rows) as a new data version; ANCHOR-Lite (rank 1) and
  WaterSet-Lite as its alternative residual branch; re-run published comparators (HydroVision-style DenseNet121
  regressor, CyFi-style LightGBM); track **P2** (rivers/sondes FNU turbidity at unseen water bodies).
- **Ruling on labels:** the frozen `lite_protocol.md` §8a rule stands for the national track. **No target-scoped
  label is added there after validation has been seen.** A scoped SOTA claim is possible only on P2, pre-registered
  before any result.
- **Gated / deferred:** CIR-Lite-R (calibration gate), SparK SSL, P3 France and P4 NLA.

## M6 HydroMet

- Pending its plan. Track R and the lever build continue.

## M4 ChemTox: update 2026-11-10 12:25 EST (replaces "On hold" above)

The user instructed the lead to resume M4 ("keep going we using fable 5.2 now"). A new M4 agent takes over from
`docs/handoff/chemtox.md`. Written before any result exists on T1, T6 or any new arm.

- **T2 (unseen taxa):** runs to completion exactly as pre-registered and is reported whatever the outcome. No
  architecture or configuration is added to it.
- **References first:** R1 monotone-constrained GBM on continuous duration, R2 interval-censored AFT GBM, R3
  multitask-by-stacking GBM, and the plan's R4–R5, on the full acute + chronic + censored frame.
- **New tracks approved:** **T1** (new primary; novel chemistry, blocked on ECFP4-Butina clusters at cutoff 0.4,
  held-out cluster block as in the plan) and **T6** (secondary; duration extrapolation, T6a/T6b). The T1 cluster
  file and its hash are frozen before any model, and before any embedding run, touches the split.
- **New targets approved:** chronic endpoints, the censored acute rows, and the acute-to-chronic ratio. The ratio
  is a reported output, not a §15.1 label, unless a pre-registered comparator exists.
- **Candidates approved:** concatenated pretrained-embedding features, TimeHead, TED (with ablations A1–A6), and
  MMoE as comparator and ensemble member. Caps as in plan §8.2: T1 384 searched configurations, T6 128, plus
  fixed baselines; every configuration counted.
- **H4 (BCF):** enters only as ablation A5 and only if its join/QC report lands first; otherwise dropped and
  recorded.
- **Amendment:** `docs/preregistration/chemtox_arch_v2_amendment.md`, covering plan §8.3 items 1–8, stamped by the
  lead before any fit on T1 or T6.
- **Compute:** GPUs 0–9 are available, so new work need not wait for the chemprop queue to drain.

## M1 Sonde: amendment stamped 2026-11-10 12:23 EST

- `docs/preregistration/sonde_arch_v2_amendment.md` (sha256 `6c7b169d…60226`) is stamped as written, with lead
  rulings R1–R5 and clarifications L1–L5 in `docs/preregistration/sonde_arch_v2_lead_rulings.md`
  (sha256 `7cbf3987…ac55`). Both files are 0444. Registry row `sonde/prereg_snapshot`, id `2921bb1ca000`.
- No validation number existed on any new arm or track at the stamp (registry and run folders checked).
- M1 is cleared to train: step-1 test, references, SONDE-HF, M1-H; GEFS pilot in a separate decode environment.
- **Stamp procedure, all workstreams (update to rule 1):** the git commit of a stamp is deferred while an
  environment pin of a pending look depends on the repository HEAD. Until then a stamp is: sha256 check, both files
  0444, and a `<ws>/prereg_snapshot` registry row. The commit and a public push must exist before any freeze on the
  tracks concerned.

## M6 HydroMet: rulings 2026-11-10 12:26 EST (replaces "Pending its plan" above)

Input: `M6_architecture_plan.md`. Written before any result exists on any arm or track approved below.

- **References first (approved to build):** R-TD (published top-down LSTM, re-run on identical data), G1, G2 PF-GBM
  and the physics family P. The grab-sample formulation for the GBMs in plan §4.0 is **mandatory parity**: the GBMs
  train on the same instantaneous rows as D1. R-EA is optional, descriptive, outside the cap, and last in the queue.
- **Candidates approved:** D1 GSO-MT (rank 1); D2 ThermOx-T temperature stage, subject to its smoke-test kill rule;
  the D2 DO stage only if gate G-DO passes; D3 stack with its references-only control stack. D4 is deferred.
- **Caps (two budgets, both fixed here; every configuration is counted in `configs_tried.jsonl`):**
  - **Architecture v2: ≤ 12 neural-network configurations**, as in the plan §8 table: D1 3 + 2 ablations (the
    "no auxiliary heads" ablation is dropped), D2 temperature stage 3 + 2 ablations, D2 DO stage ≤ 1 (gated),
    R-TD 1 fixed. GBMs: 3 per variant.
  - **Levers (EA-LSTM, `hydromet_sota_protocol.md` §4): ≤ 8** — C1a, C1b, C2a, C2b, C12 (gated) and lever 4 (≤ 3,
    gated). **Lever 3 (pretraining) is deferred and its 3 slots are retired**, not transferred: Caravan (≈ 25 GB)
    exceeds the data cap, and CAMELS-US forcings do not match the project's forcings.
  - Row A* of the lever protocol ("arch, within the remaining cap, ≤ 12 in total") is **replaced** by the
    architecture budget above. This raises the M6 total from ≤ 12 to ≤ 20 and is disclosed as a change. Facts at the
    time of this ruling (12:30 EST): the registry holds no lever or architecture result row; one lever run
    (C2a, seed 0) is in training and has not finished; the lead has not read its log.
  - Lever models are secondary arms. They can be the claim model of a track only through the claim-model rule
    written in the amendment before any result.
- **Ownership (no two agents edit the same file):**
  - *Lever engineer:* levers 1, 2 and 4, the EA England build and `hydromet_ea_protocol.md`; its existing files in
    `sentinel/hydromet_sota/*.py`, `scripts/{data,train,eval}/sota_hydromet_*.py`, `configs/hydromet_sota/C*.yaml`.
  - *Architecture engineer (new):* the amendment, references R-TD and the grab-sample GBM formulation, D1, D2, D3;
    new files only in `sentinel/hydromet_sota/{gso_mt,thermox,stack}/`, `scripts/{data,train,eval}/sota_hydromet_v2_*.py`,
    `configs/hydromet_sota/v2/`, `tests/test_hydromet_sota_v2_*.py`; handoff `docs/handoff/M6-hydromet-archv2.md`.
    It imports the lever engineer's modules read-only and consumes the EA build; it does not rebuild it.
  - *Track R and the event track:* unchanged; the queues run on their own. The lead assigns the scoring step when
    they finish.
- **Data approved:** the on-disk sources of plan §6 (WQP grab samples WY2008–21, USGS daily-value sites outside the
  FINAL split, sonde auxiliary labels, WQP logger rows), under data-hygiene rules 1–5 of plan §6, which are
  mandatory. **Oliver et al. 2024 (CC0):** approved as stream-extract-delete after a HEAD size check, inside M6's
  20 GB cap and the 60 GB floor; rows of protected groups are locked at extraction (counts only). HydroATLAS only if
  the zero-shot schema needs it (regional subsets). ERA5-Land through Open-Meteo: size the quota cost first and
  report it; at most 40 % of the shared daily budget while M1's forecast pull is running.
- **Tracks:**
  - **T-E** (EA England; zero-shot and fine-tuned): the primary SOTA opportunity. Reservation unchanged
    (`ea_england_allocation.md`). One common input schema for every model. The near-discharge stratum is defined from
    public metadata only. The amendment states, from the metadata group count, the minimum number of groups below
    which an endpoint is descriptive.
  - **T-H** (forcing-defined heatwave windows): approved as a claim endpoint inside T-E and T-B, in their Holm
    families as the plan proposes.
  - **T-B / N2 — joint look:** the v1 `M6/test_b` models and the arch-v2 claim model are frozen together and scored
    in **one** look after 2027-06-30. v1 models are secondary members. There is no earlier, separate v1 `test_b`
    freeze.
  - **T-D** (historical logger reaches): **secondary and descriptive only**, labelled "historical reconstruction".
    It cannot carry a §15.1 label: its groups are the spent Test-A groups and its years are training years. Its rows
    are locked at extraction and scored once with the frozen models.
  - T-M and T-K: deferred. Track R: unchanged; it carries no new claim.
- **Claim model and Holm families:** one claim model per track, chosen by a rule written in the amendment that uses
  inner-CV or validation data only, before the freeze. Each family is an intersection–union test over
  {R-TD, the better of G1/G2, P} with the every-seed rule. Other arms are secondary.
- **Selection criterion:** the change to 4 primary targets is approved for arch-v2 arms only and is disclosed. v1
  models stay as frozen.
- **Code:** only the unhashed paths listed in plan §8. The frozen code hash `5bd7d329…` must not change.
- **Amendment:** `docs/preregistration/hydromet_arch_v2_amendment.md`, drafted by M6 (plan step 0 said "the lead
  drafts"; rule 1 applies instead) and stamped by the lead. Before the stamp: plan steps 1–2 only.
- **Ruling 4b delivered (M3's EA use):** M3's H4 track uses EA sample-based data with its own partition file
  (`microbe_ea_ids.json`). M6 reserves Hydrology-API sonde data. There is no overlap, and each keeps its own split.

## Display names (2026-11-10, user instruction)

Main models are named SENTINEL-<one word>; the downstream set has its own family name. Canonical table:
`docs/NAMES.md`. Display names only: code identifiers, registry keys, paths and frozen files keep the old names.

## Lead rulings, afternoon of 2026-11-10 (section written 2026-11-10 13:26 EST)

Each ruling below was sent to the workstream before this section was written. No validation number on any new arm
existed when it was made. Stamps are recorded separately when they happen.

### Shared WQP label defect (found by M2, cause verified by the lead)

- **Cause.** `sentinel/data/matchups.py::load_wqp("processed")` reads every parquet under
  `processed/wqp_discrete` and recognises a result table by its columns. It therefore also reads
  `_qc_rejects.parquet` (24,647,135 rows removed by the WQP QC) and `_unmapped.parquet` (7,854,605 rows that could
  not be mapped) as ordinary results. It also never passes `depth_m` to `wqp_surface_samples`, so the
  "≤ 1 m or unknown" surface rule is not applied to processed tables.
- **One rule, project-wide (ruled for M2; intended for M7 after its counts):** labels come from the harmonised,
  QC-passed tables only (`wqp_discrete/variable=*/`); surface = `depth_m` ≤ 1.0 m or unknown; a result flagged
  "bottom" is dropped at any depth; Secchi is exempt as the function codes it. The unchanged
  `wqp_surface_samples` applies it, with `depth_m` passed as `result_depth`. A sample with no QC-passed surface
  result is dropped and the reason is recorded. The 395,900 ambiguous "Chlorophyll" rows stay out.
- **The shared module is not edited.** M2's reader is `sentinel/models/orbit/wqp_surface.py`.
- **M2.** Depth rule (option A) and sidecar exclusion approved. Before the dev_v5 build: change-log line in
  `orbit_protocol.md`, registry row with the side-table hash and counts (protected roles: counts only), data pin.
  No reported M2 number used a WQP label (dev_v1–v4 trainable tables held 0 WQP rows). The protected counts in
  `orbit_split_report_dev_v4.json` are superseded and recounted. If a test cell falls below a pre-registered floor
  the pre-registered consequence applies; the rule is not relaxed.
- **M7 exposure (metadata only).** The national NAIP matchup manifest lists both sidecar files under its sources.
  The 4-state snapshot (2026-10-24) lists `raw/wqp` and no sidecar, so the spent 4-state look is not exposed to
  the sidecar defect. P2 reads the harmonised FNU table only but is depth-blind. M7 was asked for counts by build ×
  target × role (row membership only on protected roles). **Holds until the lead rules on those counts:** the
  national-2b label build is not finalised, and the national-2 freeze candidate does not go to pre-test review.

### M5 BioAssess

- **M5-X scorer.** Approved: a dated change-log line in `bioassess_x_protocol.md` before any test data are seen;
  the scorer fits **every listed §6 comparator** at test; row 1 needs every paired lower bound > 0. bio.infer is
  included (it is §6 item 3, mandatory). GJAM and TRPI may be left out only on their pre-declared grounds, with the
  measured reason recorded. Then: tests, a new dev dry run, a regenerated freeze candidate, pre-test review.
- **M5-F choices (a)–(o).** Approved with these conditions: (a) coordinate-free context for the candidate, but the
  comparator list keeps the coordinate-using variants; (b) UK biological sample dates ≤ 2020-12-01; (e) DO
  coherence gets the standard-atmosphere altitude correction (uncorrected error ≈ 0.031 log10 near 600 m and 0.06
  near 1,150 m, the whole tolerance), constants unit-tested against the source, and the copy in
  `bioset_ma.py` is tested against the shared `hydromet/physics.py::do_sat_mgl`; (f) pool prevalence from taxa
  lists only, training rows only, offered to every context-taking comparator; (k) region context never supplies a
  locked, test or French label, and if prediction-time context uses unlabelled test-region inputs it is disclosed,
  identical for all models and reported with a no-region-context ablation; (n) plausibility ranges identical for
  every role and model; (o) resume points logged, the frozen checkpoint is hash-pinned.
- **Stamp:** after protocol v1 arrives with its hash. No French data and no validation number before it.

### M1 Sonde

- `sonde_arch_v2_deviations_part2.md` read in full by the lead (sha `408cf516…fb5ec`); no blocking issue. M1 adds
  the reference-window climatology as a dated conformance entry, reports the new hash, and the file then goes to
  independent review. A8 counts and A12 timings follow as append-only addendum lines. Part 1 and part 2 are stamped
  together. Step 1 waits for that stamp and for the full WY2008–2019 reforecast.

### M6 HydroMet levers

- **Snapshot.** Registry row `3deeb3b6bd27` (`M6-hydromet` / `hydromet_sota/prereg_snapshot`, 7 files, logged by
  the lead at 13:14 EST) holds the hashes of the lever protocol, pins, EA protocol draft and configs C1a–C2b. At
  that time the registry held 0 lever result rows.
- **Order of events (from file-write timestamps):** protocol 12:08 EST → configs 12:13 EST → first lever run folder
  12:20 EST. The protocol's header ("13:00 EST") and two "13:10 EST" lines state wrong clock times; M6 corrects
  them in place, appends a dated correction row, and logs a new snapshot row. A new snapshot row follows every
  later append, before the next lever result is scored.
- **EA England split:** no role draw is frozen. The lead rules on S after the independent review of the M6
  amendment. Until then no EA value is read for any statistic.

### M7 Lite

- `lite_arch_v2_amendment.md` v0.3 (sha `2b369018…8a56`) is frozen for review; further points go in a separate
  addendum draft.
- **Q12.** The confirmatory floor stays (≥ 100 chips in ≥ 40 water bodies; written before the index existed). The
  index has 189 chips in 38 water bodies, so **P2 is descriptive in this round**: one hardened look, no label. The
  floor is not lowered and no val or train water body moves into test. M7's label route this round is the national
  track under the frozen §8a rule. A larger P2 needs its own stamped addendum, designed from counts only.
- **To be written into the rulings file at the stamp:** a TabPFN v2 comparator through the shared loader
  (`sentinel.models.chemtox.tabpfn_model.load_tabpfn`; Q4's premise that it is not installed is wrong); conflict
  C2: the 308 teacher pairings on M2 Test-S site-dates carry no KD term or teacher-derived feature in any v2 arm or
  comparator, no M7 output is ever an input to anything scored on Test-S, and the 777 labelled chips stay in M7
  training.

### Independent pre-stamp reviews (running)

- Review 1: M2 `orbit_arch_v2_amendment.md`, M4 `chemtox_arch_v2_amendment.md`, M1 deviations part 1.
- Review 2: M3 `microbe_arch_v2_amendment.md`, M6 `hydromet_arch_v2_amendment.md`, M7 `lite_arch_v2_amendment.md`.
- Authors were told not to edit a file under review. A stamp follows only after the lead merges the review with
  the lead's own full read.

## Lead rulings and events, afternoon of 2026-11-10, part 2 (section written 2026-11-10 14:26 EST)

Appended only; nothing above this line was changed. No validation number on any new arm existed when these rulings
were made, and no amendment has been stamped since the M1 stamp (registry row `2921bb1ca000`, 12:23 EST).

### Stoppage and recovery

- Earlier this afternoon every agent and every review verifier stopped on a weekly usage-limit error. At 14:13–14:15
  EST three probe agents answered again. Agents started with default settings had been running on
  claude-opus-5-5. From now on every agent except M4 is started on the fable model by name; M4 runs on opus (the
  user's instruction).
- No file under review changed during the stoppage. Hashes re-checked at 14:14 and 14:21 EST: chemtox
  `b9326d49…`, orbit `5001373d…`, sonde deviations part 1 `031ddddc…`, part 2 `b17f3ec6…`, microbe
  `022841f3…`, hydromet `a65e6eb5…`, lite v0.3 `2b369018…`.
- The three review runs were resumed unchanged, so finished reviews were kept and only the missing verifiers
  (M2, M3, M6, M7) and the M1 part-2 review ran again.
- Reviews and verifier verdicts are kept verbatim in `docs/preregistration/reviews/arch_v2_20261110/` with an
  index (`README.md`). The workflow journals are the source.

### M4 SENTINEL-Toxin: stamp withheld, v2 ordered (sent 14:18 EST)

The verifier confirmed five blocking defects in `chemtox_arch_v2_amendment.md` (sha `b9326d49…`):

1. **Spent test rows in the new blocks.** Eight ADORE test partitions were already scored (28,265 ids). 26,808 of
   them are in the T1/T6 universe: 4,276 in T1_test (40.54 % of it) and 22,532 in the training pool (37.72 %).
   Only 6,272 T1_test rows (536 chemicals, 364 clusters) are unseen. This breaks campaign rule 1.
2. **Power figures too optimistic.** The minimum detectable effects were computed as if rows were independent. At
   the level of the hold-out unit they are 0.0474 / 0.0268 / 0.0532 (T1 / T6a / T6b), 2.5 to 5 times larger.
3. Label statistics of sealed rows were printed during the split build.
4. Auxiliary-task rows could carry test chemicals into training.
5. The split builder and a labelled intermediate lived in a temporary directory with no pinned provenance.

Rulings L1–L11 (binding for v2; they go into `chemtox_arch_v2_lead_rulings.md` at the stamp):

- **L1.** Spent ADORE test rows leave the T1/T6 universe entirely: no test block, no training pool, no auxiliary,
  multitask, pretraining or ablation set. Spent partitions are derived by code from `results/registry.jsonl`,
  with counted assertions; the id list is frozen with its sha256. A reserved-but-unscored ADORE test partition is
  reported (counts only) and not subtracted without a lead ruling.
- **L2.** Hold-out unit = connected component over (t1_cluster, parent_key family). It is also the bootstrap and
  MDE resampling unit.
- **L3.** The split is re-derived with the same label-blind procedure (same clustering, cut-off, seed, draw rule).
  New artefacts are named `…_v2`. The sealed v1 split and v1 cluster file are not touched. The v1 text is kept
  byte for byte as `chemtox_arch_v2_amendment_v1_unstamped.md`; the v2 header says v1 was reviewed and not
  stamped.
- **L4.** From counts only, M4 tabulates the unit-level MDE at intraclass correlation 0.5 and 1.0 for T1 test
  shares of 15, 20, 25 and 30 % (and the T6a/T6b unit counts) and proposes one share with a written reason. The
  lead rules at the stamp. A block that cannot detect a realistic gain carries no §15.1 label (descriptive only).
- **L5.** `unseen_taxa.py` and `unseen_taxa_guard.py` are not edited. A new power-helper module with tests
  (it reproduces the closed-form unit-level value at icc = 1 and the row-level value at icc = 0).
- **L6.** No label-derived quantity of any v2 sealed block before its FREEZE. The v1 read is recorded as a
  disclosed deviation.
- **L7.** The auxiliary-leak and pretrained-cache verdicts as ruled by the verifier, extended by L1 and L2;
  filtered auxiliary counts are given in the amendment.
- **L8.** Path, size and sha256 of the labelled scratch intermediate are recorded, then it is deleted. No pinned
  artefact lives in a temporary directory.
- **L9.** The claim that the gpboost reference is frozen is struck. `sentinel/models/chemtox/gpboost_model.py`
  is not edited. The Bio-QSARs citation is corrected.
- **L10.** The T1/T6 guard module, pins, experiment key and protected-labels artefact are built as code now and
  reviewed before any look.
- **L11.** The lead adds the v2 blocks to `sota_campaign.md` at the stamp (file name, sha256, counts).

Every other verifier verdict (27 in all, `M4_verify.json`) and every "missed" item is implemented as its ruling
text says. M4 reports the v2 sha256, a table "verdict → where addressed", the L1 counts and the L4 table. A delta
review follows, then the stamp. No validation fit meanwhile.

### M1 SENTINEL-Sonde

- Part 2 v0.2: sha `b17f3ec6…63f4`, 229 lines, verified by the lead at 13:26 EST, body closed for review
  (integrity/parity and code-vs-text lenses, then a verifier).
- GBM references: the timing run was stopped in favour of the reforecast extraction, which is on the critical
  path. Measured lower bound: one horizon at the 1.5 M row cap with 7 boosters and 6 threads takes more than 45
  minutes. M1 proposes a declared budget from timing arithmetic only; no validation number is used to choose it;
  one rule for GBM-NWP and PF-GBM; it must not handicap the references; it is stamped as an addendum line before
  any CV run.
- The part-1 review (`M1dev_review.json`) has four blocking findings: D2 (the 19-level quantile reference must
  stay a mandatory configuration and the step-1 comparator), U1 (the GFS product never reaches SONDE-HF or R-NH in
  CV because of a standardisation defect), U3 (the reference-window climatology needs the issue-window row rule
  for every model), U8 (the file must match a frozen code state, with hashes). That run had no verifier, so the
  lead checks each finding and sends one consolidated instruction when the part-2 review is back. Step 1 does not
  start before the stamp.

### M6 SENTINEL-Hydro levers

- Snapshot row `09d19d653747` (by the lever engineer, after the clock-time corrections) supersedes
  `3deeb3b6bd27`. C2a seed 0 was scored after it: rows `cc4ae2da6f60` (inner_primary_mse 0.117455) and
  `9772f206ad93` (inner_mse_7targets 0.216031). The reference ealstm_ws has mean 0.11786 (SD 0.0064): no
  visible gain. C2a gets no tuning beyond the pre-declared lever list.
- C1a and C1b (seed 0) started after the upstream forcing table was built (13:46 EST); C2b is queued.
- EA England split: still on hold.
- Architecture engineer: R-TD grab share is ruled at the stamp; K600 is imported from an existing implementation,
  not written a third time, and the shared location is fixed at the stamp; `processed/hydromet_sota_v2/` is
  accepted with pinned manifests.

### Repo-wide reserved-site guard fails for four new files (found by the lead at 14:23 EST)

`tests/test_reserved_loaders.py::test_no_unguarded_reader_of_the_shared_tiers` fails for
`scripts/data/sota_hydromet_v2_build.py`, `sentinel/hydromet_sota/gso_mt/runner.py` (M6) and
`sentinel/models/litev2/p2.py`, `sentinel/models/litev2/p2_index.py` (M7). M6 and M7 fix their files with the
shared helper (no allowlist entry) and report, counts only, how many of the 729 reserved new sites are in anything
already built. If any is, they stop and the lead rules before a rebuild.

### M2 SENTINEL-Orbit: stray ABORTED marker

- `sentinel_runs/orbit/test_looks/M2_test/ABORTED.json` (0444, 88 bytes, 2026-11-09 17:26:57 EST, sha
  `efa31da5…5a32`) is a test artefact, not a look. Evidence: its reason string is the literal used at
  `tests/test_orbit_testlook.py:201`, and lines 196–197 of that test undo the sandbox and then point the runs
  root at the real one; no `FREEZE_orbit.json` exists; the look folder holds no other file; the registry
  (73,065 rows at 14:25 EST) has no M2 claim, seal, abort or test-score row (the only M2 rows with "test" in the
  split are counts-only design rows).
- M2 fixes the test so it cannot write under the real runs root. The lead removes the marker after that fix and
  records the removal here.

### M7 SENTINEL-Lite

Both reviews of v0.3 have two blocking findings each (stale statements of fact; TabPFN declared unavailable; no
WQP label rule). The verifier is running; a v0.4 instruction follows it. Meanwhile M7 delivers the counts owed
(WQP exposure by build × target × role; Test-S overlap by keys) and prepares the label reader by importing M2's
`sentinel/models/orbit/wqp_surface.py`. The holds on national-2b and on the national-2 freeze candidate stay.

### Downstream

- **D4 (AEGIS-Tap).** The validation dry run was not byte-identical between runs (threaded forest prediction,
  differences ≤ 2.7e-15 in 8 forest-derived columns; every AUROC bit-identical). Ruling: forest prediction is made
  single-threaded, two dry runs must be byte-identical, every validation AUROC must be unchanged bit for bit, a
  new candidate is generated, and an independent delta review precedes the FREEZE and the single look.
- **D7.** MSST (mean summer stream temperature) is approved as an added comparator, score only; MAST stays with
  its corrected description (mean annual); the family is six and the candidate must beat every listed comparator.
  No freeze candidate yet. A counts-only design note for a pre-registered ecoregion-transfer track is written
  first; both tracks are frozen together with one look event; the validation numbers already seen are disclosed
  in it as prior knowledge.
- **Five completed apps.** Their FREEZE-hashed pins hold a HEAD-dependent `pip_freeze_sha256`, so a
  rescore-from-saved is refused after any commit. The pins are not edited; this is recorded as a known limitation.
- **Commit hold.** D4's environment pin no longer depends on git HEAD (`pip freeze --exclude-editable`), so the
  hold can be lifted. The lead makes the commit after inspecting the untracked files.

## Lead rulings and events, afternoon of 2026-11-10, part 3 (section written 2026-11-10 14:33 EST)

### M5 SENTINEL-Biota

- **M5-F protocol v1 received.** `docs/preregistration/bioassess_f_protocol.md`, sha256
  `480c765fb3f783b2d441f6aaeec6e6d263d42b9ccf15b2515fe23f9ef1ef099e` (85,198 bytes, 960 lines, mode 0664), hash
  checked by the lead at 14:26 and 14:30 EST. Fetch log `bioassess_f_doc_fetch_log.md` sha256
  `d120879477767b175ae4b1b78197cc3fb40cc89e3cfe7d48e0352bfee1881df8`. An independent delta review (two lenses and a
  verifier, workflow run `wf_4e69f9a1-3b0`) was started at 14:31 EST on that exact hash. M5 was told not to edit
  either file during the review. No stamp yet. No French data may be downloaded or read, and no validation number
  produced on any M5-F arm, before the stamp.
- **M5-X, GJAM comparator: "not compared, with reason".** M5 reported that GJAM failed the fidelity check on fold 0
  of both blocks (fidelity_pass FALSE) and that the measured fits took 99.3 min (X-L) and 114 min (X-U). The lead
  read `bioassess_x_protocol.md` §6 comparator 4 (lines 126–132), amendment v4.8 item 8 (lines 335–342) and the v6
  entry of 2026-11-09 (lines 371–382): the fidelity criterion (r ≥ 0.90, else "not compared, with reason") was fixed
  before any GJAM development prediction was scored. **Ruling:** the scorer lists GJAM as "not compared, with
  reason: failed the pre-declared fidelity check (v4.8 item 8 as corrected in v6)". Conditions: (a) the 2-CPU-hour
  ground of line 132 is **not** the ground used, and the fit times are recorded; (b) per block and target, both
  correlations, the row count and the path and sha256 of the fidelity files go in the scorer output and the freeze
  candidate; (c) the older "only if it cannot run within 2 CPU-hours" sentence is not edited if frozen; a dated
  change-log entry or a candidate disclosure states that item 8 / v6 add the fidelity ground, and the exclusion is
  disclosed as a limitation; (d) no GJAM prediction is scored on test rows and the check is not re-run after
  tuning; (e) no claim wording of the form "beats all joint-species models".
- **M5-X dev dry run.** Exited 0 at 14:16 EST (run folder `sentinel_runs/bioassess/x_final_dryrun_dev_20261110T115830`).
  The candidate will be regenerated after the approved scorer change; the pre-test review runs on the new hash.

### M1 SENTINEL-Sonde: review of deviations part 2 (integrity lens returned 14:31 EST)

- Reviewed file: `sonde_arch_v2_deviations_part2.md`, sha256 `b17f3ec6…63f4` (hash matched). **Verdict: REVISE.**
  Four blocking findings: P2-01 (gage height dropped on a false statement: `gage_ft_mean` is indexed at 1,258
  sites), P2-02 (GBM-NWP and PF-GBM fit only rows to 2019-09-30 while the networks train to 2021-09-30; recorded in
  neither part), P2-03 (the full-rows GBM configuration of A12 is not in the drivers, and the 2-hour fallback rule
  is not checkable), P2-04 (reference-window climatology has no "issue ≥ reference end + 42 d" row rule; the same
  defect as U3 of the part-1 review, confirmed by the lead in `prep.py`). Should-fix: P2-05 (GFS slot never reaches
  the networks; the same defect as U1, confirmed by the lead in `data_v2.py` and `sonde_v2_cv.py`), P2-06 to P2-12.
  Notes: P2-13, P2-14. The code lens and the verifier are still running.
- **Consequence.** No M1 new-architecture number can be produced yet: step 1 and CV wait for the code fixes, one
  consolidated deviations file, a code freeze with a sha256 list, a delta review and the lead stamp.

## Lead rulings and events, afternoon of 2026-11-10, part 4 (section written 2026-11-10 14:46 EST)

### M3 SENTINEL-Microbe: architecture amendment stamped (partial release)

- **Stamp.** Registry row `421854a9d225` (`M3-microbe` / `microbe/prereg_snapshot`), written by the lead at
  2026-11-10T19:41:37+00:00 (14:41:37 EST). `docs/preregistration/microbe_arch_v2_amendment.md` sha256
  `022841f3e94498efd94b4258449730472f790c5c2ef800556350cfe55d82eb6a` (845 lines, 61,069 bytes), stamped as written,
  mode 0444. `docs/preregistration/microbe_arch_v2_lead_rulings.md` sha256
  `e148a09def685f6fa730650e24ba384389daa1fe44d29c07151568da028ecd95`, mode 0444. The row pins 12 files (amendment,
  rulings, this file as it stood at the stamp with sha256 `611c55cf…9e12` and 453 lines, the M3 plan, the three M3
  protocols, the two split files and the three review files). The guard `sentinel/models/microbe/stamp.py`
  returned stamped = True after the row was written. The git commit is deferred (commit hold) and is owed before
  any M3 freeze.
- **Time in the rulings header.** The rulings file header says "19:45 UTC (14:45 EST)". The file was written a few
  minutes before the row. The binding stamp time is the registry row timestamp, 19:41:37 UTC. The file is 0444 and
  hashed in the row, so it is not edited.
- **Basis.** Two independent reviews and an adversarial verifier on the exact hash (workflow run
  `wf_5a96bec9-d7b`; files `M3_review_integrity.json`, `M3_review_design.json`, `M3_verify.json` in
  `docs/preregistration/reviews/arch_v2_20261110/`); verifier recommendation: stamp with rulings. Before the stamp
  the lead re-read the amendment in full, reproduced the header and Appendix B hashes, and confirmed that the
  registry held no M3 new-arm row and that no EA or dissolved-gas label file was on disk.
- **Scope: partial.** Released on the `lodo` track: D1 NicheInvert-S (slot #6) with ablations A0 and A1,
  `xgb_nichez_context` and the calibration diagnostic; `xgb_v4mr` (slot #7) with its twins; the parity list; the
  GEO development runs G1, G3, G7, G8, G9; D4 after its label-free confound check. **Not released:** D2
  NicheGeo-Phys, the added EA target family, H4, every EA label download, and the dissolved-gas (GAS-3) download
  and fits. Each needs a new deviations file (`microbe_arch_v2_deviations_NN.md`), reviewed and stamped by the lead.
- **Main rulings.** The cap of 8 configurations stays (slot #5 retired; #6 to #8 on `lodo` only; G2, G4, G6
  dropped). The table models `xgb_v4mr` and `xgb_v4mr_stack` cannot be M* for H-M3. Every listed comparator must be
  beaten at the sealed look. The D1 gate is evaluated once by the lead from registry rows. Designs B and C for the
  EA data are rejected. The dissolved-gas download is run by the lead and waits for the project owner's go-ahead on
  token use. The freeze candidate is due no later than 2026-11-30.
- **M3 told at 14:42 EST** (one message): step 0 (guard wiring, loader hardening, owed tests) first, then step 1.
  M3 reports the moment the first D1 development run is running.

### M2 SENTINEL-Orbit: label-rule records received (14:43 EST)

- M2 reported the records for conditions A to F of the one WQP label rule: side table
  `processed/orbit_wqp_surface/surface_samples.parquet` sha256 `8ed6a309…3a31`; registry summary row
  `242aa40e1913` (`orbit_wqp_depth_rule`); count files `orbit_wqp_depth_rule_counts_20261110.json` and
  `orbit_wqp_rule_cellcounts_20261110.json`. No pre-registered floor is crossed on the counts reported. The
  amendment is untouched (sha256 `5001373d…df8a`). `testlook.py` changed (data pins), so the v1 freeze candidate is
  stale until it is regenerated. dev_v5 is not built (waits for the 2015–2023 chip build).
- The test that wrote the stray `ABORTED.json` is fixed (temporary runs root, guarded writer). Removal of the
  marker is the lead's step and is recorded separately when done.

## part 5 (section written 2026-11-10 14:55 EST)

### M6 SENTINEL-Hydro: architecture-v2 amendment stamped (partial)

- Registry row `b37a0319964c` (`M6-hydromet` / `hydromet_arch_v2/prereg_snapshot`), timestamp 2026-11-10T19:52:24+00:00 (binding time).
- `docs/preregistration/hydromet_arch_v2_amendment.md` sha256 `a65e6eb578f5bebe001dada100cd8ea49f2383b80987cd3927014f759696e656` (448 lines, as reviewed), 0444.
- `docs/preregistration/hydromet_arch_v2_lead_rulings.md` sha256 `b144974a96c2140c933ccbe1e8d3247d9a1ffa0ad6663ee958237e33eb4179b0` (225 lines), 0444. It wins where the two differ.
- Review: `M6_review_integrity.json`, `M6_review_design.json`, `M6_verify.json` (run `wf_5a96bec9-d7b`), verifier recommendation stamp_with_rulings; hashes in the stamp row.
- Counts at the stamp (in the row): no `M6V2`, `M6EA`, `M6/test_b` registry rows; 2 lever rows `M6SOTA/inner_v1`; `configs_tried.jsonl` 102 rows, 4 arch-v2 rows, all `smoke: true`. `stamp_status()` returned stamped True.
- Scope: the stamp alone releases **no fit**. First post-stamp fit needs deviations entry 1 (new file `hydromet_arch_v2_deviations.md`: constants module with sources and tests, `pins.json`, licence rows, T-B/N2 G and G_eff, joint-look deviation, lever declaration) and the extended code guard (L10), stamped by the lead. Not released: track T-E (England; by reference, on hold), D2 grid and DO stage, Powell rows, Tier 2, any look.
- Main rulings: statics 191 (188 + 3 VAA; "194" was a slip), dynamic 30; G1 gets the §3 additions (stronger baseline), G2 = G1 + Te features + DO saturation + stacked predictions; one G-D1a gate on J_sonde; Tier 1 only; closed variant list for the cap of 12; R-TD one transcribed configuration, fitted in both label sets; comparator set {R-TD with grab, R-TD published form, G1, G2, P} (+ references-only stack for D3); added necessary condition against co-scored v1 and lever models; T-E split S = 0.50, all 1,964 stations get a status, writer run once by the lead; pre-stamp hourly-build incident recorded.
- Told at 14:53 EST: the M6 architecture engineer (stamp, answers, order of work, question on the 14:51 `g1.p1` smoke row) and the M6 lever engineer (upstream-forcing table statement, lever protocol §8 alignment, EA split fixes, lever inner-row declaration).
- Open for the lead: commission the dated novelty search for the diel observation operator; shared Powell role table (X3).

## part 6 (section written 2026-11-11 13:20 EST, covering 2026-11-10 evening to 2026-11-11 midday)

### The usage-limit stop

Every background workstream agent was killed by a model usage limit at about 2026-11-10 21:00 EST and none ran
again until about 2026-11-11 12:55 EST. The processes they had already launched kept running throughout, so the
reference fits, comparator sweeps and downloads continued unattended; nothing was lost and no agent decision was
made in that window. All eight workstream agents were resumed from their own transcripts on 2026-11-11.

Hardware changed in the same window: another user took 7 of the 10 GPUs (indices 2, 3, 4, 6, 7, 8, 9). Only 0, 1
and 5 were free at 12:45 EST on 2026-11-11; load was 29 of 48 cores and 170 GB were free on `/`. GPU 2 and every
other GPU carrying another user's process are never touched.

### M1 SENTINEL-Sonde: GBM ruling

- `docs/preregistration/sonde_arch_v2_gbm_lead_ruling_20261110.md`, sha256
  `3028fc1bd0509152e7428ad8a0ccbe27dece086027da9713bc9dae20bbf32c5b`, 0444; registry row `6cad67b32813`
  (`M1-sonde` / `sonde/lead_gbm_ruling`, 2026-11-10 20:40 EST). Rulings G-1 to G-9: full rows, frozen recipe,
  LightGBM GPU for every GBM fit of both arms, booster-level resume, no repeat of the timing benchmark. Not a
  stamp; it releases no driver. M1's deviations-v2 package, its delta review and its stamp are still owed.

### M3 SENTINEL-Biota: the V4-MR compute ruling

- `docs/preregistration/microbe_arch_v2_v4mr_lead_ruling_20261110.md`, sha256
  `1a65d47ad65398827ad64d8e784d7a819c556ec17e65c8fa6d50d9a95625469e`, 0444; registry row `fa48ff4c72bf`
  (`M3-microbe` / `microbe/lead_v4mr_ruling`, 2026-11-11T01:56:19+00:00). Rulings V-1 to V-10: `xgb_v4mr` runs on
  `lodo` exactly as amendment §3.1; the D3a CPU figure of 6 CPU-h per track is replaced on `lodo` by at most 30
  CPU-h per target family and 60 in all (projection 21.18 per family); one family at a time, P first; the runner
  edit is `CPU_BUDGET_H["xgb_v4mr"] 6.0 -> 30.0` and nothing else; the code state is frozen from the first shard
  to the G merge; the 12-thread capacity is **not** raised; family-G tables do not hold the D1 gate; D3b is not
  ruled and needs its own figure. Deviations recorded in it: Dev-a (the D3a CPU figure) and Dev-b (parallel CPU
  jobs against amendment §12 row 21 and rule 5 above).
- 544 `microbe_arch_v2_cv` rows were registered while no agent ran (WA-tol, WA-PLS, NicheInvert-S,
  NicheInvert-S[A0], TabPFNv2[V4-CLR], TabPFNv2[genus-CLR], RF[V4-CLR], V4-XGB[cap 3,000]). No `xgb_v4mr` row
  exists yet. Nobody states a D1 gate outcome before the lead's single evaluation (B6).

### M6 SENTINEL-Hydro: the code-pin record and the batch confirmation (entry-1 §3 step 4)

This subsection is the record that entry-1 lead rulings §3 step 4 requires of the lead.

- **Stamp rows of this round, in full:** `b37a0319964c` (the amendment stamp of 2026-11-10T19:52:24+00:00) and
  **`e956ed69fd42`** (`M6-hydromet` / `hydromet_arch_v2/prereg_snapshot`, 2026-11-11T01:01:16+00:00), which is the
  row that released the reference fits. Every arch-v2 run record and registry row of this round cites
  `e956ed69fd42`. Part 5 above listed only `b37a0319964c`; that omission is corrected here.
- **Code-pin record:** `docs/preregistration/hydromet_arch_v2_code_pins.json`, sha256
  **`574307a10dd417c03997890bf4022f6ac346be01de6e2a906acbfc1e350c0182`**, 0444, 18 files, written by the
  architecture engineer; its registry row is **`7744d6502add`** (`M6V2/pins_v1`, model `code_pins`,
  2026-11-11T01:40:13+00:00, value 18). The lead re-hashed all 18 files on 2026-11-10 at 20:57 EST: all 18 match.
  `assert_released()` at the same time returned stamped True, amendment sha256 `a65e6eb5…e656`, pins sha256
  `84958df5…c5d9`, 38 pinned files, `changed_under_stamp {}`, code pins `574307a1…0182`, 18 code-pinned files,
  `code_changed_under_stamp {}`.
- **The batch check.** An independent reviewer that did not write the code checked the batch against entry-1 §2.
  Verdict `confirm_with_lead_rulings`, no blocker. It verified every §2 item with file:line evidence and a unit
  test; that the batch changed exactly the 8 code and 3 test files the engineer listed and nothing else since
  `e956ed69fd42`; that all five entry points call `assert_released` before any row read and that only `--smoke`
  bypasses it, with the one F7-sanctioned exception `--smoke --d2-kill-rule` (row `afc1fbaebe0d`); that all 15
  non-smoke run directories and all 38 M6V2 rows carry the release block; 134 tests passed, 0 failed, 0 skipped;
  34 of 34 mutations killed. **The 36 `M6V2/inner_v1` reference-fit rows are not void.**
- **Confirmation.** On that basis the lead confirms the batch of entry-1 §3 step 1. The D1 grid, selection, gate
  G-D1a and the stacks are released, under the rulings written in
  `docs/preregistration/hydromet_arch_v2_batch_confirmation_20261111.md` (see below).

### M6 SENTINEL-Hydro: the reference fits and the CPU ruling

- 36 rows under `M6V2/inner_v1`: R-TD and R-TD-W in both label forms for seeds 0-2 (12 GPU fits) and G1 and G2
  seed 0 with their selections. Launch logs: `rtd` queue finished 2026-11-10 22:25 EST, `rtdw_published` 22:57,
  `rtdw_fixed` 2026-11-11 00:15, `g1_sel_s0` 02:26, `g2_sel_s0` 05:55. No run failed and none is missing from the
  registry. No validation, test or England number exists.
- `docs/preregistration/hydromet_arch_v2_cpu_lead_ruling_20261111.md`, sha256
  `0d21f6dfe13107e7b12df428648c7c0d5e14d620c3854ff9fe2bf5a810b37051`, 0444; registry row **`a15c846c1d5f`**
  (`M6-hydromet` / `hydromet_arch_v2/lead_cpu_ruling`, 2026-11-11T18:06:10+00:00). **C-1** is a reading, not a
  deviation: "≤ 2 h per process" in amendment §12 sits inside that bullet's GPU clause, and the CPU clause sets
  only "one heavy job at a time, ≤ 6 threads", so the physics chain's 3.5-hour CPU processes exceed no stamped
  limit. This corrects a lead message of the same day that had called it a deviation. **C-2** authorises a second
  concurrent CPU job of this workstream as a deviation from "CPU: one heavy job at a time", capped at two jobs of
  6 threads each under a load guard of 48, logged, released work only. **C-3** leaves the GPU rules, the disk
  floor and the D1 release gate untouched; the ruling expires with the round and extends to no other workstream.

### M7 SENTINEL-Lite: architecture-v2 stamp, 2026-11-11 13:02 EST

- Registry row **`5879088974eb`** (`M7-lite` / `lite/prereg_snapshot`, 2026-11-11T18:02:38+00:00, `written_by`
  lead, value 4). `python -m sentinel.models.litev2.stamp` returns `"stamped": true`. A v2 run whose log cites
  another `lite/prereg_snapshot` row id is void (ruling N-03).
- Stamped byte for byte: `lite_arch_v2_amendment.md` `2b369018…8a56` (DRAFT v0.3),
  `lite_arch_v2_label_addendum.md` `bcbd86de…9049`, `lite_arch_v2_deviations.md` `23e6fd87…642c`, and the lead
  rulings file `docs/preregistration/lite_arch_v2_lead_rulings.md`
  `95d5bdcbd0b2dba757fe79ab6b70be8d06fd38047db62acd91744b41be5d9c29` (929 lines), which wins where they differ.
  All four are 0444, as are `lite_split_national2b.json`, `lite_protocol.md`, `lite_national_protocol.md`,
  `labels_national2b.parquet` and the P2 index from the stamp.
- Lead's own pre-stamp checks: all 39 files of the code record `563894b9…afe1` re-hashed and matching; the eleven
  `tests/test_litev2*.py` files run by the lead gave 120 passed, 0 failed, 0 skipped;
  `tests/test_lite.py::test_scoring_hash_covers_scorer_import_closure` still fails at `tests/test_lite.py:496`
  and stays failing as the record of deviation 10; zero rows under each of the five v2 experiments.
- Review: `docs/preregistration/reviews/arch_v2_20261110/M7v2_stamp_review.{json,md}`, verdict
  `stamp_with_lead_rulings`. The rulings file carries R-01 to R-09, N-01 to N-06, the five open points of entry
  11, all 45 verifier items, rulings D-0 to D-5, the pins, the deviation-10 correction and seven lead orders
  reproduced verbatim as annexes.
- **D-0**: selection rule S is the 3-seed ensemble mean, verified by the lead in
  `scripts/train/lite_train.py:112,116,123,124-127`, `scripts/eval/lite_select.py:50-52` and the 32
  `lite_val_national2` rows; the carrier of any label stays the INT8 pipeline of seed 0.
- First v2 fit: `rerun_crop256_kd03_national2b` on GPU 0, PID 2668410, 2026-11-11 13:06:46 EST; the TabPFN memory
  test registered a cap of 10,000 rows (row `ad650fe62de2`) before it. M7 is allocated at most two GPUs.

### M2 SENTINEL-Orbit

- 1,762 `orbit_val_dev_v5` rows and 72 rows each under `orbit_power_dev_v5` and `orbit_power_dev_v5_trackU` were
  registered while no agent ran; the track-U baselines are still running. No v2 arm or comparator has been fitted.
- The independent leakage review of `orbit_v2/targets.py`, `context.py`, `devm.py` and
  `scripts/data/orbitv2_build_dev_v5m.py` returned `pass_with_lead_rulings`, no blocking finding, and **no dev_v5
  row is voided**; the six reviewed files hashed equal at the start and the end of the review. Three findings need
  lead rulings before the value-reading dev_v5m build: L-01 (no code gate ties `--mode build` to the R-21
  review), L-03 (whether the `context.py` chip-id pin is a dev_v5m artefact under R-21) and L-04 (the real-cache
  R-21 test skips until built and then neither recomputes the protected key sets nor calls `audit_artefacts`).
- Reading (b) answered: the registry holds **no** v1 selection row, so no member fit may start yet. M2 registers
  one dedicated row after the selection is final, under workstream `M2-orbit` exactly, experiment
  `orbit_arch_v2/v1_selection`, carrying the §6 rule text, the candidate sha256 and `fit_scheme = "base"`, and
  names its id to the lead before the first member fit. A dry-run row is not a selection row.
- Headline caution on the record: the best v1 arm leads the strongest published comparator by 3.9 MdSA points on
  chl-a (48.56 against 52.49) with seeds spreading 48.14 to 53.27. That is not a §15.1 claim and no write-up may
  call it one before the paired block-bootstrap interval excludes zero in at least three seeds.

### D4 AEGIS-Tap: the FREEZE file

- `docs/preregistration/FREEZE_downstream_taprisk.json` written by the lead 2026-11-11 13:11 EST, sha256
  `c8e18904cfd1a7f7cd110e1c2c0b83089c6bb34f6c0bcdae0145c39d2915241c`, 0444;
  `taprisk_look.load_freeze()` accepts it. It pins the candidate record `2bff8c92…8a1d`, the split
  `a4a1fc7b…3298`, `scoring_code_sha256` `1623ccbd…f0c3`, `registry_scoring_code_sha256` `f63ed306…3e99`, the
  24 data pins `55bb6ddf…c45b`, both path roots and the environment pin `19a9cedf…4c67`, each recomputed by the
  lead and equal to the candidate. It authorises **one** look and nothing else; a rescore, a repair or any
  further test read needs a new lead freeze file.
- **The look has not been run and is not released yet.** The stamp procedure above says "The commit and a public
  push must exist before any freeze on the tracks concerned", and the campaign never claims OSF registration — it
  claims that the plan was frozen in a public GitHub commit before testing. The local commit and the public push
  therefore come first, and the push needs a fresh GitHub token from the user, since the tokens pasted earlier
  are to be treated as compromised and revoked. The look runs only after that.

### Lead items still open at 2026-11-11 13:20 EST

M6: rulings on entry 2 (draft `22f463e8…8f27`, under independent review) and on R1 of the batch check. M4: the
pins file, the G-7 chmods, the `chemtox/prereg_snapshot` row and the ruling on the five flagged items, all of
which wait on the guard delta re-review. M2: readings (a) to (i) in a lead file, and L-01, L-03, L-04. M1: the
delta review and the stamp. M5: the X delta review and `bioassess_x_freeze.json`. Lever: the joint-look ruling
and the England T-E entry with its four open items. D4: the commit, the push, then the single look, which also
lifts the commit hold. D7: the four decisions of the design note.
