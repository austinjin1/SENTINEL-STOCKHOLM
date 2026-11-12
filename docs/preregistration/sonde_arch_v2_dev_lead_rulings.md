# M1 SENTINEL-Sonde — architecture round v2: lead rulings given by message, put on disk

**Written by:** the lead (AI-assisted), 2026-11-10 19:37 EST. This file is read-only (0444) from the moment it is
written. Its sha256 is carried by the registry row that stamps `docs/preregistration/sonde_arch_v2_deviations_v2.md`
(`M1-sonde` / `sonde/prereg_snapshot`, written by the lead); until that row exists this file is a record, not a stamp.

**Why this file exists.** Rulings A–P, the clarifications of them and the verifier conditions were sent to the M1
engineer as messages. Ruling A forbids the consolidated deviations file from citing a lead ruling that is not on
disk. §1–§3 reproduce the three messages **verbatim** (copied by script from the session record; nothing added,
removed or reworded). §4 holds the rulings of 2026-11-10 19:37 EST, which exist only in this file. §5 says how the
file is read.

**This file releases nothing.** No step-1 number, no cross-validation number and no validation number of any
architecture-v2 arm or reference may be produced before the lead stamps the consolidated deviations file. A
deviation from the stamped amendment takes effect only through that stamp.

## 0. Files this record refers to (sha256 re-computed by the lead when this file was written)

| File | sha256 |
|---|---|
| `docs/preregistration/sonde_arch_v2_amendment.md` | `6c7b169d60ad74aa941965c67641d9a9e72faea804db6ac5b5b7b77d6bd60226` |
| `docs/preregistration/sonde_arch_v2_lead_rulings.md` | `7cbf398793a6345b8c0e2aef349f82db821ef821c3c58dd0a5542bddee8aac55` |
| `docs/preregistration/sonde_arch_v2_deviations.md` | `031ddddc8bb8fecb4788ae639bc358f9c28ad042b08cb4ea3d8fd9b1ab2fcd92` |
| `docs/preregistration/sonde_arch_v2_deviations_part2.md` | `b17f3ec67af92809f2c2c2b31dca210275d7712c45d9a7a3d8e9930e788a63f4` |
| `docs/preregistration/sonde_arch_v2_gbm_timing_rule.md` | `9d627602f3868c4b8547bea128862c23effd9a56d7019c47459e24ef253ea658` |
| `docs/preregistration/m1n_protocol.md` | `e71770e5de850dbb4a4f80833aa6a91c9c52de17cbae78764c9af5afebba2b51` |
| `docs/preregistration/reviews/arch_v2_20261110/M1dev_review.json` | `ef0f26bd755a73727c31af30e1b17133381fd60ec874f9a69f7b09446af14c7d` |
| `docs/preregistration/reviews/arch_v2_20261110/M1p2_review_integrity.json` | `3735bb1566ef7193ce423ed878c220c8599bc7e3e273ec17fe50336ca0b2b7c3` |
| `docs/preregistration/reviews/arch_v2_20261110/M1p2_review_code.json` | `c7ff02e5a9362bc0b68b4fcbb70e75679763534a17f8e189311ba5de3f7fdba1` |
| `docs/preregistration/reviews/arch_v2_20261110/M1p2_verify.json` | `9ba443d93358cd7c7f69ce8deffa84d3faf6fc60cf89f6081a44171d137fb14a` |

The two unstamped deviation files (`sonde_arch_v2_deviations.md`, `sonde_arch_v2_deviations_part2.md`) stay on
disk unchanged as the reviewed record; they are superseded by the consolidated file and are never stamped.

## 1. Message of 2026-11-10 14:34 EST (19:34:28 UTC): rulings A–P

> Lead rulings for M1 Sonde after the independent reviews of deviations part 1 and part 2: both are REVISE; no step-1 or CV number until one consolidated file is stamped.
>
> Read first (verbatim reviewer output): docs/preregistration/reviews/arch_v2_20261110/M1dev_review.json (part 1; blocking D2, U1, U3, U8) and M1p2_review_integrity.json (part 2; blocking P2-01..P2-04, should-fix P2-05..P2-12). The part-2 code lens and the verifier are still running. HOLD: do not change any file under sentinel/models/sonde_v2/, scripts/train/sonde_v2_*.py or tests/test_sonde_v2*.py until I message you that the code lens has returned (it is reading that code now). Until then: read the reviews, plan the changes, do metadata-only counts and timing arithmetic, and report the GEFS extraction state (below). Do not edit sonde_arch_v2_deviations.md or _part2.md at all, now or later: they stay on disk unchanged as the reviewed record.
>
> I confirmed U1 and U3 myself in the code (data_v2.py set_products + sonde_v2_cv.py nwp_inputs: the GFS slot is standardised with all-NaN statistics, so it never reaches the networks; prep.py reference_window_climatology has no issue-row rule).
>
> RULINGS (binding; conservative reading of the stamped amendment everywhere):
>
> A. One file. Write docs/preregistration/sonde_arch_v2_deviations_v2.md. It supersedes part 1 and part 2 and is the only deviations file I stamp. It holds every deviation of both parts, corrected, plus the items below, each with: stamped text (file:line), what the code does (file:line at the frozen state), reason, effect on candidate vs references, and counts where rows change. No statement of fact that you have not checked on disk today. No reference to a lead ruling that is not on disk (P2-13): conditions rest on this file and my rulings file.
>
> B. D2 (19 levels). The stamped text is "Quantile LightGBM (19 levels)". Q19 is a mandatory declared configuration of GBM-NWP and of PF-GBM, and Q19 is the step-1 comparator. If step 1 also runs Q or LS, the 0.01 gate uses the best-scoring GBM configuration (worst case for the candidate). Q and LS are additional configurations inside the limit of 5. The "no gain" sentence is removed unless backed by train-only evidence that you cite.
>
> C. P2-01 (gage height). The sentence "no gage-height series exists in the daily tier" is false (gage_ft_mean indexed at 1,258 sites). Default ruling: conform to the stamped amendment (gage height as auxiliary input and output where gage_ft_mean exists; never a decision). If you believe it cannot be restored, give me the true reason and counts before writing the file; do not drop it on your own.
>
> D. P2-02 / U2 (GBM fit window). Every reference is fitted on the same train-block rows as the networks: issue dates through the train-block end (2021-09-30) with the 42-day embargo at the block edge. fit_end is passed explicitly from sonde_v2 code; sentinel/models/sonde/gbm.py and every other file of the original track is NOT edited (the Test-A freeze candidate hashes them). Where a method needs rows for early stopping or calibration, they come from the same 15 % in-fold basin groups as A11 (seed 20261110 + fold), not from a time slice. Same rule in step 1 and CV, written once.
>
> E. P2-03 / A12 (GBM row cap and budget). (1) A full-rows configuration must be in the declared lists of both GBM references and runnable by both drivers. (2) Write the rule that turns a measured time into a cap as exact text BEFORE the timing run that decides it: unit of timing (one variable, one horizon, all quantile levels of that configuration, 6 threads), the candidate caps in descending order (full, then fixed values you name), the subsampling rule (seeded, stratified by site), and that the first cap whose fit is within 2 h is used. (3) The timing script must time exactly those fits, on train-only rows, and write gbm_timing.json; nothing is scored. (4) Step 1 uses the same rule. (5) If Q19 under any cap does not fit the total budget, send me the arithmetic (fits x minutes, per arm) and one concrete proposal (for example fixed boosting rounds and learning rate for all GBM configurations). That proposal is a rule for the references: it goes in the consolidated file and is reviewed before the stamp, not in a later addendum. Early-stopping row rules must be identical in step 1 and CV and recorded.
>
> F. P2-04 / U3 (reference window). (1) Reference window = first 730 calendar days from the site-variable's first observation, at least 365 observed days, as m1n_protocol.md line 39. (2) Row rule for every model, candidate and reference, in step 1 and CV: a row is used for training, early stopping, calibration, ensemble weights or scoring only if its issue date >= that site-variable's reference-window end + 42 days. (3) A site-variable is usable in a build only if its whole 730-day window lies inside the dates loaded by that build; otherwise its climatology is NaN in that build (so step 1, which loads the train block only, never uses a truncated window). (4) Unit tests: no retained row violates (2); a site-variable whose window ends after the loaded range has zero rows. (5) Counts in the file (counts only; never label values or label statistics of val-block rows): site-variables and rows lost or kept per primary variable and role, for step 1 and for CV.
>
> G. U1 / P2-05 / U4 (NWP slots). Each product slot is standardised from its own finite train-block rows. A declared product that is missing, or whose coverage is below a threshold you declare in the file, is a hard error in every driver (no silent skip, no None). Unit test: non-zero GFS-slot mask on val-block batches for SONDE-HF and R-NH. If the GFS train/val table does not exist and cannot be built before step 1, say so and propose the smallest honest alternative (CV with the slots that exist, recorded as a deviation) for my ruling; do not decide it yourself.
>
> H. P2-06. The text must say exactly which arm fits and writes which targets, and the drivers must do that. DO % saturation: the stamped text calls it a deterministic transform; default ruling is conform. A trained head needs a reason and my ruling.
>
> I. P2-07. Record the bias map completely (precipitation ratio-of-means with no clip, identity fallback source 0, 3-point minimum, slope clip [0.5, 1.5]). The A8 counts are appended as a dated addendum once the full WY2008-2019 reforecast exists and before any number that uses the map.
>
> J. P2-08. R-NH is single-target: its training batches and early-stopping rows are drawn from rows that carry a label for its own target, so the reference gets a full budget of labelled rows. n_rows in the parity record = labelled rows. Stage-2 optimiser settings (weight decay included) are recorded and identical for SONDE-HF and R-NH.
>
> K. P2-09 / U5. C4 (CMAL head) is implemented as stamped, with a unit test showing C4 differs from C1 in head and loss. If you would rather strike C4, ask me first; a struck configuration is not replaced.
>
> L. P2-10 / U7. Configuration lists for every arm (candidate and references) are written to the parity log with a sha256 per list and logged to the registry BEFORE the first run of that arm; a driver refuses any configuration whose list hash is not declared. R-NH gets 5 declared configurations (parity with the candidate's 5) unless you give me a reason for fewer.
>
> M. P2-11. GBM-NWP features conform to the stamped sentence (forecast degree-days as stamped). Member mean and spread cannot exist under my ruling R3 (control member only): record that as a deviation that removes information from a reference, disclosed wherever GBM-NWP is reported.
>
> N. P2-12. Error model fitted on WY2008-2019 ending 2019-09-30 as stamped, on the full reforecast, never partial years. Step-1 evaluation asserts that every model used the same noise sample (hash equality), not only equal row indices.
>
> O. U8 (code freeze). After the fixes and green tests: write into the consolidated file a sha256 list of every file in sentinel/models/sonde_v2/, scripts/train/sonde_v2_*.py and the sonde_v2 tests, then stop editing them. Any later change is a new dated deviation that I stamp before the affected run.
>
> P. Remaining part-1 items D1, D3, D4, D5, U6, U9 and notes: each gets one line in the consolidated file: resolved (file:line) or recorded as a deviation.
>
> REPORT NOW (no code change needed): (1) GEFS extraction state: months on disk of the 144 reforecast months and of the operational period, whether the running extraction (PID 2244958, started with --pilot) continues past the pilot by itself, measured throughput and ETA for the full WY2008-2019 reforecast, projected kept size; whether the R3 pilot registry rows exist. (2) Whether a GFS (Open-Meteo) train/val table can be built and how long. (3) Your GBM timing arithmetic and the E(2)/E(5) proposals. (4) Anything in A-P you think is wrong or impossible, with evidence. Then wait for my go on code edits.
>
> Unchanged rules: no validation number on any new arm before the stamp; train-only or synthetic smoke only; new-site block never read; GPUs via the lock helper, <= 2 h per job, <= 6 CPU threads; nothing installed into sentinel2; no git commits. The Test-A rehearsal of the original track continues and is not delayed by any of this.

## 2. Message of 2026-11-10 14:54 EST (19:54:49 UTC): go on code edits, clarifications, verifier items 1–7

The message's own first line gives its time as "14:58 EST"; the session record's time is 14:54:49 EST.

> Lead to M1 (14:58 EST): GO on code edits. The part-2 code lens and the verifier are back (overall REVISE, every blocking and should-fix finding confirmed). Read both in full first: docs/preregistration/reviews/arch_v2_20261110/M1p2_review_code.json and M1p2_verify.json (keys: verified[].finding_id/status/severity_after_check/evidence/condition_for_rulings, missed[], overall_reason). Every `condition_for_rulings` in the verifier file is binding on top of A–P, plus the four `missed` items. Parts 1 and 2 stay unedited.
>
> Rulings on your points:
> - F (clarification): accepted. "First observation" = the first finite value on or after the cube start 2007-10-01. Reference window = the 730 days from it, ≥ 365 observed. Enforce with a separate issue mask (issue ≥ window end + 42 d) for every model and every use (training, early stopping, baselines, scoring); do not blank the climatology of hindcast inputs. Same build in step 1 and CV, unit test, counts by category in the file.
> - D (wording): accepted. Train rows = target day (issue + h) ≤ 2021-09-30 by the frozen issue_dates rule, no fit_end mask, no extra embargo inside the train block, identical for every model. The fit window is written to the parity log for every run.
> - E(1): confirmed, per configuration. The cap of a configuration is set by the more expensive arm (PF-GBM) and used by BOTH GBM arms. LS at full rows if its unit passes.
> - E(5) lr 0.10 / 1000 rounds: NOT ruled yet. It changes the frozen reference recipe, and a 500k cap against networks trained on all rows weakens the comparator, so I want measurements first. Do this timing-only benchmark as soon as the code is fixed (rule text written before the run, train-only rows, fixed 50 rounds per booster, no score kept except wall time and train-loss equality): (i) CPU 6 threads at full rows and at 500k; (ii) LightGBM `device=gpu` with the same max_bin as the CPU setting on one idle GPU (7, 8 or 9, through the lock helper), same rows. Nothing is installed into sentinel2; if the wheel there has no GPU support, say so and I decide on a separate env. Also check on a synthetic task that GPU and CPU fits give the same pinball loss within 1 %. Send me: s/round per booster for each, and the projected hours for step 1 and for CV at lr 0.05 / 2000. Preference order I will rule from: frozen recipe at full rows on GPU > frozen recipe with the largest passing cap > your E(5). GPUs 7–9 are idle now.
> - G: (b). GEFS-only for EVERY arm, recorded as a deviation with the true reason (the GFS table cannot exist before about 11-28; train-block overlap is 191 days, T2m only). Exactly one declared product; any arm that declares or reads a second product fails hard; remove the dead slot 2 from the network inputs. The pull may keep running; no model reads it in this round.
> - H: accepted. DO % saturation = deterministic transform from each model's DO-mean quantiles at its T-mean median, disclosed as ignoring T uncertainty. References fit and write the 8 scored variables only; the §2d auxiliary report is SONDE-HF against persistence and climatology and carries no claim against the references — say so in the file and in the report header. Gage height: input for every model that can take it; output for SONDE-HF.
> - O: accepted. M1-H model code and its CV driver come later as NEW files under their own stamped deviation, with no edit to the files in the code-freeze list. No M1-H number before that stamp; M1-H must be done before the freeze (R4).
> - D5: conform as you plan.
>
> Extra items from the verifier (all binding):
> 1. Drivers refuse to produce predictions or registry rows unless a lead-written `M1-sonde` `sonde/prereg_snapshot` row carries the current sha256 of sonde_arch_v2_deviations_v2.md and the file is read-only (F04). Test it.
> 2. Stage 2 uses weight decay 1e-5 (fix the saved-optimiser overwrite, F08); record es_max_samples 30,000.
> 3. GBM early-stopping rows drawn after the finite-target mask, one declared size, effective count per variable and horizon in the parity log (F10).
> 4. Step 1 refuses to run unless every reforecast month 2007-10 … 2019-09 is present; its eval phase fails unless the perfect-prog input hash of the HF phase equals that of the GBM phase (P2-12, F07). Record that step 1 and CV have different date axes.
> 5. Parity check step across the arms of each fold before any CV score; Chronos runner writes a parity record (F15). One declaration record with every arm's full configuration list (GBM caps and fit windows included), hash in the registry, drivers refuse anything else (P2-10).
> 6. Cumulative degree-day feature must be NaN, not 0, when the product is missing (nansum fix), both GBM arms. "Leads 1..h" = leads 1..min(h, 7), recorded (F13).
> 7. A8 counts (sources 0–3 per product, the unclipped precipitation ratio map, the 3-pair minimum, the pass-through) go INTO the consolidated file, so write its final version after the full WY2008–2019 reforecast is on disk (P2-07, F03).
>
> Order: (1) code fixes + tests green; (2) GBM timing benchmark → send me the numbers, I rule on the GBM settings; (3) when the reforecast is complete: A8 counts; (4) write docs/preregistration/sonde_arch_v2_deviations_v2.md (one file, supersedes parts 1 and 2, rule A form, code sha256 list per rule O) and send me path + sha256; no edits after that; (5) I run a delta review and stamp; (6) step 1: `--phase hf` on a GPU and the GBM phase, then eval. Report job name, GPU, PID and start time to me at once when the HF phase starts — the user is waiting for exactly that notice.
>
> No step-1 or validation number before my stamp; lock helper for every GPU job; no installs into sentinel2; no git commits.

## 3. Message of 2026-11-10 18:25 EST (23:25:53 UTC): CPU rule for the GBM timing benchmark

The message's own first line gives its time as "18:30 EST"; the session record's time is 18:25:53 EST.

> Lead (18:30 EST): rehearsal result received (12 of 12 jobs identical, plan 9e59bb9d, identity.json); I will commission the pre-test review after your code fixes are in, so the reviewed code is the final code. CPU question: allowed. The machine has 48 cores and the 1-minute load was about 29 at 18:25 EST. Run the GBM timing benchmark (6 threads) alongside the extraction only if the 1-minute load average at start is ≤ 36, and record the load average at start and end in the benchmark record. Because contention can only make the GBM look slower: a timing taken under load may not be used to justify a smaller GBM setting. If the numbers would lead you to propose a reduction, repeat the benchmark after the extraction ends and use that one.

## 4. Rulings of 2026-11-10 19:37 EST (new; they exist only in this file)

- **Q — GBM timing benchmark when the machine is loaded.** At 19:35 EST the 1-minute load average was about 70 on
  48 cores, almost all of it this project's own jobs. §7 of `sonde_arch_v2_gbm_timing_rule.md` lets the benchmark
  start only at a load of 36 or less. The lead adds one route, because contention can only lengthen a wall time:
  the benchmark **may** be run once at any load, with the 1-minute load average at start and end in its record,
  and that record may be used for **one conclusion only**: if, for the more expensive arm (PF-GBM), the time of
  the unit at **full rows** computed by §3 of the timing rule is 2 h or less (on CPU with 6 threads, or on the
  GPU), then the configuration runs at full rows on that device. A record taken at a load above 36 may **not** be
  used to choose a cap below full rows, to choose a device because the other looked slower, or to support the
  E(5) proposal. If the loaded record does not give "full rows", it is kept and marked `not_used_for_cap`, and
  the benchmark is repeated at a load of 36 or less; only that second record enters §4 of the timing rule.
  Everything else in the timing rule is unchanged (train-only rows, 50 rounds, no score, no registry row).
  The lead still rules on the GBM settings after seeing the numbers, and the ruling is entered in
  `sentinel/models/sonde_v2/declared.py` before the code list is written.
- **R — the stamp row.** When the lead stamps `sonde_arch_v2_deviations_v2.md`, the new row of
  `M1-sonde` / `sonde/prereg_snapshot` carries `extra.written_by = "lead"`, `extra.sha256` (a mapping from
  repository-relative path to sha256 that contains at least the consolidated file, this file and the lead's
  rulings file on the consolidated file) and `extra.code_sha256` (the value the M1 engineer sends with the
  consolidated file, re-computed by the lead before the stamp). Row `2921bb1ca000` stays as the stamp of the
  amendment; it does not release a driver.
- **S — order.** Benchmark → the lead's GBM ruling (a dated addition in a new lead file, not an edit of this one)
  → `declared.py` → final green test run → consolidated file with the code list (rule O) → independent delta
  review → stamp → step 1 (`--phase hf` first).

## 5. How this file is read

- Rulings A–P are read together with the clarifications of §2. Where §2 changes a letter of §1 (D, E(1), F, G, H,
  O), §2 wins. Where §3 and ruling Q differ, Q wins.
- "Verifier conditions" means every `condition_for_rulings` and every `missed` item of `M1p2_verify.json` at the
  sha256 of §0. They are binding, as §2 says.
- Message rulings on the two unstamped parts that were sent before 14:34 EST on 2026-11-10 are superseded by A–P.
  They are not on disk and are not cited. If the consolidated file needs one of them, the M1 engineer asks and the
  lead adds it in a new dated file.
- Where this file and the stamped amendment (`sonde_arch_v2_amendment.md`) or `sonde_arch_v2_lead_rulings.md`
  (R1–R5, L1–L5) differ, the stamped files hold until the consolidated deviations file is stamped; from that stamp
  the lead's rulings file on the consolidated file says which text wins.
- Nothing is appended to this file. Later rulings go into new dated files.
