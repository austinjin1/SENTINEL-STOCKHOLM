# M4 ChemTox — architecture round v2 amendment: tracks T1 and T6 (version 2)

**Version 2, written 2026-11-10. UNSTAMPED.** No fit, no validation number and no test look exists
on T1 or T6, and none may before the lead's stamp.

**Version 1 was reviewed and NOT stamped.** Three reviewers and a second-stage verifier found five
blocking defects that needed the file changed rather than a ruling: already-scored ADORE test rows
were never subtracted (4,276 sat inside v1's own `T1_test`, 40.5 % of it, and 22,532 in its
training pool, so v1's "no spent block" sentence was false); the MDE used row-independent synthetic
errors and so set the binding gate 2.5–5× too lax; label-derived SD and replicate floors of all
three sealed blocks were printed as counts-only; auxiliary chronic, censored and BCF rows carried
sealed chemicals, sealed cells and long durations into training; and the pinned builder did not
emit the pinned split and left an unsealed labelled copy in a temporary directory. V1 is preserved
byte-for-byte at `docs/preregistration/chemtox_arch_v2_amendment_v1_unstamped.md`, sha256
`b9326d49aca711f0ceeec1362c21883030a94f83dd6971a2db15cba1babc9a9e`, and the review outputs are in
`docs/preregistration/reviews/arch_v2_20261110/`. **Nothing in v1 may be quoted as
pre-registered.**

This file amends nothing in place. Every frozen artefact keeps its bytes: nothing pinned by the T2
unseen-taxa freeze (`chemtox_unseen_taxa_plan.md`, `chemtox_unseen_taxa_pins.json`,
`unseen_taxa.py`, `unseen_taxa_guard.py`, `test_blocks.parquet`, `training_pool.parquet`) or by the
M5 BioAssess freeze (`chemtox/{__init__,evaluate,features,tabpfn_model}.py`, `gpboost_model.py`) is
edited, and the v1 sealed artefacts (`chemtox_t1t6_split_v1.parquet` sha `005f7957…`,
`chemtox_t1_clusters_v1.csv` sha `a6e33634…`) are kept untouched as the record. Any change after
the stamp comes as a dated addendum appended to this file, never as a rewrite.

**Governing inputs.** `docs/preregistration/reviews/arch_v2_20261110/M4_verify.json` (27 verdicts,
4 points, 6 missed items — the verifier governs where the first-stage reviews differ), the lead's
rulings L1–L11 of 2026-11-10, `docs/plans/architecture_research/LEAD_DECISIONS_20261110.md`
§"M4 ChemTox: update 2026-11-10 12:25 EST", `docs/plans/architecture_research/M4_architecture_plan.md`,
`docs/plans/sota_campaign.md` rules 1–8, RESEARCH_PLAN §15.1 and §15.5.

**Not in scope.** T2 runs to completion exactly as pre-registered and is reported whatever the
outcome; this amendment adds no entry, configuration, architecture or input to it. T3 stays
deferred to the next EPA release. T4 stays exploratory with no budget. T5 is **blocked on a
licence** (§10.3) and recorded as blocked, not skipped. No spent block is re-scored, re-trained on
or re-claimed.

---

## 0. Where every review item is answered

| Verdict / item | Answered in |
|---|---|
| SPENT-ADORE (blocking) | §2.1, §3.1, §3.2 |
| MDE-ROW-INDEPENDENT (blocking) | §6.1, §6.2 |
| PROTECTED-LABEL-STATS (blocking) | §3.5, §6.6, §15.1 |
| AUX-LEAK (blocking) | §10.2, §4.1 |
| BUILDER-PROVENANCE (blocking) | §2.2, §15.2 |
| GUARD-GAP | §8.1, §8.2 |
| PARENT-KEY | §2.3 |
| PRETRAINED-CACHE | §10.5 |
| LOOK-ORDER-AND-FAMILY | §8.3 |
| T6B-POWER-CLAIM | §6.5 |
| DURATION-FLOOR-AND-BANDS | §3.3, §3.4, §6.7 |
| R5-NOT-PAIRABLE | §4.4 |
| SHARED-REFERENCE-GATE | §12.1 (h), §6.4 |
| SEEDS | §12.1 (d) |
| PRIMARY-METRIC-AND-CI | §12.2 |
| COMPARATOR-ROW-SETS | §4.5 |
| T6-BASELINE-LADDER | §5.4 |
| OPEN-CHOICES | §5.6 |
| MDE-DOMAIN-BYPASS | §6.3 |
| CONFIG-TOTALS | §5.5 |
| GPU-BUDGET | §5.7 |
| RESERVATION-NOT-REGISTERED | §8.5 |
| GPBOOST-FREEZE-CLAIM | §11.2 |
| R4-PARITY-CONTRADICTION | §4.3 |
| REPLICATE-FLOOR-AS-CLAIM-FLOOR (refuted) | §6.6 |
| SAME-POOL-TWO-BUDGETS | §5.8 |
| VERIFICATION-RECORD | §16 |
| Point 1 — the T6 extrapolation disclosure | §1.2, §3.4, §10.2 |
| Point 2 — the 544 total | §5.5 |
| Point 3 — T5 blocked, R5 admissible as data only | §4.4, §10.3 |
| Point 4 — no ruling-requested line; T6b's status | §6.5, §15 |
| Missed 1 — trivial baselines spend a block | §3.1 |
| Missed 2 — matched baseline-ensemble control | §5.3 |
| Missed 3 — reproduction-fidelity tolerance | §4.6 |
| Missed 4 — filtered auxiliary counts | §10.2 |
| Missed 5 — the 0.05 ensemble margin | §7.3 |
| Missed 6 — pre-FREEZE read of a third-party error table | §14 |

**The headline consequence, stated first because it changes what these tracks are for.** With the
spent-ADORE rows removed and the power model corrected, the governing minimum detectable effect is
**0.096 (T1_test), 0.103 (T6a_test), 0.103 (T6b_test)** log units. The largest difference this
project has ever measured between two parity-matched models is 0.070, and the spent prospective
block's headline difference was 0.029. **All three blocks are therefore pre-registered as
DESCRIPTIVE by default**, per lead ruling L4: a §15.1 label is available only if the realised
difference exceeds the block's governing MDE, and on this project's own track record that is
unlikely. §6.2 and §15.3 say so plainly rather than burying it.

---

## 1. The two tracks, and why they were chosen (sota_campaign rule 2)

Both justifications were recorded before any validation number on either track existed, and they
still are: nothing has been fitted. Neither track was chosen because a result favoured it — the
only results that existed when they were chosen are T2's in-flight selection scores, and those are
*unfavourable* (a mandatory comparator leads).

### 1.1 T1 — unseen chemical clusters (new PRIMARY)

A hazard-screening model's real failure mode is a chemical nobody has tested: a new pesticide, a new
pharmaceutical metabolite, a spill of something unusual. It is also the only regime in which M4 has
seen a learned representation beat the trees (chemprop 1.202 vs reference 1.287 on the
ADORE-unseen-chemical rows of the spent prospective block). The practical question it answers is the
one a regulator asks of a new substance.

### 1.2 T6 — exposure-duration extrapolation (SECONDARY)

Standard tests run 24–96 h; the questions that matter after a spill are "what happens over a week?"
and "what does a one-hour pulse do?". Regulatory extrapolation across duration is done today with
crude rules.

**The "structurally handicapped comparator" argument, restated conditionally (Point 1).** V1 argued
that T6 is the one axis where the comparator class is structurally beaten, because beyond its last
split a tree predicts a constant. The verifier showed that argument was **false as v1 was written**:
138,341 chronic rows on pool chemicals, 84,812 of them outside 24–96 h and 72,900 above 96 h, up to
26,298 h — the same maximum as T6a_test — would have reached training, with 11,773 of them in the
same (CAS, species, duration) cell as a T6a_test row. A model that has seen the shape cannot be said
to be extrapolating it.

The argument is therefore **withdrawn unless the restriction is in force**, and the restriction is
now in force and measured: §10.2's loader restricts **every** T6 auxiliary and pretraining row to
24 ≤ `duration_h` ≤ 96, and the measured maximum labelled duration any T6 model sees in training is
**96.0 h** (asserted in `tests/test_chemtox_arch_v2.py::test_t6_auxiliary_rows_carry_no_long_duration_label`).
With that in force the argument holds and is stated as: *on T6, no model — candidate or comparator —
has seen a label beyond 96 h, so predictions beyond it are extrapolation for all of them, and a tree
extrapolates as a constant while a composed time law extrapolates as a shape.*

### 1.3 T6a and T6b

T6a: unseen duration, seen chemistry. T6b: unseen duration **and** held-out chemical unit. Both are
pre-registered now as full blocks with their own looks (§6.5).

---

## 2. Pinned artefacts, the blocking unit, and the builder

### 2.1 The frozen files and their hashes

| Artefact | Path | sha256 |
|---|---|---|
| Spent-ADORE `result_id` list (keys only; 8 partitions, 28,265 ids) | `docs/preregistration/chemtox_arch_v2_spent_adore_ids_v2.json` | `3fa94b68ad96c9018737886430ae6d1efd3b1c52dc33b2d98b4857a87f5bd1d8` |
| Cluster + parent-key + hold-out-unit table (3,210 chemicals) | `docs/preregistration/chemtox_t1_clusters_v2.csv` | `a906aac1f9dd97577ff473981a1fce5aae1a509e3fb2dcb42f8726dea2d821f5` |
| Row partition, **label-free**, 9 columns, 58,824 rows | `$SENTINEL_DATA/processed/chemtox_arch_v2/chemtox_t1t6_split_v2.parquet` | `ab71aace46260980a6f011a561c2280fe0025226978508daa57f96493c5323a3` |
| Builder (emits all three of the above) | `sentinel/models/chemtox/arch_v2.py` | pinned at the stamp by `arch_v2_guard.make_pins()` |
| Power helper (L5) | `sentinel/models/chemtox/arch_v2_power.py` | pinned at the stamp |
| Single-look guard (L10) | `sentinel/models/chemtox/arch_v2_guard.py` | pinned at the stamp |
| Counts report (the source of every number in §3) | `$SENTINEL_RUNS/chemtox/arch_v2/t1t6_counts_v2.json` | `9fd67573287e62b6ff50cf4c673e2373a2795fa99bfbfe1224f5601516c6d4b4` |
| Auxiliary row counts (§10.1, §10.2) | `$SENTINEL_RUNS/chemtox/arch_v2/aux_row_counts_v2.json` | `e250e48e942b0ccee6b3f73ce1f2dd9ec8fc8372da737417b6de7e7024c59c76` |
| Power report | `$SENTINEL_RUNS/chemtox/arch_v2/t1t6_mde_v2.json` | `f67d0376fa3440e7653c77fd0a74ed78afda9a214979f44fdb5af6bc32151475` |
| H4 join/QC report with AUX-LEAK filters | `$SENTINEL_RUNS/chemtox/arch_v2/h4_bcf_join_qc_v2.json` | `7dbdd501f47c29be973f7c424b86bec5532c382b36c55f1054c1b834d4ec527c` |
| ADORE partition inventory (the L1 report) | `$SENTINEL_RUNS/chemtox/arch_v2/adore_partition_inventory_v2.json` | re-hashed at the stamp |

The split file carries **no target column** — `result_id`, `test_cas`, `t1_cluster`,
`holdout_unit`, `duration_h`, `duration_band`, `in_adore_domain`, `shared_reference`, `block` — so
every downstream power computation reads structure, never a label. The three source-code files are
pinned by `arch_v2_guard.make_pins()` at the stamp and re-verified by `verify_freeze()` before every
look. A revision writes `_v3`; this version is preserved.

### 2.2 The builder is the thing that produced the pinned files (BUILDER-PROVENANCE)

`sentinel.models.chemtox.arch_v2.build_split()` performs the subtractions, the clustering, the unit
construction, the draw, the nine-column trim **inside the function**, and the writes, and returns
the report this section quotes. It writes only to the pinned paths. **No artefact pinned here lives
in a temporary directory.** The v1 eight-column labelled intermediate
(`…/scratchpad/t1t6/chemtox_t1t6_partition.parquet`, 1,609,554 bytes, sha256
`9ef801b86207aac55bf872acfa5a2705e7158198d9a52c4d1c142c64e7736c31`) and every copy of it have been
**deleted**; a filesystem search for that basename returns nothing. The v1 label-bearing reports are
retained, bytes unchanged, renamed `t1t6_counts_v1_WITHDRAWN_label_stats.json` and
`t1t6_mde_v1_WITHDRAWN_label_stats.json`, and the v1 scripts as `*_v1_superseded.py`.

### 2.3 The hold-out unit: connected components over (cluster, parent-key family) — L2, strict

Chemicals are clustered by **ECFP4 (Morgan radius 2, 2048 bits) Butina at Tanimoto distance
cutoff 0.4**, over one canonical SMILES per `test_cas` (the most frequent SMILES recorded for that
CAS). The **hold-out unit** is then the connected component over two relations: same cluster, or
same `parent_key`. `parent_key` is the project's own definition, `prospective.parent_key` — the
skeleton block of the InChIKey of the largest fragment, charges neutralised and stereochemistry
removed — called, never modified.

Whole components are held out, which makes *"no parent-key family and no cluster spans a held-out
and a retained unit"* true **by construction**; `holdout_units()` asserts it, `build_split()`
re-asserts it on the real frame, and two tests check it. The same component is the resampling unit
of every cluster bootstrap and of every MDE in §6 (L2).

Measured: **2,153 clusters → 2,029 hold-out units**, 124 cluster pairs or chains merged by a shared
parent key, 1,583 units holding a single chemical.

**Why cutoff 0.4, and an honest exceedance.** The criterion was fixed in v1 before any of this:
*the largest cluster must hold under 10 % of the universe's rows.* Murcko scaffolds fail twice here
(46.2 % of usable acute rows are acyclic; the benzene scaffold alone covers 20 % of the endpoint),
so cluster membership rather than scaffold identity is the unit. On the v2 universe the largest
**cluster** holds **7.76 %** of rows — the criterion passes — but the largest **hold-out unit**,
after the L2 merge, holds **10.09 %** (5,938 of 58,824 rows). The cutoff is **not** re-tuned to
cure that: changing a split parameter after seeing its effect on our own power is the forking path
this whole review exists to prevent. The exceedance is instead disclosed, and it is carried
explicitly into the power calculation, which uses the exact unequal-unit-size formula precisely
because one unit this large dominates the sampling noise (§6.1). The largest unit's share **within
each block** is reported in §3.2 and is the reason the T6 blocks cannot resolve better than ~0.10
whatever the hold-out share.

**Split seed `20261110`**, unchanged from v1. The draw is **a share of UNITS**, not of rows (L4),
because unit count and not row count governs power: units are permuted under the seed and the first
`round(share × n_units)` are held out.

---

## 3. The universe, the subtractions, and the partition

### 3.1 Spent-ADORE: derived by code, not typed (SPENT-ADORE, Missed 1, L1)

`arch_v2.scored_adore_partitions()` scans `results/registry.jsonl` for any row whose `split` field
names a `<challenge>/<split>/test` partition, normalises the two spellings
(`occurrence` → `split_occurrence`, `scaffold-murcko` → `split_scaffold-murcko`), and takes the
union. **A registry row spends the partition whatever the experiment and whatever the model: a
trivial baseline reads the same held-out labels as a candidate does.** That is the lead's
conservative answer to the verifier's open question, and it is why the set is 8 partitions and not
the 4 the first-stage reviewers named.

| Partition | Scored under | Test `result_id`s |
|---|---|---|
| `t-F2F/split_occurrence` | `adore_chemtox_test`, `adore_published_baseline`, `adore_trivial_baselines` | 4,133 |
| `t-F2F/split_scaffold-murcko` | `adore_chemtox_test`, `adore_trivial_baselines` | 4,326 |
| `t-C2C/split_occurrence` | `adore_chemtox_test`, `adore_trivial_baselines` | 1,038 |
| `t-C2C/split_scaffold-murcko` | `adore_trivial_baselines` | 1,102 |
| `t-A2A/split_occurrence` | `adore_trivial_baselines` | 85 |
| `t-A2A/split_scaffold-murcko` | `adore_trivial_baselines` | 113 |
| `a-CA2F-diff/split_occurrence` | `adore_chemtox_test`, `adore_trivial_baselines` | 3,576 |
| `a-CA2F-same/split_occurrence` | `adore_trivial_baselines` | 22,538 |
| **union** | | **28,265** |

Note `chemtox_test_plan.md` §1 explicitly excluded `a-CA2F-same` from test *claims*, yet its test
labels were read to score trivial baselines. It is the largest single contributor and it is
subtracted.

**These rows leave the universe entirely** — not in any test block, not in the pool, not in any
auxiliary, multitask, pretraining or ablation row set. `build_split()` asserts it after the cut;
`auxiliary_rows()` applies the same subtraction with its own counter; and
`test_no_spent_row_is_anywhere_in_the_split` checks the pinned artefact.

### 3.2 The four subtractions, each a counted assertion, and the partition

| Step | Rows | CAS | Species | Removed |
|---|---|---|---|---|
| Usable acute (LC50/EC50, uncensored, molar value present, SMILES present) | **122,946** | 4,592 | 2,550 | — |
| − T2's six reserved classes, outright | **98,727** | 4,156 | 1,586 | 24,219 |
| − spent prospective `result_id`s | **87,290** | 3,950 | 1,585 | 11,437 |
| − **spent ADORE `result_id`s (L1)** | **60,482** | 3,277 | 1,549 | **26,808** |
| − no duration, and duration ≤ 0.25 h | **58,842** | 3,212 | 1,534 | 1,526 + 114 |
| − 2 chemicals RDKit cannot parse | **58,824** | 3,210 | 1,534 | 18 |

The first two steps reproduce the architecture plan's A.2 and A.8 figures exactly. The class-level
exclusion is a **strict superset** of T2's reserved `result_id`s, so T2's three frozen blocks stay
out **without ever opening `test_blocks.parquet`**. In-domain (`chem_structure_class` organic or
organic_multicomponent): **38,244 of 58,824**; 20,598 rows are inorganic, organometallic or
structureless.

The partition is one **row-disjoint four-way split**, so no row is ever scored twice and no block's
look spends another block's rows (rule 1). The hold-out share is **30 % of units** (§6.2 proposes
it; the lead rules at the stamp).

| Block | Rule | Rows | Units | Clusters | Chemicals | Species | Refs | Largest unit | In-domain rows / units |
|---|---|---|---|---|---|---|---|---|---|
| `train_pool` | unit retained, 24–96 h | **32,106** | 1,293 | 1,377 | 2,012 | 1,193 | 3,757 | 4,321 (13.5 %) | 18,416 / 1,189 |
| **`T1_test`** | unit held out, 24–96 h | **11,363** | **546** | 572 | 860 | 711 | 1,847 | 1,115 (9.8 %) | 10,368 / 502 |
| **`T6a_test`** | unit retained, outside 24–96 h | **11,249** | **694** | 744 | 1,027 | 545 | 1,514 | 1,617 (14.4 %) | 5,821 / 583 |
| **`T6b_test`** | unit held out, outside 24–96 h | **4,106** | **289** | 305 | 425 | 364 | 687 | 555 (13.5 %) | 3,639 / 250 |

**No label-derived statistic of any sealed block appears in this table** (PROTECTED-LABEL-STATS):
no SD, no replicate floor, no per-stratum anything. The only floor quoted anywhere in this file is
the **training pool's** 0.504, and only as an absolute-RMSE bound (§6.6).

**Shared-publication concentration**, measured from `reference_number` and written into the split as
the `shared_reference` column (structure, not a label): the share of a block's rows whose
publication also appears in the pool is **67.0 % (T1_test)**, 44.7 % (T6a), 28.2 % (T6b). The
`no shared reference` strata hold 3,745 rows / 143 units, 6,224 / 522 and 2,946 / 227. T1's 67 % is
far above T2's 16–29 %, so the reference-clustered reading is **co-primary** and the stratum has its
own gate (§12.1 (h)).

### 3.3 The duration floor and the bands (DURATION-FLOOR-AND-BANDS)

**Minimum admissible duration: 0.25 h (15 min).** Below it an aquatic LC50/EC50 record is not
distinguishable from an instantaneous dose, and both the crude baseline and the candidate's time law
diverge as t → 0: Haber's rule gives C = k/t → ∞, and the composition layer's
`-log10(-expm1(-k t))` → `-log10(k t)` → ∞. **114 rows at or below the floor are dropped from every
block, with the count recorded**; 1,526 rows carry no duration at all and are also dropped.

**Bands.** Every band is half-open `(lo, hi]` **except the training-window band, which is the closed
interval [24, 96]**, and this is a deliberate, disclosed departure from the review's literal wording
`(0,24], (24,96], …`. Under the literal reading every exactly-24 h row — **9,759 of them, the single
most common acute duration** — would fall in the *short tail* while the training window kept them,
so "band" and "window" would disagree about the same row; narrowing the window to `(24,96]` instead
would score a standard 24 h acute test as duration extrapolation, which it is not. Aligning the one
shared edge removes exactly the ambiguity the half-open wording exists to remove. `in_window` is
therefore not a separate predicate at all: it **is** `duration_band == "w24_96"`, and `build_split()`
raises if the two ever disagree on any row.

| Band | Definition | `T6a_test` rows / units | `T6b_test` rows / units | Claimable (≥ 50 units) |
|---|---|---|---|---|
| `lt24` | 0.25 < d < 24 | 3,762 / **342** | 1,610 / **138** | yes / yes |
| `w24_96` | 24 ≤ d ≤ 96 | — (training window) | — | — |
| `gt96_168` | 96 < d ≤ 168 | 3,336 / **445** | 1,165 / **184** | yes / yes |
| `gt168_720` | 168 < d ≤ 720 | 3,834 / **333** | 1,187 / **128** | yes / yes |
| `gt720_2160` | 720 < d ≤ 2160 | 267 / **68** | 126 / **39** | yes / **no** |
| `gt2160` | d > 2160 | 50 / **18** | 18 / **7** | **no** / **no** |

**No claim rests on a band with fewer than 50 resampling units.** The rule is applied as written, so
on the v2 counts T6a's 720–2160 h band clears it with 68 units while the review's parenthetical
expected it not to; T6a's `gt2160` and both of T6b's long bands do not and are **descriptive only**.
Per-band results are a **mandatory deliverable of every T6 look**, the headline is scoped to the
bands in which the sign holds, and no write-up quotes a long-range figure without the > 720 h strata
beside it.

### 3.4 Extrapolation reach, per block, and the honest scope (Point 1)

`T6a_test` reaches **273.9×** the window's upper edge (max 26,298 h); `T6b_test` reaches **77.0×**
(max 7,392 h). Those two numbers are quoted together, always, and never as a single "274×": the
long tail that justifies the figure is **50 rows in T6a and 18 in T6b**, while 3,762 T6a rows sit
below 24 h and 3,336 in 96–168 h. The duration coverage of every row set is stated separately:
primary blocks as above; **auxiliary and pretraining rows on T6 are capped at 96 h** (§10.2), so the
maximum labelled duration a T6 model sees in training is 96.0 h; on T1 auxiliary rows carry their
own durations (max 26,298 h for H3), which is admissible because T1's claim is about chemistry, not
duration, and is disclosed.

### 3.5 Selection folds mirror how each test block is built

T6's selection folds can extrapolate at most **2×** inside the pool while the test blocks reach
77–274×. Selection therefore chooses a configuration; only the sealed look tests the shape claim,
and the claim rests on the look. The alternative — putting part of the long tail into the pool — is
rejected because it shrinks the regime the track exists to test, and because it would reintroduce
exactly the leak the AUX-LEAK ruling closed.

---

## 4. References, comparators and parity

### 4.1 The parity statement (binding)

**Every input, auxiliary target and extra training row a candidate receives is given to every
mandatory comparator, on identical rows.** It is checked mechanically: the runner logs, per fold and
per entry, the training row count, the sorted feature-column names and their hash, the set of
auxiliary targets, and the AUX-LEAK removal counters of §10.2; a comparator whose row set or input
set differs from the candidate's by anything other than the estimator **fails the run**. This is the
C-M4P-1 lesson, and the general form of T2's condition (d2).

### 4.2 Mandatory references, built first, never candidates

| Ref | Model | Why mandatory |
|---|---|---|
| **R1** | Monotone-duration GBM with `monotone_constraints` on continuous `duration_h` plus a log-duration spline basis | Kills "a tree cannot be monotone in duration". On T6 it is the candidate's direct rival. **Library fixed to LightGBM** (§5.6) |
| **R2** | Interval-censored AFT GBM, `xgboost` `objective=survival:aft` with `label_lower_bound`/`label_upper_bound` | Kills "squared-error GBMs must drop censored rows" |
| **R3** | Multitask-by-stacking GBM over acute + chronic with an endpoint indicator and per-endpoint offsets | Parity for the "more targets" lever (rule 4) |
| **R4** | The concatenated embedding block goes to R1–R3 and to the re-tuned `rf`/`xgboost` arms | §4.3 fixes its exact scope |
| **R5** | TRIDENT's released per-SMILES 10×10-CV errors, never a re-run | §4.4 fixes what it can and cannot do |

If a monotone, interval-censored, multitask GBM closes most of the T6 gap, that is a publishable
negative costing no GPU time, and §15.3 records it as the most likely outcome.

### 4.3 R4's scope: §4.3 governs, the published arms stay verbatim (R4-PARITY-CONTRADICTION)

V1 said both "every comparator gets the embedding" (R4) and "the published arms stay verbatim". The
contradiction is resolved **in favour of the verbatim reading**:

* `rf_published_adore`, `xgboost_published_adore` and `gpboost_published` are run **verbatim, without
  the concatenated embedding**, and are reported as **published-form comparators only**.
* **R4 parity applies to R1–R3 and to the re-tuned `rf_emb` / `xgboost_emb` arms**, and those are
  the arms condition **(g)** is read against.
* Any embedding-fed published twin would be a **new entry** and must be enumerated inside the caps
  before the stamp. None is.

### 4.4 R5 is a descriptive cross-reference, never claim condition (a) (R5-NOT-PAIRABLE, Point 3)

**Condition (a) is read only against a published comparator actually re-run here, at full parity, on
the block's own rows** — `mmoe` or `gpboost_published`, whichever leads pool validation, **named in
writing before the look** (§5.6 fixes that this is resolved by pool validation, never by choice
after the look).

TRIDENT's released per-SMILES errors are a **descriptive, non-paired, non-parity cross-reference**,
reported with its split mismatch and its training overlap with our rows stated beside it. **It can
never satisfy (a) and can never carry a §15.1 label.** Its per-SMILES errors are **restricted to
pool chemicals until the relevant FREEZE file exists** — reading a third party's per-chemical
difficulty ranking over sealed chemistry before the look is itself a pre-look read (§14). TRIDENT is
never re-run on our rows: that measures memorisation.

### 4.5 Comparator row sets and declared fallbacks (COMPARATOR-ROW-SETS)

For each comparator this file states the rows it is evaluated on and the fallback for rows it cannot
score:

| Comparator | Evaluated on | Fallback for unscoreable rows |
|---|---|---|
| R1, R2, R3, `rf_emb`, `xgboost_emb`, `lightgbm_noemb`, `mmoe` | every row of the block | none needed |
| `rf_published_adore`, `xgboost_published_adore` | every row of the block | published-form imputation of missing published descriptors, as in the ADORE reference implementation |
| `gpboost_published` | every row of the block | population mean for a grouping level unseen in training |
| `readacross` | every row of the block | the global neighbour pool, with `widening_` counters reported |
| trivial baselines | every row of the block | global mean where a species or chemical is unseen |
| `r5_trident` | the intersecting SMILES only | **secondary and labelled as such**; never used for (a) or (b) |

**The confirmatory comparison is always read on the full block**, with an unscoreable comparator's
rows filled by its declared fallback. Any intersection-only comparison is secondary and labelled.
Condition **(g)** requires a point difference in the candidate's favour against **every** parity
comparator.

### 4.6 Reproduction-fidelity tolerances (Missed 3)

A weak re-implementation is a weak comparator, and nothing in v1 would have detected one. Fixed
before the stamp:

| Comparator | Published reference | Tolerance |
|---|---|---|
| `rf_published_adore`, `xgboost_published_adore` | ADORE `t-F2F` published RMSE 0.927 (RESEARCH_PLAN §15.2; this project already reproduced it) | **± 0.03 log units** on the same published or CV basis |
| `gpboost_published` | Zubrod et al. 2024 report no single figure reproducible on our rows | **no numeric tolerance is available**; instead the nine hyper-parameters are asserted verbatim against the manifest `chemtox_published_gpboost`, and the library difference (gpboost 1.8.0 vs the authors' 1.2.1) is disclosed |
| `mmoe` | Wu et al. 2026 report per-task figures on their own 6-task split, not on ours | **no numeric tolerance is available**; `mmoe` is instead required to beat `lightgbm_noemb` on pool validation, and if it does not it is **reported as a failed reproduction and cannot carry (a) or (b)** |
| `r5_trident` | its own released 10×10-CV per-SMILES errors | used as the fidelity check itself; no tolerance applies since nothing is re-fitted |

A comparator that misses its stated tolerance is reported as a failed reproduction and cannot carry
condition (a) or (b).

**On T1 and T6 the taxonomy is in-distribution**, so GPBoost's published species/genus/group
grouping is admissible here and `gpboost_published` is **not** the ADAPTED estimator it must be on
T2. Only the library difference is disclosed.

---

## 5. Configuration caps, with every configuration counted

### 5.1 T1 — 12 searched entries × 32 = **384 searched**

| # | Entry | Role |
|---|---|---|
| 1 | `ted` | candidate (GPU) |
| 2 | `timehead` | candidate |
| 3 | `rx_lightgbm` | candidate |
| 4 | `rx_catboost` | candidate |
| 5 | `rx_chemprop` | candidate (GPU) |
| 6 | `mmoe` | comparator + ensemble member (GPU) |
| 7 | `r1_monotone_gbm` | comparator (R1) |
| 8 | `r2_aft_gbm` | comparator (R2) |
| 9 | `r3_multitask_stack_gbm` | comparator (R3) |
| 10 | `xgboost_emb` | comparator — simple-baseline candidate |
| 11 | `rf_emb` | comparator — simple-baseline candidate |
| 12 | `lightgbm_noemb` | information control (identical to #3 without the embedding) |

### 5.2 T1 — fixed: **13 configurations across 12 entries**

`rf_published_adore` 1 · `xgboost_published_adore` 1 · `gpboost_published` **2** (fish, invertebrate
forms) · `readacross` 1 · `trivial_global_mean` 1 · `trivial_per_species_mean` 1 ·
`trivial_per_chemical_mean` 1 · `r5_trident_released_errors` 1 (no fit) · `ensemble_mean` 1 ·
`ensemble_decorrelated` 1 · **`ensemble_mean_baseline` 1** · **`ensemble_decorrelated_baseline` 1**.

### 5.3 The matched baseline-ensemble control (Missed 2)

V1 bound itself twice to ensembling parity and never enumerated the control, so condition (g) would
have read an ensemble candidate against unensembled baselines — the C-M4P-1 failure the same
paragraph cited. Fixed: **`ensemble_mean_baseline` and `ensemble_decorrelated_baseline`** are built
from the **strongest simple baselines**, with the same member count and the same 0.05 margin rule,
and are counted in §5.2. Precedent: the lead's M1 section approves "E2 with the matched
baseline-ensemble control E2-B" as a named entry with its own budget.

**No §15.1 label is available for an ensemble candidate on any block on which the matched baseline
ensemble was not built and reported.** T6 enumerates no ensemble entry, so on T6 the ensembling
clause has nothing to attach to and **no ensemble claim is available on T6 at all**.

### 5.4 T6 — 8 searched entries × 16 = **128 searched**, and the baseline ladder

Searched: `ted` · `timehead` · `rx_lightgbm_dur` (candidates) · `r1_monotone_gbm` · `r2_aft_gbm` ·
`r3_multitask_stack_gbm` · `xgboost_emb_dur` · `mmoe` (comparators). 16 rather than 32 because T6's
point is a shape claim, not a hyper-parameter search.

**T6 — fixed: 13 configurations across 12 entries** (T6-BASELINE-LADDER adds the first three):
**`trivial_per_species_mean` 1** · **`trivial_per_chemical_mean` 1** ·
**`trivial_per_species_x_band_mean` 1** · **`rf_emb_dur` 1** (added as a *fixed* entry, so the 128
searched cap is untouched) · `rf_published_adore` 1 · `xgboost_published_adore` 1 ·
`gpboost_published` 2 · `readacross` 1 · `haber_rule` 1 · `nearest_duration_carry_forward` 1 ·
`trivial_global_mean` 1 · `r5_trident_released_errors` 1.

**Condition (b) on T6 is read against whichever simple baseline leads pool validation, named in
writing before the look.**

### 5.5 The totals, reconciled (CONFIG-TOTALS, Point 2)

| | T1 | T6 | Total |
|---|---|---|---|
| Searched (cap, per-entry cap 32 / 16) | **384** | **128** | 512 |
| Fixed | 13 | 13 | 26 |
| Search + fixed | **397** | **141** | 538 |
| Ablations A1–A6, fixed fits of the single selected `ted` configuration | 6 | 6 | 12 |
| **Declared grand total** | | | **550** |

Entries: 24 on T1, 20 on T6, **44 in all**. The five figures (384, 128, 26, 12, 550) go on every
registry row of both tracks so they cannot drift, and the realised count **including failures,
discarded configurations and timed smokes** is reconciled against them in the write-up.

Binding additions: (i) the 12 ablations sit **outside** the searched caps and **may not grow** if
the pre-declared fallback cuts `ted` to 8 configurations; (ii) 397 and 141 are search-plus-fixed
totals and exclude the ablations, which is why they sum to 538 and not 550; (iii) **a second
embedding source, a re-screen of any family, or a screen-then-refit pattern is a new entry** whose
32 (T1) / 16 (T6) configurations count against the total; (iv) a failed fit still consumes its slot
and is reported as a failure. **No expansion after a validation number exists on either track.**

*The total rose from v1's 544 to 550 because §5.3 and §5.4 add six fixed entries the review
required; no searched cap moved.*

### 5.6 Open choices, closed now (OPEN-CHOICES)

* **Embedding source precedence, one source only:** **MolFormer-XL if its weight licence clears,
  else ChemBERTa-2, else the in-tree Chemeleon fallback.** The choice is recorded with its licence
  date in `docs/licenses.md` **before any fit**. Trying a second source is a new entry (§5.5 iii).
* **R1's library is fixed to LightGBM** (`monotone_constraints` on continuous `duration_h`).
* **The `mmoe`-vs-`gpboost_published` choice for condition (a) is resolved by pool validation**
  under §4.4, never by choice after the look.

### 5.7 Compute: per-entry fit counts and GPU ceilings (GPU-BUDGET)

GPUs **0–9** through `python -m sentinel.core.gpu --name <job> --min-free-mb <n> --`, never a GPU
another user's process occupies, **≤ 2 h per job with checkpoints**, one configuration × one fold per
job (the `--fold-merge` pattern proven on T2's chemprop queue). CPU ≤ 6 threads, one heavy CPU job at
a time.

| Entry | Track | Configs | Folds | Fits | GPU-h ceiling |
|---|---|---|---|---|---|
| `ted` | T1 | 32 | 4 | 128 | **30** |
| `ted` | T6 | 16 | 3 | 48 | **12** |
| `rx_chemprop` | T1 | 32 | 4 | 128 | **26** |
| `mmoe` | T1 | 32 | 4 | 128 | **12** |
| `mmoe` | T6 | 16 | 3 | 48 | **5** |
| `timehead` | T1 + T6 | 32 + 16 | 4 / 3 | 176 | **2** |
| embedding cache (one-time) | — | — | — | — | **2** |
| 3-seed refits of the frozen winners | both | — | — | — | **6** |
| **Total ceiling** | | | | | **95 GPU-h** |

Everything else is CPU. **If a per-entry ceiling is reached, the pre-declared fallback to 8
configurations per track for that entry applies automatically and is recorded.** VRAM ≤ 4 GB of
11 GB, measured with `torch.cuda.max_memory_allocated()` in the smoke, so a second workstream can
share the card. **No budget is renegotiated after any validation number exists on either track.**
Disk: `df` before every build and chunk, hard stop below **60 GB free on `/`**, M4's new artefacts
inside the **10 GB** cap (arch-v2 artefacts to date: 1.1 MB).

### 5.8 One pool, two fold structures (SAME-POOL-TWO-BUDGETS)

Restated as the review requires: *both tracks search inside the same 32,106-row pool under different
fold structures; shared entries are counted once per track because each track selects its own
winner.* The total of 550 is unchanged by this restatement.

---

## 6. Power: the corrected MDE, and what each block can and cannot detect

### 6.1 The corrected model (MDE-ROW-INDEPENDENT, L5)

`unseen_taxa.mde` draws the two synthetic predictors' errors **independently per row**, so its MDE
scales with rows rather than resampling units. The corrected helper is a **new module**,
`sentinel/models/chemtox/arch_v2_power.py` — `unseen_taxa.py` and `unseen_taxa_guard.py` are frozen
and untouched (L5) — and adds a within-unit error component parameterised by `icc`, the intra-class
correlation of the two models' errors inside a unit:

```
a_ui = sqrt(icc) * g_u + sqrt(1-icc) * z_ui            (model A)
b_ui = sqrt(icc) * g'_u + sqrt(1-icc) * z'_ui          (model B, correlated at rho)
SD(d) = s * sqrt(1-rho^2) * sqrt( sum_u n_u (1 + icc^2 (n_u - 1)) ) / N
MDE   = z_(1-alpha/2) * SD(d)
```

`icc` moves error *structure*, never error *magnitude*: each row's two errors stay marginally
N(0, s²) and correlated at `rho` whatever `icc` is, which is what makes the sweep an honest
sensitivity analysis rather than a rescaling. The design effect carries **`icc²`**, not `icc`,
because the statistic is built from squared errors. The formula collapses to the row-level
`z s √(1−ρ²)/√N` at `icc = 0` (what `unseen_taxa.mde` estimates) and to the unit-level
`z s √(1−ρ²)/√U` at `icc = 1`; `tests/test_chemtox_arch_v2_power.py` asserts both limits, checks the
derivation against Monte Carlo at every tabulated `icc`, and reproduces the verifier's own gate
values (0.047 / 0.027 / 0.053 on the v1 blocks) from counts alone.

**Two deliberate strictenings beyond the ruling's arithmetic**, both conservative and both flagged:
the exact **unequal-unit-size** sum is used rather than the mean unit size (one unit holds ~10 % of
the universe and 9.8–14.4 % of each block, which the equal-size approximation understates), and the
bootstrap check averages over 8 error realisations because a single realisation's answer at high
`icc` depends on which unit-level draws landed on the large units. The **pre-registered figure is
the analytic one**, because it is deterministic.

**`rmse_level` = 1.2** log units, T2's pre-registered value and this project's measured hold-out
range. **`rho` = 0.9** primary, `rho` = 0 reported as the pessimistic bound. **The primary `icc` is
0.5**; `icc` = 1 is the conservative cluster-level bound and `icc` = 0 is reported **only** to show
what v1 assumed.

### 6.2 The governing gate, and the L4 share sweep with a proposal

Computed from counts only (§6.6), by `scripts/eval/chemtox_arch_v2_power_report.py` reading the
label-free split. **The governing MDE per block is the stricter of the full block and its in-domain
subset** (§6.3).

| Block | Rows | Units | MDE icc 0 (v1's assumption) | **MDE icc 0.5 (primary gate)** | MDE icc 1.0 |
|---|---|---|---|---|---|
| `T1_test` | 11,363 | 546 | 0.0101 | **0.0959** | 0.1911 |
| `T6a_test` | 11,249 | 694 | 0.0134 | **0.1027** | 0.2047 |
| `T6b_test` | 4,106 | 289 | 0.0170 | **0.1033** | 0.2044 |
| `T1_test`, no shared reference | 3,745 | 143 | 0.0168 | 0.1255 | 0.2493 |
| `T6a_test`, no shared reference | 6,224 | 522 | 0.0130 | 0.1061 | 0.2111 |
| `T6b_test`, no shared reference | 2,946 | 227 | 0.0189 | 0.1039 | 0.2053 |

**L4 sweep — unit-level MDE at the governing reading, by share of units held out:**

| Share of units | T1_test rows / units | T1 gate icc 0.5 | T6a rows / units | T6a gate icc 0.5 | T6b rows / units | T6b gate icc 0.5 | pool rows |
|---|---|---|---|---|---|---|---|
| 15 % (304 u) | 5,692 / 275 | 0.1307 | 13,235 / 840 | 0.0886 | 2,120 / 143 | 0.1684 | 37,777 |
| 20 % (406 u) | 6,915 / 363 | 0.1116 | 12,757 / 795 | 0.0917 | 2,598 / 188 | 0.1421 | 36,554 |
| 25 % (507 u) | 9,164 / 448 | 0.1092 | 12,078 / 741 | 0.0962 | 3,277 / 242 | 0.1214 | 34,305 |
| **30 % (609 u)** | **11,363 / 546** | **0.0959** | **11,249 / 694** | **0.1027** | **4,106 / 289** | **0.1033** | **32,106** |

**Proposal: 30 % of units, with the reason fixed here before any result.** Three grounds, in order.
(i) T1 is the primary block and 30 % gives it its best power, monotonically: 0.1307 → 0.0959. (ii)
**30 % is the only share at which the pre-declared structural criterion — no single resampling unit
holds 10 % or more of a block's rows — is satisfied for `T1_test` (9.8 %)**; at 15–25 % the largest
unit holds 12.0–14.6 % of it. (iii) T6b, the weakest block, improves monotonically with the share
(0.1684 → 0.1033) and only at 30 % does it reach parity with the other two. The cost is accepted and
stated: the pool falls from 37,777 to 32,106 rows (−15 %), which weakens every model equally and
every comparator identically, and T6a's gate worsens from 0.0886 to 0.1027 because T6a gives units
up to T6b. T6a is secondary; T1 is primary. **The lead rules on the share at the stamp; if the
ruling differs, the artefacts are rebuilt as `_v3` and re-pinned before anything is fitted.**

### 6.3 The applicability domain is disclosed, not bypassed (MDE-DOMAIN-BYPASS)

V1 forced `in_adore_domain = True`, applying **no** domain restriction, and then compared its
figures with T2's — which were computed on the **organic-only** subset. The two were never
comparable and v1's comparison is withdrawn.

Fixed now: `in_adore_domain` (`chem_structure_class` ∈ {organic, organic_multicomponent}) is a column
of the split; **38,244 of 58,824 universe rows are in-domain**, so the restriction is material and
not cosmetic. Both readings are tabulated for every block and stratum in `t1t6_mde_v2.json`, and:

* **The claim is read on ALL rows of the block.** That is the honest hazard-screening question —
  a regulator asks about the substance in front of them, not about the organic subset — and
  restricting the block would make it a different, easier object. The in-domain subset is a
  **pre-registered secondary stratum**, reported always.
* **Condition (f) is gated by the stricter of the two MDEs.** Measured: for `T1_test` and `T6b_test`
  the in-domain subset is stricter (0.0959 vs 0.0891; 0.1033 vs 0.0938), for `T6a_test` the full
  block is (0.1027 vs 0.0600). The governing values are the bolded column of §6.2.

### 6.4 Publication structure (SHARED-REFERENCE-GATE)

Condition (f) is **split in two legs** (§12.1 (h)): the difference must exceed the governing MDE on
the full block **and** keep its sign on the `no shared reference` stratum **and** exceed that
stratum's own ρ = 0.9 MDE (0.1255 / 0.1061 / 0.1039). A claim failing the stratum leg is reported as
**qualified** and labelled *"not separable from publication structure"*. Any label that passes is
explicitly scoped as **including shared-publication rows**.

### 6.5 What each block can and cannot detect — and T6b's status (T6B-POWER-CLAIM, Point 4)

Stated plainly, as L4 requires.

* At the primary gate the blocks resolve differences of **≈ 0.10 log units**. They **cannot** resolve
  0.01–0.05, which is the range of every parity-matched differential this project has measured: the
  spent prospective headline was **0.029**, and the largest spread between the best tree and chemprop
  on T2 was **0.070**. They **can** resolve ≈ 0.10 and above — the size of T2's *information* channel
  effect (0.123), not of its architecture effects.
* **All three blocks are therefore pre-registered as DESCRIPTIVE by default.** A §15.1 label is
  available only if the realised difference exceeds the governing MDE of §6.2. If it does not, the
  result is reported as a **measured null for that block**, with the same prominence as a win, and
  no weaker claim is substituted.
* The v1 sentences *"Every block is better powered than T2's best (0.011)"* and *"better powered than
  any block M4 has ever scored"* are **struck**. The corrected statement: at the row-level `icc` = 0
  reading T1 and T6a would look better powered than T2's best and T6b would not, but **the row-level
  reading is not the gate**, and at the gate none of the three is well powered for a realistic
  effect.
* **T6b keeps its status as a full block with its own look** (§13 rule 8), because allowing a block
  to be demoted after its number is known is the larger forking path. It carries a §15.1 label only
  if its corrected unit-level MDE supports one, it occupies **its own named place in the Holm
  family** (§8.3), and a sub-MDE result is a measured null for that block — **never re-described
  after the fact as a stratum of T6a**.

### 6.6 Label-derived statistics, and the replicate floor (PROTECTED-LABEL-STATS, ST-14 refuted)

**No label-derived quantity of `T1_test`, `T6a_test` or `T6b_test` is computed, printed or stored
anywhere in this file or in any v2 artefact** (L6). The v2 counts and MDE JSONs carry counts and
keys only; the split file has no target column; the power helper takes unit-size vectors, and its
statistic is a paired difference between two synthetic predictors, so the observed target cancels
exactly and *cannot* enter — this is structural, not a promise, and
`test_signature_is_counts_only` asserts it. v1's per-block SD and replicate-floor columns are
**struck**, and v1's two reports are retained only as `*_v1_WITHDRAWN_label_stats.json`.

**The only floor quoted is the training pool's, 0.504** (pooled within-group SD over
(CAS, species, duration, endpoint) groups with ≥ 2 records), and **only as absolute-RMSE context**.
Per the refuted verdict ST-14: *for T1 and T6 the corrected MDE is the only difference gate and the
replicate floor is reported solely as absolute-RMSE context, as `chemtox_unseen_taxa_plan.md` §6
already rules; the T2 table's "Floor for a claim" column and the `unseen_taxa.mde` docstring are
superseded on this point.* That sentence is quoted in any write-up comparing T1/T6 with T2. **No
practically-relevant difference threshold is declared**, because none can be justified independently
of the data.

### 6.7 Per-band power

Band-level MDEs are in `t1t6_mde_v2.json`. The claim rule of §3.3 is mechanical: **fewer than 50
resampling units ⇒ descriptive only**, which excludes T6a `gt2160` (18 units) and T6b
`gt720_2160` (39) and `gt2160` (7).

---

## 7. Selection and ablations

### 7.1 Selection folds (validation only; no test block is read)

**T1 — 4 folds, leave-one-unit-group-out.** The pool's 1,293 hold-out units are partitioned into 4
groups of near-equal row count by greedy largest-first assignment under seed `20261110`. Each fold
trains on three groups and validates on the fourth. This mirrors how `T1_test` is built — whole
units held out, in-window durations — which is the A5 lesson from T2: a selection fold that does not
mirror the test block selects on an artefact.

**T6 — 3 duration-band hold-out folds inside the pool**, mirroring T6a: S1 trains 24–72 h and
validates > 72–96 h; S2 trains 48–96 h and validates 24–< 36 h; S3 trains 24–48 h and validates
> 48–96 h (a 2× step). **For T6b**, one combined fold holds out a unit group **and** the top band.
§3.5's disclosure stands.

**Score and winner.** Selection score = **mean micro-RMSE over the folds**, with the worse half, the
worst fold, the per-fold scores and the top-5 spread retained and reported. One winner per entry
(lowest mean; ties broken by worst fold) goes into the freeze, then is refit with **seeds 0, 1, 2**.
Fold scores are validation-only and are never compared with the ADORE-CV or T2 numbers: different
pools, different blocking, so absolute RMSE is not commensurable across them.

### 7.2 The ablation set (pre-declared; a win without it cannot be attributed)

| Ablation | What it isolates |
|---|---|
| **A1** | D-MPNN trunk with free-form duration conditioning (TRIDENT-shaped), **no composition layer** — is the mechanism doing anything? |
| **A2** | Composition layer with `B = 0` (pure anchored law, no learned residual) — the physics-only floor |
| **A3** | Duration and endpoint fed to the trunk as well — **leak test**: was the restriction load-bearing? |
| **A4** | Acute-only training (no chronic, no censored) — the multitask contribution |
| **A5** | **+ BCF head** — admissible: the join/QC report has landed (§9) |
| **A6** | **+ Q10 temperature term** with a *free* prior; no literature constant is assumed |

**Attribution rule, binding:** if **A1** scores within the block's governing MDE of the full model,
the win belongs to the trunk and the data, **the mechanistic framing is dropped from the claim and
the paper**, and that is reported in the same table as the headline. A `ted` claim requires A1 and
A2 present; if A3 improves on the full model the restriction was not load-bearing and that is
reported. The A1 test is read at the **unit-level gate**, not at a row-level value.

### 7.3 Ensembles, and the membership margin (Missed 5)

`ensemble_decorrelated` selects members greedily to maximise `mean pairwise (1 − corr(residuals))`
subject to each member being within **0.05 RMSE** of the best single model, member set fixed before
the look. `ensemble_mean` is the unweighted mean of the frozen candidate winners. §5.3's matched
baseline ensembles are built identically from the strongest simple baselines with the same member
count and margin.

**Disclosed:** 0.05 is about **five times** T1's row-level MDE (0.0097) and about **half** its
governing unit-level gate (0.0959), so ensemble members may differ from the best single model by more
than a claimable difference at the row-level reading and by less than one at the gate. The margin
**stays fixed at 0.05** and is reported beside every ensemble result.

---

## 8. The guard, the look order, and the reservation

### 8.1 A new guard module, with its own pins and experiment key (GUARD-GAP, L10)

`sentinel/models/chemtox/arch_v2_guard.py` is new code with its **own** pins file
(`docs/preregistration/chemtox_arch_v2_pins.json`), its **own** experiment key
(`chemtox_arch_v2_test`), its **own** block names and its **own** protected-labels artefact.
`unseen_taxa_guard.py` and `unseen_taxa.py` are neither edited nor imported, and a test asserts that.

**Reader discipline:** `load_training_pool()` is how every T1/T6 fit obtains rows — it reads the
pinned split and returns `block == "train_pool"` rows only, so a selection run **physically cannot**
obtain a sealed row, and it never opens `ecotox_aquatic.parquet` itself. `load_test_block()` is the
**only** reader of a sealed block and refuses without an authorisation marker. The sealed targets
live in a separate 0444 artefact built **at authorisation time**, so **before the first look the
sealed labels do not exist anywhere under the arch-v2 paths** — `test_protected_labels_do_not_exist_before_any_freeze`
asserts it.

### 8.2 The four scorer-side gaps, closed

1. **A lead-written `FREEZE` file is the only unlock.** `authorise()` refuses without
   `$SENTINEL_RUNS/chemtox/arch_v2/<block>/FREEZE`, whose sha256 must match the value the lead
   records at the stamp. No file this workstream can write substitutes for it.
2. **`BaseException` aborts** the pass before any claim, prediction, summary or registry row is
   written — `scored_pass()`, with a test that a `KeyboardInterrupt` leaves `ABORTED.json` and no
   `SCORED.json`, no predictions and no claim.
3. **A full import-closure code hash** over every repo module reachable from `sys.modules`, recorded
   as `code_sha` on the claim. This replaces the package hash that goes stale as other workstreams
   edit `sentinel/` (still an open 🟡 on the prospective track).
4. **All artefacts verified before the claim** (`verify_artifacts()` re-hashes every file against
   `SHA256SUMS`) and **access closed at sealing** (`seal_dir()`, 0444 + `SHA256SUMS`).

Plus: registry refusal if any row already exists for the block, an `O_EXCL` claim marker carrying
pid + host + `code_sha` + the FREEZE sha, and write-once `save_predictions()`.
**The guard is reviewed before any look** (L10); this amendment's stamp clears no fit, no embedding
run and no look.

### 8.3 Look order and the Holm family (LOOK-ORDER-AND-FAMILY)

* **One claim model per block**, chosen on **pool validation only** and named in writing before any
  look.
* **The Holm family is (claim model + the other candidate entries) × (the (a) and (b) comparator
  legs)**, with **m fixed in writing** at the freeze. On T1 the candidate entries are `ted`,
  `timehead`, `rx_lightgbm`, `rx_catboost`, `rx_chemprop`, `ensemble_mean` and
  `ensemble_decorrelated` — 7 — so **m = 7 × 2 = 14**. On T6 they are `ted`, `timehead` and
  `rx_lightgbm_dur` — 3, with no ensemble entry (§5.3) — so **m = 3 × 2 = 6**, applied within each
  of `T6a_test` and `T6b_test`, each of which has **its own named place** in the family (§6.5).
  Comparator-vs-comparator contrasts are diagnostics and are not in the family.
* **All three blocks' configurations, seeds, ensemble members and comparator choices are frozen, and
  all three FREEZE files are issued by the lead, before the first look.** **No refit and no
  re-selection may occur between the looks.**
* `T1_test` and `T6b_test` share held-out chemical units by construction, so **any statement spanning
  blocks additionally needs Holm across the block × comparator pairs**, and the count of shared
  chemicals is reported with it.
* One look per block, ever; a rescore is refused after scoring. Write-ups go to
  `chemtox_t1_test_results.md` and `chemtox_t6_test_results.md`, never into a frozen file.

### 8.4 Registry

Every run writes a row through `sentinel.core.registry`, carrying the five configuration figures of
§5.5, the governing MDE of the block, the seed, the AUX-LEAK removal counters, and the split sha256.

### 8.5 Reservation (RESERVATION-NOT-REGISTERED, L11)

**At the stamp the lead adds `T1_test`, `T6a_test` and `T6b_test` to `sota_campaign.md`'s "Reserved
unseen blocks added 2026-11-10" section**, naming:

* file **`chemtox_t1t6_split_v2.parquet`**, sha256
  **`ab71aace46260980a6f011a561c2280fe0025226978508daa57f96493c5323a3`**;
* rows / hold-out units **T1_test 11,363 / 546 · T6a_test 11,249 / 694 · T6b_test 4,106 / 289**;

with the standing instruction that **no workstream trains or tunes on those `result_id`s** and that
any overlap is reported to the lead as counts. (v1's `chemtox_t1t6_split_v1.parquet` sha `005f7957…`
is **superseded** and must not be registered as the reservation: its blocks contain spent rows.)

---

## 9. H4 (BCF): the report landed, so A5 is in — with filtered counts

The lead's condition was that H4 enters only as ablation A5 and only if its join/QC report lands
first. It has (`h4_bcf_join_qc_v2.json`, sha256 `7dbdd501…`), so **A5 is admitted** — as an ablation
of `ted` only, never as a track, a claim target, or an input to a comparator-asymmetric arm.

**The defect the report found, recorded because it would not have raised an error:** the BCF value
lives in **`bcf1_mean`/`bcf1_unit`**, not `conc1_mean`. On these 27,725 rows `conc1_*` is the
*exposure water concentration* (`ug/L` 16,372, `mg/L` 3,110, …), so reading it — the obvious thing to
do — would have silently trained a bioconcentration head on the exposure concentration.

**Unit and basis reconciliation.** Admissible units are only the exactly dimensionless-equivalent
ones: `L/kg` (20,577), `ml/g` (732) and blank → 21,494 rows. Everything else is excluded and
counted, including `RA` (4,498 — a relative-activity code, not a BCF), `L/g`, `ml/mg`, `kg/L`
(inverted), `NR`/`NA`. The convertible ones would add ~0.9k rows but each needs its own factor
decision, so they are excluded rather than guessed. **BCFD/BAFD is dry weight**, and dry- and
wet-weight factors differ by roughly 4–5×, so basis is a categorical feature and is never pooled
silently. Trim: `log10 BCF ∈ [−2, 7]`.

| | Rows | Chemicals | Species |
|---|---|---|---|
| Unfiltered usable (**v1's figure; may NOT be quoted as a training count**) | 17,418 | 672 | 479 |
| **Admissible for A5 on T1** (AUX-LEAK filters applied) | **10,050** | 373 | 409 |
| **Admissible for A5 on T6** (plus the 24–96 h cap) | **2,984** | 230 | 207 |

---

## 10. Targets, auxiliary rows, licences

### 10.1 Targets, with universe **and** admissible counts (Missed 4)

Only the **admissible** counts may be quoted in any write-up.

| ID | Target | Universe rows | Admissible T1 | Admissible T6 | Role |
|---|---|---|---|---|---|
| **H1** | log10 LC50/EC50 (mol/L) | 122,946 | 32,106 pool (+ sealed blocks) | same pool | the primary target |
| **H2** | the same target, interval-censored | 11,719 | **5,388** (785 chem) | **4,455** (726 chem, max 96 h) | auxiliary via R2 and `ted`'s Tobit term |
| **H3** | log10 chronic effect concentration | 225,081 | **121,678** (1,467 chem) | **45,661** (1,285 chem, max 96 h) | auxiliary task (R3, `ted`, `mmoe`) |
| **H4** | log10 BCF/BCFD/BAF | 17,418 usable | **10,050** | **2,984** | ablation A5 only (§9) |
| **H5** | log10 acute-to-chronic ratio, with an interval | 10,334 pairs in the frame | computed from **`train_pool` rows only** | same | a **reported output, never a §15.1 label** (no pre-registered comparator exists) |
| **H6** | per-row predictive sigma / calibrated interval | — | — | — | reported (coverage + NLL); table stakes against L1/L3 |
| **H7** | within-one-bin severity class | — | — | — | a reported metric (L5's finding) |

The "~21.5k censored acute rows" figure in circulation is wrong by ~1.8×: the usable universe is
11,719 and the **admissible** count is 4,455–5,388.

### 10.2 The AUX-LEAK filters, enforced as counted assertions (AUX-LEAK, L7)

`arch_v2.auxiliary_rows()` removes, in order and each with its own counter:

1. T2's six reserved classes;
2. spent prospective `result_id`s;
3. **spent ADORE `result_id`s (L1)**;
4. any row whose `test_cas` sits in a **held-out hold-out unit** — which, because the unit is the
   connected component over (cluster, parent-key family), also removes every row whose cluster or
   parent-key family touches the hold-out (L2);
5. any row whose chemistry is **absent from the T1/T6 universe**, since whether it touches the
   hold-out is then *unknown* and the conservative reading of AUX-LEAK excludes it;
6. any row sharing a **(CAS, species, duration) cell** with a sealed row;
7. for **T6 only**, any row outside 24–96 h.

The loader then **asserts** that no held-out chemistry, no sealed cell and (on T6) no out-of-window
row survived, and raises if the CAS join degenerates — a dtype round-trip had silently turned the
unit map into a no-op during development, which is the same class of silent-no-op defect T2's delta
review found twice, so the coverage check is now permanent. Measured removals for H3/T6: 35,493
reserved classes, 43,759 held-out chemistry, 13,586 unmapped chemistry, 10,565 sealed cells and
**76,017 out-of-window rows**; the maximum labelled duration surviving is **96.0 h**.

This applies to **every** auxiliary, multitask, pretraining and ablation row set — H2, H3, H4, H5 and
any future auxiliary target — for `ted`, R2, R3, `mmoe`, A4 and A5. **H5's ACR target is computed
from `train_pool` rows only.**

### 10.3 Licences (verified 2026-11-10; entered in `docs/licenses.md`)

* **TRIDENT Zenodo record 10260621 — CLEARED.** `cc-by-4.0`, access open, v2 created 2023-12-05,
  8 creators. `03_Model_error_per_SMILES_from_10x10_CV.xlsx` (4,825,857 B) and
  `02_Preprocessed_complete_data.xlsx` (30,897,579 B) — 35.7 MB, against M4's 10 GB cap; the 2.8 GB
  weights and 3.2 GB source archives are **not** downloaded. Admitted as a **licensed data read
  only**, coupled to §4.4: it cannot carry condition (a).
* **EnviroTox — NOT CLEARED.** No licence or terms page exists; the About page reads "Copyright ©
  2019 HESI. All rights reserved." and disclaims all warranties; reuse permission is not stated and
  no download file could be located. **T5 is blocked and recorded as blocked, and stays listed as
  blocked in `RESULTS.md`.** No EnviroTox file is retrieved, cached or overlap-audited until written
  permission from the rights holder is in `docs/licenses.md`; if permission arrives, T5 is a **new
  amendment** pre-registered before any EnviroTox row is read, so the overlap audit against the spent
  blocks happens before the data lands. **No write-up claims cross-source or curation-pipeline
  generalisation.**
* **REACH / ECHA** — untouched (standing lead ruling).
* **The embedding weights are gated on their own licence**, with the precedence of §5.6.

### 10.4 Reserved blocks of other workstreams

M2's Test-S, M5-F's France rows, M1's StreamPULSE and M7's P2 are not used. Any overlap discovered
between a T1/T6 row and another workstream's reserved block is reported to the lead as counts.

### 10.5 Pretrained caches and checkpoints (PRETRAINED-CACHE)

**No checkpoint, embedding cache or auxiliary row set built before the T1/T6 split may initialise or
feed any T1/T6 fit.** Pretraining rows are re-derived from `block == "train_pool"` **after** the
AUX-LEAK exclusions; the cache key records the **split file's sha256**
(`ab71aace…`) so a cache built against another split cannot be loaded.

**Disclosed limitation:** the frozen pretrained embeddings (MolFormer-XL or its fallback) were
trained by third parties on public chemical corpora that include the held-out structures. They saw
**structures, no labels and none of our rows**. That is disclosed beside every result, and
`lightgbm_noemb` (§5.1 entry 12) exists so the embedding's contribution is measurable rather than
assumed.

---

## 11. Novelty check (dated 2026-11-10) and the citation correction

### 11.1 Prior art

Metadata for L1–L11 was retrieved 2026-11-10 (Crossref / web search; architecture plan §2).
**Numbers inside L2, L3, L4 and L6 were NOT verified — only their metadata — and no number from them
is relied on here.**

| Work | What it already does | What it pre-empts | What is left |
|---|---|---|---|
| **L1 TRIDENT** (Gustavsson et al. 2024, *Sci. Adv.* 10(10):eadk6669) | SMILES transformer + DNN head **conditioned on log duration** and endpoint one-hots, trained jointly on EC50 and EC10 | duration conditioning, joint multi-endpoint training, released weights and errors | it **conditions** on duration; it does not **compose** duration through a differentiable law, and it does not evaluate duration as a hold-out |
| **L2 Liang et al.** (*ES&T* 2026-11-02, DOI 10.1021/acs.est.6c12447) | links fragmented ECOTOX records into model-ready datasets; multitask with endpoint-specific heads over shared context | **the "more ECOTOX data + multitask" framing of our own levers** | not a duration-extrapolation or cluster-hold-out evaluation. Its pooled R² ≈ 0.75 and its "multitask gains shrink with structural separation" finding are **[UNVERIFIED]** and unused |
| **L3 Wu et al. MMoE** (*JCIM* 66(20):13302) | multi-gate MoE over 6 endpoints, **EC50 ≥ EC10 ordering**, heteroscedastic uncertainty | **the obvious next architecture.** Hence `mmoe` is a comparator and ensemble member here, never a novelty claim | our ordering is derived from one latent plus a non-negative ACR reported with an interval, not imposed in output space |
| **L4 MarineTox** (Zhu et al. 2026, *ES&T* 60(21):14813) | multitask transfer from data-rich freshwater to 31 low-resource saltwater tasks | multitask transfer as the fix for data-poor taxa | a taxon-transfer claim — T2's business, not T1/T6's |
| **L5 3DMol-Tox** (Yuan et al., bioRxiv 2025.11.24.690199) | leakage-controlled benchmark; a 3D deep model **only matches GPBoost** on regression | that a fancier encoder alone will win | its own recommendation, structure-aware splitting, is what T1 implements; it does not evaluate duration |
| **L10 GUTS** (Jager et al. 2011) | the standard TKTD survival framework from which time-dependent LC50 follows | **the mechanism: NOT novel as mechanism** | making it differentiable, training it jointly across heterogeneous endpoint families, and evaluating it as a duration hold-out |

**The honest claim this amendment permits, stated before any fit:** *"mechanism-composed time and
endpoint structure, evaluated on a pre-registered duration-extrapolation block and a pre-registered
novel-chemistry block, against published duration-aware and multitask comparators at input parity."*
Not "a new architecture beats SOTA". Any label is scoped to its block, never generalised, and RMSE is
never averaged across blocks (rule 3). Given §6.5, the most likely honest output of both tracks is a
**descriptive comparison plus a measured null**.

### 11.2 The GPBoost citation, and the struck freeze claim (GPBOOST-FREEZE-CLAIM)

**V1's claim that `gpboost_model.py` is "covered by an existing freeze" is STRUCK.** It was asserted
without a named freeze or hash list. Per L9, `sentinel/models/chemtox/gpboost_model.py` is **not
edited** — it may sit in the import closure of a sealed look — and the wrong article title on its
line 11 is recorded here as a defect instead:

> **Correct citation (verified 2026-11-10, Crossref):** Zubrod JP, Galic N, Vaugeois M, Dreier DA.
> "Bio-QSARs 2.0: Unlocking a new level of predictive power for machine learning-based ecotoxicity
> predictions by exploiting chemical and biological information." *Environment International*
> **186**:108607, 2024. DOI 10.1016/j.envint.2024.108607.

**This corrected title is used in every write-up, model card and paper**, so the wrong one cannot
propagate. Sealing or correcting the file itself is the lead's change-control decision, not this
amendment's.

---

## 12. Claim conditions

### 12.1 A §15.1 label on `T1_test`, `T6a_test` or `T6b_test` requires ALL of

| | Condition |
|---|---|
| (a) | paired CI **excluding 0** against the **strongest published method actually re-run here at full parity on the block's own rows** (`mmoe` or `gpboost_published`, by pool validation, named before the look). TRIDENT cannot satisfy this (§4.4) |
| (b) | the same against the **strongest simple baseline**, by pool validation, named before the look |
| (c) | **Holm correction** within the block's family, m fixed in writing (§8.3) |
| (d) | **seeds 0, 1 and 2 for every stochastic entry, comparators and ensemble members included**; `all_seeds_better` must be **TRUE**; the reported point estimate is the **seed mean**; the paired bootstrap is run **per seed** with the per-seed CIs reported beside the seed-mean CI |
| (e) | **one look**, unlocked only by the lead-written `FREEZE` file (§8.2) |
| (f) | the difference **exceeds the block's governing MDE** of §6.2 (the stricter of full-block and in-domain, at icc 0.5, ρ 0.9) |
| (g) | **input parity**: a point difference in the candidate's favour against **every** parity comparator (§4.1, §4.5) |
| (h) | the difference **keeps its sign on the `no shared reference` stratum and exceeds that stratum's own ρ = 0.9 MDE**; otherwise the claim is **qualified** and labelled "not separable from publication structure" (§6.4) |
| (i) | for `ted`: **A1 and A2 present**, and A1 **not** within the governing MDE of the full model (§7.2) |
| (j) | for an ensemble candidate: the **matched baseline ensemble** was built and reported on that block (§5.3) |

### 12.2 Metrics and intervals (PRIMARY-METRIC-AND-CI)

* **Primary metric: micro-RMSE in log10(mol/L)** on the block's rows, as in the frozen T2 protocol §6.
* **Reported and never claimed:** macro-RMSE by chemical and by species, MAE, R², bias, and percent
  within 1 log unit.
* **Every CI is a two-sided 95 % percentile cluster bootstrap with B = 2,000, paired on identical
  rows**, resampling **hold-out units** (§2.3) — and, for the co-primary reading, `reference_number`.

---

## 13. Kill and fallback rules, pre-declared

1. **Smoke over 30 min/job, or a per-entry GPU ceiling reached → that entry drops to 8
   configurations** for that track, automatically, and it is recorded. The 12 ablations do not grow.
2. **A selection winner that is a comparator is an honest negative**, reported with the same
   prominence as a win, exactly as T2 is being reported.
3. **R5 fails → recorded, not worked around.** If the TRIDENT per-SMILES file does not contain usable
   errors, R5 is dropped and the inability to compare with TRIDENT is disclosed beside every claim.
   Condition (a) never depended on it (§4.4).
4. **The embedding licence fails → the next source in §5.6's precedence**, reported as a limitation.
5. **A fit failure is reported as a failure**, with its configuration counted (the T2 GPBoost-NaN
   precedent).
6. **Disk below 60 GB free on `/` → stop and report**, checked before every build, chunk and shard.
7. **A `BaseException` aborts the pass** with no claim and no prediction written (§8.2).
8. **A sub-MDE result on any block is a measured null for that block.** `T6b_test` is **not** demoted
   to "a stratum of T6a" after the fact; its status is fixed here as a full block with its own look.
9. **Nothing here may lower a bar.** If a result is within the MDE, the answer is "not resolvable on
   this block", never a weaker claim.
10. **A comparator missing its §4.6 reproduction tolerance** cannot carry condition (a) or (b).

---

## 14. What may happen before the stamp, and what may not

**Permitted now** (the lead's rule 2 for this round): code and unit tests; data builds whose outputs
carry counts and keys only; the P1 reference models R1–R3 on `train_pool` rows obtained through
`load_training_pool()`; smoke tests on synthetic or train-only data; licence verification.

**Forbidden until the stamp:** any fit on T1 or T6 that produces a validation number; any embedding
run that touches the split; any read of `T1_test`, `T6a_test` or `T6b_test` beyond the counts in §3.

**Forbidden until the relevant block's `FREEZE` file exists** (Missed 6): **any read of a
third-party per-chemical error or prediction table that is restricted to, or ranked over, the
chemicals of `T1_test`, `T6a_test` or `T6b_test`.** Downloading the cleared TRIDENT files and
building the per-SMILES intersection **over pool chemicals only** is permitted after the stamp.

**State at the time of writing:** no T1/T6 fit has run; no T1/T6 registry row exists; no `FREEZE`
file, authorisation marker, protected-labels artefact or `SCORED.json` exists for any new block; and
T2's three frozen blocks remain unspent (`registry_rows()` 0/0/0) and untouched by anything here.

---

## 15. Deviations, strictenings and open points for the lead

Listed rather than buried, per the lead's instruction not to deviate silently.

### 15.1 The v1 pre-stamp label read, disclosed (PROTECTED-LABEL-STATS)

V1 computed and printed, for row sets that largely overlap the v2 sealed blocks: the per-block
**SD of the target** and the per-block **replicate floor**, and MDEs derived from the target column
(though the MDE statistic is provably target-invariant and was verified so). **This was a pre-stamp
read of label-derived statistics of blocks intended to be sealed, and it should not have happened.**

**Which design choices were fixed BEFORE those statistics existed**, and so cannot have been
influenced by them: the Butina cutoff **0.4** and its "largest cluster < 10 % of rows" criterion; the
decision to hold out **whole clusters**; the **15 %** block size of the architecture plan; the
**24–96 h** training window and the use of duration bands; the existence and definition of **T6a and
T6b**; and the four-way row-disjoint partition. Every one of those predates the v1 counts run.

**Which choices were made after, and are therefore v2 choices made under review supervision:** the
**30 % unit share** (proposed in §6.2 on criteria stated there, from label-free counts), the
**0.25 h** duration floor, the **band edges**, the **hold-out unit** definition (L2, the lead's
ruling), and the **spent-ADORE subtraction** (L1, the lead's ruling).

No further label-derived quantity of a v2 sealed block is computed, printed or stored before its
`FREEZE` file exists.

### 15.2 Strictenings beyond the ruling, all conservative

1. **The exact unequal-unit-size MDE formula** instead of the ruling's equal-size arithmetic. The
   ruling's gate values (0.047 / 0.027 / 0.053 on the v1 blocks) are reproduced by a test, but the
   v2 figures use the exact sum, which is **larger** wherever units are unequal — and they are very
   unequal here.
2. **The bootstrap check averages over 8 error realisations**; the pre-registered figure is the
   deterministic analytic one.
3. **Auxiliary chemistry absent from the universe is excluded**, not merely chemistry known to touch
   the hold-out, because its relation to the hold-out is unknown.
4. **Trivial-baseline scorings spend a block** — the conservative branch of the verifier's open
   question, costing 17,108 more subtracted ids than the narrow reading.

### 15.3 The consequence the lead should weigh

**The two blocking fixes together have made these tracks descriptive.** Subtracting the spent ADORE
rows cut the universe from 85,746 to 58,824, and the corrected power model raised the gate from
~0.010 to ~0.10. At ~0.10 the blocks cannot resolve any differential this project has ever measured
between parity-matched models. Nothing in the data or the method is wrong — the honest position is
simply that **T1 and T6 as constituted are well-powered descriptive comparisons and poorly-powered
claim tracks**, and §6.5 pre-registers them that way.

The driver is **unit-size concentration**, not the subtraction: one hold-out unit holds 9.8–14.4 % of
each block's rows, and at icc = 1 no block's MDE can fall below roughly `1.025 × (largest unit
share)` however many units are held out. If the lead wants a claim-capable track, the levers are
structural, and all of them need a ruling before anything is fitted:

* (i) accept **icc = 0.5 is too conservative** and gate at a lower icc, with an argument for the
  value that is independent of our data (I have none to offer);
* (ii) change the **resampling unit** — but L2 fixed it, and a finer unit reintroduces the
  parent-family leak;
* (iii) a **tighter Butina cutoff** would break up the dominant unit, but choosing it now, after
  seeing its effect on our own power, is the forking path this review exists to prevent;
* (iv) accept the descriptive framing, report T1/T6 honestly, and treat the §15.1 label as out of
  reach on acute aquatic chemistry at this data scale.

My recommendation is **(iv)**, with the tracks run and reported as pre-registered: the references
R1–R3 are cheap, the comparison is informative, and a well-measured null on a well-powered
descriptive block is a publishable result that costs no integrity.

### 15.4 Open points needing a lead ruling

1. **The hold-out share** (§6.2): I propose 30 %. A different ruling means `_v3` artefacts.
2. **Unscored ADORE partitions** (§3.1, L1's second question): **105** challenge × split test
   partitions exist, **8 are scored** and **97 are not**, holding 241,031 test `result_id`s. **No
   unscored partition is reserved by any M4 protocol** — the four that `chemtox_test_plan.md`
   reserves are all scored, and no unscored partition's challenge and split are both named in
   `chemtox_test_plan.md`, `chemtox_prospective_plan.md` or `chemtox_unseen_taxa_plan.md`. They are
   mostly **alternative partitionings of the same rows** (for example `a-CA2F-same/split_random` has
   the same 22,538 test ids as the scored `a-CA2F-same/split_occurrence`). **Cost if the lead rules
   they also leave the universe: 4,632 further rows (7.87 %), leaving 54,192.** Per L1 I have **not**
   subtracted them.
3. **Whether `gpboost_model.py`'s wrong title is corrected and the file sealed**, or left as
   recorded here (§11.2, L9).
4. **The descriptive-vs-claim framing** of §15.3.

---

## 16. Verification record (VERIFICATION-RECORD)

| Claim in this file | How it is verified |
|---|---|
| 8 spent partitions / 28,265 ids / 26,808 rows in the old universe | derived by code and asserted in `tests/test_chemtox_arch_v2.py`; reproduces the verifier's figures exactly |
| no spent row anywhere in the split | `test_no_spent_row_is_anywhere_in_the_split` on the pinned artefact |
| no cluster or parent family spans the hold-out | `holdout_units()` + `build_split()` assertions, `test_no_parent_family_or_cluster_spans_the_holdout` |
| window band ≡ `DURATION_WINDOW` | `build_split()` raises on disagreement; `test_band_edges_and_window_agree_on_every_row` |
| the split carries no target | `test_split_file_carries_no_target`; the power script refuses a split with a label-like column |
| MDE limits and the ruling's gate values | `tests/test_chemtox_arch_v2_power.py` (18 tests), incl. Monte-Carlo checks at icc 0, 0.5, 1 |
| MDE is label-free | `test_signature_is_counts_only`, `test_label_free_flag_and_shape_invariance` |
| T6 auxiliary rows carry no label beyond 96 h | `test_t6_auxiliary_rows_carry_no_long_duration_label` |
| the guard refuses without a lead-written FREEZE | 4 guard tests; `BaseException` abort tested |
| sealed labels do not exist pre-FREEZE | `test_protected_labels_do_not_exist_before_any_freeze` |
| T2 untouched | `unseen_taxa_guard.verify_freeze()` passes; `registry_rows()` 0/0/0 |

**Test status at the time of writing: 18/18 in `test_chemtox_arch_v2_power.py`, 42/42 in
`test_chemtox_arch_v2.py`.** Every number in §§2–3, §6, §9 and §10.1 comes from a script in the repo
whose output is a pinned JSON, not from a hand calculation.
