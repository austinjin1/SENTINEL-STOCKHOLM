# M4 ChemTox arch-v2 — write-up template for a T1 / T6 look

**A template, not a result.** Written 2026-11-10, before any pool fit, so the wording of every
outcome is fixed in advance (ruling **F8**). Fill it in and save as
`chemtox_t1_test_results.md` or `chemtox_t6_test_results.md`; **never** append a result to a stamped
file (C-7).

Governing text in precedence order: `chemtox_arch_v2_lead_rulings.md` (sha256 `553be413…`),
`chemtox_arch_v2_addendum_01.md`, `chemtox_arch_v2_amendment.md` (sha256 `fb52cd4b…`). **Every
citation of the amendment cites the rulings file with it** — the amendment alone would mislead a
reader, because the rulings strike sentences it still contains.

---

## 0. The sentences that must appear, verbatim

Copy these; do not paraphrase.

> **No SOTA label is available on this block.** Under ruling F1, `T1_test`, `T6a_test` and
> `T6b_test` are **descriptive only**: their governing minimum detectable effects (0.0959 / 0.1027 /
> 0.1033 at icc 0.5, and 0.1911 / 0.2047 / 0.2044 at icc 1.0) are larger than any difference this
> project has measured between two parity-matched models. A powered confirmatory claim requires a
> new, separately stamped track under lead point (f).

> **MDE disclosure (F7).** The governing figure is the **analytic** value; the bootstrap check is
> reported beside it and is **not** a gate. The percentile bootstrap's coverage on heavy-tailed unit
> sizes has not been verified.

> **The first-text deviation (C-8).** Label-derived statistics of row sets overlapping these blocks
> were computed before the stamp, in the withdrawn first text. The design choices fixed before that
> read are listed in amendment §15.1; the withdrawn reports are retained as the record and were not
> opened for any decision of this round.

> **The 10 % criterion (B2).** The only pre-declared 10 % criterion is the largest **cluster**'s
> share of the universe's rows (7.76 %, met). **No per-block, per-unit 10 % criterion was ever
> pre-declared**, and none is claimed as met.

---

## 1. What was looked at

| | |
|---|---|
| Block | `<T1_test / T6a_test / T6b_test>` — descriptive only |
| Rows / hold-out units | `<n>` / `<u>` |
| Split file + sha256 | `chemtox_t1t6_split_v2.parquet` `ab71aace…23a3` |
| Pins file + sha256 | `<written by the lead>` |
| FREEZE sha256, and the lead's two records | `<file>` / `<registry row id>` |
| `AUTHORISED.json` pid + host + `code_sha` | `<…>` |
| Date of the single look | `<…>` |
| Claim model, named before the look | `<…>` |
| (a) comparator, by pool validation (F2) | `<mmoe / gpboost_published>`, pool score `<…>` |
| (b) baseline, by pool validation from the closed F3 list | `<…>`, pool score `<…>` |
| Holm family size *m*, fixed in writing | `<14 on T1 / 6 on T6>` |

## 2. The headline

**Outcome (F8 wording — one of exactly these five):**

| Situation | Wording to use |
|---|---|
| All of (a)–(j) met | *exceeds the pre-registered MDE on an under-powered block; no SOTA label* |
| Ahead, any condition unmet | *not resolvable on this block* |
| Ahead on the block but (h) fails | *ahead on the full block, not separable from publication structure* |
| Behind on the point estimate | *no gain* |
| Behind with a CI excluding 0 | *worse than `<comparator>`* |

> Outcome: **`<wording>`**

If **no candidate is ahead of the (a) comparator on any block**: *RESEARCH_PLAN §15.5 applies and
the next track must be harder or larger.*

**A "no gain" is reported with the same prominence as a win** — same section, same table, same
summary line. For T6 the headline is the **full block** as defined in amendment §3, never a subset
of bands chosen after the look (F9).

## 3. The numbers

| Leg | Comparator | Seed-mean paired Δ | CI (`holdout_unit`) | CI (`reference_number`) | Excludes 0 under both? | Holm p (adj.) |
|---|---|---|---|---|---|---|
| (a) | | | | | | |
| (b) | | | | | | |

* **Δ is the seed-mean paired difference** (F6): mean over seeds 0, 1, 2 of the candidate minus the
  mean over the same seeds of the comparator, on identical rows, seed *s* paired with seed *s*.
  Negative favours the candidate.
* **B = 10,000** for every interval in the Holm family; never lowered.
* Holm p per leg = `2 × min(share ≤ 0, share ≥ 0)`, each share as `(count + 1)/(B + 1)`.
* **Both clusterings must exclude 0** for legs (a) and (b) (F5).
* Per-seed CIs are reported beside the seed-mean CI, and `all_seeds_better` is stated.

Per seed, against each of the (a) and (b) comparators:

| Seed | candidate micro-RMSE | (a) comparator | (b) baseline | candidate ahead of both? |
|---|---|---|---|---|
| 0 | | | | |
| 1 | | | | |
| 2 | | | | |

> `all_seeds_better` = `<TRUE/FALSE>` (TRUE only if the point difference favours the candidate for
> each of seeds 0, 1 and 2 against each of the (a) and (b) comparators, on the full block).

## 4. Conditions (a)–(j), computed and reported in full

Reported as the pre-registered descriptive reading. **Meeting them all does not produce a label.**

| | Condition | Met? | Evidence |
|---|---|---|---|
| (a) | paired CI excludes 0 vs the strongest published method re-run here at parity | | |
| (b) | the same vs the strongest simple baseline from the closed F3 list | | |
| (c) | Holm correction within the block's family, *m* fixed in writing | | |
| (d) | seeds 0, 1, 2; `all_seeds_better` TRUE; point estimate is the seed mean | | |
| (e) | one look, unlocked only by the lead-written FREEZE | | |
| (f) | difference exceeds the block's governing **analytic** MDE | | |
| (g) | point difference in the candidate's favour against **every** parity comparator | | |
| (h) | sign kept on the `no shared reference` stratum **and** above that stratum's own MDE | | |
| (i) | for `ted`: A1 and A2 present, and A1 **not** within the governing MDE of the full model | | |
| (j) | for an ensemble: the matched baseline ensemble was built and reported on this block | | |

## 5. MDE, both figures (F7)

| Stratum | Rows | Units | Governing analytic MDE (icc 0.5) | icc 1.0 | Bootstrap check (not a gate) |
|---|---|---|---|---|---|
| full block | | | | | |
| in ADORE domain | | | | | |
| no shared reference | | | | | |

The claim is read on **all rows** of the block; the in-domain subset is a pre-registered secondary
stratum, and condition (f) is gated by the **stricter** of the two. The training pool's replicate
floor, 0.504, is quoted only as absolute-RMSE context and **never as a difference threshold**
(ST-14 refuted).

## 6. T6 only — all five bands, beside the headline (F9)

| Band | Rows | Units | ≥ 50 units? | candidate | (a) comparator | (b) baseline |
|---|---|---|---|---|---|---|
| `lt24` | | | | | | |
| `gt96_168` | | | | | | |
| `gt168_720` | | | | | | |
| `gt720_2160` | | | | | | |
| `gt2160` | | | | | | |

**No band carries a label.** Bands with fewer than 50 resampling units are **descriptive only**
(T6a `gt2160`; T6b `gt720_2160` and `gt2160`). Extrapolation reach is stated per block —
**273.9× for T6a, 77.0× for T6b** — never as a single figure, and never without the > 720 h strata
beside it. The maximum labelled duration any T6 model saw in training is **96.0 h**.

## 7. Ablations (for `ted`)

| Ablation | What it isolates | Score | Δ vs full model | Within the governing MDE? |
|---|---|---|---|---|
| A1 | no composition layer | | | |
| A2 | `B = 0`, pure anchored law | | | |
| A3 | duration + endpoint also fed to the trunk (leak test) | | | |
| A4 | acute-only training | | | |
| A5 | + BCF head | | | |
| A6 | + Q10 term, free prior | | | |

> **Attribution (binding).** If A1 is within the block's governing MDE of the full model, the win
> belongs to the trunk and the data, **the mechanistic framing is dropped from the claim and the
> paper**, and that is stated here, in this table, beside the headline.

## 8. Parity record

| Entry | Rows | Feature sha256 | Auxiliary targets | Published form (exempt from the feature check) |
|---|---|---|---|---|

Auxiliary rows came through `arch_v2_guard.load_auxiliary_rows()` / `load_h4_rows()` (addendum 01);
their exclusion reports and counts are cited here. On T6 the maximum auxiliary duration is 96 h.

## 9. Reproduction fidelity (§4.6)

| Comparator | Reference | Tolerance | Realised | Within tolerance? |
|---|---|---|---|---|
| `rf_published_adore` | 0.927 | ± 0.03 | | |
| `xgboost_published_adore` | 0.927 | ± 0.03 | | |
| `gpboost_published` | none available | — | — | nine hyper-parameters asserted verbatim |
| `mmoe` | none available | — | | must beat `lightgbm_noemb` on pool validation |

A miss is **disclosed beside the result** and never removes that arm from (a) (F2). A missing arm
cannot be the **only** arm carrying (a) or (b).

## 10. Configuration count, reconciled

| | Declared | Realised (incl. failures, discarded configurations, timed smokes) |
|---|---|---|
| T1 searched | 384 | |
| T6 searched | 128 | |
| Fixed | 26 | |
| Ablations | 12 | |
| **Grand total** | **550** | |

## 11. Disclosures that must not be dropped

- all three blocks are **descriptive only** (F1);
- TRIDENT is a **descriptive, non-paired, non-parity cross-reference**; it cannot carry (a) (R5);
- `gpboost_published` runs with gpboost 1.8.0 against the authors' 1.2.1; the corrected Bio-QSARs
  2.0 citation is used (S4, §11.2);
- the embedding is **MolFormer-XL** (apache-2.0, licence checked 2026-11-10); the frozen embeddings
  were trained by third parties on corpora that include the held-out structures — **structures, no
  labels, none of our rows** (PRETRAINED-CACHE);
- the ensemble membership margin is 0.05 RMSE, about half T1's governing gate;
- T5 (cross-source) is **blocked** on the EnviroTox licence, not skipped;
- no ADORE partition is scored before all three looks are complete (lead point (b));
- `t1t6_mde_v2.json` governs wherever this write-up's prose differs (F10).
