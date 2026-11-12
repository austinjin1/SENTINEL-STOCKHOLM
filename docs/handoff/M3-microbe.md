# Handoff — M3 SENTINEL-Microbe (updated 2026-11-10 20:36 EST = 2026-11-11 01:36 UTC)

## Architecture round v2 — where it stands (read this first)

Governing files (all in `docs/preregistration/`, stamped, read-only, never edited): `microbe_arch_v2_amendment.md`,
`microbe_arch_v2_lead_rulings.md` (wins over the amendment), `microbe_arch_v2_deviations_01.md`,
`microbe_arch_v2_dev01_lead_rulings.md` (wins over deviations 01). The append-only execution log is
`microbe_arch_v2_log.md` (entries E1–E23 so far): every ruling received, every launch, every result and every row
id is there. Read its last entries before doing anything.

- **Released:** step 1 on `lodo` only; keys A0, XGB_V4CLR_MO, CALIBRATION, NICHE_S_QMP, GEO_POWER.
  **Not released (do not start):** D2 NicheGeo-Phys, the added EA family, H4, GAS-3, `niche_s_doy`, any EA label
  download, and any slot-#8 arm (`xgb_v4mr_stack` / D3b) before the lead's `microbe/d1_gate_evaluation` row.
- **Code:** `sentinel/models/microbe/{niche,archv2,stamp,v4mr,wa}.py`, runner
  `scripts/train/microbe_arch_v2_cv.py`, tests `tests/test_microbe_archv2.py` (79) and `tests/test_microbe_niche.py`.
  `niche.py`, `archv2.py` and the runner are hashed into every fold shard: none of the three may be edited
  between the first shard of a model and its merge.
- **Results so far (development LODO, mean pooled R², 3 seeds where stochastic; experiment
  `microbe_arch_v2_cv`):** `niche_s` (D1, slot #6) P 0.186, G 0.199; `xgb_nichez_context` (control) P 0.139,
  G 0.178; `tabpfn_v4clr` P 0.305. On P, D1 is well below the existing V4 tables (0.314–0.325): a negative result
  on its face. The D1 gate is evaluated once, by the lead, from registry rows; M3 states no gate outcome.
- **Held:** `xgb_v4mr` (slot #7). Train-only timing projects 41.6 CPU-h on `lodo` (21.2 per family) against the
  stamped 6 CPU-h budget (log E23). It waits for a lead ruling; no fallback is invented by M3.
- **Running (2026-11-11 01:35 UTC):** two CPU lanes, 6 threads each (lead capacity ruling: at most 12 threads in
  total across M3 CPU jobs), launcher `sentinel_runs/microbe/logs/archv2_cpu_lane.sh`, status files
  `logs/archv2_lane_{a,b}.status`:
  lane a = `niche_s_a0` fold shards + merge, then `xgb_v4clr_3000` P fold shards + merge;
  lane b = `wa_tol` + `wa_pls` (P, G), `rf_v4clr` P, `xgb_gfo` P, `xgb_v4clr_mo` P.
  GPU (lock helper, never GPU 2): `tabpfn_v4clr` G and `tabpfn_clr` G, logs `logs/archv2_tabpfn_*_G.log`.
- **Then (lead's order):** when every B6 prerequisite is finished on `lodo`, one message to the lead with the row
  ids per model and family and the words "gate prerequisites complete". After that: `niche_s_tax`, the family G
  comparators (`archv2_queue_a.jobs` lists them), the tuned tables on G (one-thread fold shards), `xgb_clr_stack`
  only if D3b runs, the G9 ensemble, the GEO power file (once), the G10 choice row, `niche_s_qmp` only if that
  row names G1.
- **Rules that are easy to break:** a counted model is never rerun; a fold shard is fitted once; the 2024 holdout
  is never scored; no EA label file is opened; `scripts/eval/microbe_final.py` is never run without the
  lead-written `FREEZE_microbe.json`; no git commit; processes are killed by PID only.

## Before the architecture round (existing track; the text below is the state of 2026-11-10 00:30 EST)

The W6 tuned tables (`rf_clr_tuned`, `xgb_clr_tuned`) on `lodo` have finished since; the queue described in the
earlier version of this file is gone.

## State (development CV only; holdout and reserve untouched; no FREEZE)
- **Protocol log:** `docs/preregistration/microbe_protocol.md`. Every cycle was declared before its results.
  - §10.2: phase 1, fusion control, cycle 2 (enlarged track `lodo_x001`, sha `a6141698…`).
  - §10.3: cycle 3 (V4 harmonisation).
  - §10.4: cycle 4 (+context).
  - §10.5: Test-B 2025 (secondary, pre-registered).
  - §10.6: cycle-4 outcome, cross-release transfer, cycle 5.
  - §10.7: event track.
- **Mean dev-LODO R² (8 targets):**

  | Model | Mean dev-LODO R² |
  |---|---|
  | Ens[V4-Ridge + V4-XGB] | 0.325 |
  | V4-XGB+context | 0.317 |
  | V4-XGB | **0.314** |
  | Ens[Ridge + XGB] (baseline fusion) | 0.307 |
  | TabPFN | 0.291 |
  | V4-Ridge | 0.291 |
  | XGB[genus-CLR] | 0.289 |
  | XGB+context | 0.286 |
  | Ridge[genus-CLR] | 0.284 |
  | SeqClust | 0.256 |
  | RF[genus-CLR] | 0.252 |
  | SeqSet-kmer | 0.239 |

- **Controls:**
  - Permutation null: RF[genus-CLR], 20 permutations, real > all nulls on 8/8 targets (p = 1/21). V4-XGB, 3
    permutations, real > all on 8/8 (null R² −0.07 to −0.32).
  - Site-identity probe on 1,215 events: site 96.2% (logistic regression) and 92.3% (RF); domain 98.3%.
- **Cross-release transfer 002→001:** fails for every model on TN, TP and DOC (TP R² ≈ −1), which points to an era
  level shift. SC transfers: V4-XGB 0.479 vs genus XGB 0.142.
- **Built and tested, not run:**
  - test scorer `scripts/eval/microbe_final.py`;
  - guard `sentinel/models/microbe/testlook.py`;
  - tests `tests/test_microbe_testlook.py` (7 tests): lead-written FREEZE only, candidate file refused; claim
    exclusive (O_EXCL) and tied to pid/host; refused if the registry holds a test row; access closes at sealing;
    sealed SHA256SUMS checked against the registry sealed row; unlisted files refused; rescore refused once scoring
    rows exist; BaseException aborts; event-track errors recorded as `event_track_not_run`; data pins and
    artifacts verified before the claim; code hash = full import closure (incl. events.py → ew_scoring, hazards,
    early_warning, metrics, bootstrap, package `__init__`s);
  - candidate writer `scripts/eval/microbe_freeze_candidate.py`;
  - event adapter `sentinel/events/microbe.py`, `scripts/eval/events_microbe.py`,
    `docs/preregistration/events_microbe_protocol.md` (smoke-tested only; no logged event result yet).
- **Data:**
  - `emp_freshwater` (EMP R1 Deblur-150, 4,723 samples / 7 studies, 0.19 GB raw); 82% of .002 read mass appears
    exactly in EMP V4.
  - `interim/microbe_seqemb/kmer5_v4.npy` (0.9 GB).
  - The upstream per-bottle chemistry key issue is a known issue (lead, 2026-11-09); the workaround stays.
- **Novelty:** `docs/novelty_checks/microbe.md`, updated 2026-11-10. The claim wording must name V4 harmonisation,
  not "frozen DNA-LM embedding".
- **Configurations tried (dev CV):**
  - 21 DNA-encoded / SENTINEL candidate configurations on LODO-family tracks, plus 2 on cross-release;
  - about 30 baseline configurations (incl. the parity baselines added in cycles 2–4);
  - all are in the registry.

## SOTA plan (docs/plans/sota_campaign.md; for lead review before new compute)

### 1. Diagnosis: where M3 stands on validation (dev LODO, paired site- and domain-cluster bootstrap, Holm over 8)

| Comparator | V4-XGB | V4-XGB+context |
|---|---|---|
| Tier-0 season + location (`tier0_best`) | Beats on TP, SC (turbidity and chl-a positive, but they fail Holm on the domain unit) | Beats on TP, DOC, SC, turbidity, chl-a |
| RF[genus-CLR] (the published-method family: Hermans 2024, Pilgrim 2022 use RF on taxa) | Beats on DOC (+0.14) and SC (+0.11) | Beats on DOC, SC |
| XGB[genus-CLR] (strongest untuned GBM) | Beats on DOC (+0.07) and SC (+0.06) | Ties on DOC (p 0.012 / 0.042) and SC |
| Ridge[genus-CLR], TabPFN, Ens[Ridge+XGB] | Ties on every target; per-target best baselines win on NO3, turbidity, SC (point estimates) | — |
| H-M3 dev analogue (beat tier-0, RF and XGB) | 1 target (SC); DOC misses tier-0 (p 0.054) | 0 targets |

W6 tuned RF/XGB results are pending.

**Why it is not SOTA yet:**
- **(a) Power.** There are 33 sites and 19 domains. The minimum detectable ΔR² is 0.10–0.20 on dev, and the
  2024 holdout (214 events) has less power.
- **(b) Strong tier-0 for some targets.** Season + location is strong for DOC, NO3 and DO in unseen domains.
- **(c) Composition tables hold the signal.** Rule 5 asks whether the learned model has information the GBM
  lacks; for whole-amplicon embeddings the answer was no. The informative lever was the *unit* (V4-harmonised
  exact sequences), which XGB already exploits.

The realistic label is **first-in-class** (no held-out-region prior art found), which needs the same bar.

### 2. Levers, ranked by expected gain per GPU-hour

| Rank | Lever | Comparator parity | Cost |
|---|---|---|---|
| L1 | **More labelled data with parity.** Rule 4: more data comes first. | Baselines get the same rows | CPU only |
| L2 | **Unseen-test power.** Test-B 2025 (pre-registered §10.5) and a pooled 2024+2025 analysis. | — | ~0 compute |
| L3 | **EMP V4 pretraining** (cycle 6). | Comparator: same model without pretraining, and the tables | ≤ 8 GPU-h in ≤ 2 h chunks |
| L4 | **DNA-LM embeddings of V4 windows** (DNABERT-S, then DNABERT-2 if licence and size allow), with SeqClust/SeqSet. | — | ~1 GPU-h embedding, then CPU |
| L5 | **Method:** cross-fitted stacking or multi-task tables. | — | CPU, cheap |

- **L1 detail:**
  - Done: `lodo_x001` (+517 .001 events). It gave +0.01 for tables and +0.01 for V4-Ridge.
  - Next: an external geography with paired chemistry, e.g. England EA river-biofilm 16S (694 sites; access and
    paired chemistry to be checked) or US EPA NRSA / USGS 16S if released with chemistry. These are new sources,
    so they are unseen test blocks too.
  - Expected gain: largest (power-limited task).
- **L2 detail:** raises test power by about 2× when NEON RELEASE-2027 lands (about Jan 2027).
- **L3 detail:**
  - Masked-abundance or contrastive set pretraining of the SeqSet on 4,723 EMP freshwater samples plus NEON dev
    communities, on V4[:150] tokens.
  - V4 harmonisation makes EMP usable (82% read-mass overlap).
  - Expected gain: uncertain. SeqSet is currently below the tables, and Rule 5 says pretraining is the right kind
    of lever when tables tie.
- **L4 detail:** the cycle-5 k-mer V4 SeqSet is declared. Expected gain: low to moderate. Whole-amplicon DNABERT was
  worse than k-mers and genus.
- **L5 detail:** expected gain small, with a forking-path risk. Only one pre-declared configuration.

### 3. Tracks and their unseen test blocks

| Track | Unseen test block |
|---|---|
| Primary `lodo` (or `lodo_x001`) × time-out | 2024 holdout, 214 events (unscored) |
| Cross-release reserve | .001 events before 2017, 190 events (unscored) |
| Test-B (secondary) | 2025 events, after RELEASE-2027 |
| External geography (proposed, L1) | A new source (e.g. England EA). Protocol section, frozen split sha256 and power check before any result. Train on NEON, test on the external source (and the reverse). Rule 2 justification: out-of-region transfer is the practical use case. |
| Event track | The same blocks, scored in the same single pass (frozen σ/Platt) |

### 4. Caps and ETA

| Item | Cap | ETA |
|---|---|---|
| Current track | ≤ 8 further configurations: cycle 5 (2), cycle 6 EMP (≤ 4), one stacking config, one L4 config | — |
| External track | ≤ 6 configurations | Access-dependent, 1–2 weeks |
| Freeze candidate on the current path (W6 tuning, logged comparisons, selection, event dev run, candidate file) | — | ~1.5 days |
| With cycles 5 and 6 first | — | ~4 days |

**Recommendation.** Do not spend the 2024 primary block yet. The current best candidate passes only one target
on the dev analogue, and the bar needs two. Run cycles 5 and 6 (capped) and then freeze, so that one block is
scored only with the best-supported model. Test-B and the external track add unseen power afterwards. If the lead
prefers to freeze now, the freeze path is ready: candidate V4-XGB (or the §6 selection at that time), track `lodo`.

## Next steps (in order)
1. After W6, run the logged comparisons once with the final references:
   `microbe_compare.py --schemes lodo lodo_x001 release_002_to_001 --refs tier0_best rf_clr rf_clr_tuned xgb_clr xgb_clr_tuned ridge_clr tabpfn_clr ens_ridge_clr+xgb_clr rf_clr_context xgb_clr_context --pass-refs …`
2. Run the logged event dev run for the candidate(s): `scripts/eval/events_microbe.py --model <M*> --scheme <track>`.
3. Run cycles 5 and 6 if the lead approves. Then apply the §6 selection over the full DNA-encoded candidate set,
   `microbe_freeze_candidate.py`, and send it to the lead. **Never** run `microbe_final.py` before the lead writes
   `FREEZE_microbe.json`.
