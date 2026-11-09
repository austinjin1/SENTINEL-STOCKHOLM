# Track A — Calibrated label-free lead-time detector (real NEON data)

AquaSSM reconstruction-residual detector, per-site calibrated to a target sustained false-positive rate of 5% (K=3 consecutive 2-h windows = 6 h sustained). Checkpoint `aquassm_neon_ft.pt`. Label-free: no anomaly labels are used to fit the detector; the DO<4 mg/L onset scores lead time after the fact.

## Specificity (held-out clean periods)

- Truly-clean control sites ['HOPB', 'MART', 'MCRA'] pooled held-out sustained FPR: **4.6%** (Wilson 95% CI 4.3%-4.9%, n=17234).
- Mean held-out FPR across all 22 sites: 15.4% (median 4.8%).

## Early warning on specificity-validated episodes

- Episodes with a verified onset: 12. Detected before onset AND meeting the 5% held-out FPR bar: **5/12**.
- Median lead time (validated episodes): **29.2 days** (IQR 22.5-31.8; range 4.0-36.8).

| Site | Onset | Held-out FPR | Detected pre-onset | Lead (d) | Specificity-validated |
|------|-------|--------------|--------------------|----------|------------------------|
| ARIK | 2024-05-17 | 9.5% | yes | 12.8 | no |
| BARC | 2024-12-08 | 19.5% | no | — | no |
| BIGC | 2025-08-01 | 4.8% | yes | 29.2 | YES |
| COMO | 2026-01-30 | 4.9% | yes | 4.0 | YES |
| CRAM | 2023-05-03 | 73.8% | no | — | no |
| GUIL | 2026-02-10 | 37.1% | yes | 44.8 | no |
| KING | 2024-07-31 | 0.0% | yes | 36.8 | YES |
| LECO | 2025-08-03 | 4.7% | no | — | no |
| LIRO | 2023-06-09 | 20.9% | yes | 26.2 | no |
| MAYF | 2025-04-17 | 4.7% | yes | 31.8 | YES |
| MCDI | 2024-07-27 | 30.1% | yes | 40.3 | no |
| PRIN | 2024-10-25 | 4.6% | yes | 22.5 | YES |

## Matched-FPR head-to-head: rolling-CUSUM vs baseline (the rigorous same-FPR test)

The per-site baseline above over-fires on non-stationary sites. A strictly-causal
rolling-baseline multivariate CUSUM change-point detector (label-free, same AquaSSM
residual cache) was compared to a **global-threshold** re-expression of the baseline
under an **identical protocol**: both detectors calibrated to the SAME pooled
clean-control **held-out** FPR budget, FPR measured on disjoint held-out control
windows, episodes never touching calibration, and the SAME precursor guardrail
(first alarm strictly pre-onset AND >7 d from any QC-clean DO<4).

Sweeping each detector's single global threshold to a matched control-FPR budget:

| Target control-FPR | Baseline: episodes (FPR, median lead) | CUSUM: episodes (FPR, median lead) |
|--------------------|----------------------------------------|-------------------------------------|
| 3% | 2/12 (4.6% held-out FPR* → 2.6%, 11.4 d) | **9/12** (3.0%, **32.1 d**) |
| 5% | 5/12 (4.8%, 25.5 d) | **10/12** (5.0%, **37.2 d**) |
| 7% | 6/12 (6.9%, 29.0 d) | **10/12** (7.0%, **40.2 d**) |

At **every** swept operating point CUSUM detects **more** real episodes with **longer**
median lead at the **same** clean-control FPR budget — a genuine matched-FPR **Pareto
improvement**, not a higher-FPR win. The result is robust to how single-run control
sites are split (interleave vs temporal-halving; verdict identical). Threshold-free
partial-AUC of the detection ROC (episodes-detected vs control-FPR) also favors CUSUM:
pAUC[0,5%] 0.74 (CUSUM) vs 0.16 (baseline); pAUC[0,10%] 0.79 vs 0.35.

Headline (5% matched held-out control-FPR): **CUSUM 10/12 episodes, median lead 37.2 d**
vs baseline **5/12, 25.5 d**. CUSUM catches 6 episodes the baseline misses
(BARC, COMO, KING, LECO, LIRO, PRIN) and misses 1 the baseline catches (ARIK, a
short-runway site the rolling min_base warmup cannot cover) — so it is a strict win
in count but not a clean superset. All 10 CUSUM alarms verified as genuine precursors
(strictly pre-onset, non-hypoxia-near, before the first-ever DO<4 crossing). CRAM has
no usable pre-onset runway (onset at record start) and is undetectable by either.

*Caveat: one fully-clean control (MART) has a single contiguous clean run, so the
interleave split reuses its calibration windows as "held-out" (26% of the pooled
held-out denominator), making the absolute FPR slightly optimistic. A temporal-halving
split that removes this in-sample overlap gives the same Pareto verdict at the same
FPR levels (results/trackA_leadtime/aquassm_specificity/cusum_matched_fpr.json,
`robustness_temporal_split`).

Artifacts: `cusum_matched_fpr.json` (sweep + guardrail checks),
`cusum_vs_baseline_matchedFPR.json` (single-point + ROC), `rolling_cusum_results.json`.

## Honest caveats

- CUSUM's absolute pooled-control held-out FPR is slightly optimistic due to the MART single-run split fallback (above); the matched-FPR *comparison* is unaffected since both detectors share the split. Independent re-verification with a stricter split is advisable before publication.
- Small validated episode set; lead time is a case-set timing result on real NEON streams, not a population estimate, and is bounded above by the clean pre-onset runway.
- Sites that exceed the FPR target (trigger-happy on clean water) are listed but excluded from the lead-time headline — their 'early' alarms are not real early warnings.
- Only 4 real channels (pH/DO/turbidity/conductivity); temperature & ORP are zero-padded.
- Per-site calibration on the site's own clean history (operational assumption); FPR is measured on held-out clean periods interleaved with (disjoint from) the calibration runs.
- AquaSSM sensor encoder only; not a multimodal or forecast model.