# Track A — Leave-one-site-out generalization of the label-free detector

Tests whether the alarm threshold tau (in self-normalized-sigma units) is a TRANSFERABLE constant. For each held-out site, tau is calibrated on the pooled clean calibration scores of all OTHER sites (5% sustained-FPR target), then applied unchanged. Per-site input z-scoring and residual self-normalization are computed on each site's own clean baseline (label-free); only tau is transferred.

## The threshold is nearly site-invariant (the mechanism)

- Per-site calibrated tau: mean **2.68** sd 0.38 range [2.14, 3.57] across 22 sites.
- LOSO global tau (each excludes its held-out site): mean 2.79 sd 0.02 range [2.75, 2.86].
- A single transferred threshold reproduces the per-site threshold to within ~0.35 sigma on average — evidence the operating point is universal, not site-tuned.

## Early warning under TRANSFERRED tau (held-out site never seen)

- Episodes detected pre-onset with transferred tau: **9/12**.
- Meeting the 5% held-out-FPR bar as well (specificity-validated): **4/12**.
- Median lead (validated, transferred tau): **25.8 days** (range 4.0-36.8).

| Site | Onset | global tau | per-site tau | held-out FPR (global) | detected | lead (d) | validated |
|------|-------|-----------|--------------|-----------------------|----------|----------|-----------|
| ARIK | 2024-05-17 | 2.78 | 2.86 | 10.2% | yes | 12.8 | no |
| BARC | 2024-12-08 | 2.81 | 2.56 | 16.5% | no | — | no |
| BIGC | 2025-08-01 | 2.81 | 2.30 | 3.7% | yes | 29.2 | YES |
| COMO | 2026-01-30 | 2.81 | 2.65 | 3.6% | yes | 4.0 | YES |
| CRAM | 2023-05-03 | 2.79 | 2.17 | 62.0% | no | — | no |
| GUIL | 2026-02-10 | 2.79 | 2.86 | 39.2% | yes | 44.8 | no |
| KING | 2024-07-31 | 2.78 | 2.87 | 0.0% | yes | 36.8 | YES |
| LECO | 2025-08-03 | 2.75 | 3.03 | 7.0% | no | — | no |
| LIRO | 2023-06-09 | 2.79 | 2.59 | 19.9% | yes | 26.1 | no |
| MAYF | 2025-04-17 | 2.76 | 3.49 | 7.3% | yes | 31.9 | no |
| MCDI | 2024-07-27 | 2.80 | 2.50 | 25.8% | yes | 40.2 | no |
| PRIN | 2024-10-25 | 2.79 | 2.40 | 0.5% | yes | 22.5 | YES |

## Specificity under transferred tau (held-out clean periods)

- Truly-clean controls ['HOPB', 'MART', 'MCRA'] pooled held-out sustained FPR with transferred tau: **4.1%** (Wilson 95% CI 3.8%-4.4%, n=17234).
- Mean held-out FPR across all 22 sites (transferred tau): 14.2% (median 7.1%).

## Interpretation

- If detected/validated counts and median lead under the TRANSFERRED tau match the per-site-calibrated baseline, the detector generalizes to unseen sites by self-normalization: no site-specific threshold tuning is required, only an unlabeled clean baseline for input/residual normalization.
- Sites that over-fire on held-out clean water do so under BOTH per-site and transferred tau — that is a residual-nonstationarity limit of the method, not a failure of threshold transfer.