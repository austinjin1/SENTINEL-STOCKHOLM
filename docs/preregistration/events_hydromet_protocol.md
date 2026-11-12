# SENTINEL-HydroMet (M6) — event track (B output) — pre-registration

**Status:** DRAFT v0.1 (2026-11-09; approved as written by the lead the same day, including the separate file, the
`events_hydromet.py` naming and E1a OFF Test-A — no override). Written before any event score, before any track-R test look and before
`FREEZE_hydromet_trackR.json` exists. Implements `docs/plans/event_layer_plan.md` (M6 rows, §2, §4) and the user
requirement that every model output (A) water quality and (B) harmful-event detection.

**Why a separate file and not `hydromet_protocol.md` §10.** `hydromet_protocol.md` and every file matching
`sentinel/models/hydromet/*.py`, `configs/hydromet/*.yaml`, `scripts/{train,eval}/hydromet_*.py` are hashed by the
primary M6 freeze (under review), and track R must keep that hash. Any edit there would void both freezes. This
file is the event track's protocol section; it is pinned by sha in `FREEZE_events_hydromet.json`. The event code
lives only in new, unhashed paths (event plan §4.1): `sentinel/eval/events.py` (shared, being built),
`sentinel/events/hydromet.py` and `scripts/eval/events_hydromet.py`. (`scripts/eval/hydromet_events.py` would match
the hashed glob `scripts/eval/hydromet_*.py`, so it is not used.) The primary `M6/test` and `M6N/test` looks and the
candidate under review are unchanged.

## 1. Honest prior (stated first)
On the original val cell (`M6S/val_v2`, track S; cycle 1–4 EA-LSTMs and the GBM/physics baselines — `ealstm_ws_full`
was never evaluated, and nothing was evaluated on test), **no evaluated model detects low DO**: pooled critical success
index for daily-minimum DO < 5 mg/L ≤ 0.033 for every evaluated model and comparator (255 observed exceedance days), and every model is
1.2–2.7 °C too cold on days above the training 90th percentile. The event track is therefore expected to show weak
hypoxia detection; any skill must beat climatology and the thresholded GBM/physics comparators to count.

## 2. Events (definitions from the event plan §1.1, imported from `sentinel/eval/events.py`, unchanged)
| Code | Event | Definition | Variable (M6 output) |
|---|---|---|---|
| E1a | Hypoxia, acute | daily minimum DO < 3.0 mg/L (state); onset per `hazards.daily_onsets` (H1) | `do_mgl_min` |
| E1b | Hypoxia, chronic | 30-day mean of daily-mean DO < 5.5 mg/L (≥ 27 of 30 days observed) | `do_mgl_mean` |
| E7-cold | Thermal stress, coldwater | 7DADMax > 20 °C at coldwater-eligible sites (≤ 50 % of the site's train-block Jul–Aug observed days above 20 °C; ≥ 30 such days required) | `temp_c_7dadm` |

Coldwater eligibility uses only train-block (WY2008–2021) observations of the site. For FINAL test-role sites those
are protected, so eligibility is computed inside the claimed `M6E/test` look, before any event score, and logged.

## 3. Cells and power (event plan §3; power rule ≥ 30 positives and ≥ 10 clusters with positives)
- **Development/recalibration cell = track-R CV cell** (FINAL train+val-role sites of each held-out region × val
  block, predicted by the fold models that never saw that region). Plan counts: E1a 90 positives / 18 basin groups,
  E1b 91 / 12, E7-cold 155 / 24 → all powered. It is not confirmatory (it contains the val cell used for selection).
- **Test cell (`M6E/test`)**: FINAL test-role sites × Test-A (WY2024–25), scored from the **sealed `M6R/test`
  predictions**, E1b and E7-cold only. **E1a is excluded from Test-A**: the EW-v2 single pass already scored those
  hypoxia labels on the same cells (event plan §4.2). E1a gets a later look on Test-B (WY2026, Approved after
  2027-06-30) with its own marker, together with M1.
- Test-cell positives are unknown before the look (protected). If a test event has < 30 positives or < 10 basin
  groups with positives, it is reported as descriptive and its hypothesis enters Holm with p = 1.

## 4. Models and routes
- **Event model:** track-R `ealstm` (3-seed ensemble; the track-R candidate with a predictive distribution). The
  track-R `ealstm_gbmblend` is reported descriptively (the GBM half has no predictive distribution).
- **Route (i) derived — primary.** Per row, z = (threshold − μ_agg) / σ_agg in the event's direction, where μ, σ are
  the Gaussian-head outputs (seed ensemble: μ = mean of seed μ, σ² = mean of seed σ² + variance of seed μ). E1a:
  `do_mgl_min` (μ, σ); E7: `temp_c_7dadm` (μ, σ); E1b: μ_agg = 30-day mean of `do_mgl_mean` μ, σ_agg = 30-day mean
  of σ (perfect-correlation bound). P = logistic(a + b·z), with (a, b) **Platt-fitted on the CV cell only**, per event,
  then frozen.
- **Route (ii) direct — challenger, powered events only (all three on the CV cell).** Per fold, a LightGBM binary
  classifier on the `forcing_gbm` feature set (same inputs, no in-situ input) trained on the fold's training sites ×
  train block event labels; 3 configurations (`max_rows_per_site` ∈ {1000, 3000, all}) chosen on the fold's inner
  groups, as the GBM regression parity; Platt-recalibrated on the CV cell like route (i).
- **Comparators (event plan §2.3), each Platt-recalibrated on the CV cell the same way:**
  1. threshold-on-GBM: z from the fixed track-R GBM comparator (`forcing_gbm_bag` for temperature mean / DO min,
     `forcing_gbm` for 7-DADMax / DO mean; `hydromet_track_r_comparators.json`) with σ = residual SD of that model
     per target on the CV cell;
  2. threshold-on-physics: the fixed physics comparator (`air2stream_regional` / `dosat_ratio[air2stream_regional]`),
     σ as in 1; for DO this is the DO-saturation-from-temperature physical reference;
  3. climatology: day-of-year harmonic logistic (2 harmonics) of the event state, fitted per fold on the fold's
     training sites × train block (a regional climatology — a site climatology does not exist at an unsensed reach).
- Nothing is tuned on val beyond the Platt parameters (2 per model × event) and the direct head's 3-point inner grid.

## 5. Metrics and tests
- **State classification** (rows = site-days where the event state is defined): Brier score, BSS vs climatology,
  AUPRC with prevalence as chance, reliability table and ECE; basin-group bootstrap B = 2,000 (seed 0).
- **Primary hypotheses (per event):** ΔBrier = Brier(event model, derived) − Brier(comparator) for each of the three
  comparators; paired basin-group bootstrap; IUT p = max over the three; Holm over the events of the look
  (`M6E/test`: E1b, E7-cold → 2). Won = Holm rejects and all three ΔBrier < 0 with CI below 0, and ΔAUPRC > 0 vs
  climatology (point). Secondary: ΔAUPRC with CIs, derived vs direct (paired ΔBrier, ΔAUPRC), onset POD / lead time /
  false alarms per site-year at the EW-v2 alarm budget (≤ 1 alarm-day per site per 30 days) for E1a/E7, descriptive.
- Hypotheses with < 3 basin groups in the paired cell are not testable (p = 1), as in `hydromet_protocol.md` §7.

## 6. Freeze and single look
- `FREEZE_events_hydromet.json` (written by the lead after `FREEZE_hydromet_trackR.json` and before `M6R/test`
  or right after it — the event look only reads its sealed predictions): sha of `FREEZE_hydromet_trackR.json`, sha
  of this file, sha of the event code (`sentinel/eval/events.py`, `sentinel/events/hydromet.py`,
  `scripts/eval/events_hydromet.py`, `configs/events/hydromet.yaml`), the frozen Platt parameters and direct-head
  artifacts (sha), the coldwater rule, and the `M6R/test` `PREDICTIONS_SAVED.json` / `SHA256SUMS` sha once sealed.
- `M6E/test`: own O_EXCL marker and registry refusal (same pattern as `sentinel/models/hydromet/testlook.py`,
  implemented in the new event files), reads only the sealed, sha-checked `M6R/test` predictions plus the test-role
  labels, scores once. Track R and its event track are secondary results; they never change the primary M6 label.
