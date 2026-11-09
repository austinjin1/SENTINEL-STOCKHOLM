# SENTINEL v2 — Event layer plan (cross-model "event tracks")

**Status:** planning document, 2026-11-09. Nothing here is pre-registered yet. Each workstream turns its
section into a protocol section, adds a split reference and power check, and freezes it before any
event-track model selection (RESEARCH_PLAN §15.5 amendment items 5–7).
**Goal (user brief):** every model should (A) predict water quality and (B) say which harmful event is
happening or has happened. Where (A) or (B) does not apply to a modality, this plan says so; it does not
force it.
**Scope of this document:** the shared event taxonomy, the per-model A/B matrix, label-feasibility counts,
integration with frozen work, sequencing and effort, and how Fuse combines event probabilities. No code is
written here. `sentinel/eval/events.py` is specified in §1.3 and built later.

### Provenance of every number (R1)
All counts below come from scratch scripts run on 2026-11-09 (CPU, ≤ 6 threads). Throughout, `$EV` is
`/tmp/claude-1016/-home-bcheng-STOCKHOLM/dbc009e4-477d-4528-910a-e89e964c5c0f/scratchpad/events`.

| Tag | Script | Output |
|---|---|---|
| **[S1]** | `$EV/m1_counts.py` | `$EV/out/m1_event_counts.csv`, `$EV/out/m1_meta.json`, `$EV/out_m1.log` |
| **[S2]** | `$EV/multi_counts.py` | `$EV/out/multi_event_counts.csv`, `$EV/out/multi_meta.json`, `$EV/out_multi.log` |
| **[S3]** | `$EV/external_match.py` | `$EV/out/external_match_counts.csv`, `$EV/out_ext.log` |
| **[S4]** | `$EV/nrc_fishkill.py` | `$EV/out_nrc_fishkill.log` |
| **[S5]** | `$EV/split_roles.py` | `$EV/out_split_roles.log` (split sizes, freeze files, registry test-look rows) |
| **[S6]** | `$EV/side_checks.py` | `$EV/out_side_checks.log` (NRC coordinate coverage in IA/DE; NEON qPCR target genes) |

- **Registered 2026-11-09.** The S-scripts were promoted to
  `scripts/eval/events_feasibility_{sonde,multi,external,nrc_fishkill,splits,side_checks}.py` (S1–S6;
  helpers in `events_feasibility_common.py`).
  - They label through the released `sentinel/eval/events.py` v1.
  - They enforce the never-read-test-labels rule in code: an id/role/date mask is built first, and
    `ProtectedRowsError` is raised before any label column is read if a protected row survived.
  - They ran with `--register`: registry experiment `events/feasibility`, workstream `events`, 349 current
    rows. Outputs are in `$SENTINEL_RUNS/events/feasibility_*/`.
  - **Protocols must cite the registry entries, not this file.**
- **How the registered values compare with the scratch S1–S6 numbers in this document.**
  - Identical: every onset count, every positive count, every cluster count and every external match.
  - Different, for two reasons:
    1. E3b state days are now the days inside qualifying ≥ 10-day runs, as defined. The scratch run counted
       all exceedance days. §3.1 has been updated.
    2. The observed-day and site-year denominators of derived series (E1b, E3, E6, E7) now count days where
       the derived series exists. §3.1 has been updated.
  - In the multi script, a log10 round-trip (10^log10(20) > 20) briefly inflated Lite E4a and UK
    chloride/conductivity counts. It was fixed (values rounded to 6 decimals) and the affected rows were
    superseded in the registry.
- **Sanity check of the M1 pipeline:** [S1] reproduces the registered EW-v2 onset counts exactly.
  - H1 < 3 mg/L, train block: 566 at train-role sites and 101 at val-role sites.
  - H1 < 3 mg/L, val block: 84 at train-role sites and 6 at val-role sites.
  - These match `ew_v2_protocol.md` §9 and `sentinel_runs/sonde/ew_onset_counts_trainval/onset_counts.csv`.

### Test-label hygiene used for every count
No count, mask or inspection touched a test role or test block. In every script, protected rows were
removed using id, role and date columns only, before any label column was read.

| Workstream / catalogue | Rows removed before reading labels |
|---|---|
| M1/M6 | Guarded `sentinel.models.sonde.common.prepare_data` with `blocks=("train","val")`; only train- and val-role sites. Test-role sites and Test-A/B are never requested (`guard.check_access`). Approved values only. |
| M2 | `dev_v4` cache read with pyarrow filters role ∈ {train, val} × tblock ∈ {train, val}, not `buffer_excluded`, not `protected`. |
| M7 | `chip_role` ∈ {train, val} (+ `val_n` for the national split). |
| M3 | `development_ids` and collect date < 2024-01-01. |
| M5 | `load_nrsa()`, which refuses the holdout surveys, and `load_uk(include_reserved=False)`. |
| Fuse | Tier D dev sites (filter on `site`); Tier C dev sites. |
| M6 track N | Train/val-role stations, dates ≤ 2023-09-30. |
| Fish kills | KillAlert v2 test events (2018–2026), dropped by `event_id`. |
| WA toxins | ToxinWatch test lakes and every year ≥ 2022. |
| Orbit-matched HAB reports | Orbit test role and test years, via the frozen v3 rule. |
| All external catalogues | Everything after 2023-09-30, kept unseen as a future event-track test period. |

---

## 1. Shared event taxonomy (E-codes)

### 1.1 Definitions
Every event is a **criterion-anchored excursion** in the sense of `hazards.py`: it reuses a published
criterion's number, but it is not a regulatory violation finding. Where an E-code equals an existing EW-v2
hazard, the hazard definition is imported unchanged (`hazard_key`).

| Code | Event | Criterion-anchored definition | Source of the number | Defining variable(s) | Type | `hazard_key` |
|---|---|---|---|---|---|---|
| **E1a** | Hypoxia (acute) | Daily min DO < 3.0 mg/L; onset per `hazards.daily_onsets` (3 clean days ≥ 4.0, refractory 7 d). Tiers < 4.0 and < 2.0. | EPA 440/5-86-001 (1986), warmwater 1-day minimum | `do_mgl_min` (sonde); grab DO for snapshot modalities, labelled "grab" | acute | `H1_hypoxia`, `_t4`, `_t2` |
| **E1b** | Hypoxia (chronic) | 30-day mean of daily-mean DO < 5.5 mg/L (≥ 27 of 30 days observed); onset needs a 7-day pre-window ≥ 6.0 that ends before the averaging window (pre-gap 29 d, as H2), refractory 30 d. | EPA 1986, warmwater 30-day mean, other life stages | `do_mgl_mean` | chronic | — (new) |
| **E2a** | Salinisation (chronic) | Derived chloride, 4-day mean > 230 mg/L | EPA 440/5-88-001 (1988) CCC | `cl_mgl_4dmean` (site regression, `chloride.py`); lab chloride for M5 | chronic | `H2_chloride` |
| **E2b** | Salinisation (acute) | Derived chloride (daily-max SC) > 860 mg/L | EPA 1988 CMC | `cl_mgl_max` | acute | `H2_chloride_acute` |
| **E2c** | Ionic stress (diagnostic) | Specific conductance > 300 µS/cm | EPA/600/R-10/023F (2011) field-based benchmark (Central Appalachia; **not national**, secondary only) | `cond_us_cm`, `spcond` | chronic | — |
| **E3a** | Sediment plume (acute) | Daily-mean turbidity > trailing-30-day median background (d−30…d−1, ≥ 20 observed days) + 50. Snapshot variant: turbidity > 50 NTU or TSS > 80 mg/L (absolute). | IDAPA 58.01.02.250.02.e (+50 NTU instantaneous over background); EIFAC (1965) / Alabaster & Lloyd (1982) TSS 80 mg/L class boundary | `turb_fnu_mean`, `tss`, `turbidity` | acute | — (replaces the unimplemented H3) |
| **E3b** | Sediment (chronic) | Turbidity > background + 25 for ≥ 10 consecutive days | IDAPA 58.01.02.250.02.e | `turb_fnu_mean` | chronic | — |
| **E4a** | Algal bloom | chl-a > 20 µg/L. Sensitivity tiers: > 12 µg/L (WHO 2021 alert level 1) and ≥ 50 µg/L (Orbit Track E stratum). | VERA Bloom_binary (as H5b); WHO 2021 recreational alert levels | `chl_ugl_mean`, `chla`, `y_chla` | acute (state) | `H5b_chl_bloom` |
| **E4b** | Toxic cyanoHAB | Microcystins > 8 µg/L | EPA 822-R-19-001 (2019) recreational criterion (= H5c) | sample microcystin | acute | (H5c, plan text) |
| **E4c** | Reported HAB | NY DEC "confirmed" bloom-year; CA FHAB advisory ≥ Caution (CCHAB 2016 triggers). **Agency decision, not a measurement.** Partly visual, so partly circular with predicted chl-a. | Agency programmes | report | acute | — |
| **E5a** | Nitrate pollution | NO₃+NO₂-N > 10 mg/L | 40 CFR 141.62 MCL | `no3no2_mgl_mean`, lab NO₃ | acute | `H4_nitrate` |
| **E5b** | Nutrient enrichment | TP > 0.1 mg/L (streams). Secondary: NRSA ecoregional "Poor" TP/TN condition class. | EPA 1986 "Gold Book" TP recommendation; EPA NARS condition thresholds | TP, TN | chronic | — |
| **E6** | pH excursion / acidification | Daily min pH < 6.5 or daily max pH > 9.0 (snapshot: lab pH outside 6.5–9) | EPA 1986 Gold Book freshwater range | `ph_min`, `ph_max`, `ph_lab` | acute / chronic | — |
| **E7** | Thermal stress | 7DADMax > 20 °C (tiers 16 and 18 °C). Primary population: **coldwater-eligible sites**, i.e. ≤ 50% of the site's train-block Jul–Aug days above 20 °C (decided from train-block data only). Onset: 7 clean days before the averaging window, refractory 30 d. | EPA 910-B-03-002 (2003) Region 10 salmonid criteria | `temp_c_max` → `temp_c_7dadm` | chronic (7-d) | — |
| **E8** | Toxic chemical spill | (i) **occurrence:** NRC report with material reaching water (33 CFR 153 reporting); (ii) **severity:** risk quotient RQ = PEC / HC5 > 1, where HC5 comes from the M4 species-sensitivity distribution | EU TGD (2003) / Aldenberg & Jaworska (2000) SSD–HC5 convention | release amount, discharge, predicted LC50 by taxon | acute | — |
| **E9** | Fish kill | Agency-reported kill. Cause classes: `is_env_wq` (low DO, algae, temperature) vs anthropogenic. | Iowa DNR, DE DNREC | report | acute (outcome) | — |
| **E10** | Pathogen / faecal contamination | E. coli > 235 cfu/100 mL; enterococci qPCR > 1,000 CCE/100 mL | EPA-820-F-12-058 (2012 RWQC beach action values) (= H6) | culture / qPCR | acute | (H6) |
| **E11** | Invasive dreissenid establishment | First NAS record in a HUC12 | USGS NAS (D6 MusselGuard) | report | chronic | — (context only) |

**Notes on the definitions.**
- **E3 on sondes is turbidity in FNU, while the IDAPA number is in NTU.** The FNU/NTU difference is a
  documented limitation (Orbit already separates the units).
- **E3a is not attributed to a cause.** Natural storm plumes count. The B-output says "sediment plume
  occurring", not "pollution".
- **E7 is the normal summer state at warmwater sites.** In [S1] (registered) the val-block state
  prevalence is 223,023 / 742,203 ≈ 0.30 at all sites, versus 4,133 / 195,026 ≈ 0.02 at the 393
  coldwater-eligible sites. Only the coldwater-eligible population is a meaningful harmful-event track.

### 1.2 Event-level metrics (identical across models)
- **State classification** ("is the event happening on this row?").
  - AUPRC, with the row prevalence reported as the chance level.
  - Brier score and BSS against site × DOY climatology, or against the prevalence of the training cell
    for snapshot modalities.
  - Reliability table and ECE.
  - Cluster bootstrap (B = 2,000) using the workstream's resampling unit.
- **Onset detection** (only for modalities with a time series at the site: M1, M6, Fuse Tier D).
  - **Hit rate** (POD): onsets with ≥ 1 alarm in [onset − h, onset − 1 d], at a frozen false-alarm
    budget of ≤ 1 alarm-day per site per 30 days, as in EW-v2.
  - **Lead time:** days from the first alarm in the window to the onset (median and IQR).
  - **False alarms per site-year:** alarm episodes not followed by an onset within h, divided by the
    observed site-years.
  - **Placebo control:** POD before real onsets vs season-matched placebo dates, using the EW
    `ew_scoring` machinery imported unchanged.
- **Chronic events** use state classification only, at their integration window (a 30-day window, a
  survey sample, or a season).
- **Power rule** (this plan's flag): a (model, event) pair is **underpowered** if the validation cell has
  < 30 positives **or** < 10 clusters carrying positives. The cluster unit is the workstream's bootstrap
  unit: basin group, lake, location cluster, domain or station group. For onset tracks, the EW rule
  (≥ 50 evaluable test onsets and ≥ 2 basin groups) still applies to any test claim.

### 1.3 `sentinel/eval/events.py` — final API (v1)
Built 2026-11-09 (option (b) of the lead's 2026-11-09 instruction: this section was updated to the final
API; changes vs the draft are listed below). The module imports `sentinel.eval.hazards` and **never edits
it**. `hazards.py`, `chloride.py`, `early_warning.py`, `ew_scoring.py`, `ew_controls.py`, `metrics.py`,
`bootstrap.py` and `splits.py` are hashed by `FREEZE_ew.json`; events.py only imports them. It is **not**
re-exported from `sentinel/eval/__init__.py` (hashed by `bioassess_freeze.json` and Lite's `SHARED_CODE`).

```python
from sentinel.eval import hazards as HZ          # HAZARDS, Hazard, daily_onsets, issue_labels (read-only use)
__version__: str                                  # "1.0" once released

# --- taxonomy
class OnsetRule(method: Literal['hazard', 'clean_window', 'run_start'], pre_days: int = 0, pre_margin: float = 0.0,
                pre_gap: int = 0, refractory_days: int = 0)
class EventDef(code: str, name: str, kind: Literal['acute', 'chronic'], variables: tuple[str, ...], unit: str,
               direction: Literal['above', 'at_or_above', 'below', 'outside'] | None,
               threshold: float | tuple[float, float] | None, aggregation: str, duration_days: int = 1,
               tiers: tuple[str, ...] = (), hazard_key: str | None = None, onset: OnsetRule | None = None,
               anchor: str = '', label_source: Literal['measurement', 'report', 'condition_class'] = 'measurement',
               parent: str | None = None)
EVENTS: dict[str, EventDef]; EVENT_CODES: tuple[str, ...]
def get_event(ev: EventDef | str) -> EventDef

# --- labelling
def derive_series(frame: pd.DataFrame, ev: EventDef | str) -> pd.Series
def event_state(x, ev: EventDef | str) -> np.ndarray                       # float {0, 1, NaN}
def event_onsets(x, ev: EventDef | str) -> tuple[np.ndarray, np.ndarray]   # (event, onset); HZ.daily_onsets if hazard_key
def issue_labels(x, ev: EventDef | str, horizon: int) -> tuple[np.ndarray, np.ndarray]   # delegates to HZ.issue_labels
def coldwater_eligible(dates, temp_c_max, train_block: tuple[str, str] = ('2007-10-01', '2021-09-30'), *,
                       threshold: float = 20.0, max_frac: float = 0.5, min_days: int = 60) -> bool

# --- route (i): predictive distributions (space: 'linear' | 'log10' | 'ln'; the threshold is transformed)
class Gaussian(mu: np.ndarray, sigma: np.ndarray, space: str = 'linear')
class Quantiles(levels: np.ndarray, values: np.ndarray, space: str = 'linear')
class GaussianMixture(weights: np.ndarray, means: np.ndarray, sds: np.ndarray, space: str = 'linear')
class Samples(values: np.ndarray, space: str = 'linear')
def exceedance_probability(pred, ev: EventDef | str) -> np.ndarray
class Calibrator(method: str, params: dict = {}, n_fit: int = 0, n_pos_fit: int = 0, fit_sha256: str = '')
    # .transform(p) / __call__(p)
def recalibrate(p_val, y_val, *, method: str = 'platt') -> Calibrator        # 'platt' | 'isotonic'

# --- metrics (CI dicts are {point, ci_low, ci_high, n_boot_valid[, note]}; CI = NaN with < 2 clusters)
def state_metrics(y, p, *, groups, p_ref=None, n_boot: int = 2000, seed: int = 0) -> dict
def onset_metrics(issue, score, usable, onsets, *, threshold: float, h_days: int, site,
                  clusters: Mapping | None = None, n_boot: int = 2000, seed: int = 0) -> dict
def compare_routes(y, p_derived, p_direct, *, groups, n_boot: int = 2000, seed: int = 0) -> dict
def power_flag(n_pos_val: int, n_clusters_val: int, min_pos: int = 30, min_clusters: int = 10) -> str
    # 'feasible' | 'underpowered'

# --- external labels
class MatchRule(radius_km: float, window_days: tuple[int, int] = (0, 0), require_record_span: bool = False,
                same_lake: bool = False)
def match_external(events: pd.DataFrame, sites: pd.DataFrame, rule: MatchRule) -> pd.DataFrame
```

**§1.3 changes (draft → v1).** The names, the positional parameters and the required keywords of the five
entry points (`exceedance_probability`, `recalibrate`, `state_metrics`, `compare_routes`, `power_flag`) are
unchanged. The other changes are:
1. `exceedance_probability(pred, ev)`: `pred` is one of four typed objects (`Gaussian`, `Quantiles`,
   `GaussianMixture`, `Samples`). The draft's "log-space aware" is the `space` field on each object.
2. `recalibrate` gains the optional keyword `method='platt'` (`'isotonic'` also allowed). It returns a
   `Calibrator` with `.params`, `.n_fit`, `.n_pos_fit` and `.fit_sha256`. It refuses one-class input,
   labels other than 0/1, and probabilities outside [0, 1].
3. `state_metrics`, `compare_routes` and `onset_metrics` gain optional `n_boot` and `seed` keywords;
   `onset_metrics` also gains `clusters` (site → cluster).
4. **Return shapes:**
   - CI entries are dicts `{point, ci_low, ci_high, n_boot_valid}`.
   - With fewer than 2 resampling units the CI is NaN and a `note` key is added.
   - `compare_routes` returns `delta_brier` / `delta_auprc` dicts from `bootstrap.paired_bootstrap` (`diff`,
     `ci_low`, `ci_high`, `p_value`, …).
   - `power_flag` returns the strings `'feasible'` / `'underpowered'`.
5. **Taxonomy additions:**
   - Snapshot variants with their own codes (`E1a_grab`, `E2a_lab`, `E3a_tss`, `E3a_turb`, `E6_snap`).
   - Tiers (`E4a_t12`, `E4a_t50`, `E7_t16`, `E7_t18`) and `E10_qpcr`.
   - `E5b_nrsa`, a condition-class event.
   - New `EventDef` fields `label_source` and `parent`, and direction `at_or_above`.
   - E6 on daily series uses `aggregation='ph_range_excess'` (max(6.5 − pH_min, pH_max − 9) > 0). A
     single-distribution pH forecast uses `E6_snap` (`outside` (6.5, 9.0)).
6. **New helpers:** `get_event`, `EVENT_CODES`, `coldwater_eligible` (the E7 population rule).
7. **`MatchRule.window_days`** applies to a `date` column of `sites` (sample or image dates). The record
   span uses the `first` / `last` columns.

**Rules for the module.**
- It reads no data.
- `tests/test_events.py` checks that E1a/E2a/E2b/E4a/E5a are bit-identical to `hazards.daily_onsets` and
  `issue_labels`. It also re-counts the registered EW-v2 H1 onsets (566/101/84/6) on train/val.
- Workstream adapters live in **new** paths that match no freeze glob: `sentinel/events/<ws>.py`,
  `scripts/eval/events_<ws>.py`, `configs/events/*.yaml`. See §4.1.


## Released versions
| Module | Version | Released | sha256 | Mode |
|---|---|---|---|---|
| `sentinel/eval/events.py` | 1.0 | 2026-11-09 13:02 EST | `4279d2427d777405ce1cb140cab995567a8086fee5fd4776950e4de1065954ae` | 0444 |

**v1 is immutable.** Fixes and additions go into a NEW module (`sentinel/eval/events_v2.py`) and never
edit v1, because an edit would invalidate every freeze that hashes it.

Release checks:
- `tests/test_events.py` passes 31/31. This includes the real-data check that E1a reproduces the
  registered EW-v2 H1 onsets 566 / 101 / 84 / 6.
- `FREEZE_ew.json`'s scoring-code sha256 still verifies.
- The full test suite ran to completion before the release.

---

## 2. Per-model A/B matrix

**Legend.**
- **F** = feasible.
- **U** = underpowered by the §1.2 rule. Numbers are validation positives / validation clusters with
  positives, from §3.
- **n/a** = not physically plausible for the modality and timescale, or no label exists.
- **Route (i) "derived":** P(variable beyond threshold) from the model's predictive distribution.
- **Route (ii) "direct":** classifier head trained on event labels.

### 2.1 Matrix

| Model | (A) water-quality outputs | (B) events: verdict (val pos / clusters) | Route(s) | Notes on physical plausibility |
|---|---|---|---|---|
| **M1 Sonde** | 1–14 d quantile forecasts of temperature, DO mean/min, SC, pH, turbidity, NO₃ (+ derived chloride) | E1a **F** (90 onsets / 18 basins). E1b **F** (91 / 12). E3a **F** (1,202 / 29). E3b **F** (115 / 21). E6 **F** (595 / 26). E7-coldwater **F** (155 / 24). E2a **U** (6 / 3). E2b **U** (6 / 1). E5a **U** (6 / 2). E4a **n/a** for streams (0 / 0; lakes are a separate track). E9 / E8 **n/a as labels** (§3.3). | (i) and (ii) for every F event; onset + state | The only modality with true onset detection and lead time. Sensors at the site. |
| **M6 HydroMet** | Daily water temperature, 7DADMax, DO mean/min at unsensed reaches | Original val cell (val-role sites × val block): E1a **U** (6 / 3), E1b **U** (2 / 2), E7-cold **U** (25 / 4), E7-all **F** (239 / 11). Track R CV (train+val roles × val block): E1a **F** (90 / 18), E1b **F** (91 / 12), E7-cold **F** (155 / 24). Track N (grab DO < 3, val-role stations): **U** (491 positives but 9 basin groups). | (i) primary (exceedance from a predictive distribution, see §2.2). (ii) secondary. | Weather-driven nowcast. Onset metrics are possible because outputs are daily, but lead time comes only from forecast weather. |
| **M2 Orbit** | chl-a, TSS, turbidity, Secchi at each S2 overpass | E4a > 20 **F** (309 / 90 lakes; 36 blocks). E4a ≥ 50 **F** (75 / 35 lakes). E3 (TSS > 80, turbidity > 50) **U** (0 positives in `valxval_lake` and `valxany_lake`). E4b/E4c via reports: §3.3. E1, E2, E5, E6, E7 **n/a** (not optically observable; S2 has no thermal band). | (i) primary, (ii) secondary. **State only, no onset**: revisit ≥ 5 d, cloud gaps. | Track E (bloom strata at 20/50 µg/L, AUROC/AUPRC) is already pre-registered in `orbit_protocol.md` §11.1. The event track extends it rather than duplicating it. |
| **M7 Lite** | Secchi, turbidity, TSS (+ chl-a trophic class) | E4a **F** (val 104 / 43 water bodies; `val_n` 120 / 58). E3-TSS **U** (val 11 / 3; `val_n` 31 / 8). E3-turbidity **U** (val 4 / 2), but **F** on `val_n` (48 / 18). Others **n/a**. | (i) from student outputs, (ii) head. **State only.** NAIP revisit is multi-year, and the drone use case is ad hoc. | B means "is this water body in a bloom / plume in this image", not an alert. |
| **M3 Microbe** | TN, TP, DOC, NO₃, SC, turbidity, DO, chl-a (sample date) | E5b TP **F** (226 / 19 domains). E4a **F** (35 / 14 domains). E2c **F (borderline)** (268 / 10 domains; benchmark not national). E1 (daily-mean DO < 3) **U** (20 / 5). DO < 5: **U** (48 / 7). E3 **U** (25 / 8). E5a **n/a** (0 positives). E10 **n/a** (NEON qPCR `targetGene` is only 16S / ITS totals [S6]). | (i) primary, (ii) secondary. **Chronic diagnosis only.** | 16S integrates weeks to months of community response. It suits "this stream is nutrient-enriched / salinised", not acute onset. |
| **M4 ChemTox** | **A does not apply.** M4 predicts the toxicity (LC50/EC50) of a chemical to a taxon, not ambient water quality. | E8 severity only, **given a reported release**: RQ = PEC/HC5. Labels: NRC reached-water reports near train/val sonde sites, val block: 241 within 10 km in 31 basins, of which **3 have a structure-resolvable chemical** → **U** for any M4-scored analysis. There are no outcome labels: 0 fish kills within 25 km and 0–7 d of an NRC reached-water report; the county-level rule gives co-occurrences with no enrichment for anthropogenic kills ([S4]). | (i) derived only: P(RQ > 1) from SSD and model uncertainty. **(ii) n/a** (no event labels). | Report severity as a decision-support quantity with stated uncertainty. Do not claim event detection. |
| **M5 BioAssess** | Chloride, pH, conductivity, TN, TP, turbidity (inferred from invertebrates; integrates weeks to months) | NRSA dev (LOEO OOF): E2a **F** (102 / 88 locations, 8 of 9 ecoregions). E5b **F** (1,409 / 1,192). E6 **F** (82 / 74). E3 (chronic turbidity > 50) **F** (263 / 248). E2c **F** (1,973). NRSA "Poor" TP / TN / salinity **F**. UK `pool_val`: E2a **U** (4 / 3), E6 **U** (10 / 6), E5b **F** (273 / 85). E1a / E9 / E4 **n/a** (acute). | (i) primary (exceedance from the predictive distribution of the log-target). (ii) secondary. **Chronic only.** | All M5 test blocks are spent (§4.2), so a new test block is needed. |
| **Fuse** | Multi-target at NEON (sensor h1–14, lab chemistry) + existing `ev__*` heads | Tier D dev: H5b/E4a onset rows **F** (308 / 20 sites, 13 domains). E1a onset rows 235 / 12 sites / **9 domains** → **U on the domain criterion** (F on sites). E5a **n/a** (0). Tier C E4b microcystin **F** (1,987 / 611 sites / 68 HUC4). | Combination of per-model probabilities (§6) + its own direct head (already in `model.py`) | Fuse Tier D counts are **issue-row labels** (onset within 7 d), not distinct onsets. |

### 2.2 Routes: pre-register both, compare them
- **Route (i) derived.** `events.exceedance_probability(pred, ev)` turns each model's predictive
  distribution into P(beyond threshold) for each row.
  - M1: quantile forecasts. M2: ensemble or MDN. M3/M5: Gaussian on log-target, using CV residual SD by
    target. M6: see below.
  - **Calibration checks are required:** PIT/reliability and ECE on validation. Then a single Platt (or
    isotonic) recalibration is fitted on validation and frozen, exactly as EW-v2 recalibrates every method
    including climatology.
  - **M6 needs a predictive distribution.** Its models are point predictors. Options are a conditional
    residual model (quantile GBM on val-block residuals, fitted on train-role sites) or an ensemble spread.
    This must be pre-specified.
- **Route (ii) direct.** A binary head on the same inputs, trained on event labels from the training cell
  only, with the same tuning budget as the comparators (§15.5 item 4).
- **Pre-registered comparison:**
  - paired ΔBrier and ΔAUPRC (derived − direct), cluster bootstrap, Holm across the events of a workstream;
  - **primary = derived** where the model has a distributional output, because it ties (B) to (A) and
    needs the fewest labels; **direct = challenger**;
  - the choice is fixed before validation results are seen, and validation is used only to select the
    recalibration and the head's hyperparameters.
- **When a direct head is not fitted:** underpowered events get no direct head, and are reported as
  derived-only and descriptive. This follows the EW rule `min_train_onsets: 5` in `configs/sonde/events.yaml`.

### 2.3 Required comparators (every event track)
1. **Threshold-on-baseline.** Apply the event threshold to the strongest simple baseline's prediction:
   persistence/current value (M1); site-mean / DOY+location (M2, M3, M5, M7); GBM; air2stream (M6).
   The derived-probability variant uses the baseline's own residual distribution.
2. **Persistence and climatology.**
   - Site × DOY harmonic-logistic event climatology (EW-v2 implementation, imported).
   - For snapshot modalities: training-cell prevalence by lake/location and month.
   - Persistence of the last observed state where a prior observation exists.
3. **Published event-detection methods.**

| Workstream | Published comparators |
|---|---|
| **M2 blooms** | NOAA/EPA CyAN CI-cyano (Wynne et al. 2008/2010 CI; Lunetta et al. 2015). **Blocked:** needs an Earthdata login, not downloaded (`docs/datasheets/labels_cyan_availability.md`), so this is a user action. Until then: S2 MCI (Gower et al. 2005) and NDCI (Mishra & Mishra 2012) thresholds, already in `sentinel/models/orbit/baselines.py`, plus an S2-band CI analogue on B4/B5/B6 (pre-specified, labelled as an analogue). |
| **M2/M7 plume** | Nechad et al. (2010) / Dogliotti et al. (2015) single-band turbidity algorithms. |
| **M7** | RGBN band-ratio index thresholds (the existing Lite baseline family). |
| **M1/M6 hypoxia** | EW-v2 ladder (climatology, persistence, threshold-on-current-value, lag logistic, event GBM). M6 adds a DO-saturation-from-temperature physical reference. |
| **M5** | Weighted-averaging taxon-optima inference (ter Braak & Looman 1986) and EPA XC95 extirpation-concentration benchmarks (Cormier et al. 2013) for conductivity/chloride. NRSA MMI "Poor" as a reference classifier for the "impaired" composite. |
| **M3** | RF on CLR (existing baseline) thresholded. |
| **M4** | HC5 from measured ECOTOX data where ≥ 5 species exist, and US EPA Web-ICE / ECOSAR predictions as the published SSD comparator. |

---

## 3. Label feasibility numbers (train and validation roles only)

### 3.1 M1 Sonde (and M6 sensor tracks): daily tier, Approved values [S1; registered `events/feasibility`, model `labels:M1_sonde`]
- Population: 1,907 stream sites after the guard (train-role 1,601 / val-role 306), dates
  2007-10-01..2023-09-30, `sonde_split.json` sha `fd4bb14f…`.
- Validation = val block (WY2022–23) × train- and val-role sites. This is the EW calibration cell (Platt
  recalibration and thresholds are fitted on val rows). The strict val-role-only cell (= M6 original val
  cell) is in the last two columns.
- Clusters = basin groups.

| Event | Train block onsets (basins) | Train block state days | **Val onsets (basins w/ onset)** | Val state days (basins) | Val site-years | Val-role-only onsets (basins) | Flag (val) |
|---|---|---|---|---|---|---|---|
| E1a DO-min < 3 | 667 (30) | 47,601 | **90 (18)** | 6,829 (22) | 1,022 | 6 (3) | F |
| E1a tier < 4 | 950 (33) | 78,197 | **125 (22)** | 11,698 (27) | 1,022 | 5 (2) | F |
| E1a tier < 2 | 479 (26) | 28,037 | **57 (12)** | 3,336 (17) | 1,022 | 0 (0) | F |
| E1b 30-d mean DO < 5.5 | 635 (25) | 79,015 | **91 (12)** | 12,907 (15) | 998 | 2 (2) | F (borderline clusters) |
| E2a chloride 4-d > 230 | 95 (11) | 35,990 | **6 (3)** | 3,733 (7) | 64 | 1 (1) | **U** |
| E2b chloride > 860 | 83 (5) | 15,751 | **6 (1)** | 2,157 (3) | 48 | 0 (0) | **U** |
| E3a turbidity > bg + 50 | 7,684 (34) | 30,831 | **1,202 (29)** | 4,805 (29) | 596 | 65 (7) | F |
| E3b turbidity > bg + 25 ≥ 10 d | 789 runs (26) | 10,553 | **115 (21)** | 1,542 (21) | 596 | 23 (7) | F |
| E4a chl > 20 (streams) | 1 (1) | 100 | **0 (0)** | 36 (1) | 6 | 0 | **U / n/a** |
| E5a NO₃ > 10 | 37 (2) | 7,450 | **6 (2)** | 571 (2) | 113 | 4 (1) | **U** |
| E6 pH < 6.5 or > 9 | 3,570 (38) | 55,225 | **595 (26)** | 7,854 (27) | 833 | 40 (3) | F |
| E7 7DADMax > 20, all sites | 8,500 (66) | 1,246,520 | **1,517 (62)** | 223,023 (62) | 3,491 | 239 (11) | F, but mostly seasonal state |
| E7 7DADMax > 20, coldwater-eligible (393 sites) | 855 (30) | 22,646 | **155 (24)** | 4,133 (24) | 914 | 25 (4) | F |

How to read the table:
- "Train block" = train-block × (train + val roles), the `trainval` rows of [S1].
- Sites observed in the val block (derived series present): E1a 369 sites in 43 basins; E3 219 / 32;
  E6 303 / 42; E7 1,236 / 66.
- **E3a and E6 onsets are frequent.** Turbidity spikes during storms and pH has diel swings, so these
  events need a tighter pre-window or a daily-max variant if the alarm budget saturates. This is a
  Phase 1 design decision, made on validation.

### 3.2 Other workstreams [S2]
Clusters with positives are given as *with-positive / total* in the cell.

| Model | Cell (role) | Event | Labelled rows | Positives | Prevalence | Clusters w/ pos (of total) | Flag |
|---|---|---|---|---|---|---|---|
| M2 Orbit | fit (train × train years) | E4a chl > 20 | 10,235 | 2,058 | 0.201 | 837 lakes/waterbodies (2,434); 298 blocks (565) | training |
| M2 Orbit | **valxval_lake** | E4a chl > 20 | 1,159 | **309** | 0.267 | **90 lakes** (318); 36 blocks (89) | F |
| M2 Orbit | valxval_lake | E4a chl ≥ 50 | 1,159 | **75** | 0.065 | 35 lakes; 18 blocks | F |
| M2 Orbit | valxval_lake | E4a chl > 12 (WHO AL1) | 1,159 | 443 | 0.382 | 130 lakes; 50 blocks | F |
| M2 Orbit | fit | E3 TSS > 80 / turbidity > 50 NTU | 1,165 / 655 | 52 / 40 | 0.045 / 0.061 | 32 / 22 waterbodies | training |
| M2 Orbit | valxval_lake (valxany_lake) | E3 TSS > 80 | 34 (92) | **0 (0)** | 0 | 0 | **U** |
| M2 Orbit | valxval_lake (valxany_lake) | E3 turbidity > 50 NTU | 11 (40) | **0 (0)** | 0 | 0 | **U** |
| M7 Lite | train (split v2) | E4a / E3-TSS / E3-turbidity | 994 / 544 / 575 | 173 / 10 / 11 | 0.17 / 0.02 / 0.02 | 88 / 5 / 5 water bodies | training |
| M7 Lite | **val** | E4a chl > 20 | 428 | **104** | 0.243 | 43 (131) | F |
| M7 Lite | val | E3 TSS > 80 | 161 | **11** | 0.068 | 3 (37) | **U** |
| M7 Lite | val | E3 turbidity > 50 | 132 | **4** | 0.030 | 2 (26) | **U** |
| M7 Lite | **val_n** (national) | E4a / E3-TSS / E3-turbidity | 364 / 303 / 483 | **120 / 31 / 48** | 0.33 / 0.10 / 0.10 | 58 / **8** / 18 | F / **U** (clusters) / F |
| M3 Microbe | dev (LODO OOF; 1,215 dev ids, < 2024) | E5b TP > 0.1 mg/L | 1,183 | **226** | 0.191 | 19 domains (19); 33 sites (33) | F |
| M3 Microbe | dev | E4a chl > 20 | 394 | **35** | 0.089 | 14 domains (18); 16 sites | F (borderline) |
| M3 Microbe | dev | E2c SC > 300 | 730 | 268 | 0.367 | 10 domains (19); 12 sites | F (borderline) |
| M3 Microbe | dev | E1 DO (daily mean) < 3 / < 5 | 683 | **20 / 48** | 0.029 / 0.070 | **5 / 7** domains | **U** |
| M3 Microbe | dev | E3 turbidity > 50 FNU | 655 | **25** | 0.038 | **8** domains | **U** |
| M3 Microbe | dev | E5a NO₃-N > 10 | 1,180 | **0** | 0 | 0 | n/a |
| M5 NRSA | dev (LOEO OOF; 2008-09 + 2013-14) | E2a chloride > 230 | 3,946 | **102** | 0.026 | 88 locations (3,112); 8 of 9 ecoregions | F |
| M5 NRSA | dev | E2c cond > 300 | 3,892 | 1,973 | 0.507 | 1,575 locations; 9 ecoregions | F (benchmark regional) |
| M5 NRSA | dev | E5b TP > 100 µg/L | 3,946 | 1,409 | 0.357 | 1,192; 9 | F |
| M5 NRSA | dev | E6 pH out of 6.5–9 | 3,845 | **82** | 0.021 | 74; 9 | F |
| M5 NRSA | dev | E3 turbidity > 50 NTU | 3,893 | 263 | 0.068 | 248; 9 | F |
| M5 NRSA | dev | E5a NO₃+NO₂ > 10 | 220 | 55 | 0.25 | 55; **3** ecoregions | F on locations; only 220 labelled |
| M5 NRSA | dev | NRSA "Poor": TP / TN / salinity / benthic MMI | 3,730 / 3,730 / 3,676 / 3,730 | 1,995 / 1,679 / 286 / 1,676 | 0.53 / 0.45 / 0.08 / 0.45 | 1,654 / 1,378 / 239 / 1,416 locations | F |
| M5 UK | pool_fit | E2a / E5b / E6 | 26,496 / 7,766 / 52,549 | 481 / 3,402 / 378 | 0.018 / 0.44 / 0.007 | 170 / 1,086 / 212 sites | training |
| M5 UK | **pool_val** | E2a chloride > 230 | 1,448 | **4** | 0.003 | 3 sites | **U** |
| M5 UK | pool_val | E6 pH | 4,025 | **10** | 0.002 | 6 sites | **U** |
| M5 UK | pool_val | E5b TP > 100 µg/L | 535 | 273 | 0.51 | 85 sites | F |
| Fuse D | dev (26 sites, 5 domain folds) | `ev__H5b_chl_bloom__onset_h7` | 17,461 | 308 (issue rows) | 0.018 | 20 sites; 13 domains (15) | F |
| Fuse D | dev | `ev__H1_hypoxia__onset_h7` | 38,181 | 235 (issue rows) | 0.006 | 12 sites; **9** domains | **U** (domains) |
| Fuse D | dev | `ev__H1_hypoxia__any_h7` | 41,675 | 3,141 | 0.075 | 14 sites; 9 domains | **U** (domains) |
| Fuse D | dev | `ev__H4_nitrate__any_h7` | 31,231 | **0** | 0 | 0 | n/a |
| Fuse C | dev | E4b microcystin > 8 | 51,031 | 1,987 | 0.039 | 611 sites (7,855); 68 HUC4 (150) | F |
| M6 track N | val-role stations, WY2022–23 | E1 grab DO < 3 | 16,520 | 491 | 0.030 | 124 stations; **9** basin groups (12) | **U** (clusters) |
| M6 track N | train-role stations, WY2022–23 | E1 grab DO < 3 | 75,576 | 6,412 | 0.085 | 1,206 stations; 34 basin groups | training |

Notes on [S2]:
- **NRSA `acid_cond` has no values** in the development surveys (0 labelled), so the ANC/acidification
  class cannot be used.
- **NRSA field DO < 3** (134 / 3,838) exists, but M5 does not predict DO and a single field reading is an
  acute snapshot. Not used.

### 3.3 External event labels: what matching would add [S3, S4]
**Matching rules** (fixed now; sensitivities in brackets).
- **Fish kill or NRC spill → M1/M6 sonde site.**
  - The event point is ≤ 10 km straight-line from a train/val-role stream site [5, 25 km].
  - That site's daily record (first..last of DO, SC, turbidity or temperature in `sonde_daily/index.parquet`)
    spans the event date.
  - Cluster = basin group.
  - Planned refinement: NLDI same-flowline, upstream ≤ 10 km / downstream ≤ 25 km.
- **HAB report or toxin sample → Orbit/Lite lake.**
  - The point is inside or ≤ 500 m from a HydroLAKES polygon (Orbit `LAKE_MAX_DIST_M`).
  - Role and time block come from the frozen Orbit v3 rule (`sentinel.models.orbit.split.assign`),
    computed without writing Orbit's cache.
  - Image matchup window ±3 d; S2 era ≥ 2016.
  - Clusters = lake and 50-km block.

| Catalogue (eligible rows) | Target | Matched positives: train | Matched positives: val | Clusters | Verdict |
|---|---|---|---|---|---|
| Iowa fish kills (KillAlert train/val only, WY2008–2017; 181 events, 34 `is_env_wq`) | M1 sites, 10 km | 3 (2 env_wq) | **0** | 1 basin | Not usable as an M1 label |
| Delaware fish kills (63 events, 32 env_wq) | M1 sites, 10 km [25 km] | 15 (6 env_wq) [53] | **0** | 2 [3] basins | Not usable |
| NRC reached-water reports (40,112 train / 4,749 val, with coordinates) | M1 sites, 10 km | 1,521 in 50 basins | **241 in 31 basins** | — | Exploratory "spill signature" study only |
| … of which a structure-resolvable chemical (M4-scorable) | M1 sites, 10 km | 8 | **3** | 6 / 3 basins | **U** for M4 |
| NRC → fish kill (549 KillAlert train/val kills) | 0–7 d, ≤ 25 km | 0 | — | — | 0: NRC coordinates are sparse inland (136 of 7,602 Iowa and 154 of 3,676 Delaware reports have coordinates [S6]) |
| NRC → fish kill, same county, 0–7 d | — | 33 kills with a reached-water report: 11 anthropogenic vs 15 `is_env_wq` | — | — | No enrichment; not a valid E8 outcome label |
| NY HABs lake-years (S2 era) | Orbit lakes | train role × train years: 255 positive (96 lakes, 30 blocks) | **val role × val years: 48 (34 lakes, 11 blocks)** | — | Adds E4c positives at **new** lakes: only 0–4 overlap the Orbit dev lakes. **Listing-only: non-listed lake-years are not negatives.** |
| CA FHAB bloom reports ≥ Caution (S2 era) | Orbit lakes | 201 (47 lakes, 32 blocks) | **17 (6 lakes, 4 blocks)** | — | Val **U** |
| WA microcystin > 8 µg/L site-days (ToxinWatch train/val, < 2022, S2 era) | Orbit lakes | 190 (26 lakes) | **0** in val years. Val-role lakes in train years: 22 (7 lakes). | — | Val **U**. Rows of ToxinWatch's spent test may never be added (§4.2). |

**What this means.**
1. Fish kills (E9) and NRC spills (E8) **cannot be learned at model sites** from existing catalogues. The
   overlap with gauged sondes is a handful of events, all in the M1 train block. Every 2018+ kill is in
   KillAlert's untouched test block, and §15.5 item 7 bars it from any training set.
2. E9 stays D3 KillAlert's task. The event layer passes M1/M6 event probabilities to KillAlert as
   features (its §9 iteration plan already names "upstream M1/M6 predicted DO/temperature").
3. **The only external labels that add useful positives are HAB reports for M2/M7 (E4c).** They come
   with three caveats:
   - they require new S2/NAIP matchups at report lakes and dates;
   - they are positive-only, so negatives must come from in-situ chl-a ≤ 20 rows or from report-programme
     lakes with no listing in a monitored year (NY lists by lake-year);
   - CA advisories are partly visual, so they are circular with imagery.
   Pre-register them as a **secondary** label set, never the primary.

---

## 4. Integration with frozen work

### 4.1 Files that must not be touched, and where event code goes
- **Hashed by `FREEZE_ew.json`:**
  - `sentinel/eval/{metrics,bootstrap,early_warning,hazards,ew_scoring,ew_controls,chloride,splits}.py`
  - `sentinel/models/sonde/{evaluate,data,common,features,subdaily,events,guard,runner}.py`
  - `sentinel/data/{sonde_splits,sonde_daily}.py`, `sentinel/models/baselines/sonde_baselines.py`
  - `scripts/train/sonde_events.py`, `configs/sonde/{common,events}.yaml`
  - `sonde_split.json`, the frozen `event_calibration.json` and its 90 weights
  - data content hashes of `sonde_daily`, `sonde_hourly`, `forcings_daily`, `chloride_derived`
  - `processed/sites/sites.parquet`, `processed/catchment/streamcat_sites.parquet`
- **Hashed by `bioassess_freeze.json`:**
  - `sentinel/eval/{__init__,splits,bootstrap,metrics}.py`
  - all `sentinel/models/bioassess` modules listed there, and `scripts/eval/bioassess_final.py`
  - the NRSA/UK data files
  - the ChemTox modules `sentinel/models/chemtox/{__init__,evaluate,features,tabpfn_model}.py`
- **M4:** sealed prospective predictions in `sentinel_runs/chemtox/prospective_test/` (0444, SHA256SUMS)
  and the unseen-taxa pins. `severity.py` is not in a freeze list, but PAF/RQ extensions still go in a
  **new** module (`sentinel/models/chemtox/severity_events.py`) so that M4's code is never edited.
- **`FREEZE_downstream_*`:** `sentinel/models/downstream/{common,crossfit,testlook,weather}.py`, the
  per-app modules, splits and data pins. This includes `downstream_d2n_microcystin_split.json` and
  `events_wa_toxic_algae.py` via the ToxinWatch freeze.
- **Glob-hashed (adding a file changes the hash):**
  - `sentinel/models/{orbit,lite,hydromet}/*.py`
  - `configs/{orbit,lite,hydromet}/*.yaml`
  - `scripts/{train,eval}/{orbit,lite,hydromet}_*.py`
  - **Never create event files matching these globs.**
- **Where event code goes:**
  - `sentinel/eval/events.py`
  - `sentinel/events/<workstream>.py`
  - `scripts/eval/events_<workstream>.py`
  - `configs/events/*.yaml`
  - one protocol section per workstream, e.g. `docs/preregistration/events_<ws>_protocol.md`
  - its own `FREEZE_events_<ws>.json`, which pins the parent model's freeze record by sha.

### 4.2 Test blocks: spent vs available, and where a new block is needed
Spent = a test look has been scored or a single-pass marker has been claimed. Source: [S5], registry rows
whose split/experiment marks a test look, and markers on disk. §15.5 item 7 forbids adding any spent
(or frozen) test-block row to a training set.

| Workstream | Spent / frozen test blocks | Event-track test plan |
|---|---|---|
| **M1 / EW-v2** | **Test-A (test-role sites × WY2024–25): EW single pass claimed 2026-11-09 17:05 UTC.** Evidence: registry `ew_v2_test_score`/`pass_claimed`, marker `sentinel_runs/sonde/.ew_scored_test.json`. Test-B (WY2026) is reserved for the EW pooled H1 pass (`test_ab`, not before 2027-06-30). | **E1a, E2a/b, E5a:** no new test on Test-A or Test-B (same labels as H1/H2/H4). M1-derived versions go to the **live prospective Test-C** (from Nov 2026, registered separately). **E1b, E3a/b, E6, E7:** test-role sites × Test-B (WY2026, Approved after 2027-06-30), own single-pass marker. Disclosure: Test-A inputs for these variables were loaded by the EW and (later) M1 scorers but never scored for these events; using Test-B avoids the question. |
| **M6 HydroMet** | None scored. Not frozen (candidate only). | Freeze the event track **before** `M6/test`. E7-coldwater and E1b on track R test folds × Test-A in a separate look `M6E/test`. **E1a (DO-min < 3) cannot use Test-A**: those labels were scored by the EW pass. Use Test-B, together with M1. |
| **M2 Orbit** | None scored (`orbit_power_*` rows are "counts only"). Not frozen. | Add the event track to the Orbit freeze (Track E already exists). Same headline cell (test role × 2024–25) and track U. Event metrics are scored in the same look or an immediately following marker. **No new block needed if frozen first.** |
| **M7 Lite** | None scored. `FREEZE_lite_candidate.json` is a candidate (scoring-code sha PENDING). | Add E4a (and E3 on the national track) before the final freeze. Test-A, Test-B, `test_n`, `test_time_n`. |
| **M3 Microbe** | None scored. 2024+ holdout and the pre-2017 DP1.20141.001 reserve are unscored. | Add before the Microbe freeze: E5b, E4a, E2c only. The U events stay descriptive. |
| **M4 ChemTox** | **ADORE test spent (2026-10-24). Prospective ECOTOX spent (2026-11-09).** Unseen-taxa frozen, not scored. | **New block needed:** NRC reports after the event-track freeze date (prospective, e.g. 2026-12-01 → 2027-11-30), with RQ computed blind. There is no ground truth for severity, so the endpoint is descriptive (coverage of HC5 intervals vs measured ECOTOX HC5 where available). |
| **M5 BioAssess** | **All spent:** Test A/B (2018-19), C (2023-24), T (incl. UK `reserved_test`), scored 2026-11-09. NRSA 2023-24 was also scored by D5 BioIntegrity. | **New block needed.** Options: (a) the M5-X tracks (lakes / England) if the event protocol is frozen before M5-X's freeze; (b) prospective UK EA samples dated after the snapshot; (c) NRSA 2028-29. The NRSA dev-only CV numbers above are validation only. |
| **Fuse** | None scored. Splits frozen; no `fuse_freeze.json`. | Add the §6 combination rule and the event endpoints to the Fuse freeze. Tier D test sites (D03/D07/D13/D14), Tier C Test-S/Test-R. No new block. |
| **D3 KillAlert** | Test 2018–2026 untouched, not frozen. | Not an event-layer test. **Its test events are barred from every event-track training set.** |
| **D2 ToxinWatch** (+ D2n split hashed in its freeze) | **Spent** (2026-11-08/09). | WA rows of test lakes or ≥ 2022, and WQP microcystin 2021–25, are excluded from all event training. |
| **D6 MusselGuard, D1n, SafeSwim, D5 BioIntegrity** | **Spent** (registry rows 2026-11-09). | E10/E11 are not re-tested by the event layer. Their outputs are read-only Fuse inputs. |

---

## 5. Sequencing and effort
Effort is in agent-days; compute is CPU unless noted.

| Phase | Work | Effort |
|---|---|---|
| **0 (shared, first)** | **Done 2026-11-09:** `sentinel/eval/events.py` v1 (released, 0444) + `tests/test_events.py`; S1–S6 promoted to `scripts/eval/events_feasibility_*.py` and registered (`events/feasibility`); "Event layer" section in `docs/CONTRIBUTING_V2.md`. **Open:** external-label datasheets; this document → per-workstream protocol sections. | 3–4 d (≈ 1 d left) |
| **1 (unfrozen workstreams, before their freeze)** | **Orbit:** derived + direct E4a; S2 CI analogue; HAB-report secondary labels (new matchups at about 100 report lakes). | 5–7 d (+ S2 chip pulls; GPU only if the head is retrained jointly) |
| | **Lite:** E4a/E3 derived + head. | 3 d |
| | **HydroMet:** predictive distribution (quantile residual model), E1b/E7 on track R. | 4–5 d |
| | **Microbe:** E5b/E4a/E2c derived + head. | 2–3 d |
| | **Fuse:** §6 stacker + leave-one-modality-out. | 4–5 d (GPU optional) |
| **2 (frozen or spent workstreams)** | **M1:** E1b/E3/E6/E7 labels, derived from frozen M1 quantile forecasts once `FREEZE_sonde` exists, plus direct heads (reuse the EW event-GBM code by import), Test-B plan. | 5–6 d |
| | **M5:** derived from existing CV predictions (no retraining for derived); direct head; choose the new test block. | 3–4 d |
| | **M4:** `severity_events.py` (PAF = SSD CDF at PEC, P(RQ > 1)); NRC × gauge PEC table (amount_in_water_kg / Q). | 4–5 d |
| **3** | Event-track freezes, single looks in each workstream's order; the Fuse combination look last. | 1 d per workstream |

**Critical dependencies.**
- M1's derived route needs the M1 forecasting freeze.
- CyAN needs an Earthdata login from the user.
- M5 and M4 have no confirmatory test until a new block exists.
- Test-B for M1/M6 cannot be scored before 2027-06-30.

---

## 6. How Fuse combines per-model event probabilities
1. **Inputs.** For each site-day s,t and event E, model m supplies:
   - p_m(E | s, t) on the logit scale, already recalibrated on that model's own validation cell (§2.2);
   - an availability flag a_m (0 if the modality did not observe, e.g. no S2 overpass, no 16S sample in
     the integration window, no benthic sample in the season);
   - an age Δt_m since the modality's last observation.

   Chronic modalities (M3, M5) and the HAB-report and KillAlert outputs enter as slowly varying priors,
   carried forward over their integration window and down-weighted by age.
2. **Combiner (pre-registered primary).** Missing-aware logit stacking:

       logit P_fuse(E) = β₀(E, DOY) + Σ_m a_m·[w_m(E)·logit p_m + γ_m(E)·g(Δt_m)] + Σ_m (1 − a_m)·δ_m(E)

   - Fitted out-of-fold over the Fuse dev domain folds only.
   - L2-penalised, with w_m ≥ 0 (monotone in each model's evidence).
   - β₀ is the event climatology, so a site with no modality falls back to climatology.

   **Secondaries:** (a) the existing Fuse direct head (`ev__*` BCE) on fused embeddings; (b)
   reliability-weighted log-odds pooling with fixed weights from validation Brier (no fitting); (c) the
   best single modality.
3. **Event-type composites.** "Any harmful event" = 1 − Π_E (1 − P_fuse(E)) is reported only as a
   descriptive dashboard quantity. Cross-event dependence (e.g. E4a → E1a via night-time DO sag) is
   handled by letting the stacker see the other events' p_m (a one-step event graph, fixed a priori:
   E4a, E5b → E1a; E3a → E4a suppression), not by learned arbitrary graphs.
4. **Attribution and controls (as Fuse P3/P4).**
   - Leave-one-modality-out ΔBrier per event.
   - Exact Shapley over the available-modality subsets.
   - A shuffled-modality control (permute p_m across sites within date). Fusion must beat it and the best
     single modality on paired rows, with CI excluding 0.
5. **Power reality at NEON.** From [S2], E1a has 9 dev domains with positives (**U**) and E4a has 13
   (F); E5a has none. Fuse's event claim therefore rests on **E4a (Tier D)** and **E4b microcystin
   (Tier C: 1,987 positives, 68 HUC4)**. E1a fusion is reported as descriptive unless test-domain counts
   meet the rule.

---

## 7. Top recommendations
1. **Build `events.py` once and import it everywhere.** E-codes that equal EW hazards delegate to
   `hazards.py` verbatim. Never re-export it from the hashed `sentinel/eval/__init__.py`.
2. **Claim events only where the counts support it:**
   - M1/M6 E1a, E1b, E3a/b, E6, E7-coldwater;
   - M2/M7 E4a (state, not onset);
   - M3 E5b/E4a;
   - M5 chronic E2a/E5b/E6/E3 and NRSA "Poor" classes;
   - Fuse E4a/E4b.

   Everything flagged **U** is descriptive only: E2/E5a on sondes, E3 on imagery, M3 DO/turbidity,
   UK chloride/pH, Fuse E1a by domain, track N.
3. **Derived route primary, direct head challenger,** both pre-registered, with the paired comparison of
   §2.2 and mandatory recalibration and reliability checks. The direct head is skipped for U events.
4. **Do not use fish kills or NRC spills as model-site labels.** The matched counts are near zero, and the
   2018+ kills are KillAlert's test. Route E9 through KillAlert and E8 through M4 severity (descriptive,
   with a prospective NRC block).
5. **Freeze event tracks into the Orbit, Lite, HydroMet, Microbe and Fuse freezes before their first
   test look.** That avoids new test blocks. M1 (Test-A spent), M4 and M5 (all spent) need new blocks:
   Test-B / Test-C, prospective NRC, and M5-X or prospective UK EA respectively.
6. **Ask the user for an Earthdata login** so CyAN CI-cyano can be the published bloom comparator (§15.1
   needs the strongest published method).

## 8. What could not be determined here
- **Distinct onsets for Fuse Tier D.** The table counts issue-row labels; distinct-onset counts need the
  Fuse label builder.
- **Real distributional calibration of each model's outputs.** This needs validation predictions, which
  this plan did not load.
- **Whether NY/CA report lakes have usable S2 matchups** (cloud, ice, lake size ≥ 3×3 S2 pixels). This
  needs a chip pull.
- **The exact E7 coldwater rule's agreement with state designated uses.** No designated-use layer is in
  the data root.
- **An NLDI flowline refinement of the fish-kill/NRC matching.** It was not run offline; distances are
  straight-line.
- **E10 pathogen for any M1–M7 modality.** No faecal-indicator labels exist at NEON or sonde sites.
- **Whether M5-X's splits still have an unscored test block** usable for an event track. M5-X is in
  development; check with its protocol owner before writing the M5 event protocol.
