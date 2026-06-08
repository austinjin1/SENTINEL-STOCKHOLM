# Case-Study Testing Methodology

How SENTINEL was evaluated on real contamination events, and exactly which of
the 31 documented events were run through a model on real sensor data.

## Summary

SENTINEL's sensor encoder (**AquaSSM**) was run on **real USGS NWIS continuous
sensor records** for the subset of the 31 documented events that have a
continuous in-situ monitoring station at the event location. The remaining
events are real, documented incidents but **lack co-located continuous sensor
data**, so they could not be run through the model; they are included as
literature-documented context, not as model-tested results.

| | Count |
|---|---|
| Documented events (total) | 31 |
| **Run through AquaSSM on real USGS data** | **10** |
| No continuous co-located sensor station (literature only) | 15 |
| NEON sites (separate sensor pipeline) | 6 |

The 10 tested events are a **representative sample** of the 31 — they span the
same event types (HAB, hypoxia, salinity intrusion, nutrient/agricultural
runoff) and geography (Great Lakes, Gulf, West Coast, Mid-Atlantic, Midwest).

## Pipeline (per event)

1. **Data** — download USGS NWIS *instantaneous values* (IV) for the station
   over `[advisory − pre_window, advisory + post_window]`, parameters:
   DO (00300), pH (00400), specific conductance (00095), water temperature
   (00010), turbidity (63680).
2. **Preprocessing** — assemble the 6-channel AquaSSM input (pH, DO, Turb,
   SpCond, Temp, ORP=0), z-score with global WQ norms, forward/back-fill
   transient sensor gaps, build 128-step sliding windows (stride 64), require
   ≥30% valid coverage per window.
3. **Inference** — `SensorEncoder` backbone (`aquassm_full_best.pt`, val
   AUROC 0.94) → 256-d embedding → binary `AnomalyHead` → sigmoid probability
   per window.
4. **Scoring** — anomaly threshold 0.10; "first detection" = first pre-advisory
   window above threshold; lead time = advisory − first-detection time.

Code: `scripts/evaluation/reeval_case_studies_31.py` (reuses the inference
machinery in `scripts/experiments/exp1_case_studies_real.py`). Outputs:
`case_studies_v3_REAL_inference.json`.

## The 10 events tested on real data

| Event | USGS site | Records | Lead time (d), thr 0.10 |
|-------|-----------|---------|--------------------------|
| Lake Erie HAB 2023 | 04199500 | 7,199 | 59.3 |
| Gulf Dead Zone 2023 | 07374000 | 3,486 | 87.2 |
| Mississippi Salinity 2023 | 07374000 | 4,168 | 58.6 |
| SF Bay Heterosigma 2022 | 11447650 | 8,736 | 59.2 |
| Klamath River HAB 2021 | 11530500 | 7,200 | 59.2 |
| Delaware River Salinity 2022 | 01463500 | 2,181 | 57.3 |
| Iowa Nitrate Crisis 2015 | 05482000 | 7,196 | 59.3 |
| Green Bay Hypoxia | 040851385 | 24,892 | 59.2 |
| Saginaw Bay HAB | 04157005 | 8,703 | 59.3 |
| Hudson River HAB 2025 | 01359165 | 5,543 | 59.3 |

## Events NOT tested on real data (15)

No continuous co-located USGS sensor station with the required parameters:
Toledo 2014, Chesapeake blooms 2023, Grand Lake St. Marys 2009, Lake Erie 2015,
Lake Okeechobee 2016 & 2018, Utah Lake 2016 & 2018, Animas/Gold King 2015,
Neuse River, Jordan Lake, Chesapeake Hypoxia 2018, Clear Lake 2024, Tar Creek,
Lake Winnebago. These are documented from official advisories / literature.

NEON events (POSE, BLDE, MART, BARC, LECO, SUGG) use the NEON sonde pipeline.

## Interpretation and limitations (read before citing lead times)

- The lead times above are reported **at threshold 0.10**. At this threshold the
  anomaly head fires on ~100% of windows (mean probability ≈ 0.72 even at
  control sites), so the "first detection" tends to fall at the start of the
  observation window and the lead time tracks the chosen `pre_window` length
  rather than a measured precursor onset. These numbers should therefore be read
  as **"the detector was active throughout the pre-event window,"** not as a
  calibrated early-warning lead time.
- A defensible early-warning metric requires (a) an anomaly head calibrated for
  specificity (low false-positive rate on held-out clean sites at the *same*
  operating threshold) and (b) lead time measured as the first *sustained*
  exceedance above a per-site baseline. That re-analysis is future work.
- "Tested on real data" here means the AquaSSM **sensor** encoder only. The
  satellite, microbial, molecular, and behavioral encoders were validated
  separately on their own datasets; no event was run through all five modalities
  on real co-located data.
