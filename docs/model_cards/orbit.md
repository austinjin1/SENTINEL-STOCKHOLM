# Model card — SENTINEL-Orbit (M2): SatWQ-Net, Sentinel-2 L2A optical water quality

**Status:** development. No test-set result exists. Every metric on this card is either `[PENDING]` or a
registry entry (`results/registry.jsonl`, workstream `M2-orbit`).
- Protocol: `docs/preregistration/orbit_protocol.md`.
- Split rule v3: `docs/preregistration/orbit_split_v3.json`
  (sha256 `ffe288f2fafecada964fc8125dcd35ddd9b332d3a04f782d0298e01f097a5e32`).

## Intended use
- Estimates chl-a (µg/L), TSS (mg/L), turbidity (NTU/FNU) and Secchi depth (m), with predictive
  distributions, from Sentinel-2 L2A chips over inland water.
- Acts as the teacher for M7 SENTINEL-Lite.
- **Not for:** phycocyanin, CDOM, any non-optical variable, or spills.

## Model
- **SatWQ-Net** (`sentinel/models/orbit/satwq.py`):
  - spectral-token Transformer with SRF-centroid wavelength tags;
  - attention pooling over up to 64 SCL-water pixels;
  - geometry branch, plus a context branch (DOY, location) with whole-vector dropout;
  - Gaussian-mixture head, 5 components per target, in log10.
- **Training:** pretrained on GLORIA (train-role water bodies, ≤ 2021; ρ = π·Rrs, with
  atmospheric-correction-style augmentation), then fine-tuned on S2 × in-situ matchups (train-role
  water bodies, ≤ 2021).
- **Evaluated model:** an equal-weight seed ensemble.
- **Parameters:** base configuration (d = 64, 3 layers) is 182,780 parameters, computed from the model
  (`n_params` in TRAINING_LOG.md).

## Teacher interface for M7
- **Current teacher (development, 2026-10-25):** the SatWQ-Net base configuration, a 5-seed ensemble
  (`satwq_base_s0..s4`) trained on the fit cell (train role × ≤ 2021) of dev_v3.

  | Seed | Checkpoint | sha256 |
  |---|---|---|
  | s0 | `$SENTINEL_RUNS/orbit/satwq_base_s0/model.pt` | `634354e903f289692314a1f812207f8ab09a577c2e4a192a764e939d1d788961` |
  | s1 | `$SENTINEL_RUNS/orbit/satwq_base_s1/model.pt` | `6bc5cb30f2934f82b83b34b85241067fa720b8365b8aa0c0d2623dae4f4fbe22` |
  | s2 | `$SENTINEL_RUNS/orbit/satwq_base_s2/model.pt` | `125c4a261d71a2f1e2136d2a9eb6d81b4fa5848dd47ea0fb999592cd15ae90c8` |
  | s3 | `$SENTINEL_RUNS/orbit/satwq_base_s3/model.pt` | `0522e40ca23925a49d3567c1c74208eb4ddd040ae29df24c404835371ee6c348` |
  | s4 | `$SENTINEL_RUNS/orbit/satwq_base_s4/model.pt` | `f6d228d5fcb2974b09f9d37f279e00051b1c802f691d878426541794c548f8c8` |

  The manifest `docs/model_cards/orbit_teacher.json` holds these paths and hashes;
  `OrbitTeacher.from_manifest()` verifies them. If a later iteration selects a better configuration on
  validation, the manifest is rewritten and M7 should re-read it.
- **Precomputed predictions:** `$SENTINEL_RUNS/orbit/teacher/naip_teacher_teacher_preds.parquet`, one row per
  `processed/s2_chips/naip_teacher` chip (2,130 chips; sha256 in the manifest).
  - Each row has the orbit split columns `orbit_role/block/waterbody/in_lake/tblock/buffer_excluded`, so the
    student can hold out the same lakes.
  - Chips without a valid SCL-water pixel in the centre 15 × 15 have NaN predictions.
- **Python interface** (`sentinel/models/orbit/teacher.py`):
  ```python
  from sentinel.models.orbit.teacher import OrbitTeacher
  t = OrbitTeacher.from_manifest(device="cuda")                 # or "cpu"
  df = t.predict_store(paths.processed("s2_chips") / "naip_teacher")   # one row per chip
  maps = t.predict_pixels(chip, index_row)                       # {target: [64, 64] log10 median}; NaN off water
  ```
- **Columns per target** `t ∈ {chla, tss, turbidity, secchi}` (log10; units µg/L, mg/L, NTU/FNU, m):
  - `{t}_log10_q16/q50/q84/mean/var`: all 12 bands, with context (DOY, location);
  - `imgonly_{t}_…`: all bands, context off. **Recommended distillation target**: it carries no location
    prior;
  - `rgbn_imgonly_{t}_…`: B02/B03/B04/B08 only, context off. This is the drone-band mode of the same weights.
- **Per-pixel maps.** Every SCL-water pixel is predicted as a 1-pixel set. Training samples random pixel
  subsets, so this is in distribution. Spot check: over the centre window, the median of the single-pixel
  predictions agreed closely with the chip prediction (24 random chips). Expect real spatial heterogeneity
  across the full 640 m chip.
- **Inputs used.** Reflectance (harmonised DN × 1e-4), SCL, sun/view angles, AOT, JRC distance to land, and
  the chip-index lat/lon/sensing time. In-situ labels are never used.

## (B) Event output: algal-bloom state (orbit_protocol.md §12, v0.4, 2026-11-09)
- **E4a algal bloom** (chl-a > 20 µg/L; tiers > 12 and ≥ 50 µg/L), in-lake rows, state only (no onset: S2 revisit
  ≥ 5 d and cloud gaps). Primary output: P(E4a) = exceedance of the SatWQ-Net seed-ensemble mixture, Platt-recalibrated
  on the val cell (frozen). Challenger: a logistic head on the frozen fused embedding.
- **E3 sediment plume** (TSS > 80 mg/L, turbidity > 50 NTU): exceedance probability reported for display only;
  0 validation positives, so no skill claim is possible.
- Code: `sentinel/events/orbit.py` (adapter), `sentinel/eval/events.py` (shared, v1.0); artifact
  `events_orbit_<version>.pkl` (calibrators, direct heads, comparator classifiers), sha-pinned in the freeze.
- Not for: cyanobacteria-specific detection (S2 has no 620 nm band; no phycocyanin), toxin presence, onset timing.
  CyAN CI-cyano was not compared (needs an Earthdata login the user has not provided).
- Validation results: registry experiment `orbit_events_val_<version>` (development, not claims).

## Evaluation
- Headline cell: unseen lakes × 2024–2025. Not yet scored: `[PENDING]`.
- Validation (development) results are in the registry under experiments `orbit_val_dev_v3` (fit = train role × ≤ 2021)
  and `orbit_val_dev_v3_fit-recent` (iteration cycle 2). They are
  development numbers, not claims.

## Turbidity units (protocol v0.3, 2026-11-08)
- From v0.3 the turbidity target is **NTU family** (NTU and ratio NTRU) plus AquaCorr rows whose unit is
  unstated. WQP FNU values are never fitted. They are kept as `turbidity_fnu` and reported as a separate
  stratum (`data.apply_turbidity_unit_policy`).
- The **current teacher** (dev_v3, 2026-10-25) saw only AquaCorr turbidity, with the unit unstated. Its
  `turbidity_*` columns are therefore "NTU or FNU, unit unstated". The M7 agent flagged this issue. Treat
  them as NTU-like and do not compare them with FNU sonde values.
- A retrained teacher will get a **new versioned manifest**; `orbit_teacher.json` is left unchanged.

## Limitations
- AquaCorr in-situ units are unstated.
- The AquaCorr chip extraction is incomplete and US-only.
- Turbidity and TSS have few validation rows.
- Mixed S2 processing baselines.
- S2C uses S2A band centres.
