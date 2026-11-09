# Datasheet — official MDN-STREAM Sentinel-2 MSI weights (`mdn_stream_weights`)

| | |
|---|---|
| **Source** | MDN-STREAM, https://github.com/STREAM-RS/MDN-STREAM, folder `Weights/MSI/`. This is the production MDN behind NASA STREAM, from the lineage of Pahlevan et al. 2020 and 2022 |
| **Licence** | GPL-3.0 (repository). Used only as a research baseline; not redistributed |
| **Retrieved** | 2026-10-25 ~02:00 UTC via raw.githubusercontent.com |
| **Local path** | `$SENTINEL_DATA/raw/mdn_stream_weights/MSI/*.zip` (10 MB), extracted to `MSI/x2025/` |
| **Code** | `sentinel/models/orbit/mdn.py::OfficialMDN` |

## What is used
- **Zip `1b79098d…`:** 10 rounds, each a Keras `trained_model.h5` plus a `config.json`.
  - Network: 5 × 100 ReLU layers and a mixture layer (5 components; joint chl / TSS / CDOM with full
    covariance).
  - Inputs: Rrs (sr⁻¹) at 443, 490, 560, 665, 705, 740 and 783 nm, scaled with RobustScaler.
  - Targets: ln, then MinMax(−1, 1).
  - Training metadata (from its `config`): dataset `all`, training shape (2435, 7).
- **Reading.** The files are read as data with `h5py` and `json`. Nothing from the archive is
  executed. The second zip (`ba787bde…`, TF1 checkpoints with *pickled* configs) is **not** loaded.
- **Estimate.** Each round gives the mean of its highest-prior component; the ensemble value is the
  median over rounds.
- **Sanity check** (computed; manifest `mdn_stream_weights.stats`). On GLORIA S2A-simulated Rrs, chl-a
  MdSA is ≈ 28%, TSS ≈ 35% and CDOM ≈ 43%, matching published MSI values. This is **not independent**,
  because the MDN training data overlap GLORIA.
- **Use in SENTINEL-Orbit.**
  - Input: L2A water-median ρ/π for B01–B07.
  - Output: chl-a and TSS only.
  - The model is never scored on GLORIA. Overlap with our in-situ lakes, if any, favours this baseline.
