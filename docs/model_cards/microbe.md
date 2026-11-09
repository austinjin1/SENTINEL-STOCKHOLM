---
model_id: sentinel_microbe
name: SENTINEL-Microbe
modality: microbe
registry_models: ["SeqSet[seqset_dnabert]", "SeqSet[seqset_kmer]", "SeqSet[seqset_id]", "SeqEmb-Ridge[DNABERT-S]", "SeqEmb-RF[DNABERT-S]"]
registry_experiments: [microbe_phase1_cv, microbe_phase1_compare, microbe_phase1_controls]
workstream: M3
version: 0.1.0
status: in development
license: MIT
intended_use: >
  Research model estimating the co-located water quality (TN, TP, DOC, nitrate, specific
  conductance, turbidity, dissolved oxygen, chlorophyll-a) of a freshwater site from a single
  surface-water 16S rRNA amplicon community sample, and producing a primer/release-agnostic sample
  embedding (DNA-encoded sequences) for SENTINEL-Fuse. Intended for slowly varying site-state
  description at monthly/bimonthly cadence.
limitations: >
  Phase 1 only: cross-validated on NEON development events (2018-08 to 2023-12); the final tests
  (2024 time-out; pre-2017 V3V4 cross-release reserve) have not been scored. 16S is sampled monthly
  (streams) or every other month (lakes/rivers): NOT an early-warning model. Communities identify
  their site almost perfectly (site-identity probe), so only domain/site hold-out scores are
  meaningful. Expected skill is modest; see the registry before using any prediction.
---

# Model card — SENTINEL-Microbe (M3)

**Status: phase 1 (implemented; development cross-validation only).** No test score exists. This
card contains no metrics: all numbers are registry entries (workstream `M3-microbe`; experiments
`microbe_phase1_cv`, `microbe_phase1_compare`, `microbe_phase1_controls`, `microbe_encoder_checks`).

## Model details

Two DNA-encoded model families (protocol `docs/preregistration/microbe_protocol.md` §4.1, §6):

- **SeqSet** (`sentinel/models/microbe/seqset.py`): a set transformer over the 256 most abundant
  sequences of an event. Each token = projection of a frozen, label-free sequence vector
  (DNABERT-S mean-pooled last layer, 768-d; or 5-mer profile, 1,024-d; or a learned id embedding as
  the no-DNA ablation) + four abundance features (log count, relative abundance, its square root,
  within-sample centred log abundance). 2 pre-LN self-attention blocks (d = 128, 4 heads), PMA ⊕
  abundance-weighted mean pooling, sample descriptors (log reads, log richness, primer one-hot)
  after pooling, shared trunk, 8 linear heads, masked MSE on training-fold-standardised targets;
  mean of 3 seeded networks.
- **SeqEmb** (`sentinel/models/microbe/baselines.py`): relative-abundance-weighted mean of the
  frozen DNABERT-S vectors of all sequences in the event → Ridge (alpha by site-grouped inner CV)
  or random forest.
- **DNA encoder:** DNABERT-S (Zhou et al. 2024, Apache-2.0, `zhihan1996/DNABERT-S` revision
  `00e47f96…`), re-implemented in plain PyTorch (`sentinel/models/microbe/dna.py`) because the
  authors' remote code does not instantiate under transformers 5.x; parity with the reference
  modules is checked by `scripts/eval/microbe_dnabert_parity.py`. fp32 inference.

Because tokens are described by their DNA rather than by an OTU identifier or taxonomy label,
both families apply unchanged to NEON's older V3V4 ASV pipeline (DP1.20141.001) — the basis of the
cross-primer / cross-release transfer test.

## Training and evaluation data

`microbe_wq` (NEON DP1.20141.002 RELEASE-2026 16S events joined with DP1.20093.001 grab chemistry
and DP1.20288.001 sonde daily means; `scripts/data/build_microbe_wq.py`, datasheet
`docs/datasheets/microbe_wq.md`), `microbe_wq_v1` (DP1.20141.001, 2014–2018, cross-release only).
NEON data are CC BY 4.0. Event = site × local collection date; chemistry joined on the same
`parentSampleID` (field visit), sonde = daily mean at the co-located station (S2 / buoy).

## Evaluation

Frozen split `docs/preregistration/microbe_split.json` (sha256 in the protocol §10). Development
schemes: leave-one-NEON-domain-out (primary; model selection), leave-one-site-out, cross-release
V4V5 → V3V4 (2017–2018 .001 events) and reverse, random 5-fold (leakage gap only). Baselines on
identical folds: global mean, Ridge/RF on DOY + location (tier 0), site mean + DOY (random scheme
only), Ridge/RF/XGB on genus CLR, RF on presence/absence, RF on CLR + context, TabPFN v2 on genus
CLR, and pre-registered tuned RF/XGB. Controls: within-fold permutation null; site-identity probe.
Metrics: pooled R², anomaly (within-domain) R², per-domain R², RMSE, Spearman, MSE skill vs tier 0;
site-cluster and domain-cluster bootstrap CIs (B = 2,000); paired differences with Holm.

## Limitations, risks and known failure modes

- Site identity is strongly encoded in community composition; random-split skill is not evidence
  of transfer.
- Primer set is confounded with time and bioinformatics pipeline in NEON; the cross-release test
  measures all three shifts together.
- Sequencing depth differs ~10× between NEON pipelines; depth/richness descriptors are disabled in
  cross-release schemes.
- Chloroplast 16S reads are kept (a phototroph signal for chlorophyll-a), so chl-a skill may partly
  reflect algal plastids rather than bacteria.

## Licences

NEON data CC BY 4.0 (attribution: NSF NEON, RELEASE-2026). DNABERT-S Apache-2.0. TabPFN 2.2.1
(baseline only): Prior Labs Licence 1.1.

## Provenance

Every CV entry stores git state, package hash, script sha256, config hash and the data manifests;
every training run is in `TRAINING_LOG.md` / `logs/training_runs.jsonl`.
