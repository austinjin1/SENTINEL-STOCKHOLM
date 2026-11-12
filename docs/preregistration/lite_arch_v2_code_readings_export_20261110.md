# M7 SENTINEL-Lite, architecture round v2 — code readings, part 2: anchor file and multi-input export

Written Tue 10 Nov 2026 20:04 EST by the M7 workstream agent, before the lead's stamp. Second notes file of the
round; the first is `lite_arch_v2_code_readings_20261110.md` (sha256 `002d5d08…a4d4`), which is not changed.

Everything below was measured with **random weights and random tensors, or with synthetic tables**. No real row was
read, no v2 model was fitted, no validation row was loaded. This is not a deviation record.

Code: `sentinel/models/litev2/anchor_export.py`, `sentinel/models/litev2/export.py` (both outside every frozen hash
list). Tests: `tests/test_litev2_anchor_export.py` (6), `tests/test_litev2_export.py` (9).

## 1. Anchor file (§4.6: "compact float32 format written by our exporter; parity … within 1e-4 log10")

| # | Reading |
|---|---|
| X1 | One file holds every tree of the anchor (four per-target models + the pooled turbidity model, three seeds each). A prediction is the sum of a group's tree outputs divided by the number of seeds. |
| X2 | Per internal node: float32 threshold, uint8 feature index, two int8 children, one flag byte (default side, missing-value type). Per leaf: one float32. Missing values and zeros follow LightGBM's decision rule. |
| X3 | A threshold is stored as the largest float32 not above LightGBM's double threshold. For a float32 feature value the decision is then exactly LightGBM's. |
| X4 | **Parity reference.** The deployable feature code works in float32, so the unit test compares the file with LightGBM on the same float32 feature values. Result on synthetic tables: max difference 5e-8 log10. |
| X5 | The anchors used in training and validation come from LightGBM on float64 features. Against those, a row whose feature lies within float32 rounding of a threshold can differ (synthetic: 0.03 % of values differ by more than 1e-4, max 0.003). This difference is reported with the export, next to the INT8-vs-FP32 difference. |
| X6 | "All tree thresholds and leaf values" = internal nodes + leaves of the file. With the frozen `GBM_PARAMS` and full trees that is 5 × 3 × 400 × (14 + 15) = 174,000 numbers and 1.04 MB (synthetic fit; plan: about 0.2 M, about 1.1 MB). |

## 2. Multi-input wrapper (§4.6: "(chip, s, c, a) … Export recipe … as in the frozen protocol")

| # | Reading |
|---|---|
| X7 | One ONNX file per seed. Inputs: `x01` (DN / 255), `sc` (raw [s, c]; a missing value may be NaN and becomes the train mean in the graph, as in training), `anchor`, and three inputs made by the feature code from the pixels: `pool_w` (frozen centre-window weights), `water` (water fraction per output cell), `gsd`. An arm that does not use an input does not have it (A1: no `anchor`; A3: no `sc`; A5: no `water`, `gsd`). |
| X8 | Output: log10 μ and log σ per target. The auxiliary KD map head is not exported (3,320 parameters; §4.3 "not used at inference"). Exported network parameters of ANCHOR-Lite: 1,401,296. |
| X9 | Attention weights in the graph: exp(logit − max over the chip's water cells) × mask ÷ sum. This equals the masked softmax of the training code (torch vs wrapper < 1e-5; torch vs ONNX FP32 < 1e-4, tested at two image sizes) and puts no −inf in the graph. |
| X10 | MACs by the frozen counter: 1.08 GMACs per 512² tile (does not depend on the weights). |
| X11 | Calibration: the frozen 2 × 2 mosaics of train chips (`lite.export.calibration_tiles`, n = 32, seed 0), each with the [s, c], anchor and GSD of one train row. |
| X12 | The frozen `quant_pre_process` call is used unchanged (no dimension merging); the wrapper reshapes the per-cell inputs with the map's own shape so that call can infer every shape. |

## 3. Decision asked of the lead — INT8 scope (touches "Export recipe … as in the frozen protocol")

The frozen recipe quantised a single-input convolutional network; the pooling was float32 code outside the file.
The v2 file also holds the context MLPs, the pooling and the residual heads, and the anchor enters as a float input
that is added to the residual at the very end. Two readings are implemented; one must be fixed before a trained v2
model is quantised.

| | `scope="image"` (the code's default) | `scope="graph"` |
|---|---|---|
| What is INT8 | Input normalisation, encoder, decoder (all but about 0.08 M parameters, nearly all MACs) | Every node the quantiser supports, including the heads and the final "anchor + residual" sum |
| Recipe call | The frozen call, restricted to those nodes | The frozen call, as is |
| File size, ANCHOR-Lite, random weights | 1.99 MB | 1.81 MB |
| Zero-initialised model (μ must equal the anchor) | exact (unit test) | not exact |
| Error added by quantising the head arithmetic alone (residual held constant, random inputs) | 0 | median 0.005 log10, max 0.02 |

With the anchor file the pipeline is about 3.03 MB (image scope) or 2.85 MB (graph scope) against the 4 MB budget,
and about 1.58 M parameters against 3 M. Both scopes fit.

**My recommendation: `scope="image"`.** The 8-bit grid of the last sum is about 0.02 log10 wide, which adds an error
of about 1 % to every prediction for a saving of 0.18 MB. It is also the closer analogue of the frozen pipeline,
where everything after the map was float32. The choice is made here on random weights only. If the lead prefers the
literal reading, `scope="graph"` runs with no code change.

## 4. Proposed meaning of "a passing export-parity test of the multi-input INT8 pipeline of seed 0" (§7)

Stated now, before any trained model exists, for the lead's ruling:

1. ONNX FP32 vs the training code on every national validation row and every `val_p2` row: max |Δ log10 μ| < 1e-4.
2. Anchor file vs LightGBM on float32 features, same rows: max |Δ| < 1e-4 log10.
3. The INT8 file returns a finite μ and σ on every one of those rows.

The INT8-vs-FP32 metric difference on validation is **reported** (frozen §9), not a pass condition, as in the frozen
protocol. Budgets are checked separately (§4.6).

## 5. Not written yet

The export script that puts these pieces together on a trained seed-0 model (after the stamp, when one exists), the
pipeline latency including feature code and trees, and an export wrapper for ANCHOR-Set (needed only if ANCHOR-Set
is the P2 selection).
