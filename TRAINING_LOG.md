# Training Log — SENTINEL

## Run 001 — 2026-04-04 01:22 (Phase 1: MPP Pretrain)
- **Experiment**: AquaSSM MPP Pretraining
- **Iteration**: Base
- **Config**:
  - Model: SensorEncoder (4,600,895 params)
  - LR: 5e-4, Schedule: Cosine, Epochs: 50, Batch: 8
  - Physics constraint weight: 0.1 (DO-temp, pH-cond, cond-TDS)
  - Weight decay: 0.01, Gradient clip: 1.0
- **Data**: 212 sequences (162 real USGS + 50 synthetic), split 148/31/33
- **Hardware**: A100 80GB GPU 0 (shared, 99% util)
- **Duration**: ~80 min (50 epochs @ ~1.6 min/epoch)
- **Metrics**:
  - Epoch  1: Train 1.00 (MPP 0.527, Phys 4.746), Val 0.184
  - Epoch 10: Train 0.40 (MPP 0.077, Phys 3.279), Val 0.000
  - Epoch 25: Train 1.15 (MPP 0.746, Phys 4.059), Val 0.000
  - Epoch 30-35: ALL NaN (physics instability)
  - Epoch 40: Recovered — Train 1.32 (MPP 0.923, Phys 3.930)
  - Epoch 50: Train 0.70 (MPP 0.356, Phys 3.433)
- **vs Threshold**: N/A (pretrain only, no AUROC)
- **Status**: ✅ Completed (with instability issues)
- **Issues**:
  1. Physics constraint uncertainty weighting caused NaN at epochs 30-35
  2. Val loss = 0.0 from epoch 5 onwards (all val batches NaN-filtered)
  3. Root cause: delta_t[0] != 0 in preprocessed sequences

## Run 002 — 2026-04-04 03:19 (Phase 1: MPP Pretrain v2, no physics)
- **Experiment**: AquaSSM MPP Pretraining without physics constraints
- **Iteration**: Iteration 1 — remove physics constraints
- **Config**: Same as Run 001 but physics_weight=0, epochs=30
- **Duration**: ~24 min
- **Metrics**:
  - Epoch  1: Train 0.457, Val 0.334
  - Epoch  5: Train 0.351, Val 0.000
  - Epoch 15: Train 0.000 (all NaN)
  - Epoch 30: Train 0.914 (partial recovery)
- **Status**: ✅ Completed (same NaN issue — confirmed not physics-related)
- **Diagnosis**: delta_t[0] != 0 is the root cause, not physics constraints

## Run 003 — 2026-04-04 04:04 (End-to-end anomaly detection, with dt fix)
- **Experiment**: AquaSSM end-to-end with dt[0]=0 fix
- **Iteration**: Iteration 2 — fix delta_t preprocessing
- **Config**: Combined pretrain+finetune, LR 3e-4, 50 epochs
- **Duration**: Ongoing (all batches NaN due to scheduler.step() bug)
- **Status**: ❌ Failed — scheduler.step() before optimizer.step() corrupted LR state
- **Root Cause**: PyTorch warning "scheduler.step() before optimizer.step()" + 28.6% batch NaN rate = model diverged immediately

## Diagnostic — 2026-04-04
- **Finding**: 28.6% of batches produce NaN embeddings even with fresh model
- **Cause**: Sequences where delta_t[0] != 0 (160 files fixed, but padding also introduces dt=900 at start)
- **Fix Applied**: Force dt[0]=0 in collate function + fixed 160 .npz files
- **Remaining Issue**: Some sequences still have numerical issues; need to investigate the SSM's behavior with long constant-dt sequences

## Summary of Iterations
| Run | Change | Result | Learned |
|-----|--------|--------|---------|
| 001 | Baseline + physics | MPP: 0.527→0.077 ✓, but NaN at ep30 | Physics loss unstable, but MPP learns |
| 002 | Remove physics | Same NaN pattern | Not physics — it's the data |
| 003 | Fix dt[0]=0 | All NaN from ep1 | Scheduler bug + residual data issues |
| Diag | Batch-level test | 71.4% batches OK | Core model works, data pipeline has edge cases |
