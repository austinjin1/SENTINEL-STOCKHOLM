#!/bin/bash
# Post-training pipeline: runs all downstream analyses after AquaSSM full training.
# Run from /home/bcheng/SENTINEL
set -e
cd /home/bcheng/SENTINEL

echo "=== POST-TRAINING PIPELINE ==="
echo "Step 1: Copy aquassm_full_best.pt to aquassm_real_best.pt"
cp checkpoints/sensor/aquassm_full_best.pt checkpoints/sensor/aquassm_real_best.pt
echo "  Done."

echo "Step 2: Re-run NEON anomaly scan with new checkpoint"
python scripts/neon_anomaly_scan.py 2>&1 | tee logs/neon_anomaly_scan.log
echo "  Done."

echo "Step 3: Run corrected case studies (exp1 v2)"
python scripts/exp1_case_studies_v2.py 2>&1 | tee logs/exp1_case_studies_v2.log
echo "  Done."

echo "Step 4: Re-run exp9 bootstrap CI"
python scripts/exp9_bootstrap_ci.py 2>&1 | tee logs/exp9_bootstrap_ci.log
echo "  Done."

echo "Step 5: Re-run exp16 parameter attribution"
python scripts/exp16_parameter_attribution.py 2>&1 | tee logs/exp16_attribution.log
echo "  Done."

echo "Step 6: Re-run exp17 risk index"
python scripts/exp17_risk_index.py 2>&1 | tee logs/exp17_risk_index.log
echo "  Done."

echo "Step 7: Re-run exp18 seasonal analysis"
python scripts/exp18_seasonal_analysis.py 2>&1 | tee logs/exp18_seasonal.log
echo "  Done."

echo "Step 8: Re-run exp19 behavioral profile"
python scripts/exp19_behavioral_profile.py 2>&1 | tee logs/exp19_behavioral.log
echo "  Done."

echo "Step 9: Re-run exp20 cascade analysis"
python scripts/exp20_cascade_analysis.py 2>&1 | tee logs/exp20_cascade.log
echo "  Done."

echo "Step 10: Re-run AquaSSM benchmark with new checkpoint"
python scripts/benchmark_aquassm.py 2>&1 | tee logs/benchmark_aquassm.log
echo "  Done."

echo "Step 11: Compile all results into master_results.json"
python scripts/compile_results.py 2>&1 | tee logs/compile_results.log
echo "  Done."

echo "=== PIPELINE COMPLETE ==="
echo "Results saved to results/master_results.json"
