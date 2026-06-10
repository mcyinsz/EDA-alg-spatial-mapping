#!/usr/bin/env bash
# Full reproduction: main + inter-only + intra-parallel ablation + kappa sweep + plots + verification
set -e

echo "============================================"
echo "  Step 1/9: Main experiment (with intra-layer comm)"
echo "============================================"
python src/experiment.py

echo ""
echo "============================================"
echo "  Step 2/9: Main experiment plots"
echo "============================================"
python src/visualize.py

echo ""
echo "============================================"
echo "  Step 3/9: Inter-only control experiment"
echo "============================================"
python src/experiment.py --inter-only

echo ""
echo "============================================"
echo "  Step 4/9: Inter-only control plots"
echo "============================================"
python src/visualize.py --results results/experiment_results_inter_only.json

echo ""
echo "============================================"
echo "  Step 5/9: Intra-parallel ablation experiment"
echo "============================================"
python src/experiment.py --intra-parallel

echo ""
echo "============================================"
echo "  Step 6/9: Intra-parallel ablation plots"
echo "============================================"
python src/visualize.py --results results/experiment_results_intra_parallel.json

echo ""
echo "============================================"
echo "  Step 7/9: Communication sensitivity study"
echo "============================================"
python scripts/sweep_sensitivity.py 2>&1 | tee logs/sensitivity_study.log

echo ""
echo "============================================"
echo "  Step 8/9: Verify all results"
echo "============================================"
python scripts/verify_results.py

echo ""
echo "============================================"
echo "  Done! All results and plots generated."
echo "============================================"
