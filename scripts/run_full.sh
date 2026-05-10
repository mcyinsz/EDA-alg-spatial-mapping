#!/usr/bin/env bash
# Full reproduction: main experiment + inter-only control + all plots + verification
set -e

echo "============================================"
echo "  Step 1/5: Main experiment (with intra-layer comm)"
echo "============================================"
python src/experiment.py

echo ""
echo "============================================"
echo "  Step 2/5: Main experiment plots"
echo "============================================"
python src/visualize.py

echo ""
echo "============================================"
echo "  Step 3/5: Inter-only control experiment"
echo "============================================"
python src/experiment.py --inter-only

echo ""
echo "============================================"
echo "  Step 4/5: Inter-only control plots"
echo "============================================"
python src/visualize.py --results results/experiment_results_inter_only.json

echo ""
echo "============================================"
echo "  Step 5/5: Verify all results"
echo "============================================"
python scripts/verify_results.py

echo ""
echo "============================================"
echo "  Done! All results and plots generated."
echo "============================================"
