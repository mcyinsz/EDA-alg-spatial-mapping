# Full reproduction: main experiment + inter-only control + all plots + verification
# Run from repo root: powershell -ExecutionPolicy Bypass -File scripts\run_full.ps1

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Step 1/5: Main experiment (with intra-layer comm)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
python src/experiment.py

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Step 2/5: Main experiment plots" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
python src/visualize.py

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Step 3/5: Inter-only control experiment" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
python src/experiment.py --inter-only

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Step 4/5: Inter-only control plots" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
python src/visualize.py --results results/experiment_results_inter_only.json

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Step 5/5: Verify all results" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
python scripts/verify_results.py

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Done! All results and plots generated." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
