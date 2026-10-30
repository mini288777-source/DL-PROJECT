$ErrorActionPreference = "Stop"
$PYTHON = "C:\Users\lucky\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe"
$env:PYTHONPATH = "d:\DL"

Write-Host "=========================================================="
Write-Host "PHASE 8 SEPARATED PIPELINE: MASTER EXECUTION"
Write-Host "=========================================================="

Write-Host "Step 1: Running Track A Classical Baseline..."
& $PYTHON d:\DL\scripts\phase8_separated\run_track_a.py
if ($LASTEXITCODE -ne 0) { throw "Track A failed" }

Write-Host "Step 2: Running Track B Quantum Residual..."
& $PYTHON d:\DL\scripts\phase8_separated\run_track_b.py
if ($LASTEXITCODE -ne 0) { throw "Track B failed" }

Write-Host "Step 3: Running Track C Matched Classical Shadow..."
& $PYTHON d:\DL\scripts\phase8_separated\run_track_c.py
if ($LASTEXITCODE -ne 0) { throw "Track C failed" }

Write-Host "Step 4: Generating synthesis, statistics, figures, and report..."
& $PYTHON d:\DL\scripts\phase8_separated\generate_report.py
if ($LASTEXITCODE -ne 0) { throw "Report generation failed" }

Write-Host "Phase 8 Separated Pipeline finished successfully."
