# Validation script for AnomalyGuard

Write-Host "Starting AnomalyGuard Validation..." -ForegroundColor Cyan

# 1. Run Pytest
Write-Host "Running pytest..." -ForegroundColor Yellow
pytest tests/ -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "Pytest failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

# 2. Start Server
Write-Host "Starting Server..." -ForegroundColor Yellow
$serverProcess = Start-Process python -ArgumentList "-m uvicorn app.main:app --host 0.0.0.0 --port 7860" -PassThru -NoNewWindow
Start-Sleep -Seconds 5

try {
    # 3. Run Demos
    Write-Host "Running Anti-Hacking Demo..." -ForegroundColor Yellow
    python demos/demo_anti_hacking.py
    
    Write-Host "Running Realistic Scenarios Demo..." -ForegroundColor Yellow
    python demos/demo_realistic_scenarios.py

    # 4. Check Endpoints (using Invoke-RestMethod to avoid PowerShell curl alias issues)
    Write-Host "Checking Core Endpoints..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "http://localhost:7860/health" | ConvertTo-Json
    Invoke-RestMethod -Uri "http://localhost:7860/threat-intel/live" | ConvertTo-Json
    Invoke-RestMethod -Uri "http://localhost:7860/anti-hacking/report" | ConvertTo-Json
    Invoke-RestMethod -Uri "http://localhost:7860/business-impact/roi" | ConvertTo-Json

    # 5. Check Curriculum Endpoints
    Write-Host "Checking Curriculum Endpoints..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "http://localhost:7860/curriculum/status" | ConvertTo-Json
    Write-Host "  [OK] /curriculum/status" -ForegroundColor Green

    # 6. Check Compliance Endpoints
    Write-Host "Checking Compliance Endpoints..." -ForegroundColor Yellow
    Invoke-RestMethod -Uri "http://localhost:7860/compliance/audit" | ConvertTo-Json
    Write-Host "  [OK] /compliance/audit" -ForegroundColor Green
    Invoke-RestMethod -Uri "http://localhost:7860/compliance/trail" | ConvertTo-Json
    Write-Host "  [OK] /compliance/trail" -ForegroundColor Green
    Invoke-RestMethod -Uri "http://localhost:7860/compliance/dashboard" | ConvertTo-Json
    Write-Host "  [OK] /compliance/dashboard" -ForegroundColor Green

    Write-Host "`n[OK] Anomaly-Guard-: Validation Complete" -ForegroundColor Green
}
finally {
    # Stop Server
    Write-Host "Stopping Server..." -ForegroundColor Gray
    Stop-Process -Id $serverProcess.Id -Force
}
