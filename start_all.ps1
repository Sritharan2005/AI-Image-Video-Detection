# DeepGuard Unified Launcher: 3-Detector Backend + Ultra + Frontend
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Starting DeepGuard: 3-Detector Voting System           " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# Resolve Python executable: prefer active venv or local venv directory, fallback to system PATH python
$pythonCmd = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } elseif (Test-Path ".\venv\Scripts\python.exe") { ".\venv\Scripts\python.exe" } else { "python" }

# 1. Start DeepGuard Backend (Baseline & Face-Crop, Port 8000)
Write-Host "[1/3] Starting DeepGuard Backend (EfficientNet-B0 on port 8000)..." -ForegroundColor Yellow
$p8000 = Start-Process -FilePath $pythonCmd -ArgumentList "backend/main.py" -PassThru -NoNewWindow

# 2. Start Ultra Detector Backend (OpenCLIP ViT-L-14 + Sentry + Nonescape on port 8005)
Write-Host "[2/3] Starting Ultra Detector Backend (Port 8005)..." -ForegroundColor Yellow
$p8005 = Start-Process -FilePath $pythonCmd -ArgumentList "-m aidetector.cli api --port 8005 --host 127.0.0.1 --backend ultra" -PassThru -NoNewWindow

# 3. Start Frontend (Vite on port 5173)
Write-Host "[3/3] Starting Frontend UI (Vite on port 5173)..." -ForegroundColor Yellow
Start-Process -FilePath "npm" -ArgumentList "run dev"

Write-Host "`nDeepGuard is running!" -ForegroundColor Green
Write-Host "Frontend:          http://localhost:5173" -ForegroundColor White
Write-Host "DeepGuard API:     http://127.0.0.1:8000/api/status" -ForegroundColor White
Write-Host "Ultra Detector:    http://127.0.0.1:8005/health" -ForegroundColor White
Write-Host "`nPress Ctrl+C or close terminal to stop." -ForegroundColor Gray
