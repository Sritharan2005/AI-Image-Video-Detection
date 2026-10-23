# DeepGuard Launcher: Authentic Project A Backend + Project B Frontend
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Starting DeepGuard: Authentic Backend + Unified UI     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Start Authentic Backend (Project A on port 8000)
Write-Host "[1/2] Starting Project A Backend (UniversalFakeDetect on port 8000)..." -ForegroundColor Yellow
$backendProcess = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m aidetector.cli api --port 8000 --host 127.0.0.1" -PassThru -NoNewWindow

# 2. Start Frontend (Project B on port 5173)
Write-Host "[2/2] Starting Project B Frontend (Vite on port 5173)..." -ForegroundColor Yellow
Start-Process -FilePath "npm" -ArgumentList "run dev"

Write-Host "`nDeepGuard is running!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "Backend:  http://127.0.0.1:8000/health" -ForegroundColor White
Write-Host "Press Ctrl+C or close terminal to stop." -ForegroundColor Gray
