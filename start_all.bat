@echo off
title DeepGuard 3-Detector Voting System
echo ==========================================================
echo    Starting DeepGuard: 3-Detector Voting System
echo ==========================================================

echo [1/3] Starting DeepGuard Backend (Port 8000)...
start "DeepGuard Backend (8000)" cmd /k "python backend/main.py"

echo [2/3] Starting Ultra Detector Backend (Port 8005)...
start "Ultra Detector (8005)" cmd /k ".\.venv\Scripts\python.exe -m aidetector.cli api --port 8005 --host 127.0.0.1 --backend ultra"

echo [3/3] Starting Frontend (Port 5173)...
start "DeepGuard Frontend (5173)" cmd /k "npm run dev"

echo.
echo All services launched!
echo Frontend: http://localhost:5173
echo.
pause
