@echo off
echo ==========================================
echo   Starting ManavaAI Project
echo ==========================================

echo.
echo [1/2] Launching Backend Server...
start "ManavaAI Backend" cmd /k "cd backend && echo Current Directory: %cd% && echo Installing requirements... && python -m pip install -r requirements.txt && echo Starting Server... && python -m uvicorn app.main:app --reload || echo. && echo [ERROR] Backend crashed. See above for details. && pause"

echo.
echo [2/2] Launching Frontend Server...
start "ManavaAI Frontend" cmd /k "cd frontend && echo Cleaning stale locks... && (if exist .next rd /s /q .next) && echo Installing dependencies... && npm install && echo Starting Next.js... && npm run dev || echo. && echo [ERROR] Frontend crashed. See above for details. && pause"

echo.
echo ==========================================
echo   All systems launched!
echo   - Backend: http://localhost:8000
echo   - Frontend: http://localhost:3000 (usually)
echo ==========================================
pause
