@echo off
cd /d "%~dp0"
echo ===================================================
echo Starting Rakshak Credit FastAPI Backend...
echo API available at: http://127.0.0.1:8000
echo Swagger UI docs at: http://127.0.0.1:8000/docs
echo ===================================================
"%~dp0.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
