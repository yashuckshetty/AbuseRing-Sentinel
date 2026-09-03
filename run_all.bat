@echo off
REM ==============================================================================
REM AbuseRing Sentinel — One-Command Evaluator Setup & Verification (Windows)
REM ==============================================================================

echo ======================================================================
echo  AbuseRing Sentinel -- Automated Setup and Verification Gate
echo ======================================================================

REM 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    exit /b 1
)

echo [1/4] Python found:
python --version

REM 2. Install / Verify Dependencies
echo [2/4] Verifying dependencies from requirements.txt...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

REM 3. Run Full Test Suite
echo ======================================================================
echo [3/4] Running Full Test Suite (pytest)...
echo ======================================================================
python -m pytest tests/ -v
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Test suite failed. Halting launch.
    exit /b 1
)

REM 4. Start Server
echo ======================================================================
echo [4/4] Starting AbuseRing Sentinel API and Dashboard at http://localhost:8000
echo ======================================================================
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
