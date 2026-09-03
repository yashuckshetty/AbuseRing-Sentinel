#!/usr/bin/env bash
# ==============================================================================
# AbuseRing Sentinel — One-Command Evaluator Setup & Verification (Linux / macOS)
# ==============================================================================
set -e

echo "======================================================================"
echo " AbuseRing Sentinel — Automated Setup & Verification Gate"
echo "======================================================================"

# 1. Check Python version
if command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
    PYTHON_BIN="python"
else
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

echo "[1/4] Using Python binary: $($PYTHON_BIN --version)"

# 2. Setup Virtual Environment (Optional / Recommended)
if [ ! -d ".venv" ]; then
    echo "[2/4] Creating virtual environment (.venv)..."
    $PYTHON_BIN -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    PYTHON_BIN="python"
fi

# 3. Install / Confirm Dependencies
echo "[2/4] Verifying dependencies from requirements.txt..."
$PYTHON_BIN -m pip install --quiet --upgrade pip
$PYTHON_BIN -m pip install --quiet -r requirements.txt

# 4. Run Full Regression Test Suite
echo "======================================================================"
echo "[3/4] Running Full Test Suite (pytest)..."
echo "======================================================================"
$PYTHON_BIN -m pytest tests/ -v

# 5. Start API & Dashboard Server
echo "======================================================================"
echo "[4/4] Starting AbuseRing Sentinel API & Dashboard at http://localhost:8000"
echo "======================================================================"
$PYTHON_BIN -m uvicorn api.main:app --host 0.0.0.0 --port 8000
