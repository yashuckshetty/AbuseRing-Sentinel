"""
AbuseRing Sentinel — Master Clean-Slate Reproducibility Script
==============================================================
Deletes all generated data, models, and evaluation outputs, then
executes the full pipeline end-to-end to verify 100% reproducibility.
"""

import os
import sys
import glob
import shutil
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.resolve()
os.chdir(BASE_DIR)

print("=" * 80)
print("ABUSERING SENTINEL — COLD-START CLEAN REPRODUCIBILITY PASS")
print("=" * 80)

# 1. Clean generated artifacts
print("\n[STEP 1/9] Cleaning generated artifacts...", flush=True)

files_to_remove = [
    "data/events.parquet",
    "data/accounts.parquet",
    "data/labels.parquet",
    "data/split_info.json",
    "models/behavioral_lgbm.pkl",
    "models/structural_lgbm.pkl",
    "models/fused_calibrated.pkl",
    "evals/metrics.json",
    "evals/results/trajectory_results.parquet",
    "evals/results/kl_ablation_results.json",
    "evals/results/prevalence_shift_results.json",
    "evals/results/multiseed_results.json",
    "evals/results/robustness_results.json",
]

for f in files_to_remove:
    p = BASE_DIR / f
    if p.exists():
        p.unlink()
        print(f"  Removed: {f}")

for d in ["data/prevalence_low", "data/prevalence_high", "data/seed_43", "data/seed_44"]:
    dp = BASE_DIR / d
    if dp.exists():
        shutil.rmtree(dp)
        print(f"  Removed directory: {d}")

def run_cmd(cmd_list, desc):
    print(f"\n[{desc}] Running: {' '.join(cmd_list)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_list, capture_output=True, text=True, encoding="utf-8")
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"  FAILED in {dt:.2f}s!")
        print("--- STDOUT ---")
        print(res.stdout)
        print("--- STDERR ---")
        print(res.stderr)
        sys.exit(1)
    else:
        print(f"  SUCCESS in {dt:.2f}s", flush=True)
        return res.stdout

# 2. Generate Dataset
run_cmd([sys.executable, "data/simulator.py"], "STEP 2/9: Generate v2.0 Dataset")

# 3. Train Model Ladder
run_cmd([sys.executable, "-m", "models.model_suite"], "STEP 3/9: Train 5-Rung Model Ladder")

# 4. Save Robustness Stage 12a Table
run_cmd([sys.executable, "evals/save_robustness.py"], "STEP 4/9: Save Stage 12a Robustness Table")

# 5. Run Trajectory Evaluation
run_cmd([sys.executable, "evals/trajectory_eval.py"], "STEP 5/9: Run Trajectory Evaluation")

# 6. Run KL-Routing Ablation
run_cmd([sys.executable, "evals/kl_ablation_eval.py"], "STEP 6/9: Run KL-Routing Ablation")

# 7. Run Prevalence-Shift Sensitivity Analysis
run_cmd([sys.executable, "evals/prevalence_shift_eval.py"], "STEP 7/9: Run Prevalence-Shift Sensitivity")

# 8. Run Multi-Seed Robustness Evaluation
run_cmd([sys.executable, "evals/multiseed_eval.py"], "STEP 8/9: Run Multi-Seed Variance Evaluation")

# 9. Run Full Pytest Suite
run_cmd([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], "STEP 9/9: Execute Full Test Suite")

print("\n" + "=" * 80)
print("ALL PIPELINE ARTIFACTS AND EVALUATIONS REPRODUCED FROM CLEAN SLATE!")
print("=" * 80)
