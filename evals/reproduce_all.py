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
print("\n[STEP 0/17] Cleaning generated artifacts...", flush=True)

files_to_remove = [
    "data/events.parquet",
    "data/accounts.parquet",
    "data/labels.parquet",
    "data/rings.parquet",
    "data/split_info.json",
    "models/behavioral_lgbm.pkl",
    "models/structural_lgbm.pkl",
    "models/fused_calibrated.pkl",
    "models/gnn_comparison.pkl",
    "evals/metrics.json",
    "evals/results/trajectory_results.parquet",
    "evals/results/kl_ablation_results.json",
    "evals/results/prevalence_shift_results.json",
    "evals/results/multiseed_results.json",
    "evals/results/robustness_results.json",
    "evals/results/gnn_comparison_results.json",
    "evals/results/scenario_b_generalization_results.json",
    "evals/results/adversarial_results.json",
    "evals/results/dynamic_cost_results.json",
    "evals/results/capacity_constrained_results.json",
    "evals/results/ai_security_results.json",
    "evals/results/handcrafted_adversarial_results.json",
    "evals/results/gateway_latency_results.json",
]

for f in files_to_remove:
    p = BASE_DIR / f
    if p.exists():
        p.unlink()
        print(f"  Removed: {f}")

for d in ["data/prevalence_low", "data/prevalence_high", "data/seed_43", "data/seed_44", "data/scenario_b"]:
    dp = BASE_DIR / d
    if dp.exists():
        shutil.rmtree(dp)
        print(f"  Removed directory: {d}")


def run_cmd(cmd_list, desc):
    print(f"\n[{desc}] Running: {' '.join(cmd_list)}", flush=True)
    t0 = time.time()
    res = subprocess.run(cmd_list, capture_output=True, text=True, encoding="utf-8", errors="replace")
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


# 1. Generate Dataset
run_cmd([sys.executable, "data/simulator.py"], "STEP 1/17: Generate v2.0 Dataset")

# 2. Train Model Ladder
run_cmd([sys.executable, "-m", "models.model_suite"], "STEP 2/17: Train 5-Rung Model Ladder")

# 3. Save Robustness Stage 12a Table
run_cmd([sys.executable, "evals/save_robustness.py"], "STEP 3/17: Save Stage 12a Robustness Table")

# 4. Run Trajectory Evaluation
run_cmd([sys.executable, "evals/trajectory_eval.py"], "STEP 4/17: Run Trajectory Evaluation")

# 5. Run KL-Routing Ablation
run_cmd([sys.executable, "evals/kl_ablation_eval.py"], "STEP 5/17: Run KL-Routing Ablation")

# 6. Run Prevalence-Shift Sensitivity Analysis
run_cmd([sys.executable, "evals/prevalence_shift_eval.py"], "STEP 6/17: Run Prevalence-Shift Sensitivity")

# 7. Run Multi-Seed Robustness Evaluation
run_cmd([sys.executable, "evals/multiseed_eval.py"], "STEP 7/17: Run Multi-Seed Variance Evaluation")

# 8. Run GNN Comparison Evaluation
run_cmd([sys.executable, "evals/gnn_eval.py"], "STEP 8/17: Run GNN Comparison Evaluation")

# 9. Generate Scenario B Dataset
run_cmd([sys.executable, "data/simulator_scenario_b.py"], "STEP 9/17: Generate Scenario B Dataset")

# 10. Run Scenario B Generalization Evaluation
run_cmd([sys.executable, "evals/scenario_b_eval.py"], "STEP 10/17: Run Scenario B Generalization Evaluation")

# 11. Run Adversarial Evasion Evaluation
run_cmd([sys.executable, "evals/adversarial_eval.py"], "STEP 11/17: Run Adversarial Evasion Evaluation")

# 12. Run Dynamic Compounding Cost Evaluation
run_cmd([sys.executable, "evals/dynamic_cost_eval.py"], "STEP 12/17: Run Dynamic Cost Evaluation")

# 13. Run Capacity-Constrained Review Queue Evaluation
run_cmd([sys.executable, "evals/capacity_eval.py"], "STEP 13/17: Run Capacity Queue Triage Evaluation")

# 14. Run AI Security & Prompt Injection Defense Evaluation
run_cmd([sys.executable, "evals/ai_security_eval.py"], "STEP 14/17: Run AI Security Evaluation")

# 15. Run Handcrafted Adversarial Topology Battery
run_cmd([sys.executable, "evals/handcrafted_adversarial.py"], "STEP 15/17: Run Handcrafted Topology Battery")

# 16. Run Gateway Dual-Path Latency Benchmark
run_cmd([sys.executable, "evals/gateway_latency_eval.py"], "STEP 16/17: Run Gateway Latency Benchmark")

# 17. Run Full Pytest Suite
run_cmd([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"], "STEP 17/17: Execute Full Test Suite")

# Final Assertion: Verify all required health artifacts exist
required_health_files = [
    "models/behavioral_lgbm.pkl",
    "models/structural_lgbm.pkl",
    "models/fused_calibrated.pkl",
    "evals/metrics.json",
    "evals/results/trajectory_results.parquet",
    "evals/results/kl_ablation_results.json",
    "evals/results/prevalence_shift_results.json",
    "evals/results/multiseed_results.json",
    "evals/results/robustness_results.json",
    "evals/results/gnn_comparison_results.json",
    "evals/results/scenario_b_generalization_results.json",
    "evals/results/adversarial_results.json",
    "evals/results/dynamic_cost_results.json",
    "evals/results/capacity_constrained_results.json",
    "evals/results/ai_security_results.json",
    "evals/results/handcrafted_adversarial_results.json",
    "evals/results/gateway_latency_results.json",
]

missing_artifacts = [f for f in required_health_files if not (BASE_DIR / f).exists()]
if missing_artifacts:
    print(f"\nFATAL: Missing {len(missing_artifacts)} required artifacts:")
    for mf in missing_artifacts:
        print(f"  - {mf}")
    sys.exit(1)

total_checked = len(required_health_files)
print("\n" + "=" * 80)
print(f"ARTIFACT COMPLETENESS: OK ({total_checked}/{total_checked})")
print("ALL PIPELINE ARTIFACTS AND EVALUATIONS REPRODUCED FROM CLEAN SLATE!")
print("=" * 80)
