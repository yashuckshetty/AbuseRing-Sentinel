"""
Unit & Integration tests for the Adversarial Evasion & Adaptive Attacker Stress Test.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
RESULTS_PATH = Path("evals/results/adversarial_results.json")

def test_adversarial_results_file_exists():
    """Verifies that adversarial evaluation results are generated and populated."""
    assert RESULTS_PATH.exists(), "adversarial_results.json must exist"
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)
    assert "scenarios" in data
    assert data["n_test_accounts"] == 3467
    assert data["n_true_ac"] == 198
    assert len(data["scenarios"]) == 5  # baseline + 4 strategies

def test_adversarial_evasion_zero_auto_act_fps():
    """Verifies that Auto-ACT False Positives remain 0 across ALL adversarial regimes."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)
    for name, s in data["scenarios"].items():
        fp = s["decision_engine"]["auto_act_false_positives"]
        fp_rate = s["decision_engine"]["auto_act_false_positive_rate"]
        assert fp == 0, f"Scenario {name} had {fp} auto-ACT FPs (expected 0)"
        assert fp_rate == 0.0, f"Scenario {name} had non-zero FP rate {fp_rate}"

def test_adversarial_evasion_review_lane_absorption():
    """Verifies that when auto-ACT recall drops due to evasion, the REVIEW lane absorbs the accounts."""
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)
    
    baseline = data["scenarios"]["baseline"]["decision_engine"]
    s1 = data["scenarios"]["strategy_1_anti_burst"]["decision_engine"]
    s2 = data["scenarios"]["strategy_2_device_ip_hopping"]["decision_engine"]
    
    # In Strategy 1, auto-ACT drops but REVIEW increases
    assert s1["direct_auto_act_recall"] < baseline["direct_auto_act_recall"]
    assert s1["ac_breakdown"]["AC_in_REVIEW"] > baseline["ac_breakdown"]["AC_in_REVIEW"]
    assert abs(s1["effective_recall"] - baseline["effective_recall"]) <= 0.02  # Approximately preserved effective recall
    assert s1["effective_recall"] >= 0.80
    
    # In Strategy 2 (device/IP hopping), REVIEW absorbs 150 accounts
    assert s2["ac_breakdown"]["AC_in_REVIEW"] >= 150

def test_adversarial_evasion_endpoint():
    """Verifies that /api/adversarial-evasion serves the complete evaluation payload."""
    res = client.get("/api/adversarial-evasion")
    assert res.status_code == 200
    data = res.json()
    assert "scenarios" in data
    assert "strategy_4_combined_adaptive" in data["scenarios"]
    comb = data["scenarios"]["strategy_4_combined_adaptive"]["decision_engine"]
    assert comb["effective_recall"] > 0.75
    assert comb["auto_act_false_positives"] == 0
