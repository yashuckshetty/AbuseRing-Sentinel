"""
Unit and integration tests for Dynamic Compounding Loss and Break-Even Lag Modeling.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from decision.cost_model import DynamicCostConfig
from api.main import app

client = TestClient(app)
RESULTS_PATH = Path("evals/results/dynamic_cost_results.json")

def test_dynamic_cost_monotonicity_with_lag():
    """Verifies that false negative loss increases monotonically with detection lag."""
    cfg = DynamicCostConfig(c_false_negative_base=2000.0, alpha_compounding_per_day=100.0, gamma_acceleration=1.2)
    lags = [0, 5, 10, 20, 30, 60]
    losses = [cfg.calculate_fn_loss(t) for t in lags]
    for i in range(len(losses) - 1):
        assert losses[i] <= losses[i + 1], f"Loss at lag {lags[i]} ({losses[i]}) > lag {lags[i+1]} ({losses[i+1]})"
    assert losses[0] == 2000.0

def test_dynamic_cost_results_file_schema():
    """Verifies that dynamic cost evaluation results file exists and has required schema."""
    assert RESULTS_PATH.exists(), "dynamic_cost_results.json must exist"
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "cost_assumptions_simulated" in data
    assert "static_cost_baseline_rs" in data
    assert "symmetric_break_even_analysis" in data
    assert "cost_breakdown_by_component_alpha_100" in data
    assert "lag_sensitivity_curve" in data
    assert "alpha_100" in data["symmetric_break_even_analysis"]

def test_break_even_lag_monotonic_with_compounding_rate():
    """Verifies that higher compounding rates (alpha) lead to shorter break-even lags."""
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    be = data["symmetric_break_even_analysis"]
    alpha_25_be = be["alpha_25"]["break_even_lag_days_v1_hold"]
    alpha_100_be = be["alpha_100"]["break_even_lag_days_v1_hold"]
    alpha_500_be = be["alpha_500"]["break_even_lag_days_v1_hold"]
    
    assert alpha_25_be > alpha_100_be > alpha_500_be
    assert 50.0 <= alpha_500_be <= 100.0  # Around ~72.5 days for high drain under symmetric lag

def test_dynamic_cost_endpoint():
    """Verifies that /api/dynamic-cost endpoint serves the evaluation payload."""
    res = client.get("/api/dynamic-cost")
    assert res.status_code == 200
    data = res.json()
    assert "symmetric_break_even_analysis" in data
    assert data["static_cost_baseline_rs"]["behavioral_only_flat"] == 30500.0
    assert data["static_cost_baseline_rs"]["routing_flat_without_compounding"] == 149250.0
