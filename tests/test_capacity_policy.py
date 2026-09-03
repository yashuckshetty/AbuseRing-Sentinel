"""
Unit and integration tests for the Capacity-Constrained Review Queue Policy.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from policy.capacity_policy import TriageStrategy, QueueItem, ReviewQueueEngine
from api.main import app

client = TestClient(app)
RESULTS_PATH = Path("evals/results/capacity_constrained_results.json")

def test_priority_score_ranking_order():
    """Verifies that conflict-aware and score-descending triage correctly prioritize high risk."""
    item_low = QueueItem(
        account_id="ACC_LOW", p_abusive=0.10, p_benign_coord=0.80, p_benign_indep=0.10,
        p_struct_ac=0.10, p_behav_ac=0.10, sym_kl_divergence=0.10,
        n_orders=2, total_order_amount=1000.0, true_label="benign_coordinated"
    )
    item_high = QueueItem(
        account_id="ACC_HIGH", p_abusive=0.85, p_benign_coord=0.05, p_benign_indep=0.10,
        p_struct_ac=0.90, p_behav_ac=0.80, sym_kl_divergence=1.50,
        n_orders=5, total_order_amount=5000.0, true_label="abusive_coordinated"
    )
    
    ranked_conflict = ReviewQueueEngine.rank_queue([item_low, item_high], TriageStrategy.CONFLICT_AWARE)
    assert ranked_conflict[0].account_id == "ACC_HIGH"
    
    ranked_score = ReviewQueueEngine.rank_queue([item_low, item_high], TriageStrategy.SCORE_DESC)
    assert ranked_score[0].account_id == "ACC_HIGH"

def test_capacity_evaluation_metrics():
    """Verifies capacity evaluation logic computes correct recall, precision@k and exposure."""
    items = [
        QueueItem("A1", 0.9, 0.05, 0.05, 0.9, 0.9, 2.0, 3, 3000.0, "abusive_coordinated"),
        QueueItem("A2", 0.8, 0.1, 0.1, 0.8, 0.8, 1.0, 2, 2000.0, "abusive_coordinated"),
        QueueItem("B1", 0.2, 0.7, 0.1, 0.2, 0.2, 0.5, 2, 1000.0, "benign_coordinated"),
    ]
    
    # Cap at K=2
    res = ReviewQueueEngine.evaluate_capacity_limit(
        items=items,
        capacity_limit=2,
        strategy=TriageStrategy.SCORE_DESC,
        auto_act_tp=10,
        total_true_ac=20
    )
    assert res["accounts_reviewed"] == 2
    assert res["true_ac_captured_in_review"] == 2
    assert res["precision_at_k"] == 1.0
    assert res["retained_effective_recall"] == (10 + 2) / 20  # 0.60
    assert res["prevented_fraud_exposure_rs"] == 5000.0

def test_capacity_results_file_schema():
    """Verifies that capacity_constrained_results.json exists and has expected schema."""
    assert RESULTS_PATH.exists(), "capacity_constrained_results.json must exist"
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "test_population_stats" in data
    assert "capacity_limits_evaluated" in data
    assert "sym_kl_diagnostics" in data
    assert "sweep_results" in data
    assert "conflict_aware" in data["sweep_results"]
    assert "fifo" in data["sweep_results"]
    assert "random_shuffle" in data["sweep_results"]
    assert "time_of_flagging" in data["sweep_results"]
    assert "exposure_weighted" in data["sweep_results"]
    assert data["test_population_stats"]["review_queue_size"] == 779
    assert data["test_population_stats"]["true_ac_in_review"] == 124

def test_conflict_aware_outperforms_uninformative_baselines_at_k100():
    """Verifies that conflict-aware triage captures >2x recall compared to uninformative baselines at K=100."""
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    sweep = data["sweep_results"]
    k100_fifo = next(r for r in sweep["fifo"] if r["capacity_limit"] == 100)
    k100_time = next(r for r in sweep["time_of_flagging"] if r["capacity_limit"] == 100)
    k100_conflict = next(r for r in sweep["conflict_aware"] if r["capacity_limit"] == 100)
    
    # Conflict-Aware retained recall (62.63%) vs FIFO/Time (27.27%)
    assert k100_conflict["retained_effective_recall"] > k100_fifo["retained_effective_recall"] * 2.0
    assert k100_conflict["retained_effective_recall"] > k100_time["retained_effective_recall"] * 2.0
    assert k100_conflict["precision_at_k"] >= 0.80

def test_sym_kl_diagnostics_distribution():
    """Verifies that sym_KL diagnostics show selective elevation for true AC in review."""
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    diag = data["sym_kl_diagnostics"]
    assert diag["kl_multiplier_factor_distribution"]["true_ac_mean_factor"] > diag["kl_multiplier_factor_distribution"]["benign_mean_factor"]
    assert diag["rank_shift_ablation_vs_no_kl"]["mean_absolute_rank_shift"] > 5.0

def test_capacity_endpoint():
    """Verifies that /api/review-queue/capacity endpoint serves valid evaluation payload."""
    res = client.get("/api/review-queue/capacity")
    assert res.status_code == 200
    data = res.json()
    assert "sweep_results" in data
    assert "sym_kl_diagnostics" in data
    assert "conflict_aware" in data["sweep_results"]
