"""
Unit and integration tests for FastAPI backend service.
Verifies all read-only artifact endpoints and live DecisionEngine execution.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from api.main import app

BASE_DIR = Path(__file__).resolve().parent.parent
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["artifacts_loaded"] is True
    assert data["files_checked"] == 17
    assert len(data["missing_files"]) == 0


def test_model_ladder_endpoint():
    response = client.get("/api/model-ladder")
    assert response.status_code == 200
    data = response.json()
    assert data["split"] == "test"
    assert data["n_test_accounts"] == 3467
    assert data["n_true_ac"] == 198
    assert len(data["ladder"]) == 5
    assert "multi_seed_callout" in data


def test_decision_hard_bc_account():
    response = client.get("/api/decision/ACC_03653")
    assert response.status_code == 200
    data = response.json()
    assert data["account_id"] == "ACC_03653"
    assert data["decision"] == "WAIT_MONITOR"
    assert "sym_kl_divergence" in data
    assert "probabilities" in data
    assert "audit_trail" in data


def test_decision_referral_farming_account():
    response = client.get("/api/decision/ACC_04870")
    assert response.status_code == 200
    data = response.json()
    assert data["account_id"] == "ACC_04870"
    assert data["decision"] == "REVIEW"
    assert data["routing_lane"] == "conflict_review"
    assert data["sym_kl_divergence"] > 0.50
    assert data["evidence_conflict"] is True


def test_trajectory_promo_ring():
    response = client.get("/api/trajectory/PROMO_001")
    assert response.status_code == 200
    data = response.json()
    assert data["ring_id"] == "PROMO_001"
    assert len(data["checkpoints"]) == 5
    # Check that all 5 checkpoints have valid metrics
    for cp in data["checkpoints"]:
        assert "p_behav" in cp
        assert "p_struct" in cp
        assert "sym_kl" in cp
        assert "primary_decision" in cp


def test_trajectory_refarm_ring():
    response = client.get("/api/trajectory/REFARM_057")
    assert response.status_code == 200
    data = response.json()
    assert data["ring_id"] == "REFARM_057"
    assert len(data["checkpoints"]) == 5


def test_ablation_endpoint():
    response = client.get("/api/ablation")
    assert response.status_code == 200
    data = response.json()
    assert "full_test_split" in data
    assert data["full_test_split"]["threshold_ablation"]["ac_breakdown"]["AC_in_WAIT"] == 107
    assert data["full_test_split"]["kl_routing"]["ac_breakdown"]["AC_in_WAIT"] == 0


def test_prevalence_shift_endpoint():
    response = client.get("/api/prevalence-shift")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3
    # Check that all regimes maintain 0 FP in auto-ACT
    for reg in data:
        assert reg["auto_act_fp"] == 0
        assert reg["auto_act_precision"] == 1.0


def test_multi_seed_endpoint():
    response = client.get("/api/multi-seed")
    assert response.status_code == 200
    data = response.json()
    assert len(data["seeds"]) == 3
    assert "summary" in data
    assert data["summary"]["auto_act_fp"]["mean"] == 0.0


def test_robustness_endpoint():
    response = client.get("/api/robustness")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5


def test_sample_accounts_endpoint():
    response = client.get("/api/sample-accounts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 5


def test_dashboard_root_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AbuseRing Sentinel" in response.text
    assert "Core Architecture Thesis" in response.text


def test_ai_advisory_payload_structure():
    response = client.get("/api/decision/ACC_04870")
    assert response.status_code == 200
    data = response.json()
    assert "ai_advisory" in data
    ai = data["ai_advisory"]
    assert len(ai["conflict_explanation"]) > 20
    assert len(ai["key_signals"]) > 0
    assert len(ai["analyst_suggestions"]) > 0
    assert ai["boundary_checks_passed"] is True


def test_decision_engine_api_canonical_consistency():
    """Regression test: verifies API output strictly equals DecisionEngine.decide() output."""
    from api.main import CACHE
    engine = CACHE["engine"]
    fused = CACHE["fused"]
    s_te = CACHE["s_te"]
    b_te = CACHE["b_te"]

    sample_res = client.get("/api/sample-accounts")
    assert sample_res.status_code == 200
    samples = sample_res.json()

    for s in samples:
        acc_id = s["account_id"]
        api_res = client.get(f"/api/decision/{acc_id}").json()

        s_row = s_te.loc[acc_id]
        b_row = b_te.loc[acc_id]
        p_s, p_b, p_f, _ = fused.predict_proba_sub(s_te.loc[[acc_id]], b_te.loc[[acc_id]])

        dec_res = engine.decide(
            account_id=acc_id,
            p_fused=p_f[0],
            p_struct=p_s[0],
            p_behav=p_b[0],
            observation_days=float(b_row.get("account_age_days", 0)),
            n_orders=int(b_row.get("n_orders", 0)),
            as_of_ts=1707776000,
        )

        assert api_res["decision"] == dec_res.decision.value
        assert api_res["routing_lane"] == dec_res.routing_lane.value
        assert api_res["sym_kl_divergence"] == round(float(dec_res.sym_kl_divergence), 4)
        assert api_res["evidence_conflict"] == bool(dec_res.evidence_conflict)

    # Explicit check for ACC_00505: must have sym_KL < 0.50 and WAIT_MONITOR
    acc_505 = client.get("/api/decision/ACC_00505").json()
    assert acc_505["sym_kl_divergence"] < 0.50
    assert acc_505["sym_kl_divergence"] == 0.2834
    assert acc_505["decision"] == "WAIT_MONITOR"
    assert acc_505["routing_lane"] == "fused_auto"


def test_gnn_comparison_endpoint():
    """Verifies that the Rung 6 GNN comparison endpoint returns valid evaluation payload dynamically matching artifact."""
    res = client.get("/api/gnn-comparison")
    assert res.status_code == 200
    data = res.json()
    assert "gnn_test_metrics" in data
    assert "structural_lgbm_test_metrics" in data
    assert "robustness_subsets" in data

    artifact_path = Path("evals/results/gnn_comparison_results.json")
    assert artifact_path.exists()
    with open(artifact_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    assert data["gnn_test_metrics"]["model"] == expected["gnn_test_metrics"]["model"]
    assert data["gnn_test_metrics"]["precision_abusive"] == expected["gnn_test_metrics"]["precision_abusive"]
    assert data["gnn_test_metrics"]["recall_abusive"] == expected["gnn_test_metrics"]["recall_abusive"]
    assert data["gnn_test_metrics"]["f1_abusive"] == expected["gnn_test_metrics"]["f1_abusive"]
    assert data["robustness_subsets"]["referral_farming"]["n_accounts"] == expected["robustness_subsets"]["referral_farming"]["n_accounts"]
    assert data["robustness_subsets"]["referral_farming"]["n_accounts"] == 143


def test_scenario_b_endpoint():
    """Verifies that the Scenario B generalization endpoint returns valid evaluation payload."""
    res = client.get("/api/scenario-b")
    assert res.status_code == 200
    data = res.json()
    assert "standalone_models" in data
    assert "decision_engine_routing" in data
    assert "feature_transfer_stats" in data
    assert data["n_accounts"] == 1800
    assert data["n_ac"] == 270
    assert data["decision_engine_routing"]["auto_act_lane_activated"] is False
    assert data["decision_engine_routing"]["auto_act_false_positives"] == 0
    assert data["decision_engine_routing"]["auto_act_false_positive_rate"] is None
    assert "escalate" not in data["decision_engine_routing"]["routing_lanes"]


def test_adversarial_evasion_endpoint():
    """Verifies that the Adversarial Evasion endpoint returns valid evaluation payload."""
    res = client.get("/api/adversarial-evasion")
    assert res.status_code == 200
    data = res.json()
    assert "scenarios" in data
    assert data["n_test_accounts"] == 3467
    assert data["n_true_ac"] == 198
    assert "strategy_1_anti_burst" in data["scenarios"]
    assert "strategy_4_combined_adaptive" in data["scenarios"]


def test_dynamic_cost_endpoint():
    """Verifies that the Dynamic Cost endpoint returns valid evaluation payload."""
    res = client.get("/api/dynamic-cost")
    assert res.status_code == 200
    data = res.json()
    assert "symmetric_break_even_analysis" in data
    assert "lag_sensitivity_curve" in data


def test_capacity_constrained_endpoint():
    """Verifies that the Capacity-Constrained Review Queue endpoint returns valid evaluation payload."""
    res = client.get("/api/review-queue/capacity")
    assert res.status_code == 200
    data = res.json()
    assert "sweep_results" in data
    assert "conflict_aware" in data["sweep_results"]
    assert "fifo" in data["sweep_results"]


def test_graph_neighborhood_endpoint():
    """Verifies that /api/graph-neighborhood/{account_id} returns valid payload for sample accounts."""
    res = client.get("/api/graph-neighborhood/ACC_04430")
    assert res.status_code == 200
    data = res.json()
    assert data["account_id"] == "ACC_04430"
    assert len(data["nodes"]) > 0
    assert len(data["edges"]) > 0
    assert "investigation_checklist" in data


def test_gateway_spec_endpoint():
    """Verifies gateway specification endpoint returns valid dual-path contract with disclaimers."""
    res = client.get("/api/gateway/spec")
    assert res.status_code == 200
    data = res.json()
    assert "design_targets" in data
    assert "sync_path" in data["design_targets"]
    assert "async_path" in data["design_targets"]
    assert "Prototype design-target" in data["qualifier"]


def test_gateway_simulate_event_endpoint():
    """Verifies simulate-event dual-path execution flow through API."""
    payload = {
        "event": "payment.authorized",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test_api_01",
                    "amount": 50000,
                    "currency": "INR",
                    "notes": {
                        "account_id": "ACC_04870",
                        "device_id": "DEV_API_1"
                    }
                }
            }
        }
    }
    res = client.post("/api/gateway/simulate-event", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["account_id"] == "ACC_04870"
    assert "sync_authorization" in data
    assert "async_enrichment" in data
    assert data["sync_authorization"]["action"] in ["ALLOW", "CHALLENGE_2FA", "BLOCK"]
    assert data["async_enrichment"]["authoritative_decision"] in ["ACT", "REVIEW", "WAIT_MONITOR", "ABSTAIN"]
    assert "Prototype design-target" in data["sync_authorization"]["qualifier"]
    assert "Prototype design-target" in data["async_enrichment"]["qualifier"]


def test_gateway_benchmark_endpoint():
    """Verifies gateway benchmark endpoint executes and includes mandatory qualifiers."""
    res = client.get("/api/gateway/benchmark?n_trials=10")
    assert res.status_code == 200
    data = res.json()
    assert "sync_path" in data
    assert "async_path" in data
    assert "Prototype design-target" in data["qualifier"]


def test_temporal_escalation_summary_endpoint():
    """Verifies /api/temporal-escalation/summary endpoint across 19 rings."""
    res = client.get("/api/temporal-escalation/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["n_rings_evaluated"] == 19
    assert "summary_metrics" in data
    assert data["summary_metrics"]["blended_mean_lead_time_vs_complete_days"] >= 0.0
    assert "pre_positioned_sleeper_rings" in data["summary_metrics"]
    assert "active_formation_rings" in data["summary_metrics"]
    assert "Evaluated across the full population of N=19 late-forming rings" in data["qualifier"]


def test_temporal_escalation_ring_endpoint():
    """Verifies /api/temporal-escalation/ring/{ring_id} endpoint for PROMO_001."""
    res = client.get("/api/temporal-escalation/ring/PROMO_001")
    assert res.status_code == 200
    data = res.json()
    assert data["ring_id"] == "PROMO_001"
    assert "checkpoint_history" in data
    assert len(data["checkpoint_history"]) == 5


def test_handcrafted_adversarial_summary_endpoint():
    """Verifies /api/handcrafted-adversarial/summary endpoint across 25 topologies."""
    res = client.get("/api/handcrafted-adversarial/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["total_topologies_evaluated"] == 25
    assert data["overall_sentinel_effective_recall_pct"] >= 80.0
    assert "Independent out-of-distribution structural stress battery" in data["qualifier"]


def test_handcrafted_adversarial_topology_endpoint():
    """Verifies /api/handcrafted-adversarial/topology/{topo_id} endpoint."""
    res = client.get("/api/handcrafted-adversarial/topology/TOPO_01_DENSE_CLIQUE_CAMO")
    assert res.status_code == 200
    data = res.json()
    assert data["topo_id"] == "TOPO_01_DENSE_CLIQUE_CAMO"
    assert "decision_breakdown" in data


def test_gateway_latency_results_artifact():
    """Verifies gateway_latency_results.json artifact exists, parses, and contains required fields."""
    artifact_path = BASE_DIR / "evals" / "results" / "gateway_latency_results.json"
    assert artifact_path.exists(), "gateway_latency_results.json must exist"
    with open(artifact_path, "r") as f:
        data = json.load(f)
    assert "qualifier" in data
    assert "Prototype design-target measured in a local single-machine mock environment" in data["qualifier"]
    assert "sync_path" in data
    assert "async_path" in data
    for path_key in ["sync_path", "async_path"]:
        assert "p50_ms" in data[path_key]
        assert "p95_ms" in data[path_key]
        assert "p99_ms" in data[path_key]
        assert data[path_key]["p50_ms"] >= 0.0
        assert data[path_key]["p95_ms"] >= 0.0
        assert data[path_key]["p99_ms"] >= 0.0