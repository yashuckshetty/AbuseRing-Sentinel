"""
Unit and integration tests for FastAPI backend service.
Verifies all read-only artifact endpoints and live DecisionEngine execution.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["artifacts_loaded"] is True
    assert data["files_checked"] == 9
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

