"""AI Boundary Test Suite - 8 tests verifying LLM cannot violate constraints."""
import json, sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from ai.evidence_reasoner import EvidenceGapReasoner, validate_llm_output, build_evidence_payload

@pytest.fixture
def reasoner():
    return EvidenceGapReasoner(mock=True)

def _make_payload(account_id="ACC_00001", spd=4, burst=5, promo=0.8, n_orders=6,
                   known_rings=None, shared_entities=None):
    struct_feats = pd.Series({"shared_payout_degree": spd, "multi_signal_edges": 2,
                               "connected_component_size": 8, "referral_degree": 1,
                               "shared_device_degree": 2, "shared_ip_degree": 1})
    behav_feats = pd.Series({"n_orders": n_orders, "n_returns": 0, "return_rate": 0.0,
                              "promo_rate": promo, "has_promo": int(promo > 0),
                              "burst_score": burst, "account_age_days": 15.0, "n_referrals_received": 2,
                              "n_referrals_sent": 0})
    return build_evidence_payload(account_id=account_id, struct_feats=struct_feats,
        behav_feats=behav_feats, p_fused=[0.05, 0.10, 0.85], p_struct=[0.03, 0.05, 0.92],
        p_behav=[0.25, 0.35, 0.40], conflict_flags=True, as_of_ts=1703000000,
        known_ring_ids=known_rings or [], known_shared_entity_ids=shared_entities or [])

def test_mock_response_passes_boundary(reasoner):
    result = reasoner.analyze(account_id="ACC_00001",
        struct_feats=pd.Series({"shared_payout_degree": 4, "multi_signal_edges": 2,
                                 "connected_component_size": 8, "referral_degree": 1}),
        behav_feats=pd.Series({"n_orders": 6, "n_returns": 0, "return_rate": 0.0,
                                "promo_rate": 0.8, "has_promo": 1, "burst_score": 5,
                                "account_age_days": 15.0, "n_referrals_received": 2, "n_referrals_sent": 0}),
        p_fused=[0.05, 0.10, 0.85], p_struct=[0.03, 0.05, 0.92], p_behav=[0.25, 0.35, 0.40],
        conflict_flag=True, as_of_ts=1703000000)
    assert result["boundary_valid"], f"Boundary violations: {result['boundary_violations']}"
    print(f"\n[PASS] Mock response passes all boundary checks")

def test_no_risk_score_in_output(reasoner):
    bad_output = {"risk_score": 0.87, "conflict_explanation": "high risk",
                  "key_signals": [], "analyst_suggestions": [], "qualitative_assessment": "bad"}
    payload = _make_payload()
    valid, violations = validate_llm_output(bad_output, payload)
    assert not valid, "Should have failed: risk_score is forbidden"
    assert any("risk_score" in v.lower() for v in violations)
    print(f"\n[PASS] risk_score correctly flagged as violation")

def test_no_hallucinated_account_ids(reasoner):
    bad_output = {"conflict_explanation": "Account ACC_99999 is suspicious",
                  "key_signals": ["ACC_99999 device match"], "analyst_suggestions": [],
                  "qualitative_assessment": "suspicious"}
    payload = _make_payload(account_id="ACC_00001")
    valid, violations = validate_llm_output(bad_output, payload)
    assert not valid, "Should have failed: ACC_99999 not in payload"
    assert any("hallucinated" in v.lower() for v in violations)
    print(f"\n[PASS] Hallucinated account IDs correctly detected")

def test_no_forbidden_actions(reasoner):
    bad_output = {"conflict_explanation": "This account is an abuser",
                  "key_signals": [], "analyst_suggestions": ["block this account immediately"],
                  "qualitative_assessment": "block user"}
    payload = _make_payload()
    valid, violations = validate_llm_output(bad_output, payload)
    assert not valid, "Should have failed: 'block' is a forbidden action"
    assert any("block" in v.lower() for v in violations)
    print(f"\n[PASS] Forbidden action 'block' correctly detected")

def test_no_hallucinated_ring_ids(reasoner):
    bad_output = {"conflict_explanation": "Part of ring PROMO_999 which is a fraud ring",
                  "key_signals": [], "analyst_suggestions": [], "qualitative_assessment": "ring member"}
    payload = _make_payload(known_rings=["PROMO_001"])
    valid, violations = validate_llm_output(bad_output, payload)
    assert not valid, "Should have failed: PROMO_999 not in known_ring_ids"
    assert any("ring" in v.lower() for v in violations)
    print(f"\n[PASS] Hallucinated ring ID correctly detected")

def test_missing_fields_handled_gracefully(reasoner):
    incomplete_output = {"conflict_explanation": "some explanation"}
    payload = _make_payload()
    valid, violations = validate_llm_output(incomplete_output, payload)
    assert not valid, "Should fail: missing required keys"
    assert any("missing" in v.lower() for v in violations)
    print(f"\n[PASS] Missing fields correctly detected")

def test_ambiguous_evidence_no_fabrication(reasoner):
    result = reasoner.analyze(account_id="ACC_00002",
        struct_feats=pd.Series({"shared_payout_degree": 0, "multi_signal_edges": 0,
                                 "connected_component_size": 1, "referral_degree": 0}),
        behav_feats=pd.Series({"n_orders": 2, "n_returns": 0, "return_rate": 0.0,
                                "promo_rate": 0.0, "has_promo": 0, "burst_score": 0,
                                "account_age_days": 3.0, "n_referrals_received": 0, "n_referrals_sent": 0}),
        p_fused=[0.7, 0.2, 0.1], p_struct=[0.6, 0.3, 0.1], p_behav=[0.4, 0.5, 0.1],
        conflict_flag=True, as_of_ts=1703000000)
    assert result["boundary_valid"]
    output_text = json.dumps(result["llm_output"])
    import re
    hallucinated_accs = set(re.findall(r"\bACC_\d{5}\b", output_text)) - {"ACC_00002"}
    assert not hallucinated_accs, f"Hallucinated accounts in ambiguous case: {hallucinated_accs}"
    print(f"\n[PASS] No fabrication in ambiguous evidence case")

def test_full_pipeline_does_not_modify_scores(reasoner):
    """Final integration: running LLM must not change p_abusive in the payload."""
    import copy
    p_fused = [0.05, 0.10, 0.85]
    p_struct = [0.03, 0.05, 0.92]
    p_behav = [0.25, 0.35, 0.40]
    original_p_fused = copy.deepcopy(p_fused)
    result = reasoner.analyze(account_id="ACC_00003",
        struct_feats=pd.Series({"shared_payout_degree": 4, "multi_signal_edges": 2,
                                 "connected_component_size": 8, "referral_degree": 1}),
        behav_feats=pd.Series({"n_orders": 5, "n_returns": 0, "return_rate": 0.0,
                                "promo_rate": 0.8, "has_promo": 1, "burst_score": 3,
                                "account_age_days": 12.0, "n_referrals_received": 1, "n_referrals_sent": 0}),
        p_fused=p_fused, p_struct=p_struct, p_behav=p_behav, conflict_flag=True, as_of_ts=1703000000)
    assert p_fused == original_p_fused, "LLM call mutated the original p_fused list!"
    llm_out = result.get("llm_output", {})
    assert "risk_score" not in llm_out, "LLM wrote a risk_score"
    print(f"\n[PASS] LLM call did not mutate or inject scores")

def test_no_hallucinated_entity_ids(reasoner):
    """Assert that unshared/hallucinated device, IP, payout, or instrument IDs trigger boundary violations."""
    bad_output_dev = {
        "conflict_explanation": "Device DEV_99999 is linked to abuse",
        "key_signals": [], "analyst_suggestions": [], "qualitative_assessment": "suspicious"
    }
    payload = _make_payload(shared_entities=["DEV_00001", "IP_00002"])
    valid, violations = validate_llm_output(bad_output_dev, payload)
    assert not valid, "Should have failed: DEV_99999 not in payload"
    assert any("device" in v.lower() for v in violations)

    bad_output_pay = {
        "conflict_explanation": "Linked to cashout destination PAY_UNSEEN_999",
        "key_signals": [], "analyst_suggestions": [], "qualitative_assessment": "suspicious"
    }
    valid, violations = validate_llm_output(bad_output_pay, payload)
    assert not valid, "Should have failed: PAY_UNSEEN_999 not in payload"
    assert any("payout" in v.lower() for v in violations)
    print(f"\n[PASS] Hallucinated entity IDs (DEV/PAY) correctly detected")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

