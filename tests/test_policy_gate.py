"""Policy gate integration test — updated for routing-based DecisionEngine (v2.0).

Tests verify the policy gate interface contract. Decision logic is driven by
the new engine which uses sym_KL(p_struct, p_behav) for conflict detection,
not an explicit `conflict_flag` scalar. Test cases are constructed to produce
the expected routing outcome via the probability vectors, not a flag.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from policy.policy_gate import PolicyGate, PolicyDecision
from decision.decision_engine import Decision, DecisionEngine


@pytest.fixture
def gate():
    return PolicyGate(write_audit_log=False)


def make_test_case(p_abusive, n_orders=5, force_conflict=False):
    """
    Build a test input for the policy gate.

    force_conflict=True constructs strongly disagreeing p_struct / p_behav
    vectors (sym_KL > 0.5) so the engine routes to REVIEW.
    force_conflict=False constructs identical vectors (sym_KL = 0) so the
    engine routes entirely on p_fused argmax.
    """
    p_bc = (1 - p_abusive) * 0.4
    p_bi = max(1.0 - p_abusive - p_bc, 0.0)
    p_fused = [p_bi, p_bc, p_abusive]

    if force_conflict:
        # struct says benign, behav says abusive -> high KL
        p_struct = [0.85, 0.10, 0.05]
        p_behav  = [0.02, 0.04, 0.94]
    else:
        # identical vectors -> KL = 0 (low conflict)
        p_struct = list(p_fused)
        p_behav  = list(p_fused)

    return {
        "p_fused":     p_fused,
        "p_struct":    p_struct,
        "p_behav":     p_behav,
        "conflict_flag": force_conflict,  # passed to gate but engine ignores it
        "struct_feats": pd.Series({"shared_payout_degree": 3, "multi_signal_edges": 1}),
        "behav_feats":  pd.Series({"n_orders": n_orders, "account_age_days": 10.0}),
    }


def test_act_decision_for_high_confidence(gate):
    """Low-conflict + p_abusive >= 0.70 must yield ACT."""
    tc = make_test_case(p_abusive=0.80, force_conflict=False, n_orders=5)
    result = gate.process(account_id="ACC_test_01", as_of_ts=1703000000, **tc)
    assert result.final_decision == "ACT", f"Expected ACT, got {result.final_decision}"


def test_review_decision_for_conflict(gate):
    """High KL conflict (struct disagrees with behav) must yield REVIEW regardless of p_abusive."""
    tc = make_test_case(p_abusive=0.80, force_conflict=True, n_orders=5)
    result = gate.process(account_id="ACC_test_02", as_of_ts=1703000000, **tc)
    assert result.final_decision == "REVIEW", f"Expected REVIEW (conflict), got {result.final_decision}"


def test_abstain_for_insufficient_evidence(gate):
    """n_orders < 2 must yield ABSTAIN regardless of score."""
    tc = make_test_case(p_abusive=0.90, force_conflict=False, n_orders=1)
    result = gate.process(account_id="ACC_test_03", as_of_ts=1703000000, **tc)
    assert result.final_decision == "ABSTAIN", f"Expected ABSTAIN, got {result.final_decision}"


def test_conflict_forces_review(gate):
    """Evidence conflict (high sym_KL) must force REVIEW even at p_abusive >= 0.70."""
    tc = make_test_case(p_abusive=0.85, force_conflict=True, n_orders=6)
    result = gate.process(account_id="ACC_test_04", as_of_ts=1703000000, **tc)
    assert result.final_decision == "REVIEW", f"Expected REVIEW (conflict), got {result.final_decision}"


def test_ai_advisory_does_not_change_decision(gate):
    """The AI advisory text must not alter the final decision."""
    tc = make_test_case(p_abusive=0.80, force_conflict=True, n_orders=5)
    result = gate.process(account_id="ACC_test_05", as_of_ts=1703000000, **tc)
    # Decision must still be REVIEW (conflict), not ACT
    assert result.final_decision == "REVIEW"
    # AI advisory exists but does not change decision
    if result.ai_advisory:
        assert result.ai_boundary_valid, f"AI boundary violated: {result.ai_violations}"


def test_audit_trail_emitted(gate):
    """Every decision must have a complete audit trail."""
    tc = make_test_case(p_abusive=0.45, force_conflict=False, n_orders=3)
    result = gate.process(account_id="ACC_test_06", as_of_ts=1703000000, **tc)
    required_keys = {
        "as_of_ts", "p_abusive", "e_cost_act", "e_cost_review", "e_cost_wait",
        "ai_note", "cost_config_note",
    }
    missing = required_keys - set(result.audit_trail.keys())
    assert not missing, f"Audit trail missing keys: {missing}"


def test_wait_for_low_confidence(gate):
    """Low-conflict + p_abusive < THRESHOLD_ACT must yield WAIT_MONITOR."""
    tc = make_test_case(p_abusive=0.10, force_conflict=False, n_orders=5)
    result = gate.process(account_id="ACC_test_07", as_of_ts=1703000000, **tc)
    assert result.final_decision == "WAIT_MONITOR", f"Expected WAIT_MONITOR, got {result.final_decision}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
