"""
Unit and Regression Tests for Longitudinal Escalation State Machine
====================================================================
Verifies:
  1. Temporal risk state transitions across ring lifecycle trajectories.
  2. Escalation lead-time calculations across the 19 late-forming rings.
  3. Proactive quarantine trigger mechanics under evidence divergence.
  4. Regression: DecisionEngine.decide() outputs are respected as sole authority.
  5. Mandatory synthetic data qualifier inclusion on all temporal policy outputs.
"""

import pytest
import pandas as pd
import numpy as np

from policy.temporal_escalation import (
    LongitudinalEscalationPolicy,
    TemporalRiskState,
    EscalationTransition,
    RingEscalationTrace,
)
from decision.decision_engine import (
    DecisionEngine,
    Decision,
    RoutingLane,
    sym_kl_divergence,
)


@pytest.fixture
def trajectory_dataset():
    df = pd.read_parquet("evals/results/trajectory_results.parquet")
    return df


@pytest.fixture
def escalation_policy():
    return LongitudinalEscalationPolicy(
        quarantine_review_threshold_pct=0.30,
        action_threshold_pct=0.50
    )


def test_evaluate_single_ring_trajectory(trajectory_dataset, escalation_policy):
    """Verify trace generation for a specific late-forming ring (e.g. PROMO_001)."""
    promo_df = trajectory_dataset[trajectory_dataset["ring_id"] == "PROMO_001"]
    assert len(promo_df) > 0

    trace: RingEscalationTrace = escalation_policy.evaluate_ring_trajectory(promo_df)
    assert trace.ring_id == "PROMO_001"
    assert "promo" in trace.ring_type
    assert trace.initial_state == TemporalRiskState.DORMANT_BASELINE
    assert trace.terminal_state in [TemporalRiskState.QUARANTINE_HOLD, TemporalRiskState.ENFORCED_ACTION]
    assert len(trace.transitions) >= 1
    assert len(trace.checkpoint_history) == 5
    assert "Evaluated across the full population of N=19 late-forming rings" in trace.qualifier


def test_evaluate_all_19_rings_summary(trajectory_dataset, escalation_policy):
    """Verify population-level evaluation across all 19 late-forming rings."""
    summary = escalation_policy.evaluate_all_rings(trajectory_dataset)

    assert summary["n_rings_evaluated"] == 19
    assert "promo_abuse" in summary["ring_types_evaluated"]
    assert "referral_farming" in summary["ring_types_evaluated"]
    assert "return_abuse" in summary["ring_types_evaluated"]

    metrics = summary["summary_metrics"]
    assert metrics["blended_mean_lead_time_vs_complete_days"] >= 0.0
    assert "pre_positioned_sleeper_rings" in metrics
    assert "active_formation_rings" in metrics
    assert metrics["pre_positioned_sleeper_rings"]["mean_lead_time_vs_complete_days"] > metrics["active_formation_rings"]["mean_lead_time_vs_complete_days"]
    assert len(summary["ring_traces"]) == 19
    assert "Evaluated across the full population of N=19 late-forming rings" in summary["qualifier"]


def test_referral_farming_divergence_tripwire(trajectory_dataset, escalation_policy):
    """
    Verify that unseen referral farming rings (e.g. REFARM_057) trigger DIVERGENT_REVIEW
    or QUARANTINE_HOLD via symmetric KL divergence even when structural model is near 0.
    """
    refarm_df = trajectory_dataset[trajectory_dataset["ring_id"] == "REFARM_057"]
    trace = escalation_policy.evaluate_ring_trajectory(refarm_df)

    # Must transition out of baseline into Divergent Review or Quarantine Hold
    states = [h["state"] for h in trace.checkpoint_history]
    assert any(s in [TemporalRiskState.DIVERGENT_REVIEW.value, TemporalRiskState.QUARANTINE_HOLD.value] for s in states)
    assert trace.escalation_lead_time_vs_complete_days is not None


def test_regression_decision_engine_authority():
    """
    REGRESSION TEST: Verify that the state machine consumes read-only
    DecisionEngine outputs without modifying decision logic or enums.
    """
    # Verify that Decision and RoutingLane enums are untouched
    assert list(Decision) == ["ACT", "REVIEW", "WAIT_MONITOR", "ABSTAIN"]
    assert list(RoutingLane) == ["conflict_review", "fused_auto", "abstain"]

    # Verify that canonical sym_kl_divergence is imported and intact
    p = np.array([0.9, 0.05, 0.05])
    q = np.array([0.05, 0.05, 0.9])
    kl = sym_kl_divergence(p, q)
    assert kl > 2.0
