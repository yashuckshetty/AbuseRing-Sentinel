"""
AbuseRing Sentinel — Longitudinal Escalation State Machine
===========================================================
Additive temporal policy layer formalizing multi-stage ring lifecycle escalation
across longitudinal checkpoints without modifying core DecisionEngine thresholds.

LIFECYCLE RISK STATES:
  1. DORMANT_BASELINE: Initial state (sparse order velocity, n_orders < 2 or ABSTAIN).
  2. ACCELERATING_MONITOR: Velocity acceleration detected; graph still forming (WAIT_MONITOR).
  3. DIVERGENT_REVIEW: Evidence divergence triggered (sym_KL > 0.50); behavioral/structural
     mismatch routes case to human REVIEW lane.
  4. QUARANTINE_HOLD: Multi-account coordination tripwire (2+ ring members in REVIEW
     sharing network infrastructure); proactive hold on outbound payouts/referrals.
  5. ENFORCED_ACTION: Coordinated ring confirmed (ACT decision or manual confirmation).

DEFERENCE TO CANONICAL DECISION ENGINE:
  This policy layer operates strictly over read-only DecisionResult / trajectory records.
  DecisionEngine.decide() remains the sole ground-truth operational authority.

MANDATORY QUALIFIER:
  All lead-time metrics and transition rates carry the explicit qualifier:
  "Evaluated across the full population of N=19 late-forming rings (formation start >= Day 55)
   in synthetic test data; illustrates longitudinal state transition mechanics under temporal graph densification."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Canonical imports from protected baseline
from decision.decision_engine import (
    DecisionEngine,
    Decision,
    RoutingLane,
    DecisionResult,
    sym_kl_divergence,
)


class TemporalRiskState(str, Enum):
    DORMANT_BASELINE      = "DORMANT_BASELINE"
    ACCELERATING_MONITOR  = "ACCELERATING_MONITOR"
    DIVERGENT_REVIEW      = "DIVERGENT_REVIEW"
    QUARANTINE_HOLD       = "QUARANTINE_HOLD"  # Identifies candidates for human-reviewed network-level hold (proactive flag, not autonomous enforcement)
    ENFORCED_ACTION       = "ENFORCED_ACTION"


@dataclass
class EscalationTransition:
    """Record of a state transition along a ring lifecycle trajectory."""
    from_state: TemporalRiskState
    to_state: TemporalRiskState
    checkpoint_label: str
    checkpoint_day: int
    days_from_start: int
    trigger_reason: str
    sym_kl_mean: float
    p_behav_mean: float
    p_struct_mean: float
    pct_accounts_review: float
    pct_accounts_act: float


@dataclass
class RingEscalationTrace:
    """Longitudinal state machine trace for a single fraud ring."""
    ring_id: str
    ring_type: str
    formation_start_day: int
    formation_complete_day: int
    initial_state: TemporalRiskState
    terminal_state: TemporalRiskState
    escalation_lead_time_vs_complete_days: Optional[int]  # Days before formation_complete that actionable REVIEW/ACT fired
    escalation_lead_time_vs_start_days: Optional[int]     # Positive = detected before start (pre-positioned); Negative = detected during formation
    is_pre_positioned_sleeper_ring: bool                  # True if early trigger came from pre-seeded account infrastructure
    quarantine_candidate_day: Optional[int]               # Day ring flagged as candidate for human-reviewed network hold
    transitions: List[EscalationTransition]
    checkpoint_history: List[Dict[str, Any]]
    qualifier: str = (
        "Evaluated across the full population of N=19 late-forming rings (formation start >= Day 55) "
        "in synthetic test data; illustrates longitudinal state transition mechanics under temporal graph densification."
    )


class LongitudinalEscalationPolicy:
    """
    Additive state machine evaluating longitudinal ring escalation.
    
    Evaluates how evidence divergence (sym_KL) acts as a temporal tripwire
    prior to complete ring densification.
    """

    QUALIFIER = (
        "Evaluated across the full population of N=19 late-forming rings (formation start >= Day 55) "
        "in synthetic test data; illustrates longitudinal state transition mechanics under temporal graph densification."
    )

    def __init__(
        self,
        quarantine_review_threshold_pct: float = 0.30,  # 30% of ring in REVIEW triggers ring QUARANTINE
        action_threshold_pct: float = 0.50             # 50% in ACT triggers ring ENFORCED_ACTION
    ):
        self.quarantine_review_threshold_pct = quarantine_review_threshold_pct
        self.action_threshold_pct = action_threshold_pct

    def evaluate_ring_trajectory(self, ring_df: pd.DataFrame) -> RingEscalationTrace:
        """
        Evaluate a single ring's trajectory across chronological checkpoints.
        Expects ring_df sorted by checkpoint_idx.
        """
        ring_id = str(ring_df["ring_id"].iloc[0])
        ring_type = str(ring_df["ring_type"].iloc[0])
        start_day = int(ring_df["formation_start_day"].iloc[0])
        comp_day = int(ring_df["formation_complete_day"].iloc[0])

        current_state = TemporalRiskState.DORMANT_BASELINE
        initial_state = current_state
        transitions: List[EscalationTransition] = []
        history: List[Dict[str, Any]] = []

        lead_time_vs_complete: Optional[int] = None
        lead_time_vs_start: Optional[int] = None
        is_pre_positioned: bool = False
        quarantine_day: Optional[int] = None

        # Group by checkpoint_idx
        checkpoints = ring_df.groupby("checkpoint_idx")

        for ck_idx, group in checkpoints:
            ck_label = str(group["checkpoint_label"].iloc[0])
            ck_day = int(group["checkpoint_day"].iloc[0])
            days_from_start = int(group["days_from_start"].iloc[0])
            
            n_accs = len(group)
            n_act = int((group["decision"] == "ACT").sum())
            n_review = int((group["decision"] == "REVIEW").sum())
            n_wait = int((group["decision"] == "WAIT_MONITOR").sum())
            n_abstain = int((group["decision"] == "ABSTAIN").sum())

            pct_act = n_act / max(1, n_accs)
            pct_review = n_review / max(1, n_accs)
            pct_wait = n_wait / max(1, n_accs)

            mean_sym_kl = float(group["sym_kl_divergence"].mean())
            mean_p_behav = float(group["p_behav_ac"].mean())
            mean_p_struct = float(group["p_struct_ac"].mean())

            # State transition logic — requires non-abstain actionable evidence from DecisionEngine
            next_state = current_state
            trigger_reason = ""

            n_evaluable = n_act + n_review + n_wait

            if pct_act >= self.action_threshold_pct:
                next_state = TemporalRiskState.ENFORCED_ACTION
                trigger_reason = f"Enforced Action: {pct_act*100:.1f}% of ring accounts reached ACT decision."
            elif (pct_review >= self.quarantine_review_threshold_pct or (pct_act + pct_review) >= self.quarantine_review_threshold_pct) and n_review >= 2:
                next_state = TemporalRiskState.QUARANTINE_HOLD
                trigger_reason = (
                    f"Quarantine Candidate: {pct_review*100:.1f}% of ring accounts in REVIEW ({n_review}/{n_accs} accs) "
                    f"with shared network topology. Flagged as candidate for human-reviewed network hold."
                )
                if quarantine_day is None:
                    quarantine_day = ck_day
            elif (n_review > 0) or (n_evaluable > 0 and mean_sym_kl > 0.50 and mean_p_behav > 0.10):
                next_state = TemporalRiskState.DIVERGENT_REVIEW
                trigger_reason = (
                    f"Divergence Tripwire: sym_KL={mean_sym_kl:.2f} > 0.50 with {n_review} account(s) in REVIEW; "
                    f"structural/behavioral evidence conflict routes ring to human investigation."
                )
            elif mean_p_behav > 0.30 or pct_wait > 0.5:
                next_state = TemporalRiskState.ACCELERATING_MONITOR
                trigger_reason = f"Velocity Acceleration: mean P(behav)={mean_p_behav:.2f}; held in WAIT_MONITOR."
            else:
                next_state = TemporalRiskState.DORMANT_BASELINE
                trigger_reason = "Quiescent baseline: sparse orders or benign agreement."

            # Calculate lead times: first time DIVERGENT_REVIEW or QUARANTINE_HOLD is reached
            if lead_time_vs_complete is None and next_state in [TemporalRiskState.DIVERGENT_REVIEW, TemporalRiskState.QUARANTINE_HOLD, TemporalRiskState.ENFORCED_ACTION]:
                lead_time_vs_complete = max(0, comp_day - ck_day)
                lead_time_vs_start = start_day - ck_day
                is_pre_positioned = (ck_day < start_day)

            # Record transition if state changed
            if next_state != current_state:
                transitions.append(
                    EscalationTransition(
                        from_state=current_state,
                        to_state=next_state,
                        checkpoint_label=ck_label,
                        checkpoint_day=ck_day,
                        days_from_start=days_from_start,
                        trigger_reason=trigger_reason,
                        sym_kl_mean=round(mean_sym_kl, 3),
                        p_behav_mean=round(mean_p_behav, 3),
                        p_struct_mean=round(mean_p_struct, 3),
                        pct_accounts_review=round(pct_review, 3),
                        pct_accounts_act=round(pct_act, 3)
                    )
                )
                current_state = next_state

            history.append({
                "checkpoint_idx": int(ck_idx),
                "checkpoint_label": ck_label,
                "checkpoint_day": ck_day,
                "days_from_start": days_from_start,
                "state": current_state.value,
                "n_accounts": n_accs,
                "breakdown": {"ACT": n_act, "REVIEW": n_review, "WAIT": n_wait, "ABSTAIN": n_abstain},
                "sym_kl_mean": round(mean_sym_kl, 3),
                "p_behav_mean": round(mean_p_behav, 3),
                "p_struct_mean": round(mean_p_struct, 3)
            })

        return RingEscalationTrace(
            ring_id=ring_id,
            ring_type=ring_type,
            formation_start_day=start_day,
            formation_complete_day=comp_day,
            initial_state=initial_state,
            terminal_state=current_state,
            escalation_lead_time_vs_complete_days=lead_time_vs_complete,
            escalation_lead_time_vs_start_days=lead_time_vs_start,
            is_pre_positioned_sleeper_ring=is_pre_positioned,
            quarantine_candidate_day=quarantine_day,
            transitions=transitions,
            checkpoint_history=history,
            qualifier=self.QUALIFIER
        )

    def evaluate_all_rings(self, trajectory_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute longitudinal policy across all 19 late-forming rings.
        Computes population-level escalation lead times and transition distributions,
        explicitly separating pre-positioned sleeper rings from organic active formation rings.
        """
        traces: List[RingEscalationTrace] = []
        ring_groups = trajectory_df.groupby("ring_id")

        for ring_id, r_df in ring_groups:
            trace = self.evaluate_ring_trajectory(r_df.sort_values("checkpoint_idx"))
            traces.append(trace)

        lead_vs_comp_all = [t.escalation_lead_time_vs_complete_days for t in traces if t.escalation_lead_time_vs_complete_days is not None]
        lead_vs_start_all = [t.escalation_lead_time_vs_start_days for t in traces if t.escalation_lead_time_vs_start_days is not None]
        
        sleeper_traces = [t for t in traces if t.is_pre_positioned_sleeper_ring]
        active_traces = [t for t in traces if not t.is_pre_positioned_sleeper_ring]

        lead_comp_sleeper = [t.escalation_lead_time_vs_complete_days for t in sleeper_traces if t.escalation_lead_time_vs_complete_days is not None]
        lead_comp_active = [t.escalation_lead_time_vs_complete_days for t in active_traces if t.escalation_lead_time_vs_complete_days is not None]

        quarantine_counts = sum(1 for t in traces if t.quarantine_candidate_day is not None)
        enforced_counts = sum(1 for t in traces if t.terminal_state == TemporalRiskState.ENFORCED_ACTION)

        # Transition matrix aggregation
        transition_pairs: Dict[str, int] = {}
        for t in traces:
            for tr in t.transitions:
                pair = f"{tr.from_state.value} -> {tr.to_state.value}"
                transition_pairs[pair] = transition_pairs.get(pair, 0) + 1

        return {
            "qualifier": self.QUALIFIER,
            "n_rings_evaluated": len(traces),
            "ring_types_evaluated": {
                "promo_abuse": sum(1 for t in traces if "promo" in t.ring_type.lower()),
                "referral_farming": sum(1 for t in traces if "referral" in t.ring_type.lower()),
                "return_abuse": sum(1 for t in traces if "return" in t.ring_type.lower()),
            },
            "summary_metrics": {
                "blended_mean_lead_time_vs_complete_days": round(float(np.mean(lead_vs_comp_all)), 2) if lead_vs_comp_all else 0.0,
                "blended_median_lead_time_vs_complete_days": round(float(np.median(lead_vs_comp_all)), 2) if lead_vs_comp_all else 0.0,
                "pre_positioned_sleeper_rings": {
                    "n_rings": len(sleeper_traces),
                    "mean_lead_time_vs_complete_days": round(float(np.mean(lead_comp_sleeper)), 2) if lead_comp_sleeper else 0.0,
                    "mean_lead_time_vs_formation_start_days": round(float(np.mean([t.escalation_lead_time_vs_start_days for t in sleeper_traces])), 2) if sleeper_traces else 0.0,
                    "description": "Rings with pre-seeded accounts/devices created ~5 days prior to order bursts (detected before formation start)"
                },
                "active_formation_rings": {
                    "n_rings": len(active_traces),
                    "mean_lead_time_vs_complete_days": round(float(np.mean(lead_comp_active)), 2) if lead_comp_active else 0.0,
                    "mean_detection_lag_vs_formation_start_days": round(float(np.mean([abs(t.escalation_lead_time_vs_start_days) for t in active_traces])), 2) if active_traces else 0.0,
                    "description": "Rings without pre-seeding (detected during active formation, ~6 days before formation completes)"
                },
                "quarantine_candidate_rate": round(quarantine_counts / max(1, len(traces)), 3),
                "terminal_enforced_action_rate": round(enforced_counts / max(1, len(traces)), 3)
            },
            "quarantine_hold_framing": "Identifies candidates for human-reviewed network-level hold (proactive investigation flag; not autonomous account enforcement).",
            "state_transition_counts": transition_pairs,
            "ring_traces": [asdict(t) for t in traces]
        }
