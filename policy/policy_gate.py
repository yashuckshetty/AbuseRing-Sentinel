"""
AbuseRing Sentinel — Deterministic Policy Gate
===============================================
Sits between the AI advisory component and any downstream action.
FULLY DETERMINISTIC — LLM output is advisory only and cannot influence the
numeric decision.

Invariants:
  1. Decision (ACT/REVIEW/WAIT/ABSTAIN) is made by the cost model only.
  2. LLM report is APPENDED to the audit trail as advisory text only.
  3. The gate cannot be bypassed — every decision flows through here.
  4. Full audit trail is emitted for every case.
  5. No irreversible action is ever taken. All outputs are human review flags.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from decision.decision_engine import Decision, DecisionResult, DecisionEngine, CostConfig
from ai.evidence_reasoner import EvidenceGapReasoner


@dataclass
class PolicyDecision:
    """Final output of the policy gate — the only thing that leaves to human reviewers."""
    account_id: str
    final_decision: str
    p_abusive: float
    evidence_conflict: bool
    decision_rationale: str
    ai_advisory: Optional[str]
    ai_boundary_valid: bool
    ai_violations: list
    audit_trail: dict
    created_at: float


class PolicyGate:
    """
    Deterministic policy gate.
    The LLM is invoked only when evidence_conflict=True.
    The LLM output NEVER modifies the decision — only adds advisory text.
    """

    AUDIT_DIR = Path("policy/audit_log")

    def __init__(
        self,
        decision_engine: Optional[DecisionEngine] = None,
        reasoner: Optional[EvidenceGapReasoner] = None,
        write_audit_log: bool = True,
    ):
        self.engine   = decision_engine or DecisionEngine()
        self.reasoner = reasoner or EvidenceGapReasoner(mock=True)
        self.write_audit_log = write_audit_log
        if write_audit_log:
            self.AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        account_id: str,
        p_fused: list,
        p_struct: list,
        p_behav: list,
        conflict_flag: bool,
        struct_feats: pd.Series,
        behav_feats: pd.Series,
        as_of_ts: int,
        known_ring_ids: Optional[list] = None,
        known_shared_entity_ids: Optional[list] = None,
    ) -> PolicyDecision:
        """Full policy gate pipeline: cost-model decision → LLM advisory → audit."""
        p_bi, p_bc, p_ac = float(p_fused[0]), float(p_fused[1]), float(p_fused[2])
        struct_sub = float(p_struct[2])
        behav_sub  = float(p_behav[2])

        n_orders  = int(behav_feats.get("n_orders", 0))
        obs_days  = float(behav_feats.get("account_age_days", 0))

        # ── STEP 1: Deterministic cost-model decision (routing-based) ─────────
        decision_result = self.engine.decide(
            account_id=account_id,
            p_fused=np.array(p_fused),
            p_struct=np.array(p_struct),
            p_behav=np.array(p_behav),
            observation_days=obs_days,
            n_orders=n_orders,
            as_of_ts=as_of_ts,
        )

        # ── STEP 2: LLM advisory (only if conflict, does NOT change decision) ─
        ai_advisory = None
        ai_boundary_valid = True
        ai_violations = []

        if conflict_flag:
            try:
                ai_result = self.reasoner.analyze(
                    account_id=account_id,
                    struct_feats=struct_feats,
                    behav_feats=behav_feats,
                    p_fused=p_fused,
                    p_struct=p_struct,
                    p_behav=p_behav,
                    conflict_flag=conflict_flag,
                    as_of_ts=as_of_ts,
                    known_ring_ids=known_ring_ids,
                    known_shared_entity_ids=known_shared_entity_ids,
                )
                llm_out = ai_result.get("llm_output", {})
                ai_advisory = (
                    f"Conflict explanation: {llm_out.get('conflict_explanation', 'N/A')}. "
                    f"Assessment: {llm_out.get('qualitative_assessment', 'N/A')}. "
                    f"Analyst suggestions: {'; '.join(llm_out.get('analyst_suggestions', [])[:2])}."
                )
                ai_boundary_valid = ai_result.get("boundary_valid", True)
                ai_violations = ai_result.get("boundary_violations", [])
            except Exception as e:
                ai_advisory = f"AI advisory unavailable: {str(e)}"
                ai_boundary_valid = False
                ai_violations = [str(e)]

        # ── STEP 3: Build full audit trail ───────────────────────────────────
        audit = {
            **decision_result.audit_trail,
            "ai_advisory": ai_advisory,
            "ai_boundary_valid": ai_boundary_valid,
            "ai_boundary_violations": ai_violations,
            "ai_note": (
                "AI output is ADVISORY ONLY. Final decision determined by "
                "deterministic cost model, not by AI component."
            ),
            "decision_pipeline_version": "1.0.0",
        }

        policy_decision = PolicyDecision(
            account_id=account_id,
            final_decision=decision_result.decision.value,
            p_abusive=round(p_ac, 4),
            evidence_conflict=conflict_flag,
            decision_rationale=decision_result.rationale,
            ai_advisory=ai_advisory,
            ai_boundary_valid=ai_boundary_valid,
            ai_violations=ai_violations,
            audit_trail=audit,
            created_at=float(time.time()),
        )

        if self.write_audit_log:
            log_path = self.AUDIT_DIR / f"{account_id}_{as_of_ts}.json"
            with open(log_path, "w") as f:
                json.dump(asdict(policy_decision), f, indent=2, default=str)

        return policy_decision

    def batch_process(
        self,
        account_ids: list,
        p_fused: np.ndarray,
        p_struct: np.ndarray,
        p_behav: np.ndarray,
        conflict_flags: np.ndarray,
        struct_df: pd.DataFrame,
        behav_df: pd.DataFrame,
        as_of_ts: int,
    ) -> list:
        results = []
        for i, acc in enumerate(account_ids):
            s = struct_df.loc[acc] if acc in struct_df.index else pd.Series()
            b = behav_df.loc[acc] if acc in behav_df.index else pd.Series()
            result = self.process(
                account_id=acc,
                p_fused=p_fused[i].tolist(),
                p_struct=p_struct[i].tolist(),
                p_behav=p_behav[i].tolist(),
                conflict_flag=bool(conflict_flags[i]),
                struct_feats=s,
                behav_feats=b,
                as_of_ts=as_of_ts,
            )
            results.append(result)
        return results


if __name__ == "__main__":
    gate = PolicyGate(write_audit_log=False)
    result = gate.process(
        account_id="ACC_00001",
        p_fused=[0.05, 0.10, 0.85],
        p_struct=[0.03, 0.07, 0.90],
        p_behav=[0.20, 0.40, 0.40],
        conflict_flag=True,
        struct_feats=pd.Series({
            "shared_payout_degree": 4, "multi_signal_edges": 2,
            "connected_component_size": 8, "referral_degree": 1,
        }),
        behav_feats=pd.Series({
            "n_orders": 5, "n_returns": 0, "return_rate": 0.0,
            "promo_rate": 0.8, "burst_score": 3, "account_age_days": 15.0,
        }),
        as_of_ts=1703000000,
    )
    print(f"Decision: {result.final_decision}")
    print(f"AI advisory: {str(result.ai_advisory)[:200]}")
    print(f"Boundary valid: {result.ai_boundary_valid}")
