"""
AbuseRing Sentinel — Curated Representative Account Cases
==========================================================
Single-source repository of the 6 canonical representative cases evaluated
across live API endpoints, dashboard interactive selectors, and offline demo CLI.
"""

from typing import List, Dict, Any

CURATED_CASES: List[Dict[str, Any]] = [
    {
        "account_id": "ACC_03653",
        "category": "Hard BC (Benign Family + Injected Shared Payout)",
        "description": "Family group sharing a payout edge. Both structural and behavioral models independently and correctly classify this account as benign (P(struct)[AC]=0.00, P(behav)[AC]=0.00; P(fused)[Benign Coord]=0.9999). Because both models strongly AGREE (sym_KL=0.0005, well below the 0.50 conflict threshold), the system confidently resolves to WAIT_MONITOR without requiring human review. This serves as the clean agreement counterexample to the referral/sleeper conflict cases.",
        "expected_decision": "WAIT_MONITOR",
        "expected_lane": "fused_auto",
    },
    {
        "account_id": "ACC_04870",
        "category": "Referral Farming (Unseen Ring Topology)",
        "description": "True referral-farming ring member (REFARM_057). Never seen in train/val. Zero shared payout infrastructure (P(struct)[AC]=0.00), but high referral velocity (P(behav)[AC]=0.94). Extreme evidence DISAGREEMENT (sym_KL=10.22 >> 0.50) safely routes account to REVIEW for human adjudication rather than suppressing it as a false negative.",
        "expected_decision": "REVIEW",
        "expected_lane": "conflict_review",
    },
    {
        "account_id": "ACC_04430",
        "category": "Sleeper Account (Sparse Behavioral Evidence)",
        "description": "Pre-positioned sleeper account with mature structural connections (P(struct)[AC]=1.00) but sparse initial order velocity (P(behav)[AC]=0.63). Evidence DISAGREEMENT (sym_KL=1.55 >> 0.50) triggers conflict routing into REVIEW.",
        "expected_decision": "REVIEW",
        "expected_lane": "conflict_review",
    },
    {
        "account_id": "ACC_04295",
        "category": "Promo Abuse Ring Member (Concordant High Risk)",
        "description": "Coordinated promo ring member where both models strongly agree on abuse (P(struct)[AC]=1.00, P(behav)[AC]=1.00; sym_KL=0.0010). High confidence + near-zero conflict safely executes direct auto-ACT.",
        "expected_decision": "ACT",
        "expected_lane": "fused_auto",
    },
    {
        "account_id": "ACC_00505",
        "category": "Benign Independent (Standard Customer)",
        "description": "Standard independent customer with no shared entities and normal order cadence. Both models agree as benign (P(struct)[AC]=0.00, P(behav)[AC]=0.00; P(fused)[Benign Indep]=0.9997). Low divergence (sym_KL=0.28 < 0.50) safely routes to WAIT_MONITOR.",
        "expected_decision": "WAIT_MONITOR",
        "expected_lane": "fused_auto",
    },
    {
        "account_id": "ACC_04987",
        "category": "Cold-Start Account (Insufficient Orders)",
        "description": "New account with only 1 order (n_orders=1). Deterministic evidence gate enforces ABSTAIN regardless of model output (P(behav)[AC]=0.94), preventing premature automated enforcement.",
        "expected_decision": "ABSTAIN",
        "expected_lane": "abstain",
    },
]
