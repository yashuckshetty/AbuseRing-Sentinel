"""AbuseRing Sentinel - Evidence-Gap Reasoner (LLM Component)
Safety: LLM cannot write risk score, reference entities not in payload, or trigger actions.
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))

SYSTEM_PROMPT = """You are a payment fraud analyst assistant helping human reviewers evaluate flagged accounts.
Analyze the provided evidence payload and respond with JSON only.

Strict Constraints:
1. Only reference entity IDs explicitly present in the input payload.
2. Do NOT generate or suggest any numeric "risk_score" or probability.
3. Do NOT recommend operational actions (never use words like 'block', 'terminate', 'suspend', 'ban').
4. Never state categorical conclusions like 'this is fraud' or 'this account is an abuser'.
5. Frame all observations objectively as evidence for human analyst judgment (e.g., 'Evidence suggests...', 'Human reviewer should inspect...').

Output JSON keys required:
- conflict_explanation: string describing the alignment or divergence between structural and behavioral signals
- key_signals: list of strings summarizing the primary observed metrics
- analyst_suggestions: list of strings suggesting specific data points for human verification
- qualitative_assessment: string summarizing whether evidence indicates potential coordination, benign sharing, or ambiguous signal"""

from decision.decision_engine import sym_kl_divergence


def build_evidence_payload(account_id, struct_feats, behav_feats, p_fused, p_struct, p_behav,
                            conflict_flags, as_of_ts, known_ring_ids=None, known_shared_entity_ids=None):
    p_f = np.array([float(x) for x in p_fused])
    p_s = np.array([float(x) for x in p_struct])
    p_b = np.array([float(x) for x in p_behav])
    kl_val = round(sym_kl_divergence(p_s, p_b), 4)

    return {
        "account_id": account_id,
        "structural_sub_score": round(float(p_s[2]), 4),
        "behavioral_sub_score": round(float(p_b[2]), 4),
        "p_abusive": round(float(p_f[2]), 4),
        "p_benign_coord": round(float(p_f[1]), 4),
        "p_benign_indep": round(float(p_f[0]), 4),
        "sym_kl_divergence": kl_val,
        "evidence_conflict": bool(conflict_flags or (kl_val > 0.50)),
        "conflict_reason": f"Structural={p_s[2]:.3f} vs Behavioral={p_b[2]:.3f} (sym_KL={kl_val:.3f})",
        "n_orders": int(behav_feats.get("n_orders", 0)),
        "n_returns": int(behav_feats.get("n_returns", 0)),
        "return_rate": round(float(behav_feats.get("return_rate", 0)), 3),
        "promo_rate": round(float(behav_feats.get("promo_rate", 0)), 3),
        "has_promo": int(behav_feats.get("has_promo", 0)),
        "n_referrals_received": int(behav_feats.get("n_referrals_received", 0)),
        "n_referrals_sent": int(behav_feats.get("n_referrals_sent", 0)),
        "burst_score": int(behav_feats.get("burst_score", 0)),
        "account_age_days": round(float(behav_feats.get("account_age_days", 0)), 1),
        "observation_window_days": round(float(behav_feats.get("account_age_days", 0)), 1),
        "shared_payout_degree": int(struct_feats.get("shared_payout_degree", 0)),
        "shared_device_degree": int(struct_feats.get("shared_device_degree", 0)),
        "shared_ip_degree": int(struct_feats.get("shared_ip_degree", 0)),
        "referral_degree": int(struct_feats.get("referral_degree", 0)),
        "multi_signal_edges": int(struct_feats.get("multi_signal_edges", 0)),
        "connected_component_size": int(struct_feats.get("connected_component_size", 1)),
        "known_ring_ids": known_ring_ids or [],
        "known_shared_entity_ids": (known_shared_entity_ids or [])[:10],
        "as_of_ts": as_of_ts,
    }

def _mock_response(payload):
    acc_id = payload["account_id"]
    spd = payload.get("shared_payout_degree", 0)
    sdd = payload.get("shared_device_degree", 0)
    sid = payload.get("shared_ip_degree", 0)
    ref_d = payload.get("referral_degree", 0)
    beh_sub = payload.get("behavioral_sub_score", 0.0)
    str_sub = payload.get("structural_sub_score", 0.0)
    p_ac = payload.get("p_abusive", 0.0)
    p_bc = payload.get("p_benign_coord", 0.0)
    burst = payload.get("burst_score", 0)
    promo = payload.get("promo_rate", 0.0)
    n_orders = payload.get("n_orders", 0)
    sym_kl = payload.get("sym_kl_divergence", 0.0)
    is_conflict = payload.get("evidence_conflict", False) or (sym_kl > 0.50)

    # ── Branch 1: Conflict / REVIEW Lane (sym_KL > 0.50) ─────────────────────
    if is_conflict:
        if str_sub > beh_sub:
            # Structural dominant: e.g. sleeper account or dormant ring member
            explanation = (
                f"Account {acc_id} exhibits strong structural graph links "
                f"(structural_sub_score={str_sub:.3f}, shared_payout_degree={spd}, shared_device_degree={sdd}) "
                f"while maintaining low individual behavioral velocity (behavioral_sub_score={beh_sub:.3f}, burst_score={burst}). "
                f"This pattern is consistent with sleeper accounts or dormant ring nodes pre-positioned for coordinated activity."
            )
            assessment = "Strong graph connectivity paired with low transaction velocity; warrants human review for shared cashout risk."
            suggestions = [
                f"Review other accounts linked to shared payout destinations with {acc_id}",
                f"Audit historical device and IP associations for known high-risk cluster membership",
            ]
        else:
            # Behavioral dominant: e.g. referral farming or isolated promo extraction
            explanation = (
                f"Account {acc_id} demonstrates elevated behavioral velocity "
                f"(behavioral_sub_score={beh_sub:.3f}, burst_score={burst}, promo_rate={promo:.3f}) "
                f"despite limited structural payout co-sharing (shared_payout_degree={spd}). "
                f"This divergence is characteristic of referral-farming or isolated promotional extraction "
                f"where shared cashout infrastructure has not yet manifested."
            )
            assessment = "Elevated behavioral activity with minimal payout-sharing; human review recommended to evaluate referral/promotional validity."
            suggestions = [
                f"Inspect referral chain and recipient account creation times linked to {acc_id}",
                f"Verify whether order timestamps correlate with promotional campaign launch windows",
            ]

    # ── Branch 2: High-Confidence Abusive (Convergent / Auto-ACT) ─────────────
    elif p_ac >= 0.70:
        explanation = (
            f"Account {acc_id} exhibits convergent high-risk signals across both structural "
            f"(structural_sub_score={str_sub:.3f}, shared_payout_degree={spd}) and behavioral "
            f"(behavioral_sub_score={beh_sub:.3f}, burst_score={burst}, promo_rate={promo:.3f}) modalities."
        )
        assessment = "Strong convergent evidence of coordinated abusive activity across graph and behavioral indicators."
        suggestions = [
            f"Cross-reference payout destination cluster with previous confirmed abuse cases",
            f"Review velocity of related accounts operating within the same burst window",
        ]

    # ── Branch 3: Benign Coordination / hard_bc ──────────────────────────────
    elif p_bc >= 0.50 or (spd > 0 or sdd > 0 or sid > 0):
        explanation = (
            f"Account {acc_id} shares infrastructure (shared_payout_degree={spd}, shared_device_degree={sdd}, shared_ip_degree={sid}) "
            f"but displays standard non-abusive purchasing patterns (behavioral_sub_score={beh_sub:.3f}, burst_score={burst}, promo_rate={promo:.3f}). "
            f"Evidence is consistent with benign family/household coordination rather than systematic exploitation."
        )
        assessment = "Consistent with legitimate family/household coordination; low probability of malicious intent."
        suggestions = [
            f"Confirm whether linked accounts share a common delivery address or household profile",
            f"Check if payment methods represent joint family accounts or standard shared instruments",
        ]

    # ── Branch 4: Insufficient Evidence / Cold-Start ─────────────────────────
    elif n_orders < 2:
        explanation = (
            f"Account {acc_id} has limited order history (n_orders={n_orders}). "
            f"Insufficient observations to establish reliable behavioral or graph patterns."
        )
        assessment = "Cold-start account with insufficient evidence; deferred from active evaluation."
        suggestions = [
            f"Allow account to accumulate operational history before re-evaluating",
        ]

    # ── Branch 5: Baseline Monitoring ────────────────────────────────────────
    else:
        explanation = (
            f"Account {acc_id} displays low connectivity (shared_payout_degree={spd}, shared_device_degree={sdd}) "
            f"and baseline transaction metrics (n_orders={n_orders}, burst_score={burst}, promo_rate={promo:.3f}). "
            f"No prominent structural or behavioral anomaly is currently evident."
        )
        assessment = "Standard profile; no significant divergence detected between structural and behavioral signals."
        suggestions = [
            f"Monitor account transaction velocity as additional order history accumulates",
            f"Check for newly emerging device or IP co-occurrences during future scoring cycles",
        ]

    key_signals = [
        f"shared_payout_degree={spd}",
        f"behavioral_sub_score={beh_sub:.4f}",
        f"structural_sub_score={str_sub:.4f}",
        f"burst_score={burst}",
        f"sym_kl={sym_kl:.4f}",
        f"n_orders={n_orders}",
    ]

    return {
        "conflict_explanation": explanation,
        "key_signals": key_signals,
        "analyst_suggestions": suggestions,
        "qualitative_assessment": assessment,
        "_is_mock": True,
    }

def call_llm(payload, api_key=None):
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "mock":
        return _mock_response(payload)
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        user_message = "Analyze the following account evidence and respond with JSON only:\n\n" + json.dumps(payload, indent=2)
        response = model.generate_content([{"role": "user", "parts": [SYSTEM_PROMPT + "\n\n" + user_message]}],
                                           generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text.strip())
    except Exception as e:
        return {"conflict_explanation": f"LLM call failed: {str(e)}", "key_signals": [],
                "analyst_suggestions": [], "qualitative_assessment": "API_ERROR"}

def validate_llm_output(llm_output, payload):
    violations = []
    required_keys = {"conflict_explanation", "key_signals", "analyst_suggestions", "qualitative_assessment"}
    missing = required_keys - set(llm_output.keys())
    if missing:
        violations.append(f"Missing required output keys: {missing}")
    
    output_text = json.dumps(llm_output)
    
    # Account ID hallucination check
    acc_pattern = re.compile(r'\bACC_\d{5}\b')
    mentioned_accounts = set(acc_pattern.findall(output_text))
    hallucinated_accounts = mentioned_accounts - {payload["account_id"]}
    if hallucinated_accounts:
        violations.append(f"Hallucinated account IDs not in payload: {hallucinated_accounts}")
    
    # Ring ID hallucination check
    ring_pattern = re.compile(r'\b(PROMO|RETURN|REFARM)_\d{3}\b')
    mentioned_rings = set(ring_pattern.findall(output_text))
    allowed_rings = set(payload.get("known_ring_ids", []))
    hallucinated_rings = mentioned_rings - allowed_rings
    if hallucinated_rings:
        violations.append(f"Hallucinated ring IDs not in payload: {hallucinated_rings}")

    # Entity ID hallucination check (Devices, IPs, Payouts, Instruments)
    dev_pattern = re.compile(r'\bDEV_\d{5}\b')
    ip_pattern = re.compile(r'\bIP_\d{5}\b')
    pay_pattern = re.compile(r'\bPAY_\w+\b')
    instr_pattern = re.compile(r'\bINSTR_\d{5}\b')

    allowed_entities = set(payload.get("known_shared_entity_ids", []))
    
    mentioned_devs = set(dev_pattern.findall(output_text)) - allowed_entities
    if mentioned_devs:
        violations.append(f"Hallucinated device IDs not in payload: {mentioned_devs}")
        
    mentioned_ips = set(ip_pattern.findall(output_text)) - allowed_entities
    if mentioned_ips:
        violations.append(f"Hallucinated IP IDs not in payload: {mentioned_ips}")
        
    mentioned_pays = set(pay_pattern.findall(output_text)) - allowed_entities
    if mentioned_pays:
        violations.append(f"Hallucinated payout IDs not in payload: {mentioned_pays}")
        
    mentioned_instrs = set(instr_pattern.findall(output_text)) - allowed_entities
    if mentioned_instrs:
        violations.append(f"Hallucinated instrument IDs not in payload: {mentioned_instrs}")

    # Risk score check
    if re.search(r'"risk_score"\s*:', output_text, re.IGNORECASE):
        violations.append("LLM wrote a 'risk_score' key - forbidden.")
        
    # Forbidden operational actions
    for action in ["block", "terminate", "suspend", "ban", "auto-"]:
        if action.lower() in output_text.lower():
            violations.append(f"LLM recommended forbidden action: '{action}'")
            
    return len(violations) == 0, violations

class EvidenceGapReasoner:
    def __init__(self, api_key=None, mock=False):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.mock = mock or not self.api_key

    def analyze(self, account_id, struct_feats, behav_feats, p_fused, p_struct, p_behav,
                conflict_flag, as_of_ts, known_ring_ids=None, known_shared_entity_ids=None):
        payload = build_evidence_payload(account_id=account_id, struct_feats=struct_feats,
            behav_feats=behav_feats, p_fused=p_fused, p_struct=p_struct, p_behav=p_behav,
            conflict_flags=conflict_flag, as_of_ts=as_of_ts, known_ring_ids=known_ring_ids,
            known_shared_entity_ids=known_shared_entity_ids)
        effective_key = "mock" if self.mock else self.api_key
        llm_output = call_llm(payload, api_key=effective_key)
        is_valid, violations = validate_llm_output(llm_output, payload)
        return {
            "account_id": account_id,
            "payload": payload,
            "llm_output": llm_output,
            "boundary_valid": is_valid,
            "boundary_violations": violations,
            "note": "LLM output is advisory only. Final decision made by deterministic policy gate."
        }

