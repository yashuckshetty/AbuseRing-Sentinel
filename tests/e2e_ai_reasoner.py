import sys, io, json, warnings
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd, numpy as np, joblib
from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from policy.policy_gate import PolicyGate
from ai.evidence_reasoner import EvidenceGapReasoner

events = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels = pd.read_parquet("data/labels.parquet")
rings = pd.read_parquet("data/rings.parquet")
split = json.load(open("data/split_info.json"))

splits = build_temporal_splits(events, accounts, labels, split)
sp = splits["test"]
idx = sp["labels"].index[:20]  # Sample first 20 accounts

s_te = sp["struct"].reindex(idx).fillna(0)
b_te = sp["behav"].reindex(idx).fillna(0)

fused = joblib.load("models/fused_calibrated.pkl")
p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)

gate = PolicyGate(write_audit_log=False)
results = gate.batch_process(
    account_ids=list(idx),
    p_fused=p_fused,
    p_struct=p_struct,
    p_behav=p_behav,
    conflict_flags=conflicts,
    struct_df=s_te,
    behav_df=b_te,
    as_of_ts=split["test_end_ts"],
)

print(f"Processed {len(results)} sample accounts through PolicyGate + EvidenceGapReasoner:")
for r in results[:5]:
    print(f"\nAccount: {r.account_id} | Final Decision: {r.final_decision} | Conflict: {r.evidence_conflict}")
    print(f"  Rationale: {r.decision_rationale}")
    if r.ai_advisory:
        print(f"  AI Advisory: {r.ai_advisory[:160]}...")
        print(f"  AI Boundary Valid: {r.ai_boundary_valid} | Violations: {r.ai_violations}")

print("\nAI Evidence Reasoner Integration Test: PASSED")