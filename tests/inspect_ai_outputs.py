import sys, io, json, warnings
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from decision.decision_engine import DecisionEngine, Decision
from ai.evidence_reasoner import EvidenceGapReasoner, validate_llm_output, build_evidence_payload
from policy.policy_gate import PolicyGate

events = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels = pd.read_parquet("data/labels.parquet")
rings = pd.read_parquet("data/rings.parquet")
split = json.load(open("data/split_info.json"))

splits = build_temporal_splits(events, accounts, labels, split)
sp = splits["test"]
idx = sp["labels"].index
s_te = sp["struct"].reindex(idx).fillna(0)
b_te = sp["behav"].reindex(idx).fillna(0)

fused = joblib.load("models/fused_calibrated.pkl")
p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)

engine = DecisionEngine(kl_conflict_threshold=0.5)
n_orders_arr = b_te["n_orders"].fillna(0).astype(int).values
obs_days_arr = b_te["account_age_days"].fillna(0).values

results = engine.decide_batch(
    account_ids=list(idx),
    p_fused_matrix=p_fused,
    p_struct_matrix=p_struct,
    p_behav_matrix=p_behav,
    observation_days=obs_days_arr,
    n_orders_arr=n_orders_arr,
    as_of_ts=split["test_end_ts"],
)

res_dict = {r.account_id: r for r in results}
labels_meta = labels.set_index("account_id")
ring_acc = rings.drop_duplicates("account_id").set_index("account_id")

reasoner = EvidenceGapReasoner(mock=True)

# -----------------------------------------------------------------------------
# Case 1: Real Referral-Farming Account in REVIEW lane
# -----------------------------------------------------------------------------
print("================================================================================")
print("1. REAL REFERRAL-FARMING ACCOUNT IN REVIEW LANE (UNSEEN STRUCTURE)")
print("================================================================================")
ref_accs = [
    acc for acc in idx 
    if acc in ring_acc.index 
    and ring_acc.loc[acc, "ring_type"] == "referral_farming"
    and res_dict[acc].decision == Decision.REVIEW
]
target_ref = ref_accs[0]
r_ref = res_dict[target_ref]
pos_ref = list(idx).index(target_ref)

out_ref = reasoner.analyze(
    account_id=target_ref,
    struct_feats=s_te.loc[target_ref],
    behav_feats=b_te.loc[target_ref],
    p_fused=p_fused[pos_ref].tolist(),
    p_struct=p_struct[pos_ref].tolist(),
    p_behav=p_behav[pos_ref].tolist(),
    conflict_flag=r_ref.evidence_conflict,
    as_of_ts=split["test_end_ts"],
    known_ring_ids=[ring_acc.loc[target_ref, "ring_id"]] if target_ref in ring_acc.index else [],
)

print(f"Account ID: {target_ref}")
print(f"Decision: {r_ref.decision.value} | Lane: {r_ref.routing_lane.value} | sym_KL: {r_ref.sym_kl_divergence:.3f}")
print(f"P(abusive)={r_ref.p_abusive:.3f} | P(struct)={r_ref.structural_sub_score:.3f} | P(behav)={r_ref.behavioral_sub_score:.3f}")
print("\n--- Evidence Payload sent to Reasoner ---")
print(json.dumps(out_ref["payload"], indent=2))
print("\n--- Literal Generated Reasoner Output ---")
print(json.dumps(out_ref["llm_output"], indent=2))
print(f"\nBoundary Valid: {out_ref['boundary_valid']} | Violations: {out_ref['boundary_violations']}")
print()

# -----------------------------------------------------------------------------
# Case 2: Real hard_bc Account (Benign-Dense Stress Test)
# -----------------------------------------------------------------------------
print("================================================================================")
print("2. REAL hard_bc ACCOUNT (BENIGN-COORDINATED WITH INJECTED PAYOUT)")
print("================================================================================")
hard_bc_accs = [
    acc for acc in idx 
    if acc in labels_meta.index
    and labels_meta.loc[acc, "counterfactual_subset"] == "hard_bc"
]
target_hbc = hard_bc_accs[0]
r_hbc = res_dict[target_hbc]
pos_hbc = list(idx).index(target_hbc)

out_hbc = reasoner.analyze(
    account_id=target_hbc,
    struct_feats=s_te.loc[target_hbc],
    behav_feats=b_te.loc[target_hbc],
    p_fused=p_fused[pos_hbc].tolist(),
    p_struct=p_struct[pos_hbc].tolist(),
    p_behav=p_behav[pos_hbc].tolist(),
    conflict_flag=r_hbc.evidence_conflict,
    as_of_ts=split["test_end_ts"],
    known_ring_ids=[],
)

print(f"Account ID: {target_hbc}")
print(f"Decision: {r_hbc.decision.value} | Lane: {r_hbc.routing_lane.value} | sym_KL: {r_hbc.sym_kl_divergence:.3f}")
print(f"P(abusive)={r_hbc.p_abusive:.3f} | P(struct)={r_hbc.structural_sub_score:.3f} | P(behav)={r_hbc.behavioral_sub_score:.3f}")
print("\n--- Evidence Payload sent to Reasoner ---")
print(json.dumps(out_hbc["payload"], indent=2))
print("\n--- Literal Generated Reasoner Output ---")
print(json.dumps(out_hbc["llm_output"], indent=2))
print(f"\nBoundary Valid: {out_hbc['boundary_valid']} | Violations: {out_hbc['boundary_violations']}")
print()

# -----------------------------------------------------------------------------
# Case 4: Adversarial Case with Empty/Sparse Structural Data
# -----------------------------------------------------------------------------
print("================================================================================")
print("4. ADVERSARIAL CASE: ZERO GRAPH EDGES / SPARSE METRICS")
print("================================================================================")
adv_account = "ACC_99991"
sparse_struct = pd.Series({
    "shared_payout_degree": 0, "multi_signal_edges": 0, "shared_device_degree": 0,
    "shared_ip_degree": 0, "referral_degree": 0, "community_size": 1, "connected_component_size": 1
})
sparse_behav = pd.Series({
    "n_orders": 3, "n_returns": 0, "return_rate": 0.0, "promo_rate": 0.0, "burst_score": 1,
    "account_age_days": 2.0, "n_referrals_received": 0, "n_referrals_sent": 0
})

out_adv = reasoner.analyze(
    account_id=adv_account,
    struct_feats=sparse_struct,
    behav_feats=sparse_behav,
    p_fused=[0.85, 0.10, 0.05],
    p_struct=[0.90, 0.08, 0.02],
    p_behav=[0.80, 0.12, 0.08],
    conflict_flag=False,
    as_of_ts=split["test_end_ts"],
    known_ring_ids=[],
    known_shared_entity_ids=[],
)

print(f"Account ID: {adv_account}")
print("\n--- Evidence Payload sent to Reasoner ---")
print(json.dumps(out_adv["payload"], indent=2))
print("\n--- Literal Generated Reasoner Output ---")
print(json.dumps(out_adv["llm_output"], indent=2))
print(f"\nBoundary Valid: {out_adv['boundary_valid']} | Violations: {out_adv['boundary_violations']}")