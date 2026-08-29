import sys, io, json, warnings
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd, numpy as np, joblib
from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from decision.decision_engine import DecisionEngine, Decision
from ai.evidence_reasoner import EvidenceGapReasoner

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

# ── 1. Real Sleeper Account ──────────────────────────────────────────────────
sleeper_accs = [
    acc for acc in idx 
    if acc in ring_acc.index and ring_acc.loc[acc, "is_sleeper"] == True
]
target_sleep = sleeper_accs[0]
r_sleep = res_dict[target_sleep]
pos_sleep = list(idx).index(target_sleep)

out_sleep = reasoner.analyze(
    account_id=target_sleep,
    struct_feats=s_te.loc[target_sleep],
    behav_feats=b_te.loc[target_sleep],
    p_fused=p_fused[pos_sleep].tolist(),
    p_struct=p_struct[pos_sleep].tolist(),
    p_behav=p_behav[pos_sleep].tolist(),
    conflict_flag=r_sleep.evidence_conflict,
    as_of_ts=split["test_end_ts"],
    known_ring_ids=[ring_acc.loc[target_sleep, "ring_id"]],
)

print("================================================================================")
print("1. REAL SLEEPER ACCOUNT TEST-SPLIT EVALUATION")
print("================================================================================")
print(f"Account ID: {target_sleep}")
print(f"Decision: {r_sleep.decision.value} | Routing Lane: {r_sleep.routing_lane.value}")
print(f"Metrics: P(struct)={r_sleep.structural_sub_score:.4f}, P(behav)={r_sleep.behavioral_sub_score:.4f}, sym_KL={r_sleep.sym_kl_divergence:.4f}")
print("\n[LITERAL GENERATED TEXT]:")
print(json.dumps(out_sleep["llm_output"], indent=2))
print(f"\nBoundary Valid: {out_sleep['boundary_valid']} | Violations: {out_sleep['boundary_violations']}")
print()

# ── 2. Real Referral-Farming Account ─────────────────────────────────────────
ref_accs = [
    acc for acc in idx 
    if acc in ring_acc.index and ring_acc.loc[acc, "ring_type"] == "referral_farming"
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
    known_ring_ids=[ring_acc.loc[target_ref, "ring_id"]],
)

print("================================================================================")
print("2. REAL REFERRAL-FARMING ACCOUNT TEST-SPLIT EVALUATION")
print("================================================================================")
print(f"Account ID: {target_ref}")
print(f"Decision: {r_ref.decision.value} | Routing Lane: {r_ref.routing_lane.value}")
print(f"Metrics: P(struct)={r_ref.structural_sub_score:.4f}, P(behav)={r_ref.behavioral_sub_score:.4f}, sym_KL={r_ref.sym_kl_divergence:.4f}")
print("\n[LITERAL GENERATED TEXT]:")
print(json.dumps(out_ref["llm_output"], indent=2))
print(f"\nBoundary Valid: {out_ref['boundary_valid']} | Violations: {out_ref['boundary_violations']}")
print()

# ── 3. Real hard_bc Account ──────────────────────────────────────────────────
hard_bc_accs = [
    acc for acc in idx 
    if acc in labels_meta.index and labels_meta.loc[acc, "counterfactual_subset"] == "hard_bc"
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

print("================================================================================")
print("3. REAL hard_bc ACCOUNT TEST-SPLIT EVALUATION")
print("================================================================================")
print(f"Account ID: {target_hbc}")
print(f"Decision: {r_hbc.decision.value} | Routing Lane: {r_hbc.routing_lane.value}")
print(f"Metrics: P(struct)={r_hbc.structural_sub_score:.4f}, P(behav)={r_hbc.behavioral_sub_score:.4f}, P(benign_coord)={out_hbc['payload']['p_benign_coord']:.4f}, sym_KL={r_hbc.sym_kl_divergence:.4f}")
print("\n[LITERAL GENERATED TEXT]:")
print(json.dumps(out_hbc["llm_output"], indent=2))
print(f"\nBoundary Valid: {out_hbc['boundary_valid']} | Violations: {out_hbc['boundary_violations']}")
print()