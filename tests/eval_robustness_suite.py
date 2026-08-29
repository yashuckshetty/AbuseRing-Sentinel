import sys, io, json, warnings
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from decision.decision_engine import DecisionEngine, Decision, RoutingLane

# Load data
events = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels = pd.read_parquet("data/labels.parquet")
rings = pd.read_parquet("data/rings.parquet")
split = json.load(open("data/split_info.json"))

# Build test split
splits = build_temporal_splits(events, accounts, labels, split)
sp = splits["test"]
idx = sp["labels"].index
s_te = sp["struct"].reindex(idx).fillna(0)
b_te = sp["behav"].reindex(idx).fillna(0)
y_te = sp["labels"]["label"].values
labels_test = sp["labels"]

# Load models
fused = joblib.load("models/fused_calibrated.pkl")
behav = joblib.load("models/behavioral_lgbm.pkl")
struct = joblib.load("models/structural_lgbm.pkl")

p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)
pred_behav = behav.predict(b_te[BEHAVIORAL_FEATURES].fillna(0))
pred_struct = struct.predict(s_te[STRUCTURAL_FEATURES].fillna(0))

# Decision engine
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

res_df = pd.DataFrame([
    {
        "account_id": r.account_id,
        "decision": r.decision.value,
        "routing_lane": r.routing_lane.value,
        "p_abusive": r.p_abusive,
        "sym_kl": r.sym_kl_divergence,
        "evidence_conflict": r.evidence_conflict,
        "n_orders": r.n_orders,
        "struct_sub": r.structural_sub_score,
        "behav_sub": r.behavioral_sub_score,
    }
    for r in results
]).set_index("account_id")

# Merge test split metadata
labels_meta = labels.set_index("account_id")
df = labels_test.join(labels_meta[["label_true", "partial_signal", "counterfactual_subset"]])
df = df.join(res_df).join(b_te[["n_orders", "account_age_days"]], rsuffix="_b")

# Merge ring info (many-to-one or one-to-one)
ring_acc = rings.drop_duplicates(subset=["account_id"]).set_index("account_id")
df = df.join(ring_acc[["ring_id", "ring_type", "is_sleeper", "is_varied_payout"]])

print("================================================================================")
print("STAGE 12a: ROBUSTNESS SUITE EVALUATION")
print("================================================================================")
print(f"Total test accounts: {len(df)}")
print(f"Total AC test accounts: {(df['label_true'] == 'abusive_coordinated').sum()}")
print(f"Total BC test accounts: {(df['label_true'] == 'benign_coordinated').sum()}")
print(f"Total BI test accounts: {(df['label_true'] == 'benign_independent').sum()}")
print()

# -----------------------------------------------------------------------------
# 1. Unseen Ring Structure: referral-farming rings
# -----------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print("1. UNSEEN RING STRUCTURE: REFERRAL-FARMING RINGS")
print("--------------------------------------------------------------------------------")
ac_df = df[df["label_true"] == "abusive_coordinated"]
for r_type in ["promo", "return", "referral_farming"]:
    sub = ac_df[ac_df["ring_type"] == r_type]
    n_sub = len(sub)
    if n_sub == 0:
        print(f"Ring Type: {r_type} -> 0 active in test window")
        continue
    act_c = (sub["decision"] == "ACT").sum()
    rev_c = (sub["decision"] == "REVIEW").sum()
    wait_c = (sub["decision"] == "WAIT_MONITOR").sum()
    abs_c = (sub["decision"] == "ABSTAIN").sum()
    
    eff_rec = (act_c + rev_c) / n_sub
    direct_act = act_c / n_sub
    rev_rec = rev_c / n_sub
    print(f"Ring Type: {r_type:18s} (N={n_sub:2d})")
    print(f"  Decisions: ACT={act_c:2d} ({direct_act*100:5.1f}%), REVIEW={rev_c:2d} ({rev_rec*100:5.1f}%), WAIT={wait_c:2d}, ABSTAIN={abs_c:2d} ({abs_c/n_sub*100:5.1f}%)")
    print(f"  Effective Recall (ACT+REVIEW): {eff_rec*100:5.1f}%")
    print(f"  Mean P(behav)={sub['behav_sub'].mean():.3f} | Mean P(struct)={sub['struct_sub'].mean():.3f} | Mean sym_KL={sub['sym_kl'].mean():.3f}")
    print()

# -----------------------------------------------------------------------------
# 2. Sparse Evidence: sleeper accounts (partial_signal=True / is_sleeper=True)
# -----------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print("2. SPARSE EVIDENCE: SLEEPER ACCOUNTS")
print("--------------------------------------------------------------------------------")
sleepers = ac_df[(ac_df["partial_signal"] == True) | (ac_df["is_sleeper"] == True)]
n_sleep = len(sleepers)
act_s = (sleepers["decision"] == "ACT").sum()
rev_s = (sleepers["decision"] == "REVIEW").sum()
wait_s = (sleepers["decision"] == "WAIT_MONITOR").sum()
abs_s = (sleepers["decision"] == "ABSTAIN").sum()

print(f"Active Sleepers in test window: N={n_sleep}")
print(f"  Decisions: ACT={act_s:2d} ({act_s/n_sleep*100:5.1f}%), REVIEW={rev_s:2d} ({rev_s/n_sleep*100:5.1f}%), WAIT={wait_s:2d}, ABSTAIN={abs_s:2d} ({abs_s/n_sleep*100:5.1f}%)")
print(f"  Effective Recall: {(act_s+rev_s)/n_sleep*100:5.1f}%")
print(f"  Mean P(behav)={sleepers['behav_sub'].mean():.3f} | Mean P(struct)={sleepers['struct_sub'].mean():.3f} | Mean sym_KL={sleepers['sym_kl'].mean():.3f}")
print(f"  Routing lanes: {dict(sleepers['routing_lane'].value_counts())}")
print()

# -----------------------------------------------------------------------------
# 3. Benign-Dense Stress Test: hard_bc counterfactual subset
# -----------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print("3. BENIGN-DENSE STRESS TEST: hard_bc COUNTERFACTUAL SUBSET")
print("--------------------------------------------------------------------------------")
bc_df = df[df["label_true"] == "benign_coordinated"]
hard_bc = bc_df[bc_df["counterfactual_subset"] == "hard_bc"]
reg_bc = bc_df[bc_df["counterfactual_subset"] != "hard_bc"]

print(f"General Benign Coordinated (reg_bc): N={len(reg_bc)}")
print(f"  Decisions: {dict(reg_bc['decision'].value_counts())}")
print(f"  Routing lanes: {dict(reg_bc['routing_lane'].value_counts())}")
fp_act_reg = (reg_bc["decision"] == "ACT").sum()
rev_reg = (reg_bc["decision"] == "REVIEW").sum()
print(f"  Auto-ACT FP Rate: {fp_act_reg/len(reg_bc)*100:5.2f}% ({fp_act_reg}/{len(reg_bc)})")
print(f"  Sent to REVIEW Queue Rate: {rev_reg/len(reg_bc)*100:5.2f}% ({rev_reg}/{len(reg_bc)})")
print(f"  Mean P(behav)={reg_bc['behav_sub'].mean():.3f} | Mean P(struct)={reg_bc['struct_sub'].mean():.3f} | Mean sym_KL={reg_bc['sym_kl'].mean():.3f}")
print()

print(f"Hard BC (hard_bc with shared payout): N={len(hard_bc)}")
print(f"  Decisions: {dict(hard_bc['decision'].value_counts())}")
print(f"  Routing lanes: {dict(hard_bc['routing_lane'].value_counts())}")
fp_act_hard = (hard_bc["decision"] == "ACT").sum()
rev_hard = (hard_bc["decision"] == "REVIEW").sum()
print(f"  Auto-ACT FP Rate: {fp_act_hard/len(hard_bc)*100:5.2f}% ({fp_act_hard}/{len(hard_bc)})")
print(f"  Sent to REVIEW Queue Rate: {rev_hard/len(hard_bc)*100:5.2f}% ({rev_hard}/{len(hard_bc)})")
print(f"  Mean P(behav)={hard_bc['behav_sub'].mean():.3f} | Mean P(struct)={hard_bc['struct_sub'].mean():.3f} | Mean sym_KL={hard_bc['sym_kl'].mean():.3f}")
print()

# -----------------------------------------------------------------------------
# 4. Low-Signal Abuse: varied_payout_ac subset
# -----------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print("4. LOW-SIGNAL ABUSE: varied_payout_ac SUBSET")
print("--------------------------------------------------------------------------------")
var_ac = ac_df[ac_df["counterfactual_subset"] == "varied_payout_ac"]
n_var = len(var_ac)
act_v = (var_ac["decision"] == "ACT").sum()
rev_v = (var_ac["decision"] == "REVIEW").sum()
wait_v = (var_ac["decision"] == "WAIT_MONITOR").sum()
abs_v = (var_ac["decision"] == "ABSTAIN").sum()

print(f"Varied Payout AC in test window: N={n_var}")
print(f"  Decisions: ACT={act_v:2d} ({act_v/n_var*100:5.1f}%), REVIEW={rev_v:2d} ({rev_v/n_var*100:5.1f}%), WAIT={wait_v:2d}, ABSTAIN={abs_v:2d} ({abs_v/n_var*100:5.1f}%)")
print(f"  Effective Recall: {(act_v+rev_v)/n_var*100:5.1f}%")
print(f"  Mean P(behav)={var_ac['behav_sub'].mean():.3f} | Mean P(struct)={var_ac['struct_sub'].mean():.3f} | Mean sym_KL={var_ac['sym_kl'].mean():.3f}")
print(f"  Routing lanes: {dict(var_ac['routing_lane'].value_counts())}")
print()

# -----------------------------------------------------------------------------
# 5. Cold-Start: n_orders < 2 population
# -----------------------------------------------------------------------------
print("--------------------------------------------------------------------------------")
print("5. COLD-START: n_orders < 2 POPULATION ACROSS ALL CLASSES")
print("--------------------------------------------------------------------------------")
cold = df[df["n_orders"] < 2]
print(f"Total cold-start accounts (n_orders < 2): N={len(cold)}")
print(f"  Class breakdown: {dict(cold['label_true'].value_counts())}")
print(f"  Decisions: {dict(cold['decision'].value_counts())}")
print(f"  Routing lanes: {dict(cold['routing_lane'].value_counts())}")
print(f"  % ABSTAINed: {(cold['decision'] == 'ABSTAIN').mean()*100:.1f}%")