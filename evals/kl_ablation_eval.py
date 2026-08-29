import sys, io, json, os, warnings
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from decision.decision_engine import DecisionEngine, Decision, RoutingLane
from models.fused_model import FusedCalibratedClassifier

# 1. Load data & models
events = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels = pd.read_parquet("data/labels.parquet")
rings = pd.read_parquet("data/rings.parquet")
split = json.load(open("data/split_info.json"))

splits = build_temporal_splits(events, accounts, labels, split)
test_split = splits["test"]
idx = test_split["labels"].index
y_true_str = test_split["labels"]["label_str"].values
y_true_ac = (y_true_str == "abusive_coordinated")

s_te = test_split["struct"].reindex(idx).fillna(0)
b_te = test_split["behav"].reindex(idx).fillna(0)
n_orders = b_te["n_orders"].astype(int).values
obs_days = b_te["account_age_days"].values
as_of_ts = split["test_end_ts"]

fused = joblib.load("models/fused_calibrated.pkl")
p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)

# 2. Decision Path A: KL-Routing (The Proposed System)
engine_kl = DecisionEngine(kl_conflict_threshold=0.5)
dec_kl = engine_kl.decide_batch(
    account_ids=list(idx),
    p_fused_matrix=p_fused,
    p_struct_matrix=p_struct,
    p_behav_matrix=p_behav,
    observation_days=obs_days,
    n_orders_arr=n_orders,
    as_of_ts=as_of_ts,
)

# 3. Decision Path B: Simple Threshold-Only Ablation (No KL Routing / No REVIEW lane)
# Rule: ABSTAIN if n_orders < 2; else ACT if p_fused[AC] >= 0.70; else WAIT_MONITOR
dec_ablation = []
for i, acc in enumerate(idx):
    if n_orders[i] < 2:
        d = Decision.ABSTAIN
    elif p_fused[i, 2] >= 0.70:
        d = Decision.ACT
    else:
        d = Decision.WAIT_MONITOR
    dec_ablation.append(d)

# 4. Compare Full Test Population Performance
C_FP = 500.0
C_FN = 2000.0
C_REVIEW = 150.0
C_WAIT_PER_DAY = 50.0
TEST_WINDOW_DAYS = 18.0

def eval_system(decisions, name, has_review=True):
    decs = np.array([d.value if hasattr(d, "value") else d for d in decisions])
    
    n_total = len(decs)
    n_ac_total = int(y_true_ac.sum())
    n_non_ac_total = n_total - n_ac_total
    
    act_mask = (decs == "ACT")
    rev_mask = (decs == "REVIEW")
    wait_mask = (decs == "WAIT_MONITOR")
    abs_mask = (decs == "ABSTAIN")
    
    tp_act = int((act_mask & y_true_ac).sum())
    fp_act = int((act_mask & ~y_true_ac).sum())
    
    tp_rev = int((rev_mask & y_true_ac).sum())
    fp_rev = int((rev_mask & ~y_true_ac).sum())
    
    fn_wait = int((wait_mask & y_true_ac).sum())
    fn_abs = int((abs_mask & y_true_ac).sum())
    
    direct_prec = float(tp_act / (tp_act + fp_act)) if (tp_act + fp_act) > 0 else 0.0
    direct_rec = float(tp_act / n_ac_total)
    effective_rec = float((tp_act + tp_rev) / n_ac_total)
    
    cost_fp = float(fp_act * C_FP)
    cost_review = float((tp_rev + fp_rev) * C_REVIEW)
    cost_fn_wait = float(fn_wait * C_FN)
    cost_abs_wait = float(fn_abs * TEST_WINDOW_DAYS * C_WAIT_PER_DAY)
    cost_abs_fn = float(fn_abs * C_FN)
    
    total_cost_best = cost_fp + cost_review + cost_fn_wait + cost_abs_wait
    total_cost_worst = cost_fp + cost_review + cost_fn_wait + cost_abs_fn
    
    return {
        "name": name,
        "n_total": int(n_total),
        "n_ac": int(n_ac_total),
        "counts": {
            "ACT": int(act_mask.sum()),
            "REVIEW": int(rev_mask.sum()),
            "WAIT_MONITOR": int(wait_mask.sum()),
            "ABSTAIN": int(abs_mask.sum()),
        },
        "ac_breakdown": {
            "AC_in_ACT": int(tp_act),
            "AC_in_REVIEW": int(tp_rev),
            "AC_in_WAIT": int(fn_wait),
            "AC_in_ABSTAIN": int(fn_abs),
        },
        "non_ac_breakdown": {
            "NonAC_in_ACT_FP": int(fp_act),
            "NonAC_in_REVIEW": int(fp_rev),
            "NonAC_in_WAIT": int((wait_mask & ~y_true_ac).sum()),
            "NonAC_in_ABSTAIN": int((abs_mask & ~y_true_ac).sum()),
        },
        "direct_precision": round(direct_prec, 4),
        "direct_recall": round(direct_rec, 4),
        "effective_recall": round(effective_rec, 4),
        "costs": {
            "cost_fp": cost_fp,
            "cost_review": cost_review,
            "cost_fn_wait": cost_fn_wait,
            "cost_abs_wait": cost_abs_wait,
            "cost_abs_fn": cost_abs_fn,
            "total_cost_best": total_cost_best,
            "total_cost_worst": total_cost_worst,
        }
    }

res_kl = eval_system([d.decision for d in dec_kl], "KL-Routing Engine (KL=0.5)")
res_abl = eval_system(dec_ablation, "Threshold-Only Ablation (No KL / No Review)", has_review=False)

# 5. Evaluate on Specific Robustness Subsets
ring_acc_df = rings.drop_duplicates("account_id").set_index("account_id")

# A. Referral farming (unseen structure)
refarm_accs = [a for a in idx if a in ring_acc_df.index and ring_acc_df.loc[a, "ring_type"] == "referral_farming"]
# B. Sleeper accounts
sleeper_accs = [a for a in idx if a in ring_acc_df.index and ring_acc_df.loc[a, "is_sleeper"] == True]
# C. Hard BC (counterfactual benign family with injected shared payout)
if "counterfactual_subset" in labels.columns:
    cf_lookup = labels.set_index("account_id")["counterfactual_subset"].to_dict()
    hard_bc_accs = [a for a in idx if cf_lookup.get(a) == "hard_bc"]
elif "is_hard_bc" in labels.columns:
    hard_bc_lookup = labels.set_index("account_id")["is_hard_bc"].to_dict()
    hard_bc_accs = [a for a in idx if hard_bc_lookup.get(a, False) == True]
else:
    hard_bc_accs = [a for a in idx if test_split["labels"].loc[a, "label_str"] == "benign_coordinated" and s_te.loc[a, "shared_payout_degree"] > 0]

def eval_subset(subset_accs, name):
    sub_indices = [list(idx).index(a) for a in subset_accs]
    kl_decs = [dec_kl[i].decision.value for i in sub_indices]
    abl_decs = [dec_ablation[i].value for i in sub_indices]
    
    kl_counts = {str(k): int(v) for k, v in pd.Series(kl_decs).value_counts().items()}
    abl_counts = {str(k): int(v) for k, v in pd.Series(abl_decs).value_counts().items()}
    
    return {
        "subset_name": name,
        "n_accounts": int(len(subset_accs)),
        "kl_routing": kl_counts,
        "threshold_ablation": abl_counts,
    }

sub_refarm = eval_subset(refarm_accs, "Referral Farming (Unseen Structure)")
sub_sleeper = eval_subset(sleeper_accs, "Sleeper Accounts (Sparse Evidence)")
sub_hard_bc = eval_subset(hard_bc_accs, "Hard BC (Benign Family + Shared Payout)")

print("="*80)
print("KL-ROUTING ABLATION EXPERIMENT RESULTS")
print("="*80)

print("\n--- FULL TEST SPLIT COMPARISON (N=3,467, True AC=198) ---")
print(f"{'Metric':<32} | {'KL-Routing (KL=0.5)':<25} | {'Threshold-Only Ablation':<25}")
print("-" * 88)
print(f"{'Direct Auto-ACT Precision':<32} | {res_kl['direct_precision']*100:>23.2f}% | {res_abl['direct_precision']*100:>23.2f}%")
print(f"{'Direct Auto-ACT Recall':<32} | {res_kl['direct_recall']*100:>23.2f}% | {res_abl['direct_recall']*100:>23.2f}%")
print(f"{'Effective Recall (ACT+REV)':<32} | {res_kl['effective_recall']*100:>23.2f}% | {res_abl['effective_recall']*100:>23.2f}%")
print(f"{'False Positives in Auto-ACT':<32} | {res_kl['non_ac_breakdown']['NonAC_in_ACT_FP']:>24d} | {res_abl['non_ac_breakdown']['NonAC_in_ACT_FP']:>24d}")
print(f"{'Total True AC in Auto-ACT':<32} | {res_kl['ac_breakdown']['AC_in_ACT']:>24d} | {res_abl['ac_breakdown']['AC_in_ACT']:>24d}")
print(f"{'Total True AC in REVIEW':<32} | {res_kl['ac_breakdown']['AC_in_REVIEW']:>24d} | {res_abl['ac_breakdown']['AC_in_REVIEW']:>24d}")
print(f"{'Total True AC in WAIT (Missed/FN)':<32} | {res_kl['ac_breakdown']['AC_in_WAIT']:>24d} | {res_abl['ac_breakdown']['AC_in_WAIT']:>24d}")
print(f"{'Total True AC in ABSTAIN':<32} | {res_kl['ac_breakdown']['AC_in_ABSTAIN']:>24d} | {res_abl['ac_breakdown']['AC_in_ABSTAIN']:>24d}")
print(f"{'Total REVIEW Queue Volume':<32} | {res_kl['counts']['REVIEW']:>24d} | {res_abl['counts']['REVIEW']:>24d}")
print(f"{'Total Simulated Cost (Best Case)':<32} | Rs {res_kl['costs']['total_cost_best']:>21,.2f} | Rs {res_abl['costs']['total_cost_best']:>21,.2f}")
print(f"{'Total Simulated Cost (Worst Case)':<32} | Rs {res_kl['costs']['total_cost_worst']:>21,.2f} | Rs {res_abl['costs']['total_cost_worst']:>21,.2f}")

print("\n--- ROBUSTNESS SUBSETS BEHAVIOR ---")
for sub in [sub_hard_bc, sub_refarm, sub_sleeper]:
    print(f"\nSubset: {sub['subset_name']} (N={sub['n_accounts']})")
    print(f"  KL-Routing Decisions:         {sub['kl_routing']}")
    print(f"  Threshold-Ablation Decisions: {sub['threshold_ablation']}")

# Save full results to JSON and records to Parquet
output = {
    "full_test_split": {
        "kl_routing": res_kl,
        "threshold_ablation": res_abl,
    },
    "subsets": {
        "hard_bc": sub_hard_bc,
        "referral_farming": sub_refarm,
        "sleeper_accounts": sub_sleeper,
    }
}
os.makedirs("evals/results", exist_ok=True)
with open("evals/results/kl_ablation_results.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print("\nSaved full ablation results to evals/results/kl_ablation_results.json")