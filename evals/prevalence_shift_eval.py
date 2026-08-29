import sys, io, json, os, warnings, shutil
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from data.simulator import generate_dataset, DEFAULT_SEED, N_ACCOUNTS
from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from decision.decision_engine import DecisionEngine, Decision, RoutingLane
from models.fused_model import FusedCalibratedClassifier

# Load already-trained baseline models (trained on ~15% baseline train split)
fused = joblib.load("models/fused_calibrated.pkl")
behav = joblib.load("models/behavioral_lgbm.pkl")
struct = joblib.load("models/structural_lgbm.pkl")
engine = DecisionEngine(kl_conflict_threshold=0.5)

C_FP = 500.0
C_FN = 2000.0
C_REVIEW = 150.0
C_WAIT_PER_DAY = 50.0
TEST_WINDOW_DAYS = 18.0

regimes = [
    {
        "name": "Low Prevalence (~3% AC)",
        "prevalence": {"benign_independent": 0.70, "benign_coordinated": 0.27, "abusive_coordinated": 0.03},
        "dir": "data/prevalence_low",
        "generate": True,
    },
    {
        "name": "Baseline Prevalence (~15% AC)",
        "prevalence": {"benign_independent": 0.60, "benign_coordinated": 0.25, "abusive_coordinated": 0.15},
        "dir": "data",
        "generate": False, # already generated
    },
    {
        "name": "High Prevalence (~30% AC)",
        "prevalence": {"benign_independent": 0.45, "benign_coordinated": 0.25, "abusive_coordinated": 0.30},
        "dir": "data/prevalence_high",
        "generate": True,
    },
]

results = []

for reg in regimes:
    reg_name = reg["name"]
    reg_dir = reg["dir"]
    
    if reg["generate"]:
        print(f"\n--- Generating dataset for regime: {reg_name} in {reg_dir} ---", flush=True)
        os.makedirs(reg_dir, exist_ok=True)
        generate_dataset(
            seed=DEFAULT_SEED,
            abuse_prevalence=reg["prevalence"],
            output_dir=reg_dir,
            counterfactual_hard_bc=True,
            counterfactual_varied_payout=True,
            n_accounts=N_ACCOUNTS,
        )
        
    events = pd.read_parquet(os.path.join(reg_dir, "events.parquet"))
    accounts = pd.read_parquet(os.path.join(reg_dir, "accounts.parquet"))
    labels = pd.read_parquet(os.path.join(reg_dir, "labels.parquet"))
    split = json.load(open(os.path.join(reg_dir, "split_info.json")))
    
    print(f"\n--- Evaluating regime: {reg_name} ---", flush=True)
    splits = build_temporal_splits(events, accounts, labels, split)
    test_sp = splits["test"]
    idx = test_sp["labels"].index
    
    y_true_str = test_sp["labels"]["label_str"].values
    y_true_ac = (y_true_str == "abusive_coordinated")
    n_ac = int(y_true_ac.sum())
    n_total = len(idx)
    actual_prev = float(n_ac / n_total)
    
    s_te = test_sp["struct"].reindex(idx).fillna(0)
    b_te = test_sp["behav"].reindex(idx).fillna(0)
    n_orders = b_te["n_orders"].astype(int).values
    obs_days = b_te["account_age_days"].values
    as_of_ts = split["test_end_ts"]
    
    p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)
    
    dec_results = engine.decide_batch(
        account_ids=list(idx),
        p_fused_matrix=p_fused,
        p_struct_matrix=p_struct,
        p_behav_matrix=p_behav,
        observation_days=obs_days,
        n_orders_arr=n_orders,
        as_of_ts=as_of_ts,
    )
    
    decs = np.array([d.decision.value for d in dec_results])
    
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
    direct_rec = float(tp_act / n_ac) if n_ac > 0 else 0.0
    effective_rec = float((tp_act + tp_rev) / n_ac) if n_ac > 0 else 0.0
    direct_f1 = float(2 * direct_prec * direct_rec / (direct_prec + direct_rec)) if (direct_prec + direct_rec) > 0 else 0.0
    
    # Standalone behavioral model evaluation on this regime
    pred_behav = np.argmax(p_behav, axis=1)
    behav_tp = int(((pred_behav == 2) & y_true_ac).sum())
    behav_fp = int(((pred_behav == 2) & ~y_true_ac).sum())
    behav_prec = float(behav_tp / (behav_tp + behav_fp)) if (behav_tp + behav_fp) > 0 else 0.0
    behav_rec = float(behav_tp / n_ac) if n_ac > 0 else 0.0
    behav_f1 = float(2 * behav_prec * behav_rec / (behav_prec + behav_rec)) if (behav_prec + behav_rec) > 0 else 0.0
    
    cost_fp = float(fp_act * C_FP)
    cost_review = float((tp_rev + fp_rev) * C_REVIEW)
    cost_fn_wait = float(fn_wait * C_FN)
    cost_abs_wait = float(fn_abs * TEST_WINDOW_DAYS * C_WAIT_PER_DAY)
    cost_abs_fn = float(fn_abs * C_FN)
    
    total_cost_best = cost_fp + cost_review + cost_fn_wait + cost_abs_wait
    total_cost_worst = cost_fp + cost_review + cost_fn_wait + cost_abs_fn
    
    # Behavioral only cost on this regime
    behav_fn = n_ac - behav_tp
    behav_cost = behav_fp * C_FP + behav_fn * C_FN
    
    results.append({
        "regime_name": reg_name,
        "target_ac_prevalence": reg["prevalence"]["abusive_coordinated"],
        "test_total_accounts": n_total,
        "test_true_ac": n_ac,
        "actual_ac_prevalence": round(actual_prev, 4),
        "auto_act_precision": round(direct_prec, 4),
        "auto_act_recall": round(direct_rec, 4),
        "effective_recall": round(effective_rec, 4),
        "auto_act_fp": fp_act,
        "fp_rate_on_non_ac": round(float(fp_act / (n_total - n_ac)), 6) if (n_total - n_ac) > 0 else 0.0,
        "routing_distribution": {
            "ACT": int(act_mask.sum()),
            "REVIEW": int(rev_mask.sum()),
            "WAIT_MONITOR": int(wait_mask.sum()),
            "ABSTAIN": int(abs_mask.sum()),
        },
        "ac_routing": {
            "ACT": tp_act,
            "REVIEW": tp_rev,
            "WAIT_MONITOR": fn_wait,
            "ABSTAIN": fn_abs,
        },
        "review_queue_volume": int(rev_mask.sum()),
        "behavioral_only": {
            "precision": round(behav_prec, 4),
            "recall": round(behav_rec, 4),
            "f1": round(behav_f1, 4),
            "fp": behav_fp,
            "fn": behav_fn,
            "cost": float(behav_cost),
        },
        "routing_total_cost_best": total_cost_best,
        "routing_total_cost_worst": total_cost_worst,
    })

print("\n" + "="*80)
print("PREVALENCE-SHIFT EXPERIMENT RESULTS")
print("="*80)

print(f"{'Regime':<25} | {'True AC (Prev)':<15} | {'Auto-ACT Prec':<14} | {'Auto-ACT Rec':<13} | {'Eff. Recall':<12} | {'Auto-ACT FP':<12} | {'Routing Cost (Best)':<20} | {'Behav Cost':<12}")
print("-" * 135)
for r in results:
    prev_str = f"{r['test_true_ac']} ({r['actual_ac_prevalence']*100:.1f}%)"
    print(f"{r['regime_name']:<25} | {prev_str:<15} | {r['auto_act_precision']*100:>12.2f}% | {r['auto_act_recall']*100:>11.2f}% | {r['effective_recall']*100:>10.2f}% | {r['auto_act_fp']:>12d} | Rs {r['routing_total_cost_best']:>16,.2f} | Rs {r['behavioral_only']['cost']:>9,.2f}")

os.makedirs("evals/results", exist_ok=True)
with open("evals/results/prevalence_shift_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved prevalence shift results to evals/results/prevalence_shift_results.json")