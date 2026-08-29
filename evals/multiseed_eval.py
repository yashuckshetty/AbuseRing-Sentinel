import sys, io, json, os, warnings
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from data.simulator import generate_dataset, N_ACCOUNTS
from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from models.model_suite import MajorityClassBaseline, RuleBasedBaseline, build_lgbm_model, FusedCalibratedClassifier, compute_metrics
from decision.decision_engine import DecisionEngine, Decision, RoutingLane

SEEDS = [42, 43, 44]
C_FP = 500.0
C_FN = 2000.0
C_REVIEW = 150.0
C_WAIT_PER_DAY = 50.0
TEST_WINDOW_DAYS = 18.0

seed_results = []

for s in SEEDS:
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING SEED {s}", flush=True)
    print(f"=======================================================", flush=True)
    
    seed_dir = f"data/seed_{s}" if s != 42 else "data"
    
    if s != 42:
        os.makedirs(seed_dir, exist_ok=True)
        generate_dataset(
            seed=s,
            abuse_prevalence=None,
            output_dir=seed_dir,
            counterfactual_hard_bc=True,
            counterfactual_varied_payout=True,
            n_accounts=N_ACCOUNTS,
        )
        
    events = pd.read_parquet(os.path.join(seed_dir, "events.parquet"))
    accounts = pd.read_parquet(os.path.join(seed_dir, "accounts.parquet"))
    labels = pd.read_parquet(os.path.join(seed_dir, "labels.parquet"))
    split = json.load(open(os.path.join(seed_dir, "split_info.json")))
    
    splits = build_temporal_splits(events, accounts, labels, split)
    
    def get_arrays(split_name):
        sp = splits[split_name]; idx = sp["labels"].index
        s_mat = sp["struct"].reindex(idx).fillna(0); b_mat = sp["behav"].reindex(idx).fillna(0)
        y = sp["labels"]["label"].values; return s_mat, b_mat, y
        
    s_tr, b_tr, y_tr = get_arrays("train")
    s_te, b_te, y_te = get_arrays("test")
    
    # Train models on train split
    behav_lgbm = build_lgbm_model(seed=s)
    behav_lgbm.fit(b_tr[BEHAVIORAL_FEATURES].fillna(0), y_tr)
    
    struct_lgbm = build_lgbm_model(seed=s)
    struct_lgbm.fit(s_tr[STRUCTURAL_FEATURES].fillna(0), y_tr)
    
    fused = FusedCalibratedClassifier(conflict_kl_threshold=0.3)
    fused.struct_model = struct_lgbm
    fused.behav_model = behav_lgbm
    fused.classes_ = np.arange(3)
    
    # Score on test split
    p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)
    
    m_behav = compute_metrics(y_te, behav_lgbm.predict(b_te[BEHAVIORAL_FEATURES].fillna(0)), p_behav, "test", "behavioral_lgbm")
    m_struct = compute_metrics(y_te, struct_lgbm.predict(s_te[STRUCTURAL_FEATURES].fillna(0)), p_struct, "test", "structural_lgbm")
    m_fused = compute_metrics(y_te, np.argmax(p_fused, axis=1), p_fused, "test", "fused_calibrated")
    
    # Run DecisionEngine
    engine = DecisionEngine(kl_conflict_threshold=0.5)
    idx_te = list(splits["test"]["labels"].index)
    n_orders = b_te["n_orders"].astype(int).values
    obs_days = b_te["account_age_days"].values
    as_of_ts = split["test_end_ts"]
    
    dec_results = engine.decide_batch(
        account_ids=idx_te,
        p_fused_matrix=p_fused,
        p_struct_matrix=p_struct,
        p_behav_matrix=p_behav,
        observation_days=obs_days,
        n_orders_arr=n_orders,
        as_of_ts=as_of_ts,
    )
    
    y_true_str = splits["test"]["labels"]["label_str"].values
    y_true_ac = (y_true_str == "abusive_coordinated")
    n_ac = int(y_true_ac.sum())
    
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
    
    auto_prec = float(tp_act / (tp_act + fp_act)) if (tp_act + fp_act) > 0 else 0.0
    auto_rec = float(tp_act / n_ac) if n_ac > 0 else 0.0
    eff_rec = float((tp_act + tp_rev) / n_ac) if n_ac > 0 else 0.0
    
    cost_fp = float(fp_act * C_FP)
    cost_review = float((tp_rev + fp_rev) * C_REVIEW)
    cost_fn_wait = float(fn_wait * C_FN)
    cost_abs_wait = float(fn_abs * TEST_WINDOW_DAYS * C_WAIT_PER_DAY)
    cost_abs_fn = float(fn_abs * C_FN)
    
    routing_cost_best = cost_fp + cost_review + cost_fn_wait + cost_abs_wait
    routing_cost_worst = cost_fp + cost_review + cost_fn_wait + cost_abs_fn
    
    # Behavioral model cost
    behav_preds = behav_lgbm.predict(b_te[BEHAVIORAL_FEATURES].fillna(0))
    behav_fp = int(((behav_preds == 2) & ~y_true_ac).sum())
    behav_tp = int(((behav_preds == 2) & y_true_ac).sum())
    behav_fn = n_ac - behav_tp
    behav_cost = behav_fp * C_FP + behav_fn * C_FN
    
    seed_results.append({
        "seed": s,
        "n_test_accounts": len(idx_te),
        "n_true_ac": n_ac,
        "behavioral_precision": m_behav["precision_abusive"],
        "behavioral_recall": m_behav["recall_abusive"],
        "behavioral_f1": m_behav["f1_abusive"],
        "structural_precision": m_struct["precision_abusive"],
        "structural_recall": m_struct["recall_abusive"],
        "structural_f1": m_struct["f1_abusive"],
        "fused_precision": m_fused["precision_abusive"],
        "fused_recall": m_fused["recall_abusive"],
        "fused_f1": m_fused["f1_abusive"],
        "auto_act_precision": auto_prec,
        "auto_act_recall": auto_rec,
        "auto_act_fp": fp_act,
        "effective_recall": eff_rec,
        "review_queue_volume": int(rev_mask.sum()),
        "behavioral_total_cost": float(behav_cost),
        "routing_total_cost_best": float(routing_cost_best),
        "routing_total_cost_worst": float(routing_cost_worst),
    })

df_seeds = pd.DataFrame(seed_results)

print("\n" + "="*80)
print("MULTI-SEED VARIANCE SUMMARY (Seeds: 42, 43, 44)")
print("="*80)

def print_stat(name, col_name, is_pct=True, is_currency=False):
    vals = df_seeds[col_name].values
    mean_val = vals.mean()
    min_val = vals.min()
    max_val = vals.max()
    if is_currency:
        print(f"{name:<35}: Mean = Rs {mean_val:>10,.2f}  [Min: Rs {min_val:>10,.2f}, Max: Rs {max_val:>10,.2f}]")
    elif is_pct:
        print(f"{name:<35}: Mean = {mean_val*100:>6.2f}%       [Min: {min_val*100:>6.2f}%, Max: {max_val*100:>6.2f}%]")
    else:
        print(f"{name:<35}: Mean = {mean_val:>6.2f}        [Min: {min_val:>6.2f}, Max: {max_val:>6.2f}]")

print_stat("Behavioral Model F1 (AC)", "behavioral_f1", is_pct=False)
print_stat("Behavioral Model Precision", "behavioral_precision")
print_stat("Behavioral Model Recall", "behavioral_recall")
print("-" * 75)
print_stat("Structural Model F1 (AC)", "structural_f1", is_pct=False)
print_stat("Structural Model Precision", "structural_precision")
print_stat("Structural Model Recall", "structural_recall")
print("-" * 75)
print_stat("Fused Model F1 (AC)", "fused_f1", is_pct=False)
print_stat("Fused Model Precision", "fused_precision")
print_stat("Fused Model Recall", "fused_recall")
print("-" * 75)
print_stat("Auto-ACT Precision", "auto_act_precision")
print_stat("Auto-ACT Recall", "auto_act_recall")
print_stat("Effective Recall (Routing)", "effective_recall")
print_stat("Auto-ACT False Positives", "auto_act_fp", is_pct=False)
print("-" * 75)
print_stat("Behavioral-Only Total Cost", "behavioral_total_cost", is_currency=True)
print_stat("Routing Total Cost (Best Case)", "routing_total_cost_best", is_currency=True)
print_stat("Routing Total Cost (Worst Case)", "routing_total_cost_worst", is_currency=True)

os.makedirs("evals/results", exist_ok=True)
with open("evals/results/multiseed_results.json", "w") as f:
    json.dump({
        "seeds": seed_results,
        "summary": {
            col: {
                "mean": round(float(df_seeds[col].mean()), 4),
                "min": round(float(df_seeds[col].min()), 4),
                "max": round(float(df_seeds[col].max()), 4),
            }
            for col in df_seeds.columns if col != "seed"
        }
    }, f, indent=2)
print("\nSaved multi-seed results to evals/results/multiseed_results.json")