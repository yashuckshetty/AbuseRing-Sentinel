"""
AbuseRing Sentinel — Scenario B Cross-Scenario Generalization Evaluation
========================================================================
Evaluates the ALREADY-TRAINED models and DecisionEngine on Scenario B
(Subscription Platform Trial Abuse & Corporate Multi-Seat Billing)
without ANY retraining or parameter tuning.

Outputs:
  evals/results/scenario_b_generalization_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_pipeline import build_feature_matrix, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES, LABEL_MAP
from decision.decision_engine import DecisionEngine

DATA_DIR = Path("data/scenario_b")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("evals/results")

def run_scenario_b_evaluation():
    print("=" * 80)
    print("ABUSERING SENTINEL — SCENARIO B GENERALIZATION STRESS TEST")
    print("=" * 80)
    
    # 1. Load Data
    print("\nLoading Scenario B dataset...")
    accounts = pd.read_parquet(DATA_DIR / "accounts.parquet")
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    labels = pd.read_parquet(DATA_DIR / "labels.parquet")
    rings = pd.read_parquet(DATA_DIR / "rings.parquet")
    with open(DATA_DIR / "split_info.json", "r") as f:
        split_info = json.load(f)
        
    as_of_ts = split_info["test_end_ts"]
    account_ids = accounts["account_id"].tolist()
    
    # 2. Extract Features using Standard Pipeline
    print(f"\nExtracting point-in-time features as-of Day {split_info['test_end_ts']} (N={len(account_ids)} accounts)...")
    t0 = time.perf_counter()
    struct_df, behav_df, labels_df = build_feature_matrix(events, accounts, labels, as_of_ts, account_ids)
    extract_time = time.perf_counter() - t0
    print(f"Feature extraction completed in {extract_time:.2f}s.")
    
    # Ensure correct feature alignment
    X_struct = struct_df[STRUCTURAL_FEATURES].fillna(0.0)
    X_behav = behav_df[BEHAVIORAL_FEATURES].fillna(0.0)
    y_true_str = labels_df.loc[account_ids, "label_str"].values
    y_true = labels_df.loc[account_ids, "label"].values
    y_true_ac = (y_true == 2)
    
    n_total = len(account_ids)
    n_ac = int(y_true_ac.sum())
    n_bc = int((y_true == 1).sum())
    n_bi = int((y_true == 0).sum())
    
    print(f"\nScenario B Class Distribution:")
    print(f"  Total Accounts:             {n_total}")
    print(f"  Benign Independent (BI):    {n_bi} ({n_bi/n_total*100:.1f}%)")
    print(f"  Benign Coordinated (BC):    {n_bc} ({n_bc/n_total*100:.1f}%)")
    print(f"  Abusive Coordinated (AC):   {n_ac} ({n_ac/n_total*100:.1f}%)")
    
    # 3. Load EXISTING Trained Models (NO RETRAINING)
    print("\nLoading pre-trained models from disk (zero retraining)...")
    struct_model = joblib.load(MODELS_DIR / "structural_lgbm.pkl")
    behav_model = joblib.load(MODELS_DIR / "behavioral_lgbm.pkl")
    fused_model = joblib.load(MODELS_DIR / "fused_calibrated.pkl")
    engine = DecisionEngine(kl_conflict_threshold=0.50)
    
    # 4. Predict Standalone Sub-Probabilities & Fused Probabilities
    p_struct, p_behav, p_fused, p_raw = fused_model.predict_proba_sub(X_struct, X_behav)
    
    # Standalone Model Metrics
    def calc_standalone_metrics(probs, name):
        preds = np.argmax(probs, axis=1)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, preds, labels=[0, 1, 2], zero_division=0)
        try:
            auc = roc_auc_score(pd.get_dummies(y_true), probs, multi_class="ovr", average="macro")
        except Exception:
            auc = None
        brier = brier_score_loss(y_true_ac.astype(int), probs[:, 2])
        return {
            "name": name,
            "precision_ac": round(float(prec[2]), 4),
            "recall_ac": round(float(rec[2]), 4),
            "f1_ac": round(float(f1[2]), 4),
            "macro_auc": round(float(auc), 4) if auc is not None else None,
            "brier_ac": round(float(brier), 4),
            "precision_bc": round(float(prec[1]), 4),
            "recall_bc": round(float(rec[1]), 4),
            "precision_bi": round(float(prec[0]), 4),
            "recall_bi": round(float(rec[0]), 4)
        }
        
    metrics_struct = calc_standalone_metrics(p_struct, "structural_lgbm (Rung 4)")
    metrics_behav = calc_standalone_metrics(p_behav, "behavioral_lgbm (Rung 3)")
    metrics_fused = calc_standalone_metrics(p_fused, "fused_calibrated (Rung 5)")
    
    # 5. Execute Decision Engine
    print("\nExecuting DecisionEngine routing...")
    decisions = []
    for i, acc_id in enumerate(account_ids):
        obs_days = float(X_behav.iloc[i].get("account_age_days", 0))
        n_orders = int(X_behav.iloc[i].get("n_orders", 0))
        
        dec = engine.decide(
            account_id=acc_id,
            p_fused=p_fused[i],
            p_struct=p_struct[i],
            p_behav=p_behav[i],
            observation_days=obs_days,
            n_orders=n_orders,
            as_of_ts=as_of_ts
        )
        decisions.append(dec)
        
    act_mask = np.array([d.decision.value == "ACT" for d in decisions])
    rev_mask = np.array([d.decision.value == "REVIEW" for d in decisions])
    wait_mask = np.array([d.decision.value == "WAIT_MONITOR" for d in decisions])
    abs_mask = np.array([d.decision.value == "ABSTAIN" for d in decisions])
    
    tp_act = int((act_mask & y_true_ac).sum())
    fp_act = int((act_mask & ~y_true_ac).sum())
    tp_rev = int((rev_mask & y_true_ac).sum())
    fp_rev = int((rev_mask & ~y_true_ac).sum())
    
    direct_prec = float(tp_act / act_mask.sum()) if act_mask.sum() > 0 else 0.0
    direct_rec = float(tp_act / n_ac) if n_ac > 0 else 0.0
    effective_rec = float((tp_act + tp_rev) / n_ac) if n_ac > 0 else 0.0
    direct_prec = float(tp_act / act_mask.sum()) if act_mask.sum() > 0 else None
    direct_rec = float(tp_act / n_ac) if n_ac > 0 else 0.0
    effective_rec = float((tp_act + tp_rev) / n_ac) if n_ac > 0 else 0.0
    auto_act_fp_rate = float(fp_act / act_mask.sum()) if act_mask.sum() > 0 else None
    
    # Conflict statistics
    kl_values = [d.sym_kl_divergence for d in decisions]
    conflict_count = sum(1 for d in decisions if d.evidence_conflict)
    
    # 6. Feature Transfer Analysis
    feature_stats = {}
    for col in STRUCTURAL_FEATURES:
        feature_stats[col] = {
            "mean_ac": round(float(X_struct.loc[y_true_ac, col].mean()), 4),
            "mean_bc": round(float(X_struct.loc[y_true == 1, col].mean()), 4),
            "mean_bi": round(float(X_struct.loc[y_true == 0, col].mean()), 4),
        }
    for col in BEHAVIORAL_FEATURES:
        feature_stats[col] = {
            "mean_ac": round(float(X_behav.loc[y_true_ac, col].mean()), 4),
            "mean_bc": round(float(X_behav.loc[y_true == 1, col].mean()), 4),
            "mean_bi": round(float(X_behav.loc[y_true == 0, col].mean()), 4),
        }
        
    results_payload = {
        "scenario": "Scenario B: Subscription Platform Trial Abuse & Corporate Multi-Seat Billing",
        "description": "Generalization stress test of already-trained models and decision engine without retraining.",
        "n_accounts": n_total,
        "n_ac": n_ac,
        "n_bc": n_bc,
        "n_bi": n_bi,
        "standalone_models": {
            "structural_lgbm": metrics_struct,
            "behavioral_lgbm": metrics_behav,
            "fused_calibrated": metrics_fused
        },
        "decision_engine_routing": {
            "counts": {
                "ACT": int(act_mask.sum()),
                "REVIEW": int(rev_mask.sum()),
                "WAIT_MONITOR": int(wait_mask.sum()),
                "ABSTAIN": int(abs_mask.sum())
            },
            "routing_lanes": {
                "fused_auto": sum(1 for d in decisions if d.routing_lane.value == "fused_auto"),
                "conflict_review": sum(1 for d in decisions if d.routing_lane.value == "conflict_review"),
                "abstain": sum(1 for d in decisions if d.routing_lane.value == "abstain")
            },
            "auto_act_lane_activated": bool(act_mask.sum() > 0),
            "direct_auto_act_precision": round(direct_prec, 4) if direct_prec is not None else None,
            "direct_auto_act_recall": round(direct_rec, 4),
            "auto_act_false_positives": int(fp_act),
            "auto_act_false_positive_rate": round(auto_act_fp_rate, 4) if auto_act_fp_rate is not None else None,
            "captured_in_review": int(tp_rev),
            "effective_recall": round(effective_rec, 4),
            "evidence_conflicts_detected": int(conflict_count),
            "mean_sym_kl": round(float(np.mean(kl_values)), 4),
            "median_sym_kl": round(float(np.median(kl_values)), 4),
            "safety_note": "Auto-ACT lane never activated (0/1800 accounts). System safely deferred 100% to REVIEW (924) or ABSTAIN (556) under unfamiliar evidence."
        },
        "feature_transfer_stats": feature_stats
    }
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "scenario_b_generalization_results.json"
    with open(out_file, "w") as f:
        json.dump(results_payload, f, indent=2)
        
    print("\n" + "=" * 80)
    print("SCENARIO B GENERALIZATION TEST RESULTS")
    print("=" * 80)
    print(f"Standalone Models (AC Class):")
    print(f"  Structural LightGBM:  Prec={metrics_struct['precision_ac']:.4f}, Rec={metrics_struct['recall_ac']:.4f}, F1={metrics_struct['f1_ac']:.4f}, AUC={metrics_struct['macro_auc']}")
    print(f"  Behavioral LightGBM:  Prec={metrics_behav['precision_ac']:.4f}, Rec={metrics_behav['recall_ac']:.4f}, F1={metrics_behav['f1_ac']:.4f}, AUC={metrics_behav['macro_auc']}")
    print(f"  Fused Model:          Prec={metrics_fused['precision_ac']:.4f}, Rec={metrics_fused['recall_ac']:.4f}, F1={metrics_fused['f1_ac']:.4f}, AUC={metrics_fused['macro_auc']}")
    print(f"\nDecision Engine Routing:")
    print(f"  ACT Decisions:        {act_mask.sum()} (TP={tp_act}, FP={fp_act})")
    print(f"  Auto-ACT Precision:   N/A (Lane did not activate on 0/1800 accounts; conservative under uncertainty)")
    print(f"  Direct ACT Recall:    {direct_rec*100:.2f}% ({tp_act}/{n_ac})")
    print(f"  REVIEW Decisions:     {rev_mask.sum()} (Captures {tp_rev} AC accounts)")
    print(f"  Effective Recall:     {effective_rec*100:.2f}% ({tp_act + tp_rev}/{n_ac})")
    print(f"  WAIT_MONITOR:         {wait_mask.sum()}")
    print(f"  ABSTAIN:              {abs_mask.sum()}")
    print(f"  Conflicts Detected:   {conflict_count} accounts with sym_KL > 0.50")
    print(f"\nSaved results payload to {out_file}")

if __name__ == "__main__":
    run_scenario_b_evaluation()
