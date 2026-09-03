"""
AbuseRing Sentinel - GNN Evaluation (Rung 6 Comparison Baseline)
Trains 2-Layer GraphSAGE on Train split graph (Days 1-54), evaluates on Test split graph (Days 73-90),
and benchmarks against structural_lgbm (Rung 4) across overall performance and robustness subsets.
"""

import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize

from features.feature_pipeline import build_temporal_splits
from graph.temporal_graph import build_graph_as_of
from models.gnn_model import GNNClassifier

LABEL_MAP = {"benign_independent": 0, "benign_coordinated": 1, "abusive_coordinated": 2}
INV_LABEL_MAP = {0: "benign_independent", 1: "benign_coordinated", 2: "abusive_coordinated"}

def run_gnn_evaluation():
    print("=" * 80)
    print("ABUSERING SENTINEL — RUNG 6: GNN COMPARISON BASELINE EVALUATION")
    print("=" * 80)

    # 1. Load data
    events = pd.read_parquet("data/events.parquet")
    accounts = pd.read_parquet("data/accounts.parquet")
    labels = pd.read_parquet("data/labels.parquet")
    with open("data/split_info.json") as f:
        split_info = json.load(f)

    # 2. Build temporal splits & temporal graphs
    print("\nExtracting temporal feature splits...")
    splits = build_temporal_splits(events, accounts, labels, split_info)
    
    train_sp = splits["train"]
    val_sp = splits["val"]
    test_sp = splits["test"]

    print("Constructing point-in-time NetworkX graphs for each temporal cut...")
    t0_g = time.perf_counter()
    G_train = build_graph_as_of(events, split_info["train_end_ts"])
    G_val = build_graph_as_of(events, split_info["val_end_ts"])
    G_test = build_graph_as_of(events, split_info["test_end_ts"])
    print(f"Graph construction completed in {time.perf_counter() - t0_g:.2f}s.")
    print(f"  Train Graph: {G_train.number_of_nodes()} nodes, {G_train.number_of_edges()} edges")
    print(f"  Val Graph:   {G_val.number_of_nodes()} nodes, {G_val.number_of_edges()} edges")
    print(f"  Test Graph:  {G_test.number_of_nodes()} nodes, {G_test.number_of_edges()} edges")

    # Align data matrices
    X_train = train_sp["struct"].values
    y_train = train_sp["labels"]["label"].values
    train_acc_ids = list(train_sp["struct"].index)

    X_val = val_sp["struct"].values
    y_val = val_sp["labels"]["label"].values
    val_acc_ids = list(val_sp["struct"].index)

    X_test = test_sp["struct"].values
    y_test = test_sp["labels"]["label"].values
    test_acc_ids = list(test_sp["struct"].index)

    # 3. Train GNN Classifier
    print("\nTraining 2-Layer GraphSAGE Classifier on Train split (Days 1-54)...")
    gnn = GNNClassifier(hidden_dim=32, lr=0.01, weight_decay=1e-4, epochs=150, dropout=0.1, random_state=42)
    gnn.fit(
        X=X_train, y=y_train, G=G_train, account_ids=train_acc_ids,
        X_val=X_val, y_val=y_val, G_val=G_val, val_account_ids=val_acc_ids
    )
    print(f"GNN Training complete in {gnn.training_time_sec:.2f}s.")
    print(f"Initial Train Loss: {gnn.history[0]['train_loss']} -> Final Train Loss: {gnn.history[-1]['train_loss']}")
    print(f"Best Val Loss: {min(h['val_loss'] for h in gnn.history if h['val_loss'] is not None):.5f}")

    # 4. Measure Inference Latency
    t0_inf = time.perf_counter()
    y_pred_probs = gnn.predict_proba(X_test, G_test, test_acc_ids)
    inf_time_total = time.perf_counter() - t0_inf
    inf_per_batch = round((inf_time_total / len(X_test)) * 1000, 3)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 5. Compute Test Set Metrics
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2], zero_division=0)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc_macro = roc_auc_score(y_test_bin, y_pred_probs, average="macro", multi_class="ovr")

    gnn_metrics = {
        "model": "gnn_graphsage_rung6",
        "split": "test",
        "precision_abusive": round(float(prec[2]), 4),
        "recall_abusive": round(float(rec[2]), 4),
        "f1_abusive": round(float(f1[2]), 4),
        "precision_benign_coord": round(float(prec[1]), 4),
        "recall_benign_coord": round(float(rec[1]), 4),
        "f1_benign_coord": round(float(f1[1]), 4),
        "precision_benign_indep": round(float(prec[0]), 4),
        "recall_benign_indep": round(float(rec[0]), 4),
        "f1_benign_indep": round(float(f1[0]), 4),
        "roc_auc_macro": round(float(auc_macro), 4),
        "training_time_sec": gnn.training_time_sec,
        "inference_ms_per_account": inf_per_batch,
        "total_test_accounts": len(y_test)
    }

    # 6. Benchmark against Structural LightGBM (Rung 4)
    struct_lgbm = joblib.load("models/structural_lgbm.pkl")
    t0_lgbm = time.perf_counter()
    y_lgbm_probs = struct_lgbm.predict_proba(test_sp["struct"])
    lgbm_inf_per_acc = round(((time.perf_counter() - t0_lgbm) / len(test_sp["struct"])) * 1000, 3)
    y_lgbm_pred = np.argmax(y_lgbm_probs, axis=1)
    prec_lgbm, rec_lgbm, f1_lgbm, _ = precision_recall_fscore_support(y_test, y_lgbm_pred, labels=[0, 1, 2], zero_division=0)
    auc_lgbm = roc_auc_score(y_test_bin, y_lgbm_probs, average="macro", multi_class="ovr")

    struct_lgbm_metrics = {
        "model": "structural_lgbm_rung4",
        "split": "test",
        "precision_abusive": round(float(prec_lgbm[2]), 4),
        "recall_abusive": round(float(rec_lgbm[2]), 4),
        "f1_abusive": round(float(f1_lgbm[2]), 4),
        "roc_auc_macro": round(float(auc_lgbm), 4),
        "inference_ms_per_account": lgbm_inf_per_acc
    }

    # 7. Robustness Subsets Evaluation (Canonical Definitions matching kl_ablation_eval.py)
    rings = pd.read_parquet("data/rings.parquet")
    ring_acc_df = rings.drop_duplicates("account_id").set_index("account_id")
    subsets_eval = {}
    
    # Subset 1: Referral Farming (unseen structure)
    refarm_accs = [
        a for a in test_acc_ids 
        if a in ring_acc_df.index and ring_acc_df.loc[a, "ring_type"] == "referral_farming"
    ]
    if refarm_accs:
        sub_indices = [test_acc_ids.index(a) for a in refarm_accs]
        gnn_sub_rec = float(np.mean(y_pred[sub_indices] == 2))
        lgbm_sub_rec = float(np.mean(y_lgbm_pred[sub_indices] == 2))
        subsets_eval["referral_farming"] = {
            "n_accounts": len(refarm_accs),
            "gnn_recall": round(gnn_sub_rec, 4),
            "struct_lgbm_recall": round(lgbm_sub_rec, 4)
        }

    # Subset 2: Hard BC (Injected Shared Payout / Counterfactual Benign Family)
    if "counterfactual_subset" in labels.columns:
        cf_lookup = labels.set_index("account_id")["counterfactual_subset"].to_dict()
        hard_bc_accs = [a for a in test_acc_ids if cf_lookup.get(a) == "hard_bc"]
    elif "is_hard_bc" in labels.columns:
        hard_bc_lookup = labels.set_index("account_id")["is_hard_bc"].to_dict()
        hard_bc_accs = [a for a in test_acc_ids if hard_bc_lookup.get(a, False) == True]
    else:
        hard_bc_accs = [
            a for a in test_acc_ids 
            if test_sp["labels"].loc[a, "label_str"] == "benign_coordinated" and 
            test_sp["struct"].loc[a, "shared_payout_degree"] > 0
        ]
        
    if hard_bc_accs:
        sub_indices = [test_acc_ids.index(a) for a in hard_bc_accs]
        # In hard BC, false positives are predictions of Abusive Coordinated (class 2)
        gnn_fp_rate = float(np.mean(y_pred[sub_indices] == 2))
        lgbm_fp_rate = float(np.mean(y_lgbm_pred[sub_indices] == 2))
        subsets_eval["hard_bc"] = {
            "n_accounts": len(hard_bc_accs),
            "gnn_false_positive_rate": round(gnn_fp_rate, 4),
            "struct_lgbm_false_positive_rate": round(lgbm_fp_rate, 4)
        }

    # Subset 3: Sleeper Accounts (sparse structural links)
    sleeper_accs = [
        a for a in test_acc_ids 
        if a in ring_acc_df.index and ring_acc_df.loc[a, "is_sleeper"] == True
    ]
    if sleeper_accs:
        sub_indices = [test_acc_ids.index(a) for a in sleeper_accs]
        gnn_sub_rec = float(np.mean(y_pred[sub_indices] == 2))
        lgbm_sub_rec = float(np.mean(y_lgbm_pred[sub_indices] == 2))
        subsets_eval["sleeper_accounts"] = {
            "n_accounts": len(sleeper_accs),
            "gnn_recall": round(gnn_sub_rec, 4),
            "struct_lgbm_recall": round(lgbm_sub_rec, 4)
        }

    # Assemble and save artifact
    output = {
        "gnn_test_metrics": gnn_metrics,
        "structural_lgbm_test_metrics": struct_lgbm_metrics,
        "robustness_subsets": subsets_eval,
        "training_convergence": gnn.history[::15]  # Sampled history
    }

    out_file = Path("evals/results/gnn_comparison_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Save model artifact
    joblib.dump(gnn, "models/gnn_comparison.pkl")

    print("\n" + "=" * 80)
    print("GNN vs. STRUCTURAL LIGHTGBM HEAD-TO-HEAD COMPARISON (TEST SPLIT)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Structural LightGBM (Rung 4)':<30} | {'2-Layer GraphSAGE (Rung 6)':<30}")
    print("-" * 96)
    print(f"{'Abusive Precision':<30} | {struct_lgbm_metrics['precision_abusive']:<30.4f} | {gnn_metrics['precision_abusive']:<30.4f}")
    print(f"{'Abusive Recall':<30} | {struct_lgbm_metrics['recall_abusive']:<30.4f} | {gnn_metrics['recall_abusive']:<30.4f}")
    print(f"{'Abusive F1 Score':<30} | {struct_lgbm_metrics['f1_abusive']:<30.4f} | {gnn_metrics['f1_abusive']:<30.4f}")
    print(f"{'Macro ROC-AUC':<30} | {struct_lgbm_metrics['roc_auc_macro']:<30.4f} | {gnn_metrics['roc_auc_macro']:<30.4f}")
    print(f"{'Inference Latency':<30} | {lgbm_inf_per_acc:<27.3f} ms | {inf_per_batch:<27.3f} ms")
    print(f"{'Training Time':<30} | {'~0.15s (CPU)':<30} | {f'{gnn.training_time_sec:.2f}s (CPU)':<30}")
    print("-" * 96)
    print("\nRobustness Subsets:")
    for sub, vals in subsets_eval.items():
        print(f"  [{sub.upper()}] (N={vals['n_accounts']}):")
        for k, v in vals.items():
            if k != "n_accounts":
                print(f"    {k}: {v}")

    print(f"\nArtifacts saved to {out_file} and models/gnn_comparison.pkl.")

if __name__ == "__main__":
    run_gnn_evaluation()
