"""AbuseRing Sentinel - Model Suite (baseline ladder)"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
from typing import Optional
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, log_loss, confusion_matrix)
from sklearn.preprocessing import label_binarize
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).parent.parent))
from features.feature_pipeline import (STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES, ALL_FEATURES,
    LABEL_MAP, LABEL_MAP_INV, build_temporal_splits)

SEED = 42
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

class MajorityClassBaseline:
    name = "majority_class"
    def __init__(self): self.majority_class_ = None
    def fit(self, X, y):
        self.majority_class_ = int(np.argmax(np.bincount(y))); return self
    def predict(self, X): return np.full(len(X), self.majority_class_, dtype=int)
    def predict_proba(self, X):
        p = np.zeros((len(X), 3)); p[:, self.majority_class_] = 1.0; return p

class RuleBasedBaseline:
    name = "rule_based"
    def fit(self, X, y): return self
    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            spd = row.get("shared_payout_degree", 0); pr = row.get("promo_rate", 0)
            nref = row.get("n_referrals_received", 0); nret = row.get("n_returns", 0)
            deg = row.get("degree", 0)
            if spd >= 2 and pr > 0.5: preds.append(2)
            elif spd >= 2 and nref >= 1: preds.append(2)
            elif spd >= 1 and nret >= 1: preds.append(2)
            elif deg >= 3 and spd == 0: preds.append(1)
            else: preds.append(0)
        return np.array(preds)
    def predict_proba(self, X):
        preds = self.predict(X); proba = np.zeros((len(preds), 3))
        for i, p in enumerate(preds): proba[i, p] = 1.0
        return proba

from models.fused_model import FusedCalibratedClassifier, build_lgbm_model

def compute_metrics(y_true, y_pred, y_proba, split_name, model_name):
    class_names = ["benign_indep", "benign_coord", "abusive_coord"]
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_abusive = f1_score(y_true, y_pred, average=None, zero_division=0)[2]
    prec_abusive = precision_score(y_true, y_pred, average=None, zero_division=0)[2]
    recall_abusive = recall_score(y_true, y_pred, average=None, zero_division=0)[2]
    try:
        y_bin = label_binarize(y_true, classes=[0,1,2])
        auc_ovr = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except Exception: auc_ovr = float("nan")
    try: ll = log_loss(y_true, y_proba)
    except Exception: ll = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cost_config = json.load(open("data/cost_config.json"))
    c_fp = cost_config["c_false_positive"]; c_fn = cost_config["c_false_negative"]
    fp_count = int(((y_pred == 2) & (y_true != 2)).sum())
    fn_count = int(((y_pred != 2) & (y_true == 2)).sum())
    total_cost = fp_count * c_fp + fn_count * c_fn
    metrics = {"model": model_name, "split": split_name,
               "precision_macro": round(precision_macro, 4), "recall_macro": round(recall_macro, 4),
               "f1_macro": round(f1_macro, 4), "f1_abusive": round(f1_abusive, 4),
               "precision_abusive": round(prec_abusive, 4), "recall_abusive": round(recall_abusive, 4),
               "auc_ovr": round(auc_ovr, 4) if not np.isnan(auc_ovr) else None,
               "log_loss": round(ll, 4) if not np.isnan(ll) else None,
               "fp_count": fp_count, "fn_count": fn_count, "total_cost_rs": round(total_cost, 2),
               "cost_fp_rs": c_fp, "cost_fn_rs": c_fn,
               "n_total": len(y_true), "n_abusive_true": int((y_true == 2).sum())}
    print(f"\n{'='*60}\nModel: {model_name}  |  Split: {split_name}\n{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"AUC (OvR): {auc_ovr:.4f}\nLog-loss: {ll:.4f}")
    print(f"FP count: {fp_count}  (cost: Rs{fp_count*c_fp:,.0f} SIMULATED)")
    print(f"FN count: {fn_count}  (cost: Rs{fn_count*c_fn:,.0f} SIMULATED)")
    print(f"Total cost: Rs{total_cost:,.0f} SIMULATED")
    print(f"Confusion matrix:\n{cm}")
    return metrics

def run_all_models(splits):
    all_metrics = []
    def get_arrays(split_name):
        sp = splits[split_name]; idx = sp["labels"].index
        s = sp["struct"].reindex(idx).fillna(0); b = sp["behav"].reindex(idx).fillna(0)
        y = sp["labels"]["label"].values; return s, b, y
    s_tr, b_tr, y_tr = get_arrays("train")
    s_va, b_va, y_va = get_arrays("val")
    s_te, b_te, y_te = get_arrays("test")
    X_tr = pd.concat([s_tr, b_tr], axis=1)
    X_va = pd.concat([s_va, b_va], axis=1)
    X_te = pd.concat([s_te, b_te], axis=1)

    print("\n\n=== RUNG 0: Majority Class Baseline ===")
    maj = MajorityClassBaseline(); maj.fit(X_tr, y_tr)
    for sname, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
        all_metrics.append(compute_metrics(y, maj.predict(X), maj.predict_proba(X), sname, "majority_class"))

    print("\n\n=== RUNG 1: Rule-Based Baseline ===")
    rb = RuleBasedBaseline(); rb.fit(X_tr, y_tr)
    for sname, X, y in [("val", X_va, y_va), ("test", X_te, y_te)]:
        all_metrics.append(compute_metrics(y, rb.predict(X), rb.predict_proba(X), sname, "rule_based"))

    print("\n\n=== RUNG 2: Structural-Only LightGBM ===")
    struct_lgbm = build_lgbm_model()
    struct_lgbm.fit(s_tr[STRUCTURAL_FEATURES].fillna(0), y_tr)
    for sname, s, y in [("val", s_va, y_va), ("test", s_te, y_te)]:
        all_metrics.append(compute_metrics(y, struct_lgbm.predict(s[STRUCTURAL_FEATURES].fillna(0)),
            struct_lgbm.predict_proba(s[STRUCTURAL_FEATURES].fillna(0)), sname, "structural_lgbm"))
    joblib.dump(struct_lgbm, MODEL_DIR / "structural_lgbm.pkl")

    print("\n\n=== RUNG 3: Behavioral-Only LightGBM ===")
    behav_lgbm = build_lgbm_model()
    behav_lgbm.fit(b_tr[BEHAVIORAL_FEATURES].fillna(0), y_tr)
    for sname, b, y in [("val", b_va, y_va), ("test", b_te, y_te)]:
        all_metrics.append(compute_metrics(y, behav_lgbm.predict(b[BEHAVIORAL_FEATURES].fillna(0)),
            behav_lgbm.predict_proba(b[BEHAVIORAL_FEATURES].fillna(0)), sname, "behavioral_lgbm"))
    joblib.dump(behav_lgbm, MODEL_DIR / "behavioral_lgbm.pkl")

    print("\n\n=== RUNG 4: Fused Calibrated Classifier ===")
    fused = FusedCalibratedClassifier(conflict_kl_threshold=0.3)
    fused.fit(s_tr, b_tr, y_tr)
    for sname, s, b, y in [("val", s_va, b_va, y_va), ("test", s_te, b_te, y_te)]:
        p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s, b)
        pred = np.argmax(p_fused, axis=1)
        m = compute_metrics(y, pred, p_fused, sname, "fused_calibrated")
        m["evidence_conflict_rate"] = round(float(conflicts.mean()), 4)
        m["n_conflicts"] = int(conflicts.sum())
        all_metrics.append(m)
        print(f"  Evidence conflicts: {conflicts.sum()}/{len(conflicts)} ({conflicts.mean()*100:.1f}%)")
    joblib.dump(fused, MODEL_DIR / "fused_calibrated.pkl")
    return all_metrics

def save_metrics_table(all_metrics, output_path="evals/metrics.json"):
    Path("evals").mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nMetrics saved to {output_path}")

if __name__ == "__main__":
    print("Loading data...")
    events = pd.read_parquet("data/events.parquet")
    accounts = pd.read_parquet("data/accounts.parquet")
    labels = pd.read_parquet("data/labels.parquet")
    split = json.load(open("data/split_info.json"))
    print("Building temporal splits (this may take a minute)...")
    splits = build_temporal_splits(events, accounts, labels, split)
    print("\nRunning all models...")
    all_metrics = run_all_models(splits)
    save_metrics_table(all_metrics)
    print("\n\n=== BASELINE LADDER SUMMARY (test split) ===")
    test_metrics = [m for m in all_metrics if m["split"] == "test"]
    print(f"{'Model':<25} {'F1-macro':>9} {'F1-abuse':>9} {'Prec-abuse':>11} {'Rec-abuse':>10} {'AUC':>7} {'Cost(Rs)':>10}")
    print("-" * 85)
    for m in test_metrics:
        print(f"{m['model']:<25} {m['f1_macro']:>9.4f} {m['f1_abusive']:>9.4f} "
              f"{m['precision_abusive']:>11.4f} {m['recall_abusive']:>10.4f} "
              f"{str(m['auc_ovr']):>7} {m['total_cost_rs']:>10,.0f}")
