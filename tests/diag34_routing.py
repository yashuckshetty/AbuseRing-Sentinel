import sys; sys.path.insert(0, ".")
import json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import joblib
from models.model_suite import FusedCalibratedClassifier
from features.feature_pipeline import STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES, build_temporal_splits
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize

events   = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels   = pd.read_parquet("data/labels.parquet")
split    = json.load(open("data/split_info.json"))
LABEL_ENC = {"benign_independent":0,"benign_coordinated":1,"abusive_coordinated":2}

splits = build_temporal_splits(events, accounts, labels, split)
sp = splits["test"]
idx   = sp["labels"].index
s_te  = sp["struct"].reindex(idx).fillna(0)
b_te  = sp["behav"].reindex(idx).fillna(0)
y_te  = sp["labels"]["label"].values

fused = joblib.load("models/fused_calibrated.pkl")
p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)

ac_mask = (y_te == 2)
print(f"=== DIAG 3: P(AC) on TRUE-AC test accounts ({ac_mask.sum()} accounts) ===")
for name, probs in [("p_struct[:,2]", p_struct[ac_mask,2]),
                    ("p_behav[:,2]",  p_behav[ac_mask,2]),
                    ("p_fused[:,2]",  p_fused[ac_mask,2])]:
    print(f"{name}: mean={probs.mean():.3f} median={np.median(probs):.3f} "
          f"p10={np.percentile(probs,10):.3f} p25={np.percentile(probs,25):.3f} "
          f"pct>0.5={100*(probs>0.5).mean():.1f}%")
print(f"Conflict rate on true-AC: {conflicts[ac_mask].mean()*100:.1f}%")

print(f"\n=== DIAG 4: ROUTING-BASED FUSION ===")
print("Strategy: if KL(struct,behav) > threshold --> REVIEW lane")
print("          else --> argmax(p_fused) --> ACT/WAIT_MONITOR\n")

# Routing thresholds to test
kl_thresholds = [0.20, 0.30, 0.50]

for kl_thresh in kl_thresholds:
    # Recompute conflict flags at this threshold
    eps = 1e-9
    def kl_div(p, q):
        p = np.clip(p, eps, 1); q = np.clip(q, eps, 1)
        return np.sum(p * np.log(p / q), axis=1)
    sym_kl = (kl_div(p_struct, p_behav) + kl_div(p_behav, p_struct)) / 2
    routed_to_review = sym_kl > kl_thresh
    routed_to_fused  = ~routed_to_review

    # For non-conflict accounts: use p_fused argmax
    # For conflict accounts: label as REVIEW (treat as flag, not a class prediction)
    # REVIEW is handled separately in the decision engine -- here we measure:
    # (a) among review-routed AC: they are flagged (correctly) even if not auto-ACT
    # (b) among fused-routed: F1, prec, recall at argmax
    
    pred_full = np.argmax(p_fused, axis=1).copy()
    # For fused-routed subset only
    fused_ac_tp = int(((pred_full == 2) & (y_te == 2) & routed_to_fused).sum())
    fused_ac_fp = int(((pred_full == 2) & (y_te != 2) & routed_to_fused).sum())
    fused_ac_fn = int(((pred_full != 2) & (y_te == 2) & routed_to_fused).sum())

    review_ac_tp = int((routed_to_review & (y_te == 2)).sum())  # AC sent to review
    review_ac_fp = int((routed_to_review & (y_te != 2)).sum())  # non-AC sent to review
    total_review = int(routed_to_review.sum())
    total_ac = int((y_te == 2).sum())

    # Combined: REVIEW counts as correct flag for recall (human will catch it)
    # FN = AC accounts that went through fused lane AND got pred!=2
    effective_tp = fused_ac_tp + review_ac_tp   # AC accounts caught (either ACT or REVIEW)
    effective_fn = fused_ac_fn                   # AC accounts in fused lane that were missed
    effective_fp = fused_ac_fp                   # non-AC accounts auto-ACT'd incorrectly
    recall_effective = effective_tp / max(total_ac, 1)
    precision_effective = fused_ac_tp / max(fused_ac_tp + fused_ac_fp, 1)

    print(f"KL threshold={kl_thresh}:")
    print(f"  Routed to REVIEW: {total_review} ({100*total_review/len(y_te):.1f}% of all)")
    print(f"    AC in REVIEW:      {review_ac_tp} ({100*review_ac_tp/total_ac:.1f}% of all AC)")
    print(f"    non-AC in REVIEW:  {review_ac_fp}")
    print(f"  Fused lane (low conflict):")
    print(f"    AC auto-ACT (TP):  {fused_ac_tp}")
    print(f"    non-AC auto-ACT (FP): {fused_ac_fp}")
    print(f"    AC missed (FN):    {fused_ac_fn}")
    print(f"  Effective recall (AC flagged any way): {recall_effective:.4f}")
    print(f"  Auto-ACT precision (fused lane only):  {precision_effective:.4f}")
    fn_cost = effective_fn * 2000
    fp_cost = effective_fp * 500
    review_cost = total_review * 200  # assumed review cost Rs200/account
    total_cost = fn_cost + fp_cost + review_cost
    print(f"  Cost: FN={fn_cost:,} + FP={fp_cost:,} + Review={review_cost:,} = Rs{total_cost:,}")
    print()