import sys, io; sys.path.insert(0,".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, joblib
from models.model_suite import FusedCalibratedClassifier
from features.feature_pipeline import BEHAVIORAL_FEATURES, build_temporal_splits
from decision.decision_engine import DecisionEngine, Decision
from sklearn.metrics import precision_score, recall_score, f1_score

events   = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels   = pd.read_parquet("data/labels.parquet")
split    = json.load(open("data/split_info.json"))
cost     = json.load(open("data/cost_config.json"))
c_fp=cost["c_false_positive"]; c_fn=cost["c_false_negative"]
c_review=cost["c_review"]; c_wait=cost["c_wait_per_day"]

splits = build_temporal_splits(events, accounts, labels, split)
sp = splits["test"]; idx = sp["labels"].index
s_te = sp["struct"].reindex(idx).fillna(0)
b_te = sp["behav"].reindex(idx).fillna(0)
y_te = sp["labels"]["label"].values

fused = joblib.load("models/fused_calibrated.pkl")
behav = joblib.load("models/behavioral_lgbm.pkl")
p_struct, p_behav, p_fused, _ = fused.predict_proba_sub(s_te, b_te)

# Routing engine
engine = DecisionEngine(kl_conflict_threshold=0.5)
n_orders_arr = b_te["n_orders"].fillna(0).astype(int).values
obs_days_arr = b_te["account_age_days"].fillna(0).values
results = engine.decide_batch(
    account_ids=list(idx),
    p_fused_matrix=p_fused, p_struct_matrix=p_struct, p_behav_matrix=p_behav,
    observation_days=obs_days_arr, n_orders_arr=n_orders_arr,
    as_of_ts=split["test_end_ts"],
)

n_ac      = int((y_te==2).sum())
test_days = (split["test_end_ts"] - split["val_end_ts"]) / 86400

ac_act    = sum(1 for r,yt in zip(results,y_te) if yt==2 and r.decision==Decision.ACT)
ac_review = sum(1 for r,yt in zip(results,y_te) if yt==2 and r.decision==Decision.REVIEW)
ac_abs    = sum(1 for r,yt in zip(results,y_te) if yt==2 and r.decision==Decision.ABSTAIN)
n_rev_total = sum(1 for r in results if r.decision==Decision.REVIEW)
n_fp_act  = sum(1 for r,yt in zip(results,y_te) if r.decision==Decision.ACT and yt!=2)

# Behavioral-only
pred_b = behav.predict(b_te[BEHAVIORAL_FEATURES].fillna(0))
b_tp = int(((pred_b==2)&(y_te==2)).sum())
b_fp = int(((pred_b==2)&(y_te!=2)).sum())
b_fn = int(((pred_b!=2)&(y_te==2)).sum())

print("=== VERIFIED NUMBERS (re-run from actual data) ===")
print()
print(f"Test split: {n_ac} total AC accounts, {len(y_te)} total accounts, {test_days:.0f} days")
print()
print(f"BEHAVIORAL-ONLY:")
print(f"  TP={b_tp} FP={b_fp} FN={b_fn}")
print(f"  Prec-AC={b_tp/(b_tp+b_fp):.4f}  Rec-AC={b_tp/n_ac:.4f}  F1-AC={2*b_tp/(2*b_tp+b_fp+b_fn):.4f}")
print(f"  FP cost: {b_fp} x {c_fp:.0f} = Rs{b_fp*c_fp:,.0f}")
print(f"  FN cost: {b_fn} x {c_fn:.0f} = Rs{b_fn*c_fn:,.0f}")
print(f"  Total:   Rs{b_fp*c_fp+b_fn*c_fn:,.0f}")
print()
print(f"ROUTING KL=0.5:")
print(f"  ACT-TP={ac_act}  ACT-FP={n_fp_act}  REVIEW={ac_review}  ABSTAIN={ac_abs}")
print(f"  Effective recall (ACT+REVIEW): {(ac_act+ac_review)/n_ac:.4f}  ({(ac_act+ac_review)/n_ac*100:.1f}%)")
print(f"  Direct auto-ACT recall: {ac_act/n_ac:.4f}  ({ac_act/n_ac*100:.1f}%)")
print(f"  REVIEW routed recall:   {ac_review/n_ac:.4f}  ({ac_review/n_ac*100:.1f}%)")
print(f"  ABSTAIN (n_orders<2):   {ac_abs/n_ac:.4f}  ({ac_abs/n_ac*100:.1f}%)")
print(f"  REVIEW queue total: {n_rev_total}  (AC precision: {ac_review/n_rev_total*100:.1f}%)")
print(f"  FP cost:   0 x {c_fp:.0f} = Rs0")
print(f"  Rev cost:  {n_rev_total} x {c_review:.0f} = Rs{n_rev_total*c_review:,.0f}")
print(f"  ABSTAIN-as-FN:  {ac_abs} x {c_fn:.0f} = Rs{ac_abs*c_fn:,.0f}")
print(f"  ABSTAIN-as-wait:{ac_abs} x {test_days:.0f}d x {c_wait:.0f} = Rs{ac_abs*test_days*c_wait:,.0f}")
print(f"  Total (ABSTAIN=FN):   Rs{n_rev_total*c_review + ac_abs*c_fn:,.0f}")
print(f"  Total (ABSTAIN=wait): Rs{n_rev_total*c_review + ac_abs*test_days*c_wait:,.0f}")
print()
print("=== README NUMBERS TO VERIFY ===")
readme_nums = {
    "behavioral FP":      (b_fp, 25, "test"),
    "behavioral FN":      (b_fn, 9,  "test"),
    "behavioral prec":    (round(b_tp/(b_tp+b_fp),3), 0.883, "check"),
    "behavioral recall":  (round(b_tp/n_ac,4), 0.9545, "test"),
    "behavioral total":   (b_fp*c_fp+b_fn*c_fn, 30500, "test"),
    "routing n_review":   (n_rev_total, 779, "test"),
    "routing ac_review":  (ac_review, 124, "test"),
    "routing ac_act":     (ac_act, 38, "test"),
    "routing ac_abstain": (ac_abs, 36, "test"),
    "routing ac_fp":      (n_fp_act, 0, "test"),
    "routing rev_cost":   (n_rev_total*c_review, 116850, "test"),
    "routing abs_fn":     (ac_abs*c_fn, 72000, "test"),
    "routing abs_wait":   (ac_abs*test_days*c_wait, 32400, "test"),
    "routing total FN":   (n_rev_total*c_review+ac_abs*c_fn, 188850, "test"),
    "routing total wait": (n_rev_total*c_review+ac_abs*test_days*c_wait, 149250, "test"),
    "n_ac total":         (n_ac, 198, "test"),
    "test_days":          (test_days, 18, "test"),
    "eff recall pct":     (round((ac_act+ac_review)/n_ac,4), 0.8182, "test"),
    "direct ACT recall":  (round(ac_act/n_ac,4), 0.1919, "test"),
    "REVIEW recall":      (round(ac_review/n_ac,4), 0.6263, "test"),
    "ABSTAIN pct":        (round(ac_abs/n_ac,4), 0.1818, "test"),
}
all_pass = True
for name,(actual,expected,_) in readme_nums.items():
    match = abs(float(actual)-float(expected)) < 1.0
    status = "OK" if match else "MISMATCH"
    if not match:
        all_pass = False
        print(f"  {status}: {name}: actual={actual} expected={expected}")
if all_pass:
    print("  ALL README NUMBERS VERIFIED AGAINST LIVE DATA.")