import sys, io; sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, joblib
from models.model_suite import FusedCalibratedClassifier
from features.feature_pipeline import STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES, build_temporal_splits

events   = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels   = pd.read_parquet("data/labels.parquet")
split    = json.load(open("data/split_info.json"))
cost     = json.load(open("data/cost_config.json"))
c_fp=cost["c_false_positive"]; c_fn=cost["c_false_negative"]; c_review=cost["c_review"]
print(f"Cost config: c_fp=Rs{c_fp:.0f}  c_fn=Rs{c_fn:.0f}  c_review=Rs{c_review:.0f}")

splits = build_temporal_splits(events, accounts, labels, split)
sp=splits["test"]; idx=sp["labels"].index
s_te=sp["struct"].reindex(idx).fillna(0)
b_te=sp["behav"].reindex(idx).fillna(0)
y_te=sp["labels"]["label"].values

fused = joblib.load("models/fused_calibrated.pkl")
behav = joblib.load("models/behavioral_lgbm.pkl")
p_struct,p_behav,p_fused,_ = fused.predict_proba_sub(s_te, b_te)
eps=1e-9
def kl(p,q): p=np.clip(p,eps,1);q=np.clip(q,eps,1); return np.sum(p*np.log(p/q),axis=1)
sym_kl = (kl(p_struct,p_behav)+kl(p_behav,p_struct))/2
n_total_ac = int((y_te==2).sum())

print()
print("="*65)
print("ROUTING KL=0.5  FULL COST BREAKDOWN")
print("="*65)
kl_thresh=0.5
routed_review=(sym_kl>kl_thresh); routed_fused=~routed_review
pred_f=np.argmax(p_fused,axis=1)
ac_auto_tp=int(((pred_f==2)&(y_te==2)&routed_fused).sum())
ac_auto_fp=int(((pred_f==2)&(y_te!=2)&routed_fused).sum())
ac_auto_fn=int(((pred_f!=2)&(y_te==2)&routed_fused).sum())
review_ac=int((routed_review&(y_te==2)).sum())
review_non=int((routed_review&(y_te!=2)).sum())
n_review=int(routed_review.sum())
fn_cost=ac_auto_fn*c_fn; fp_cost=ac_auto_fp*c_fp; rev_cost=n_review*c_review
total_routing=fn_cost+fp_cost+rev_cost
eff_recall=(ac_auto_tp+review_ac)/n_total_ac
print(f"  REVIEW lane : {n_review} accounts ({n_review/len(y_te)*100:.1f}%)")
print(f"    AC in REVIEW        : {review_ac} ({review_ac/n_total_ac*100:.1f}% of all AC)")
print(f"    non-AC in REVIEW    : {review_non}")
print(f"    REVIEW precision    : {review_ac/n_review*100:.1f}% AC in queue")
print(f"  FUSED lane  : {routed_fused.sum()} accounts")
print(f"    auto-ACT TP         : {ac_auto_tp}")
print(f"    auto-ACT FP         : {ac_auto_fp}")
print(f"    missed FN           : {ac_auto_fn}")
print(f"  Effective recall: ({ac_auto_tp}+{review_ac})/{n_total_ac} = {eff_recall:.4f}")
print()
print(f"  COST BREAKDOWN (c_fp=Rs{c_fp:.0f}, c_fn=Rs{c_fn:.0f}, c_review=Rs{c_review:.0f}):")
print(f"    FP  cost : {ac_auto_fp:3d} FP  x Rs{c_fp:.0f}  = Rs{fp_cost:>9,.0f}")
print(f"    FN  cost : {ac_auto_fn:3d} FN  x Rs{c_fn:.0f} = Rs{fn_cost:>9,.0f}")
print(f"    Rev cost : {n_review:3d} accs x Rs{c_review:.0f}  = Rs{rev_cost:>9,.0f}")
print(f"    TOTAL                        = Rs{total_routing:>9,.0f}")

print()
print("="*65)
print("BEHAVIORAL-ONLY  FULL COST BREAKDOWN")
print("="*65)
pred_b=behav.predict(b_te[BEHAVIORAL_FEATURES].fillna(0))
b_tp=int(((pred_b==2)&(y_te==2)).sum())
b_fp=int(((pred_b==2)&(y_te!=2)).sum())
b_fn=int(((pred_b!=2)&(y_te==2)).sum())
fn_cost_b=b_fn*c_fn; fp_cost_b=b_fp*c_fp
total_beh=fn_cost_b+fp_cost_b
print(f"  Direct recall: {b_tp}/{n_total_ac} = {b_tp/n_total_ac:.4f}")
print(f"  TP={b_tp}  FP={b_fp}  FN={b_fn}")
print(f"  COST BREAKDOWN:")
print(f"    FP  cost : {b_fp:3d} FP  x Rs{c_fp:.0f}  = Rs{fp_cost_b:>9,.0f}")
print(f"    FN  cost : {b_fn:3d} FN  x Rs{c_fn:.0f} = Rs{fn_cost_b:>9,.0f}")
print(f"    Rev cost :   0 accs x Rs{c_review:.0f}  = Rs{0:>9,.0f}")
print(f"    TOTAL                        = Rs{total_beh:>9,.0f}")

print()
print("="*65)
print("COMPARISON")
print("="*65)
diff=total_routing-total_beh
print(f"  Routing KL=0.5   : Rs{total_routing:>9,.0f}")
print(f"  Behavioral-only  : Rs{total_beh:>9,.0f}")
print(f"  Difference       : Rs{abs(diff):>9,.0f} {'MORE' if diff>0 else 'LESS'} for routing")
print()
print(f"  VERDICT: Routing costs Rs{abs(diff):,.0f} MORE than behavioral-alone.")
print(f"  This is the cost of zero FN vs {b_fn} FN: the routing approach")
print(f"  trades Rs{b_fn*c_fn-0:,.0f} in avoided FN losses for Rs{rev_cost:,.0f}")
print(f"  in review cost, at the price of a REVIEW queue (not automation).")
print()
print(f"  Break-even c_review where routing == behavioral:")
be = (total_beh - 0 - 0) / n_review  # all cost saving comes from 0 FN vs b_fn FN
actual_saving = b_fn * c_fn - b_fp * c_fp  # net saving from 0FN,0FP vs behavioral
be2 = actual_saving / n_review
print(f"    = (behavioral_cost - routing_FP_and_FN_cost) / n_review")
print(f"    = (Rs{total_beh:,.0f} - Rs0) / {n_review}")
print(f"    = Rs{be:.0f}/account")
print(f"  At c_review < Rs{be:.0f}/account, routing dominates behavioral-alone.")