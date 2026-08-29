import sys, io; sys.path.insert(0, "."); sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import json, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
from models.model_suite import FusedCalibratedClassifier
from features.feature_pipeline import STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES, build_temporal_splits
from decision.decision_engine import DecisionEngine, Decision, RoutingLane

events   = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels   = pd.read_parquet("data/labels.parquet")
split    = json.load(open("data/split_info.json"))

splits = build_temporal_splits(events, accounts, labels, split)
sp=splits["test"]; idx=sp["labels"].index
s_te=sp["struct"].reindex(idx).fillna(0)
b_te=sp["behav"].reindex(idx).fillna(0)
y_te=sp["labels"]["label"].values
y_str=sp["labels"]["label_str"].values
label_map_inv={0:"benign_independent",1:"benign_coordinated",2:"abusive_coordinated"}

fused=joblib.load("models/fused_calibrated.pkl")
p_struct,p_behav,p_fused,_=fused.predict_proba_sub(s_te,b_te)

engine = DecisionEngine(kl_conflict_threshold=0.5)

# Build n_orders from behavioral features
n_orders_arr = b_te["n_orders"].fillna(0).astype(int).values
obs_days_arr = b_te["account_age_days"].fillna(0).values
account_ids  = list(idx)

print("Running decision_engine.decide_batch on test split ...")
results = engine.decide_batch(
    account_ids=account_ids,
    p_fused_matrix=p_fused,
    p_struct_matrix=p_struct,
    p_behav_matrix=p_behav,
    observation_days=obs_days_arr,
    n_orders_arr=n_orders_arr,
    as_of_ts=split["test_end_ts"],
)

# Aggregate
summary = engine.routing_summary(results)
print(f"\nDecision counts: {summary['decision_counts']}")
print(f"Decision fractions: {summary['decision_fractions']}")
print(f"Routing lane counts: {summary['routing_lane_counts']}")
print(f"Simulated review cost: Rs{summary['simulated_review_cost']:,.0f}")

# Per-class breakdown
from collections import Counter
dec_by_true_class = {0:Counter(), 1:Counter(), 2:Counter()}
lane_by_true_class= {0:Counter(), 1:Counter(), 2:Counter()}
for r, yt in zip(results, y_te):
    dec_by_true_class[yt][r.decision.value] += 1
    lane_by_true_class[yt][r.routing_lane.value] += 1

print("\nDecisions by true class:")
for cls, name in [(0,"BI"),(1,"BC"),(2,"AC")]:
    n_cls = (y_te==cls).sum()
    print(f"  {name} (n={n_cls}): {dict(dec_by_true_class[cls])}")
print("\nRouting lane by true class:")
for cls, name in [(0,"BI"),(1,"BC"),(2,"AC")]:
    print(f"  {name}: {dict(lane_by_true_class[cls])}")

# Key metrics
ac_results = [(r, yt) for r, yt in zip(results, y_te) if yt==2]
ac_act     = sum(1 for r,_ in ac_results if r.decision==Decision.ACT)
ac_review  = sum(1 for r,_ in ac_results if r.decision==Decision.REVIEW)
ac_wait    = sum(1 for r,_ in ac_results if r.decision==Decision.WAIT_MONITOR)
ac_abstain = sum(1 for r,_ in ac_results if r.decision==Decision.ABSTAIN)
n_ac = len(ac_results)
print(f"\nAC accounts ({n_ac} total):")
print(f"  -> ACT (auto-ACT recall):    {ac_act}/{n_ac} = {ac_act/n_ac:.4f}")
print(f"  -> REVIEW (routed recall):   {ac_review}/{n_ac} = {ac_review/n_ac:.4f}")
print(f"  -> WAIT_MONITOR (missed):    {ac_wait}/{n_ac} = {ac_wait/n_ac:.4f}")
print(f"  -> ABSTAIN (insufficient):   {ac_abstain}/{n_ac} = {ac_abstain/n_ac:.4f}")
print(f"  Effective recall (ACT+REVIEW): {(ac_act+ac_review)/n_ac:.4f}")
print(f"  REPORTING: '{(ac_act+ac_review)/n_ac*100:.0f}% recall via routing")
print(f"    (direct auto-ACT recall: {ac_act/n_ac*100:.1f}%, remaining {ac_review/n_ac*100:.1f}% correctly routed to human REVIEW)'")

# Non-AC auto-ACT FP check
non_ac_act = sum(1 for r, yt in zip(results, y_te) if r.decision==Decision.ACT and yt!=2)
print(f"\nauto-ACT FP (non-AC predicted ACT): {non_ac_act}")
print(f"auto-ACT precision: {ac_act/(ac_act+non_ac_act) if (ac_act+non_ac_act)>0 else 'N/A'}")
print("\nDecision engine end-to-end: OK")