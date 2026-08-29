import sys, io, json, os, warnings
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

SECONDS_PER_DAY = 86400

from graph.temporal_graph import (
    build_graph_as_of,
    extract_account_structural_features,
    extract_account_behavioral_features,
)
from features.feature_pipeline import (
    STRUCTURAL_FEATURES,
    BEHAVIORAL_FEATURES,
)
from decision.decision_engine import DecisionEngine, Decision, RoutingLane
from models.fused_model import FusedCalibratedClassifier

# 1. Load data
events = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels = pd.read_parquet("data/labels.parquet")
rings = pd.read_parquet("data/rings.parquet")
split = json.load(open("data/split_info.json"))

SIM_START_TS = split["sim_start_ts"]

# 2. Load models
fused = joblib.load("models/fused_calibrated.pkl")
behav = joblib.load("models/behavioral_lgbm.pkl")
struct = joblib.load("models/structural_lgbm.pkl")
engine = DecisionEngine(kl_conflict_threshold=0.5)

# 3. Filter late-forming rings (start_day >= 55)
unique_rings = rings.drop_duplicates("ring_id").copy()
late_rings = unique_rings[unique_rings["ring_formation_start_day"] >= 55].sort_values("ring_formation_start_day")

print(f"=== TRAJECTORY EVALUATION: {len(late_rings)} LATE-FORMING RINGS (start_day >= 55) ===", flush=True)
print(f"Ring types: {dict(late_rings['ring_type'].value_counts())}", flush=True)

# Build map of (ring_id, checkpoint_idx) -> checkpoint_day
ring_ck_map = {}
day_to_eval_accs = {}

for _, r_row in late_rings.iterrows():
    ring_id = r_row["ring_id"]
    start_day = int(r_row["ring_formation_start_day"])
    comp_day = int(r_row["ring_formation_complete_day"])
    
    t1 = max(1, start_day - 5)
    t2 = start_day
    t3 = int((start_day + comp_day) / 2)
    t4 = comp_day
    t5 = min(90, comp_day + 5)
    
    ck_list = [
        ("T1_pre_start", t1, 1),
        ("T2_start",     t2, 2),
        ("T3_midpoint",  t3, 3),
        ("T4_complete",  t4, 4),
        ("T5_post_comp", t5, 5),
    ]
    ring_ck_map[ring_id] = ck_list
    
    acc_ids = rings[rings["ring_id"] == ring_id]["account_id"].tolist()
    for ck_name, ck_day, ck_idx in ck_list:
        if ck_day not in day_to_eval_accs:
            day_to_eval_accs[ck_day] = set()
        day_to_eval_accs[ck_day].update(acc_ids)

unique_days = sorted(day_to_eval_accs.keys())
print(f"Evaluating across {len(unique_days)} unique checkpoint days...", flush=True)

# Precompute scored inferences per (day, account_id)
day_acc_results = {}

for ck_day in unique_days:
    as_of_ts = SIM_START_TS + ck_day * SECONDS_PER_DAY
    acc_list = sorted(list(day_to_eval_accs[ck_day]))
    
    G = build_graph_as_of(events, as_of_ts)
    struct_df = extract_account_structural_features(G, acc_list).set_index("account_id")
    behav_df = extract_account_behavioral_features(events, as_of_ts, acc_list, accounts).set_index("account_id")
    
    s_mat = struct_df[STRUCTURAL_FEATURES].reindex(acc_list).fillna(0)
    b_mat = behav_df[BEHAVIORAL_FEATURES].reindex(acc_list).fillna(0)
    
    p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_mat, b_mat)
    
    n_orders_arr = b_mat["n_orders"].fillna(0).astype(int).values
    obs_days_arr = b_mat["account_age_days"].fillna(0).values
    
    dec_results = engine.decide_batch(
        account_ids=acc_list,
        p_fused_matrix=p_fused,
        p_struct_matrix=p_struct,
        p_behav_matrix=p_behav,
        observation_days=obs_days_arr,
        n_orders_arr=n_orders_arr,
        as_of_ts=as_of_ts,
    )
    
    for i, acc in enumerate(acc_list):
        dr = dec_results[i]
        day_acc_results[(ck_day, acc)] = {
            "n_orders": int(n_orders_arr[i]),
            "p_struct_ac": float(p_struct[i, 2]),
            "p_behav_ac": float(p_behav[i, 2]),
            "p_fused_ac": float(p_fused[i, 2]),
            "sym_kl_divergence": float(dr.sym_kl_divergence),
            "routing_lane": dr.routing_lane.value,
            "decision": dr.decision.value,
            "evidence_conflict": bool(dr.evidence_conflict),
        }

print("Inference scoring complete. Compiling trajectory table...", flush=True)

records = []
for _, r_row in late_rings.iterrows():
    ring_id = r_row["ring_id"]
    ring_type = r_row["ring_type"]
    start_day = int(r_row["ring_formation_start_day"])
    comp_day = int(r_row["ring_formation_complete_day"])
    
    ring_members = rings[rings["ring_id"] == ring_id].copy()
    acc_ids = ring_members["account_id"].tolist()
    ck_list = ring_ck_map[ring_id]
    
    for ck_name, ck_day, ck_idx in ck_list:
        for acc in acc_ids:
            mem_info = ring_members[ring_members["account_id"] == acc].iloc[0]
            res = day_acc_results[(ck_day, acc)]
            
            records.append({
                "ring_id": ring_id,
                "ring_type": ring_type,
                "formation_start_day": start_day,
                "formation_complete_day": comp_day,
                "checkpoint_label": ck_name,
                "checkpoint_idx": ck_idx,
                "checkpoint_day": ck_day,
                "days_from_start": ck_day - start_day,
                "account_id": acc,
                "is_sleeper": bool(mem_info.get("is_sleeper", False)),
                "is_varied_payout": bool(mem_info.get("is_varied_payout", False)),
                "n_orders": res["n_orders"],
                "p_struct_ac": res["p_struct_ac"],
                "p_behav_ac": res["p_behav_ac"],
                "p_fused_ac": res["p_fused_ac"],
                "sym_kl_divergence": res["sym_kl_divergence"],
                "routing_lane": res["routing_lane"],
                "decision": res["decision"],
                "evidence_conflict": res["evidence_conflict"],
            })

traj_df = pd.DataFrame(records)

os.makedirs("evals/results", exist_ok=True)
traj_df.to_parquet("evals/results/trajectory_results.parquet", index=False)
print(f"Saved {len(traj_df)} trajectory records to evals/results/trajectory_results.parquet", flush=True)

# -----------------------------------------------------------------------------
# 4. Compute Aggregate Trajectory Statistics
# -----------------------------------------------------------------------------
print("\n" + "="*80, flush=True)
print("AGGREGATE TRAJECTORY STATISTICS", flush=True)
print("="*80, flush=True)

ring_stats = []
for ring_id, grp in traj_df.groupby("ring_id"):
    r_type = grp["ring_type"].iloc[0]
    start_day = grp["formation_start_day"].iloc[0]
    comp_day = grp["formation_complete_day"].iloc[0]
    
    ck_decisions = []
    ck_lanes = []
    ck_kls = []
    ck_str = []
    ck_beh = []
    
    for ck_idx in range(1, 6):
        ck_sub = grp[grp["checkpoint_idx"] == ck_idx]
        decs = set(ck_sub["decision"])
        lanes = set(ck_sub["routing_lane"])
        
        if "ACT" in decs:
            top_dec = "ACT"
        elif "REVIEW" in decs:
            top_dec = "REVIEW"
        elif "WAIT_MONITOR" in decs:
            top_dec = "WAIT_MONITOR"
        else:
            top_dec = "ABSTAIN"
            
        ck_decisions.append(top_dec)
        ck_lanes.append(",".join(sorted(lanes)))
        ck_kls.append(ck_sub["sym_kl_divergence"].mean())
        ck_str.append(ck_sub["p_struct_ac"].mean())
        ck_beh.append(ck_sub["p_behav_ac"].mean())
        
    has_evolved = len(set(ck_decisions)) > 1 or len(set(ck_lanes)) > 1
    
    first_rev_day = None
    first_act_day = None
    
    for ck_idx in range(1, 6):
        ck_sub = grp[grp["checkpoint_idx"] == ck_idx]
        ck_day = ck_sub["checkpoint_day"].iloc[0]
        if any(ck_sub["decision"] == "REVIEW") and first_rev_day is None:
            first_rev_day = ck_day - start_day
        if any(ck_sub["decision"] == "ACT") and first_act_day is None:
            first_act_day = ck_day - start_day
            
    never_escalated = not any(d in ["REVIEW", "ACT"] for d in ck_decisions)
    
    ring_stats.append({
        "ring_id": ring_id,
        "ring_type": r_type,
        "n_members": grp["account_id"].nunique(),
        "start_day": start_day,
        "comp_day": comp_day,
        "has_evolved": has_evolved,
        "first_review_days_from_start": first_rev_day,
        "first_act_days_from_start": first_act_day,
        "never_escalated": never_escalated,
        "t1_dec": ck_decisions[0], "t2_dec": ck_decisions[1], "t3_dec": ck_decisions[2], "t4_dec": ck_decisions[3], "t5_dec": ck_decisions[4],
        "t1_kl": ck_kls[0], "t2_kl": ck_kls[1], "t3_kl": ck_kls[2], "t4_kl": ck_kls[3], "t5_kl": ck_kls[4],
        "t1_str": ck_str[0], "t3_str": ck_str[2], "t5_str": ck_str[4],
        "t1_beh": ck_beh[0], "t3_beh": ck_beh[2], "t5_beh": ck_beh[4],
    })

stat_df = pd.DataFrame(ring_stats)

n_total_rings = len(stat_df)
pct_evolved = (stat_df["has_evolved"].mean()) * 100
pct_never_esc = (stat_df["never_escalated"].mean()) * 100

rev_days = stat_df["first_review_days_from_start"].dropna()
act_days = stat_df["first_act_days_from_start"].dropna()

print(f"Total Rings Evaluated: {n_total_rings}", flush=True)
print(f"% Rings with Evolving Decisions across Checkpoints: {pct_evolved:.1f}% ({stat_df['has_evolved'].sum()}/{n_total_rings})", flush=True)
print(f"% Rings that NEVER leave WAIT/ABSTAIN: {pct_never_esc:.1f}% ({stat_df['never_escalated'].sum()}/{n_total_rings})", flush=True)
print(f"% Rings reaching REVIEW: {(~stat_df['first_review_days_from_start'].isna()).mean()*100:.1f}% ({len(rev_days)}/{n_total_rings})", flush=True)
print(f"% Rings reaching ACT:    {(~stat_df['first_act_days_from_start'].isna()).mean()*100:.1f}% ({len(act_days)}/{n_total_rings})", flush=True)

if len(rev_days) > 0:
    print(f"Time to First REVIEW (days from start): Mean={rev_days.mean():.1f}d | Min={rev_days.min():.1f}d | Max={rev_days.max():.1f}d", flush=True)
if len(act_days) > 0:
    print(f"Time to First ACT (days from start):    Mean={act_days.mean():.1f}d | Min={act_days.min():.1f}d | Max={act_days.max():.1f}d", flush=True)

print("\nMean sym_KL divergence progression across checkpoints (T1 -> T5):", flush=True)
print(f"  T1 (Pre-Start -5d):  sym_KL = {stat_df['t1_kl'].mean():.4f} (Mean P(struct)={stat_df['t1_str'].mean():.3f}, P(behav)={stat_df['t1_beh'].mean():.3f})", flush=True)
print(f"  T2 (Start Day):      sym_KL = {stat_df['t2_kl'].mean():.4f}", flush=True)
print(f"  T3 (Midpoint):       sym_KL = {stat_df['t3_kl'].mean():.4f} (Mean P(struct)={stat_df['t3_str'].mean():.3f}, P(behav)={stat_df['t3_beh'].mean():.3f})", flush=True)
print(f"  T4 (Complete Day):   sym_KL = {stat_df['t4_kl'].mean():.4f}", flush=True)
print(f"  T5 (Post-Comp +5d):  sym_KL = {stat_df['t5_kl'].mean():.4f} (Mean P(struct)={stat_df['t5_str'].mean():.3f}, P(behav)={stat_df['t5_beh'].mean():.3f})", flush=True)

print("\nBreakdown by Ring Type:", flush=True)
for r_type, grp in stat_df.groupby("ring_type"):
    print(f"\nRing Type: {r_type} (N={len(grp)})", flush=True)
    print(f"  % Evolved: {grp['has_evolved'].mean()*100:.1f}%", flush=True)
    print(f"  % Reaching REVIEW: {(~grp['first_review_days_from_start'].isna()).mean()*100:.1f}%", flush=True)
    print(f"  % Reaching ACT:    {(~grp['first_act_days_from_start'].isna()).mean()*100:.1f}%", flush=True)
    print(f"  % Never Escalated: {grp['never_escalated'].mean()*100:.1f}%", flush=True)
    r_rev = grp['first_review_days_from_start'].dropna()
    if len(r_rev) > 0:
        print(f"  Mean Days to Review: {r_rev.mean():.1f}d", flush=True)
    r_act = grp['first_act_days_from_start'].dropna()
    if len(r_act) > 0:
        print(f"  Mean Days to ACT:    {r_act.mean():.1f}d", flush=True)
    print(f"  sym_KL progression: T1={grp['t1_kl'].mean():.3f} -> T3={grp['t3_kl'].mean():.3f} -> T5={grp['t5_kl'].mean():.3f}", flush=True)

# -----------------------------------------------------------------------------
# 5. Print Concrete Example Ring Trajectories
# -----------------------------------------------------------------------------
print("\n" + "="*80, flush=True)
print("CONCRETE EXAMPLE RING TRAJECTORIES VERBATIM", flush=True)
print("="*80, flush=True)

example_rings = ["PROMO_001", "RETURN_027", "REFARM_057"]
for ring_id in example_rings:
    sub = traj_df[traj_df["ring_id"] == ring_id]
    r_type = sub["ring_type"].iloc[0]
    start_d = sub["formation_start_day"].iloc[0]
    comp_d = sub["formation_complete_day"].iloc[0]
    n_m = sub["account_id"].nunique()
    
    print(f"\n--- Example Ring: {ring_id} ({r_type.upper()}) | Members: {n_m} | Formation: Day {start_d} -> Day {comp_d} ---", flush=True)
    for ck_idx in range(1, 6):
        ck_sub = sub[sub["checkpoint_idx"] == ck_idx]
        ck_label = ck_sub["checkpoint_label"].iloc[0]
        ck_day = ck_sub["checkpoint_day"].iloc[0]
        days_from = ck_sub["days_from_start"].iloc[0]
        
        dec_counts = dict(ck_sub["decision"].value_counts())
        lane_counts = dict(ck_sub["routing_lane"].value_counts())
        avg_str = ck_sub["p_struct_ac"].mean()
        avg_beh = ck_sub["p_behav_ac"].mean()
        avg_fused = ck_sub["p_fused_ac"].mean()
        avg_kl = ck_sub["sym_kl_divergence"].mean()
        avg_ord = ck_sub["n_orders"].mean()
        
        print(f"  Checkpoint {ck_idx} ({ck_label:13s} Day {ck_day:2d}, offset {days_from:+3d}d):", flush=True)
        print(f"    Decisions: {dec_counts}", flush=True)
        print(f"    Lanes:     {lane_counts}", flush=True)
        print(f"    Metrics:   Avg Orders={avg_ord:.1f} | P(struct)={avg_str:.3f} | P(behav)={avg_beh:.3f} | P(fused)={avg_fused:.3f} | sym_KL={avg_kl:.3f}", flush=True)
