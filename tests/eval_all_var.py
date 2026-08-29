import sys, io, json, warnings
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

import pandas as pd, numpy as np, joblib
from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from decision.decision_engine import DecisionEngine, Decision

events = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels = pd.read_parquet("data/labels.parquet")
split = json.load(open("data/split_info.json"))

splits = build_temporal_splits(events, accounts, labels, split)
fused = joblib.load("models/fused_calibrated.pkl")
engine = DecisionEngine(kl_conflict_threshold=0.5)

print("=== VARIED PAYOUT AC EVALUATION ACROSS ALL TEMPORAL SPLITS ===")
for split_name in ["train", "val", "test"]:
    sp = splits[split_name]
    idx = sp["labels"].index
    s = sp["struct"].reindex(idx).fillna(0)
    b = sp["behav"].reindex(idx).fillna(0)
    y = sp["labels"]["label"].values
    
    p_struct, p_behav, p_fused, _ = fused.predict_proba_sub(s, b)
    n_orders = b["n_orders"].fillna(0).astype(int).values
    obs_days = b["account_age_days"].fillna(0).values
    
    results = engine.decide_batch(
        account_ids=list(idx),
        p_fused_matrix=p_fused,
        p_struct_matrix=p_struct,
        p_behav_matrix=p_behav,
        observation_days=obs_days,
        n_orders_arr=n_orders,
        as_of_ts=split[f"{split_name}_end_ts"],
    )
    
    res_df = pd.DataFrame([
        {
            "account_id": r.account_id,
            "decision": r.decision.value,
            "routing_lane": r.routing_lane.value,
            "sym_kl": r.sym_kl_divergence,
            "struct_sub": r.structural_sub_score,
            "behav_sub": r.behavioral_sub_score,
        }
        for r in results
    ]).set_index("account_id")
    
    m_df = labels.set_index("account_id").join(res_df).loc[idx]
    var_sub = m_df[m_df["counterfactual_subset"] == "varied_payout_ac"]
    n_v = len(var_sub)
    if n_v > 0:
        act_c = (var_sub["decision"] == "ACT").sum()
        rev_c = (var_sub["decision"] == "REVIEW").sum()
        wait_c = (var_sub["decision"] == "WAIT_MONITOR").sum()
        abs_c = (var_sub["decision"] == "ABSTAIN").sum()
        eff_rec = (act_c + rev_c) / n_v
        print(f"Split {split_name:5s} (N={n_v:2d}): ACT={act_c:2d} ({act_c/n_v*100:5.1f}%), REVIEW={rev_c:2d} ({rev_c/n_v*100:5.1f}%), WAIT={wait_c:2d}, ABSTAIN={abs_c:2d} | Eff. Recall={eff_rec*100:5.1f}%")
        print(f"  Mean P(behav)={var_sub['behav_sub'].mean():.3f} | Mean P(struct)={var_sub['struct_sub'].mean():.3f} | Mean sym_KL={var_sub['sym_kl'].mean():.3f}")
