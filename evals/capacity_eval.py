"""
AbuseRing Sentinel — Capacity-Constrained Review Queue Evaluation
==================================================================
Simulates fixed daily manual review capacity limits (K = 25, 50, 100, 200, 300, 500, 779)
on the canonical Test split (N=3,467, 779 cases routed to REVIEW, 124 True AC).

Compares 7 triage policies:
  1. FIFO (Natural list arrival order)
  2. RANDOM_SHUFFLE (Seed 42, true uninformative baseline)
  3. TIME_OF_FLAGGING (Chronological order of first triggering order in test window)
  4. SCORE_DESC (P_fused(AC) descending)
  5. VAR_FINANCIAL (P_fused * Order Exposure)
  6. EXPOSURE_WEIGHTED (P_fused * sqrt(Order Exposure) — sym_KL ablation)
  7. CONFLICT_AWARE (P_fused * (1 + log(1 + sym_KL)) * sqrt(Order Exposure))

Outputs:
  evals/results/capacity_constrained_results.json
"""

import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_pipeline import build_temporal_splits
from decision.decision_engine import DecisionEngine, Decision
from policy.capacity_policy import TriageStrategy, QueueItem, ReviewQueueEngine

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("evals/results")

def run_capacity_evaluation():
    print("=" * 80)
    print("ABUSERING SENTINEL — CAPACITY-CONSTRAINED REVIEW QUEUE EVALUATION")
    print("=" * 80)

    # 1. Load canonical test split
    print("\nLoading canonical data and test split...")
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    accounts = pd.read_parquet(DATA_DIR / "accounts.parquet")
    labels = pd.read_parquet(DATA_DIR / "labels.parquet")
    split_info = json.load(open(DATA_DIR / "split_info.json"))

    splits = build_temporal_splits(events, accounts, labels, split_info)
    test_split = splits["test"]
    idx = list(test_split["labels"].index)
    y_true_str = test_split["labels"]["label_str"].values
    y_true_ac = (y_true_str == "abusive_coordinated")
    n_ac_total = int(y_true_ac.sum())  # 198
    as_of_ts = split_info["test_end_ts"]

    s_te = test_split["struct"].reindex(idx).fillna(0.0)
    b_te = test_split["behav"].reindex(idx).fillna(0.0)
    obs_days = b_te["account_age_days"].values
    n_orders = b_te["n_orders"].astype(int).values
    mean_amounts = b_te["mean_order_amount"].values
    total_amounts = n_orders * mean_amounts

    # Extract exact chronological first order timestamp in test window per account
    test_events = events[
        (events["timestamp"] > split_info["val_end_ts"]) &
        (events["timestamp"] <= split_info["test_end_ts"]) &
        (events["event_type"] == "order_placed")
    ]
    first_order_ts = test_events.groupby("account_id")["timestamp"].min().to_dict()

    # 2. Run DecisionEngine routing
    fused_model = joblib.load(MODELS_DIR / "fused_calibrated.pkl")
    engine = DecisionEngine(kl_conflict_threshold=0.50)
    p_struct, p_behav, p_fused, _ = fused_model.predict_proba_sub(s_te, b_te)

    decisions = engine.decide_batch(
        account_ids=idx,
        p_fused_matrix=p_fused,
        p_struct_matrix=p_struct,
        p_behav_matrix=p_behav,
        observation_days=obs_days,
        n_orders_arr=n_orders,
        as_of_ts=as_of_ts
    )

    # 3. Extract items routed to REVIEW
    review_queue_items = []
    auto_act_tp = 0
    auto_act_fp = 0

    for i, d in enumerate(decisions):
        if d.decision == Decision.ACT:
            if y_true_ac[i]:
                auto_act_tp += 1
            else:
                auto_act_fp += 1
        elif d.decision == Decision.REVIEW:
            acc = d.account_id
            item = QueueItem(
                account_id=acc,
                p_abusive=float(d.p_abusive),
                p_benign_coord=float(d.p_benign_coord),
                p_benign_indep=float(d.p_benign_indep),
                p_struct_ac=float(d.structural_sub_score),
                p_behav_ac=float(d.behavioral_sub_score),
                sym_kl_divergence=float(d.sym_kl_divergence),
                n_orders=int(n_orders[i]),
                total_order_amount=float(total_amounts[i]),
                flag_timestamp=float(first_order_ts.get(acc, as_of_ts)),
                true_label=str(y_true_str[i])
            )
            review_queue_items.append(item)

    n_review_total = len(review_queue_items)  # 779
    ac_in_review_total = sum(1 for it in review_queue_items if it.is_true_ac)  # 124

    print(f"\nDecision Engine Test Split Routing:")
    print(f"  Auto-ACT Decisions:         {auto_act_tp + auto_act_fp} (TP={auto_act_tp}, FP={auto_act_fp})")
    print(f"  Cases Routed to REVIEW:     {n_review_total} accounts")
    print(f"  True AC in REVIEW:          {ac_in_review_total} accounts")

    # 4. sym_KL Diagnostics & Ablation Analysis
    sym_kl_vals = np.array([it.sym_kl_divergence for it in review_queue_items])
    kl_factors = 1.0 + np.log1p(sym_kl_vals)
    ac_mask = np.array([it.is_true_ac for it in review_queue_items])

    # Rank shift analysis between Exposure-Weighted (no KL) and Conflict-Aware (with KL)
    scores_no_kl = np.array([it.p_abusive * np.sqrt(max(1.0, it.total_order_amount)) for it in review_queue_items])
    scores_with_kl = np.array([it.p_abusive * (1.0 + np.log1p(it.sym_kl_divergence)) * np.sqrt(max(1.0, it.total_order_amount)) for it in review_queue_items])

    # Ranks (1 = highest score)
    ranks_no_kl = pd.Series(scores_no_kl).rank(ascending=False, method="min").values
    ranks_with_kl = pd.Series(scores_with_kl).rank(ascending=False, method="min").values
    rank_shifts = ranks_no_kl - ranks_with_kl  # positive = promoted by KL

    spearman_corr = float(pd.Series(scores_no_kl).corr(pd.Series(scores_with_kl), method="spearman"))

    sym_kl_diagnostics = {
        "sym_kl_distribution": {
            "min": round(float(np.min(sym_kl_vals)), 4),
            "p25": round(float(np.percentile(sym_kl_vals, 25)), 4),
            "median": round(float(np.median(sym_kl_vals)), 4),
            "p75": round(float(np.percentile(sym_kl_vals, 75)), 4),
            "max": round(float(np.max(sym_kl_vals)), 4),
            "mean": round(float(np.mean(sym_kl_vals)), 4),
            "std": round(float(np.std(sym_kl_vals)), 4)
        },
        "kl_multiplier_factor_distribution": {
            "formula": "1 + log(1 + sym_KL)",
            "min": round(float(np.min(kl_factors)), 4),
            "mean": round(float(np.mean(kl_factors)), 4),
            "max": round(float(np.max(kl_factors)), 4),
            "true_ac_mean_factor": round(float(np.mean(kl_factors[ac_mask])), 4),
            "benign_mean_factor": round(float(np.mean(kl_factors[~ac_mask])), 4),
            "selective_ratio": round(float(np.mean(kl_factors[ac_mask]) / np.mean(kl_factors[~ac_mask])), 4)
        },
        "rank_shift_ablation_vs_no_kl": {
            "mean_absolute_rank_shift": round(float(np.mean(np.abs(rank_shifts))), 2),
            "max_rank_promotion": int(np.max(rank_shifts)),
            "max_rank_demotion": int(np.min(rank_shifts)),
            "spearman_rank_correlation": round(spearman_corr, 4)
        }
    }

    # 5. Evaluate Capacity Sweep across all 7 strategies
    capacity_limits = [25, 50, 100, 200, 300, 500, n_review_total]
    strategies = [
        TriageStrategy.FIFO,
        TriageStrategy.RANDOM_SHUFFLE,
        TriageStrategy.TIME_OF_FLAGGING,
        TriageStrategy.SCORE_DESC,
        TriageStrategy.VAR_FINANCIAL,
        TriageStrategy.EXPOSURE_WEIGHTED,
        TriageStrategy.CONFLICT_AWARE
    ]

    sweep_results = {strat.value: [] for strat in strategies}

    for strat in strategies:
        for k in capacity_limits:
            res = ReviewQueueEngine.evaluate_capacity_limit(
                items=review_queue_items,
                capacity_limit=k,
                strategy=strat,
                auto_act_tp=auto_act_tp,
                total_true_ac=n_ac_total,
                c_review_unit=150.0
            )
            sweep_results[strat.value].append(res)

    # 6. Compile payload
    payload = {
        "title": "AbuseRing Sentinel — Capacity-Constrained Review Queue Triage Evaluation",
        "description": "Simulation of human reviewer queue management under fixed daily capacity limits (K cases/day).",
        "test_population_stats": {
            "n_test_accounts": len(idx),
            "n_true_ac": n_ac_total,
            "auto_act_direct_tp": auto_act_tp,
            "auto_act_direct_fp": auto_act_fp,
            "review_queue_size": n_review_total,
            "true_ac_in_review": ac_in_review_total,
            "random_expected_ac_per_100_reviews": round(100 * (ac_in_review_total / n_review_total), 2)
        },
        "capacity_limits_evaluated": capacity_limits,
        "strategies_evaluated": [s.value for s in strategies],
        "sym_kl_diagnostics": sym_kl_diagnostics,
        "sweep_results": sweep_results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "capacity_constrained_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # 7. Display Executive Summary Table
    print("\n" + "=" * 125)
    print("CAPACITY-CONSTRAINED TRIAGE EFFICIENCY SUMMARY (RETAINED EFFECTIVE RECALL)")
    print("=" * 125)
    header = f"{'Capacity Limit (K)':<20} | {'FIFO (Arrival)':<15} | {'Random (S=42)':<15} | {'Time-Flagging':<15} | {'Score-Desc':<12} | {'Exp (No KL)':<12} | {'Conflict-Aware'}"
    print(header)
    print("-" * 125)
    
    for i, k in enumerate(capacity_limits):
        k_label = f"K = {k} reviews/day" if k < n_review_total else f"K = {k} (Full Queue)"
        fifo_rec = sweep_results["fifo"][i]["retained_effective_recall"] * 100
        rand_rec = sweep_results["random_shuffle"][i]["retained_effective_recall"] * 100
        time_rec = sweep_results["time_of_flagging"][i]["retained_effective_recall"] * 100
        score_rec = sweep_results["score_desc"][i]["retained_effective_recall"] * 100
        exp_rec = sweep_results["exposure_weighted"][i]["retained_effective_recall"] * 100
        conf_rec = sweep_results["conflict_aware"][i]["retained_effective_recall"] * 100
        print(f"{k_label:<20} | {fifo_rec:>13.2f}% | {rand_rec:>13.2f}% | {time_rec:>13.2f}% | {score_rec:>10.2f}% | {exp_rec:>10.2f}% | {conf_rec:>14.2f}%")

    print(f"\nArtifact saved to {out_file}")

if __name__ == "__main__":
    run_capacity_evaluation()
