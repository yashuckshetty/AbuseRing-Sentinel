"""
AbuseRing Sentinel — Adversarial Evasion & Adaptive Attacker Stress Test
========================================================================
Simulates active fraudster evasion strategies on the canonical Test Split
(Days 73-90, N=3,467 accounts, True AC=198) without retraining models.

Evaluates 4 distinct adversarial adaptation regimes:
  1. Anti-Burst Staggering (Temporal Evasion)
  2. Device/IP Hopping (Footprint Dilution / Graph Evaporation)
  3. Benign Camouflage (Promo/Return Masking)
  4. Combined Multi-Vector Adaptive Evasion (Full Attack)

Outputs:
  evals/results/adversarial_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES
from decision.decision_engine import DecisionEngine, Decision, RoutingLane

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("evals/results")

def run_adversarial_evaluation():
    print("=" * 80)
    print("ABUSERING SENTINEL — ADVERSARIAL EVASION & ATTACKER STRESS TEST")
    print("=" * 80)

    # 1. Load canonical test split
    print("\nLoading canonical data and building temporal splits...")
    events = pd.read_parquet(DATA_DIR / "events.parquet")
    accounts = pd.read_parquet(DATA_DIR / "accounts.parquet")
    labels = pd.read_parquet(DATA_DIR / "labels.parquet")
    rings = pd.read_parquet(DATA_DIR / "rings.parquet")
    split_info = json.load(open(DATA_DIR / "split_info.json"))

    splits = build_temporal_splits(events, accounts, labels, split_info)
    test_split = splits["test"]
    idx = list(test_split["labels"].index)
    y_true_str = test_split["labels"]["label_str"].values
    y_true_ac = (y_true_str == "abusive_coordinated")
    n_ac_total = int(y_true_ac.sum())
    n_total = len(idx)

    s_orig = test_split["struct"].reindex(idx).fillna(0.0).copy()
    b_orig = test_split["behav"].reindex(idx).fillna(0.0).copy()
    obs_days = b_orig["account_age_days"].values
    as_of_ts = split_info["test_end_ts"]

    # 2. Load trained models & DecisionEngine
    fused_model = joblib.load(MODELS_DIR / "fused_calibrated.pkl")
    engine = DecisionEngine(kl_conflict_threshold=0.50)

    # Helper function to evaluate decision engine on a feature variant
    def evaluate_scenario(s_df, b_df, scenario_name, description):
        p_struct, p_behav, p_fused, _ = fused_model.predict_proba_sub(s_df, b_df)
        n_orders_arr = b_df["n_orders"].astype(int).values

        decisions = engine.decide_batch(
            account_ids=idx,
            p_fused_matrix=p_fused,
            p_struct_matrix=p_struct,
            p_behav_matrix=p_behav,
            observation_days=obs_days,
            n_orders_arr=n_orders_arr,
            as_of_ts=as_of_ts
        )

        act_mask = np.array([d.decision == Decision.ACT for d in decisions])
        rev_mask = np.array([d.decision == Decision.REVIEW for d in decisions])
        wait_mask = np.array([d.decision == Decision.WAIT_MONITOR for d in decisions])
        abs_mask = np.array([d.decision == Decision.ABSTAIN for d in decisions])

        tp_act = int((act_mask & y_true_ac).sum())
        fp_act = int((act_mask & ~y_true_ac).sum())
        tp_rev = int((rev_mask & y_true_ac).sum())
        fp_rev = int((rev_mask & ~y_true_ac).sum())
        fn_wait = int((wait_mask & y_true_ac).sum())
        fn_abs = int((abs_mask & y_true_ac).sum())

        direct_prec = float(tp_act / act_mask.sum()) if act_mask.sum() > 0 else None
        direct_rec = float(tp_act / n_ac_total)
        effective_rec = float((tp_act + tp_rev) / n_ac_total)
        auto_act_fp_rate = float(fp_act / act_mask.sum()) if act_mask.sum() > 0 else 0.0

        # Standalone predictions on AC
        s_pred_ac = (np.argmax(p_struct, axis=1) == 2)
        b_pred_ac = (np.argmax(p_behav, axis=1) == 2)
        f_pred_ac = (np.argmax(p_fused, axis=1) == 2)

        s_rec_ac = float((s_pred_ac & y_true_ac).sum() / n_ac_total)
        b_rec_ac = float((b_pred_ac & y_true_ac).sum() / n_ac_total)
        f_rec_ac = float((f_pred_ac & y_true_ac).sum() / n_ac_total)

        kl_ac = [decisions[i].sym_kl_divergence for i in range(n_total) if y_true_ac[i]]
        mean_kl_ac = float(np.mean(kl_ac)) if kl_ac else 0.0
        median_kl_ac = float(np.median(kl_ac)) if kl_ac else 0.0

        return {
            "name": scenario_name,
            "description": description,
            "standalone_recall_ac": {
                "structural_lgbm": round(s_rec_ac, 4),
                "behavioral_lgbm": round(b_rec_ac, 4),
                "fused_calibrated": round(f_rec_ac, 4)
            },
            "decision_engine": {
                "direct_auto_act_recall": round(direct_rec, 4),
                "direct_auto_act_precision": round(direct_prec, 4) if direct_prec is not None else None,
                "auto_act_false_positives": fp_act,
                "auto_act_false_positive_rate": round(auto_act_fp_rate, 4),
                "effective_recall": round(effective_rec, 4),
                "counts": {
                    "ACT": int(act_mask.sum()),
                    "REVIEW": int(rev_mask.sum()),
                    "WAIT_MONITOR": int(wait_mask.sum()),
                    "ABSTAIN": int(abs_mask.sum())
                },
                "ac_breakdown": {
                    "AC_in_ACT": tp_act,
                    "AC_in_REVIEW": tp_rev,
                    "AC_in_WAIT_escaped": fn_wait,
                    "AC_in_ABSTAIN_gated": fn_abs
                },
                "mean_sym_kl_on_ac": round(mean_kl_ac, 4),
                "median_sym_kl_on_ac": round(median_kl_ac, 4)
            }
        }

    scenarios = {}

    # ── Regime 0: Baseline (No Evasion) ──────────────────────────────────────
    print("\n[0/4] Evaluating Baseline Canonical Test Split...")
    scenarios["baseline"] = evaluate_scenario(
        s_orig, b_orig,
        "Baseline (Unperturbed Canonical Test)",
        "Canonical test split (Days 73-90) with naturally formed rings without intentional evasion."
    )

    # ── Regime 1: Anti-Burst Order Staggering (Temporal Evasion) ───────────────
    print("\n[1/4] Evaluating Strategy 1: Anti-Burst Order Staggering...")
    s_s1 = s_orig.copy()
    b_s1 = b_orig.copy()
    # For AC accounts, clamp burst_score to 1, expand order_days_active, lower mean_daily_orders
    ac_mask = y_true_ac
    b_s1.loc[ac_mask, "burst_score"] = 1.0
    b_s1.loc[ac_mask, "mean_daily_orders"] = 1.0
    b_s1.loc[ac_mask, "order_days_active"] = b_s1.loc[ac_mask, "n_orders"]
    scenarios["strategy_1_anti_burst"] = evaluate_scenario(
        s_s1, b_s1,
        "Strategy 1: Anti-Burst Order Staggering (Temporal Evasion)",
        "Fraudsters intentionally inject delays between orders, flattening velocity bursts to 1 order/day."
    )

    # ── Regime 2: Device/IP Hopping (Footprint Dilution) ───────────────────────
    print("\n[2/4] Evaluating Strategy 2: Device/IP Hopping (Footprint Dilution)...")
    s_s2 = s_orig.copy()
    b_s2 = b_orig.copy()
    # For AC accounts, zero out shared device and shared IP links (only shared payout remains)
    s_s2.loc[ac_mask, "shared_device_degree"] = 0.0
    s_s2.loc[ac_mask, "shared_ip_degree"] = 0.0
    s_s2.loc[ac_mask, "multi_signal_edges"] = 0.0
    s_s2.loc[ac_mask, "degree"] = s_s2.loc[ac_mask, "shared_payout_degree"] + s_s2.loc[ac_mask, "shared_instrument_degree"] + s_s2.loc[ac_mask, "referral_degree"]
    scenarios["strategy_2_device_ip_hopping"] = evaluate_scenario(
        s_s2, b_s2,
        "Strategy 2: Device & IP Hopping (Footprint Dilution)",
        "Fraudsters randomize devices and route via proxy IPs, leaving only payout/instrument co-sharing."
    )

    # ── Regime 3: Benign Camouflage (Promo/Return Masking) ────────────────────
    print("\n[3/4] Evaluating Strategy 3: Benign Camouflage (Promo/Return Masking)...")
    s_s3 = s_orig.copy()
    b_s3 = b_orig.copy()
    # For AC accounts, dilute promo voucher usage and return rates to mimic standard shoppers
    b_s3.loc[ac_mask, "promo_rate"] = 0.15
    b_s3.loc[ac_mask, "has_promo"] = 0.0
    b_s3.loc[ac_mask, "return_rate"] = 0.05
    b_s3.loc[ac_mask, "n_returns"] = 0.0
    scenarios["strategy_3_benign_camouflage"] = evaluate_scenario(
        s_s3, b_s3,
        "Strategy 3: Benign Camouflage (Promo & Return Masking)",
        "Fraudsters place standard orders without coupons or returns to blend into normal consumer cadences."
    )

    # ── Regime 4: Full Multi-Vector Adaptive Evasion (Combined Attacks) ────────
    print("\n[4/4] Evaluating Strategy 4: Full Multi-Vector Adaptive Evasion...")
    s_s4 = s_s2.copy()  # Has device/IP hopping
    b_s4 = b_orig.copy()
    b_s4.loc[ac_mask, "burst_score"] = 1.0
    b_s4.loc[ac_mask, "mean_daily_orders"] = 1.0
    b_s4.loc[ac_mask, "order_days_active"] = b_s4.loc[ac_mask, "n_orders"]
    b_s4.loc[ac_mask, "promo_rate"] = 0.15
    b_s4.loc[ac_mask, "has_promo"] = 0.0
    b_s4.loc[ac_mask, "return_rate"] = 0.05
    b_s4.loc[ac_mask, "n_returns"] = 0.0
    scenarios["strategy_4_combined_adaptive"] = evaluate_scenario(
        s_s4, b_s4,
        "Strategy 4: Full Multi-Vector Adaptive Evasion (Combined)",
        "Sophisticated attacker simultaneously staggering timing, rotating device/IPs, and diluting promos."
    )

    # Compile and save
    payload = {
        "title": "AbuseRing Sentinel — Adversarial Evasion & Attacker Stress Test",
        "description": "Evaluation of existing pre-trained models and DecisionEngine against 4 adversarial evasion strategies without retraining.",
        "n_test_accounts": n_total,
        "n_true_ac": n_ac_total,
        "scenarios": scenarios
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "adversarial_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print("ADVERSARIAL EVASION EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Regime':<35} | {'Auto-ACT Rec':<12} | {'Eff. Recall':<12} | {'Auto-ACT FP':<12} | {'AC in REVIEW':<12} | {'AC Escaped'}")
    print("-" * 105)
    for k, v in scenarios.items():
        de = v["decision_engine"]
        print(f"{v['name'][:35]:<35} | {de['direct_auto_act_recall']*100:>10.2f}% | {de['effective_recall']*100:>10.2f}% | {de['auto_act_false_positives']:>12d} | {de['ac_breakdown']['AC_in_REVIEW']:>12d} | {de['ac_breakdown']['AC_in_WAIT_escaped']:>10d}")

    print(f"\nArtifact saved to {out_file}")

if __name__ == "__main__":
    run_adversarial_evaluation()
