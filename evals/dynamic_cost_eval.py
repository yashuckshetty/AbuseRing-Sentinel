"""
AbuseRing Sentinel — Dynamic Compounding Loss & Break-Even Lag Evaluation
==========================================================================
Evaluates the economic impact of time-to-detection on total platform loss,
comparing Behavioral-Only (Rung 3) against the KL-Routing Engine across
various detection lag horizons and illustrative compounding exposure rates.

Outputs:
  evals/results/dynamic_cost_results.json
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision.cost_model import DynamicCostConfig

RESULTS_DIR = Path("evals/results")

def run_dynamic_cost_evaluation():
    print("=" * 80)
    print("ABUSERING SENTINEL — DYNAMIC COMPOUNDING LOSS EVALUATION (SYMMETRIC)")
    print("=" * 80)

    config = DynamicCostConfig()
    print(f"\nBaseline Cost Parameters (Simulated):")
    print(f"  C_FP:                     Rs {config.c_false_positive:.2f}")
    print(f"  C_FN (Base):              Rs {config.c_false_negative_base:.2f}")
    print(f"  C_REVIEW:                 Rs {config.c_review:.2f}")
    print(f"  C_WAIT (per day):         Rs {config.c_wait_per_day:.2f}")
    print(f"  Alpha (compounding/day):  Rs {config.alpha_compounding_per_day:.2f} (Illustrative)")
    print(f"  Gamma (acceleration):     {config.gamma_acceleration:.2f} (Illustrative)")

    # 1. Population & Split Statistics on Test Split (N=3,467, True AC=198)
    # Behavioral-Only (Rung 3):
    #   FP = 25 accounts (25 * Rs 500 = Rs 12,500)
    #   FN = 9 missed AC accounts (accrue lag t unmitigated)
    #   Review = 0
    behav_fp = 25
    behav_fn = 9

    # KL-Routing Engine (Symmetric Dynamic Evaluation):
    #   Auto-ACT: 38 AC accounts intercepted at t=0 (Loss = Rs 0)
    #   Review Queue: 779 total reviews (779 * Rs 150 = Rs 116,850 review labor)
    #     - 124 True AC accounts in review
    #   ABSTAIN Gate: 36 True AC accounts (1-order cold-start accounts)
    #     - Observed evaluation window lag: t_abstain = 18.0 days
    #     - Monitoring holding cost: 36 * 18 * Rs 50 = Rs 32,400
    #     - Dynamic FN loss on 36 accounts at t=18.0 days: 36 * L(18.0, alpha)
    #   Escaped FN: 0 accounts
    routing_reviews = 779
    routing_review_labor_cost = routing_reviews * config.c_review  # Rs 116,850
    routing_abstain_ac = 36
    t_abstain_lag = 18.0
    routing_monitoring_cost = routing_abstain_ac * t_abstain_lag * config.c_wait_per_day  # Rs 32,400
    
    # Review Queue Turnaround Latency (Illustrative assumption: 2.0 days average queue resolution)
    t_review_latency = 2.0
    ac_in_review = 124

    # 2. Lag Sensitivity Curves (Lag t from 0 to 180 days)
    lags = list(range(0, 181, 5))
    alphas = [25.0, 50.0, 100.0, 200.0, 500.0]

    lag_curve_data = []
    for t in lags:
        row = {"detection_lag_days": t}
        for alpha in alphas:
            cfg = DynamicCostConfig(alpha_compounding_per_day=alpha)
            # Behavioral-only cost at lag t
            behav_cost = cfg.evaluate_dynamic_cost(
                n_fp=behav_fp,
                fn_lag_days=[float(t) for _ in range(behav_fn)],
                n_review=0,
                n_wait_days=0
            )
            row[f"behav_cost_alpha_{int(alpha)}"] = round(behav_cost, 2)
            
            # Symmetric Routing Cost Variant 1: Immediate-Hold (t_review=0d, t_abstain=18d)
            abstain_fn_losses_v1 = [cfg.calculate_fn_loss(t_abstain_lag) for _ in range(routing_abstain_ac)]
            routing_cost_v1 = (
                routing_review_labor_cost +
                routing_monitoring_cost +
                sum(abstain_fn_losses_v1)
            )
            row[f"routing_cost_v1_hold_alpha_{int(alpha)}"] = round(routing_cost_v1, 2)

            # Symmetric Routing Cost Variant 2: Queue Latency Exposure (t_review=2d, t_abstain=18d)
            review_latency_drain = ac_in_review * (cfg.calculate_fn_loss(t_review_latency) - cfg.c_false_negative_base)
            routing_cost_v2 = routing_cost_v1 + review_latency_drain
            row[f"routing_cost_v2_latency_alpha_{int(alpha)}"] = round(routing_cost_v2, 2)

        lag_curve_data.append(row)

    # 3. Compute Symmetric Break-Even Lags
    break_even_results = {}
    for alpha in alphas:
        cfg = DynamicCostConfig(alpha_compounding_per_day=alpha)
        
        # Routing Symmetric Cost Variant 1 (Immediate Hold during review)
        abstain_fn_losses_v1 = [cfg.calculate_fn_loss(t_abstain_lag) for _ in range(routing_abstain_ac)]
        routing_cost_v1 = routing_review_labor_cost + routing_monitoring_cost + sum(abstain_fn_losses_v1)

        # Routing Symmetric Cost Variant 2 (2-day Review Queue Turnaround Exposure)
        review_latency_drain = ac_in_review * (cfg.calculate_fn_loss(t_review_latency) - cfg.c_false_negative_base)
        routing_cost_v2 = routing_cost_v1 + review_latency_drain

        # Search fine-grained lag from 0 to 500 days
        t_fine = np.linspace(0, 500, 5001)
        costs_behav = [
            cfg.evaluate_dynamic_cost(
                n_fp=behav_fp,
                fn_lag_days=[float(t_val) for _ in range(behav_fn)],
                n_review=0,
                n_wait_days=0
            )
            for t_val in t_fine
        ]
        
        # Break-even vs Variant 1 (Hold)
        idx_v1 = np.where(np.array(costs_behav) >= routing_cost_v1)[0]
        be_v1 = round(float(t_fine[idx_v1[0]]), 2) if len(idx_v1) > 0 else None
        
        # Break-even vs Variant 2 (Queue Latency)
        idx_v2 = np.where(np.array(costs_behav) >= routing_cost_v2)[0]
        be_v2 = round(float(t_fine[idx_v2[0]]), 2) if len(idx_v2) > 0 else None

        break_even_results[f"alpha_{int(alpha)}"] = {
            "alpha_compounding_per_day": alpha,
            "gamma_acceleration": cfg.gamma_acceleration,
            "routing_cost_v1_hold_rs": round(routing_cost_v1, 2),
            "routing_cost_v2_latency_rs": round(routing_cost_v2, 2),
            "break_even_lag_days_v1_hold": be_v1,
            "break_even_lag_days_v2_latency": be_v2,
            "single_missed_account_loss_at_break_even_rs": round(cfg.calculate_fn_loss(be_v1), 2) if be_v1 is not None else None
        }

    # Detailed component breakdown for standard alpha=100 (auditable reference)
    cfg_100 = DynamicCostConfig(alpha_compounding_per_day=100.0)
    abstain_loss_100 = sum(cfg_100.calculate_fn_loss(t_abstain_lag) for _ in range(routing_abstain_ac))
    latency_loss_100 = ac_in_review * (cfg_100.calculate_fn_loss(t_review_latency) - cfg_100.c_false_negative_base)
    
    component_breakdown_100 = {
        "alpha_compounding_per_day": 100.0,
        "gamma_acceleration": 1.2,
        "routing_engine": {
            "review_labor_cost_rs": round(routing_review_labor_cost, 2),
            "review_labor_formula": f"{routing_reviews} reviews * Rs {config.c_review}",
            "monitoring_holding_cost_rs": round(routing_monitoring_cost, 2),
            "monitoring_holding_formula": f"{routing_abstain_ac} accounts * {t_abstain_lag} days * Rs {config.c_wait_per_day}/day",
            "abstain_compounding_loss_rs": round(abstain_loss_100, 2),
            "abstain_compounding_formula": f"{routing_abstain_ac} accounts * L({t_abstain_lag}d, alpha=100)",
            "review_queue_latency_exposure_rs": round(latency_loss_100, 2),
            "review_queue_latency_formula": f"{ac_in_review} AC in review * (L({t_review_latency}d, alpha=100) - C_0)",
            "total_variant_1_immediate_hold_rs": round(routing_review_labor_cost + routing_monitoring_cost + abstain_loss_100, 2),
            "total_variant_2_latency_exposure_rs": round(routing_review_labor_cost + routing_monitoring_cost + abstain_loss_100 + latency_loss_100, 2)
        },
        "behavioral_only": {
            "fp_cost_rs": round(behav_fp * config.c_false_positive, 2),
            "fp_formula": f"{behav_fp} FPs * Rs {config.c_false_positive} (immediate, non-compounding)",
            "fn_base_cost_rs": round(behav_fn * config.c_false_negative_base, 2),
            "fn_base_formula": f"{behav_fn} FNs * Rs {config.c_false_negative_base} (at lag t=0)",
            "total_flat_baseline_rs": round(behav_fp * config.c_false_positive + behav_fn * config.c_false_negative_base, 2)
        }
    }

    results_payload = {
        "title": "AbuseRing Sentinel — Symmetric Dynamic Compounding Loss Model",
        "description": "Symmetric economic evaluation of detection lag and compounding exposure comparing Behavioral-Only vs. KL-Routing.",
        "cost_assumptions_simulated": {
            "c_false_positive_rs": config.c_false_positive,
            "c_false_negative_base_rs": config.c_false_negative_base,
            "c_review_rs": config.c_review,
            "c_wait_per_day_rs": config.c_wait_per_day,
            "gamma_acceleration": config.gamma_acceleration,
            "t_abstain_lag_days": t_abstain_lag,
            "t_review_queue_turnaround_days": t_review_latency,
            "note": "All compounding growth rates and queue turnarounds are explicitly illustrative assumptions for sensitivity analysis."
        },
        "static_cost_baseline_rs": {
            "behavioral_only_flat": round(config.evaluate_static_cost(behav_fp, behav_fn), 2),
            "routing_flat_without_compounding": 149250.0
        },
        "cost_breakdown_by_component_alpha_100": component_breakdown_100,
        "symmetric_break_even_analysis": break_even_results,
        "lag_sensitivity_curve": lag_curve_data
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "dynamic_cost_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print("\n" + "=" * 80)
    print("SYMMETRIC BREAK-EVEN DETECTION LAG ANALYSIS (BEHAVIORAL-ONLY vs. ROUTING)")
    print("=" * 80)
    print(f"{'Compounding Rate (Alpha)':<25} | {'Routing Cost (Hold)':<20} | {'Break-Even Lag (Hold)':<22} | {'Break-Even Lag (2d Latency)'}")
    print("-" * 95)
    for k, v in break_even_results.items():
        be_h_str = f"{v['break_even_lag_days_v1_hold']:>10.2f} days" if v['break_even_lag_days_v1_hold'] is not None else "  >500 days"
        be_l_str = f"{v['break_even_lag_days_v2_latency']:>10.2f} days" if v['break_even_lag_days_v2_latency'] is not None else "  >500 days"
        print(f"Rs {v['alpha_compounding_per_day']:>6.1f} / day (gamma={v['gamma_acceleration']}) | Rs {v['routing_cost_v1_hold_rs']:>12,.2f} | {be_h_str:<22} | {be_l_str}")

    print(f"\nArtifact saved to {out_file}")

if __name__ == "__main__":
    run_dynamic_cost_evaluation()
