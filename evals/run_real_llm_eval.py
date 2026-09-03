"""
AbuseRing Sentinel - Real LLM Verification Experiment
Evaluates the 6 curated benchmark accounts using live Google Gemini API calls vs. deterministic mock mode.
Validates boundary constraints, records real latency, and saves individual artifacts to ai/sample_outputs/real_llm/.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib

from features.feature_pipeline import build_temporal_splits
from decision.decision_engine import DecisionEngine
from ai.evidence_reasoner import EvidenceGapReasoner, validate_llm_output, _mock_response

SAMPLE_ACCOUNTS = [
    {"account_id": "ACC_03653", "name": "Hard BC", "lane": "fused_auto", "decision": "WAIT_MONITOR"},
    {"account_id": "ACC_04870", "name": "Referral Farming", "lane": "conflict_review", "decision": "REVIEW"},
    {"account_id": "ACC_04430", "name": "Sleeper Account", "lane": "conflict_review", "decision": "REVIEW"},
    {"account_id": "ACC_04295", "name": "Promo Abuse", "lane": "fused_auto", "decision": "ACT"},
    {"account_id": "ACC_00505", "name": "Benign Independent", "lane": "fused_auto", "decision": "WAIT_MONITOR"},
    {"account_id": "ACC_04987", "name": "Cold-Start", "lane": "abstain", "decision": "ABSTAIN"},
]

def run_experiment():
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set!")
        return

    print("=" * 80)
    print("ABUSERING SENTINEL — REAL LLM VERIFICATION EXPERIMENT")
    print("=" * 80)

    # 1. Load data and models
    print("Loading test split and trained models...")
    events = pd.read_parquet("data/events.parquet")
    accounts = pd.read_parquet("data/accounts.parquet")
    labels = pd.read_parquet("data/labels.parquet")
    with open("data/split_info.json") as f:
        split_info = json.load(f)

    splits = build_temporal_splits(events, accounts, labels, split_info)
    test_sp = splits["test"]
    s_te = test_sp["struct"]
    b_te = test_sp["behav"]
    labels_te = test_sp.get("labels")

    fused = joblib.load("models/fused_calibrated.pkl")
    engine = DecisionEngine(kl_conflict_threshold=0.50)

    reasoner_real = EvidenceGapReasoner(api_key=api_key, mock=False)
    reasoner_mock = EvidenceGapReasoner(mock=True)

    out_dir = Path("ai/sample_outputs/real_llm")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    print(f"\nExecuting Live Gemini API calls across {len(SAMPLE_ACCOUNTS)} accounts...")
    print("-" * 80)

    for item in SAMPLE_ACCOUNTS:
        acc_id = item["account_id"]
        s_row = s_te.loc[acc_id]
        b_row = b_te.loc[acc_id]
        obs_days = float(b_row.get("account_age_days", 0))
        n_orders = int(b_row.get("n_orders", 0))
        as_of_ts = split_info["test_end_ts"]

        p_s, p_b, p_f, _ = fused.predict_proba_sub(s_te.loc[[acc_id]], b_te.loc[[acc_id]])
        p_s = p_s[0]
        p_b = p_b[0]
        p_f = p_f[0]

        dec_res = engine.decide(
            account_id=acc_id,
            p_fused=p_f,
            p_struct=p_s,
            p_behav=p_b,
            observation_days=obs_days,
            n_orders=n_orders,
            as_of_ts=as_of_ts,
        )

        conflict_flag = bool(dec_res.evidence_conflict)

        # Mock call
        t0_mock = time.perf_counter()
        mock_analysis = reasoner_mock.analyze(
            account_id=acc_id,
            struct_feats=s_row,
            behav_feats=b_row,
            p_fused=list(p_f),
            p_struct=list(p_s),
            p_behav=list(p_b),
            conflict_flag=conflict_flag,
            as_of_ts=as_of_ts,
        )
        mock_latency = round(time.perf_counter() - t0_mock, 4)

        # Real LLM call
        t0_real = time.perf_counter()
        real_analysis = reasoner_real.analyze(
            account_id=acc_id,
            struct_feats=s_row,
            behav_feats=b_row,
            p_fused=list(p_f),
            p_struct=list(p_s),
            p_behav=list(p_b),
            conflict_flag=conflict_flag,
            as_of_ts=as_of_ts,
        )
        real_latency = round(time.perf_counter() - t0_real, 4)

        payload = real_analysis["payload"]
        real_output = real_analysis["llm_output"]
        mock_output = mock_analysis["llm_output"]
        is_valid = real_analysis["boundary_valid"]
        violations = real_analysis["boundary_violations"]

        print(f"[{item['name']}] {acc_id}: Decision={dec_res.decision.value}, Lane={dec_res.routing_lane.value}")
        print(f"  Real Latency: {real_latency:.2f}s | Mock Latency: {mock_latency * 1000:.2f}ms | Boundary Valid: {is_valid}")
        if violations:
            print(f"  VIOLATIONS: {violations}")

        record = {
            "account_id": acc_id,
            "category": item["name"],
            "decision": dec_res.decision.value,
            "routing_lane": dec_res.routing_lane.value,
            "sym_kl_divergence": dec_res.sym_kl_divergence,
            "evidence_conflict": conflict_flag,
            "probabilities": {
                "p_abusive_structural": round(float(p_s[2]), 4),
                "p_abusive_behavioral": round(float(p_b[2]), 4),
                "p_abusive_fused": round(float(p_f[2]), 4),
                "p_benign_coord_fused": round(float(p_f[1]), 4),
                "p_benign_indep_fused": round(float(p_f[0]), 4),
            },
            "input_payload": payload,
            "real_llm_response": real_output,
            "mock_response": mock_output,
            "validation": {
                "boundary_passed": is_valid,
                "violations": violations,
            },
            "latency": {
                "real_llm_seconds": real_latency,
                "mock_seconds": mock_latency,
            },
        }

        # Save artifact file
        out_file = out_dir / f"{acc_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)

        results.append(record)

    summary_file = out_dir / "experiment_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE — SUMMARY SAVED TO ai/sample_outputs/real_llm/")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
