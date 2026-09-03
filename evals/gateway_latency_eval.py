#!/usr/bin/env python3
"""
AbuseRing Sentinel — Gateway Latency Benchmark Evaluation Script
Evaluates synchronous in-line and asynchronous near-line gateway execution latency
using the production GatewayEventAdapter over test split accounts and features.
"""

import os
import sys
import json
import joblib
import time
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from decision.decision_engine import DecisionEngine
from features.feature_pipeline import build_temporal_splits
from gateway.adapter import GatewayEventAdapter, GatewayPaymentEvent, GatewayEventType


def run_gateway_latency_eval(n_iterations: int = 100) -> dict:
    print("=" * 70)
    print(" AbuseRing Sentinel — Gateway Dual-Path Latency Benchmark")
    print("=" * 70)

    # 1. Load trained models
    print("[1/4] Loading models and decision engine...")
    behav_model = joblib.load(BASE_DIR / "models" / "behavioral_lgbm.pkl")
    struct_model = joblib.load(BASE_DIR / "models" / "structural_lgbm.pkl")
    fused_model = joblib.load(BASE_DIR / "models" / "fused_calibrated.pkl")

    engine = DecisionEngine(kl_conflict_threshold=0.50)

    adapter = GatewayEventAdapter(
        decision_engine=engine,
        behavioral_model=behav_model,
        structural_model=struct_model,
        fused_model=fused_model
    )

    # 2. Load dataset and build test features
    print("[2/4] Loading dataset and constructing test feature matrices...")
    events = pd.read_parquet(BASE_DIR / "data" / "events.parquet")
    accounts = pd.read_parquet(BASE_DIR / "data" / "accounts.parquet")
    labels = pd.read_parquet(BASE_DIR / "data" / "labels.parquet")
    with open(BASE_DIR / "data" / "split_info.json", "r") as f:
        split_info = json.load(f)

    splits = build_temporal_splits(events, accounts, labels, split_info)
    test_sp = splits["test"]
    idx = list(test_sp["labels"].index)
    s_te = test_sp["struct"].reindex(idx).fillna(0)
    b_te = test_sp["behav"].reindex(idx).fillna(0)

    # 3. Prepare test events
    print(f"[3/4] Preparing sample gateway payment events across {len(idx)} test accounts...")
    test_events: List[Tuple[GatewayPaymentEvent, pd.Series, pd.Series]] = []
    
    # Sample up to 50 diverse accounts from test split
    sample_size = min(50, len(idx))
    sample_indices = np.linspace(0, len(idx) - 1, sample_size, dtype=int)

    for i in sample_indices:
        acc_id = idx[i]
        s_row = s_te.iloc[i]
        b_row = b_te.iloc[i]
        
        # Synthesize realistic payment event payload from observation
        evt = GatewayPaymentEvent(
            event_id=f"evt_bench_{acc_id}",
            event_type=GatewayEventType.PAYMENT_AUTHORIZED,
            account_id=acc_id,
            amount_inr=float(b_row.get("amount_mean", 500.0) if pd.notnull(b_row.get("amount_mean")) else 500.0),
            currency="INR",
            timestamp=int(split_info["test_end_ts"]),
            ip_address="192.168.1.100",
            device_id=f"DEV_{acc_id[:8]}"
        )
        test_events.append((evt, s_row, b_row))

    # 4. Execute Benchmark
    print(f"[4/4] Benchmarking dual-path execution over {n_iterations} iterations...")
    bench_results = adapter.benchmark_dual_path(test_events, n_iterations=n_iterations)

    qualifier = (
        "Prototype design-target measured in a local single-machine mock environment "
        "(in-memory adapter processing synthetic test data, not live distributed gateway "
        "traffic or remote database latency). Values are machine-dependent."
    )

    output_payload = {
        "qualifier": qualifier,
        "n_iterations": n_iterations,
        "sync_path": {
            "design_budget_ms": 30.0,
            "design_target_ms": "< 30.0 ms",
            "p50_ms": bench_results["sync_path"]["measured_p50_ms"],
            "p95_ms": bench_results["sync_path"]["measured_p95_ms"],
            "p99_ms": bench_results["sync_path"]["measured_p99_ms"],
            "mean_ms": bench_results["sync_path"]["measured_mean_ms"],
            "meets_design_target": bench_results["sync_path"]["meets_design_target"]
        },
        "async_path": {
            "design_budget_ms": 500.0,
            "design_target_ms": "< 500.0 ms",
            "p50_ms": bench_results["async_path"]["measured_p50_ms"],
            "p95_ms": bench_results["async_path"]["measured_p95_ms"],
            "p99_ms": bench_results["async_path"]["measured_p99_ms"],
            "mean_ms": bench_results["async_path"]["measured_mean_ms"],
            "meets_design_target": bench_results["async_path"]["meets_design_target"]
        }
    }

    out_path = BASE_DIR / "evals" / "results" / "gateway_latency_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Results written to: {out_path}")
    print(f"Sync  p50: {output_payload['sync_path']['p50_ms']} ms | p99: {output_payload['sync_path']['p99_ms']} ms")
    print(f"Async p50: {output_payload['async_path']['p50_ms']} ms | p99: {output_payload['async_path']['p99_ms']} ms")
    return output_payload


if __name__ == "__main__":
    run_gateway_latency_eval(n_iterations=100)
