"""
AbuseRing Sentinel — FastAPI Service
====================================
Minimal, read-only presentation layer over verified evaluation artifacts
and real-time DecisionEngine policy gating.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).parent.parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from features.feature_pipeline import (
    build_temporal_splits,
    STRUCTURAL_FEATURES,
    BEHAVIORAL_FEATURES,
)
from graph.temporal_graph import build_graph_as_of
from decision.decision_engine import DecisionEngine, Decision, RoutingLane
from ai.evidence_reasoner import EvidenceGapReasoner, sym_kl_divergence
from policy.policy_gate import PolicyGate
from models.fused_model import FusedCalibratedClassifier
from gateway.adapter import (
    GatewayEventAdapter,
    GatewayPaymentEvent,
    GatewayEventType,
    SyncAction,
)
from policy.temporal_escalation import (
    LongitudinalEscalationPolicy,
    TemporalRiskState,
)

app = FastAPI(
    title="AbuseRing Sentinel API",
    description="Evidence-Disagreement Routing & Temporal Evolving Risk Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for models and test split data
CACHE: Dict[str, Any] = {}

def load_artifacts():
    """Load and cache models, test split features, and lookup tables on startup."""
    print("Loading models and datasets...", flush=True)
    models_dir = BASE_DIR / "models"
    data_dir = BASE_DIR / "data"

    fused = joblib.load(models_dir / "fused_calibrated.pkl")
    behav = joblib.load(models_dir / "behavioral_lgbm.pkl")
    struct = joblib.load(models_dir / "structural_lgbm.pkl")

    engine = DecisionEngine(kl_conflict_threshold=0.5)
    reasoner = EvidenceGapReasoner(mock=True)
    gate = PolicyGate(decision_engine=engine, reasoner=reasoner, write_audit_log=False)

    events = pd.read_parquet(data_dir / "events.parquet")
    accounts = pd.read_parquet(data_dir / "accounts.parquet")
    labels = pd.read_parquet(data_dir / "labels.parquet")
    split = json.load(open(data_dir / "split_info.json"))

    splits = build_temporal_splits(events, accounts, labels, split)
    test_sp = splits["test"]
    test_labels = test_sp["labels"]
    idx = list(test_labels.index)

    s_te = test_sp["struct"].reindex(idx).fillna(0)
    b_te = test_sp["behav"].reindex(idx).fillna(0)

    # Precompute probability matrices for test split
    p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)

    # Curated sample accounts
    labels_full = labels.set_index("account_id")
    
    sample_accounts = [
        {
            "account_id": "ACC_03653",
            "category": "Hard BC (Benign Family + Injected Shared Payout)",
            "description": "Family group sharing a payout edge. Both structural and behavioral models independently and correctly classify this account as benign (P(struct)[AC]=0.00, P(behav)[AC]=0.00; P(fused)[Benign Coord]=0.9999). Because both models strongly AGREE (sym_KL=0.0005, well below the 0.50 conflict threshold), the system confidently resolves to WAIT_MONITOR without requiring human review. This serves as the clean agreement counterexample to the referral/sleeper conflict cases.",
            "expected_decision": "WAIT_MONITOR",
            "expected_lane": "fused_auto",
        },
        {
            "account_id": "ACC_04870",
            "category": "Referral Farming (Unseen Ring Topology)",
            "description": "True referral-farming ring member (REFARM_057). Never seen in train/val. Zero shared payout infrastructure (P(struct)[AC]=0.00), but high referral velocity (P(behav)[AC]=0.94). Extreme evidence DISAGREEMENT (sym_KL=10.22 >> 0.50) safely routes account to REVIEW for human adjudication rather than suppressing it as a false negative.",
            "expected_decision": "REVIEW",
            "expected_lane": "conflict_review",
        },
        {
            "account_id": "ACC_04430",
            "category": "Sleeper Account (Sparse Behavioral Evidence)",
            "description": "Pre-positioned sleeper account with mature structural connections (P(struct)[AC]=1.00) but sparse initial order velocity (P(behav)[AC]=0.63). Evidence DISAGREEMENT (sym_KL=1.55 >> 0.50) triggers conflict routing into REVIEW.",
            "expected_decision": "REVIEW",
            "expected_lane": "conflict_review",
        },
        {
            "account_id": "ACC_04295",
            "category": "Promo Abuse Ring Member (Concordant High Risk)",
            "description": "Coordinated promo ring member where both models strongly agree on abuse (P(struct)[AC]=1.00, P(behav)[AC]=1.00; sym_KL=0.0010). High confidence + near-zero conflict safely executes direct auto-ACT.",
            "expected_decision": "ACT",
            "expected_lane": "fused_auto",
        },
        {
            "account_id": "ACC_00505",
            "category": "Benign Independent (Standard Customer)",
            "description": "Standard independent customer with no shared entities and normal order cadence. Both models agree as benign (P(struct)[AC]=0.00, P(behav)[AC]=0.00; P(fused)[Benign Indep]=0.9997). Low divergence (sym_KL=0.28 < 0.50) safely routes to WAIT_MONITOR.",
            "expected_decision": "WAIT_MONITOR",
            "expected_lane": "fused_auto",
        },
        {
            "account_id": "ACC_04987",
            "category": "Cold-Start Account (Insufficient Orders)",
            "description": "New account with only 1 order (n_orders=1). Deterministic evidence gate enforces ABSTAIN regardless of model output (P(behav)[AC]=0.94), preventing premature automated enforcement.",
            "expected_decision": "ABSTAIN",
            "expected_lane": "abstain",
        },
    ]

    CACHE["fused"] = fused
    CACHE["behav"] = behav
    CACHE["struct"] = struct
    CACHE["engine"] = engine
    CACHE["reasoner"] = reasoner
    CACHE["gate"] = gate
    CACHE["test_sp"] = test_sp
    CACHE["test_labels"] = test_labels
    CACHE["idx"] = idx
    CACHE["idx_lookup"] = {acc: i for i, acc in enumerate(idx)}
    CACHE["s_te"] = s_te
    CACHE["b_te"] = b_te
    CACHE["p_struct"] = p_struct
    CACHE["p_behav"] = p_behav
    CACHE["p_fused"] = p_fused
    CACHE["conflicts"] = conflicts
    CACHE["split_info"] = split
    CACHE["labels_full"] = labels_full
    CACHE["sample_accounts"] = sample_accounts
    
    # Build and cache test temporal graph as-of test_end_ts
    G_test = build_graph_as_of(events, split["test_end_ts"])
    CACHE["graph_test"] = G_test

    # Instantiate Gateway Event Adapter (Dual-Path Bridge)
    adapter = GatewayEventAdapter(
        decision_engine=engine,
        behavioral_model=behav,
        structural_model=struct,
        fused_model=fused,
    )
    CACHE["gateway_adapter"] = adapter
    print("Artifacts loaded successfully (including test graph and gateway adapter).", flush=True)

@app.on_event("startup")
def startup_event():
    load_artifacts()


# ── 1. HEALTH ENDPOINT ────────────────────────────────────────────────────────
@app.get("/api/health")
def get_health():
    """Confirms all required artifact files and models load from disk."""
    if not CACHE:
        load_artifacts()

    required_files = [
        "models/behavioral_lgbm.pkl",
        "models/structural_lgbm.pkl",
        "models/fused_calibrated.pkl",
        "evals/metrics.json",
        "evals/results/trajectory_results.parquet",
        "evals/results/kl_ablation_results.json",
        "evals/results/prevalence_shift_results.json",
        "evals/results/multiseed_results.json",
        "evals/results/robustness_results.json",
        "evals/results/gnn_comparison_results.json",
        "evals/results/scenario_b_generalization_results.json",
        "evals/results/adversarial_results.json",
        "evals/results/dynamic_cost_results.json",
        "evals/results/capacity_constrained_results.json",
        "evals/results/ai_security_results.json",
        "evals/results/handcrafted_adversarial_results.json",
    ]
    missing = [f for f in required_files if not (BASE_DIR / f).exists()]
    
    return {
        "status": "healthy" if not missing else "degraded",
        "artifacts_loaded": len(missing) == 0 and "fused" in CACHE,
        "files_checked": len(required_files),
        "missing_files": missing,
        "test_accounts_loaded": len(CACHE.get("idx", [])),
    }


# ── 2. MODEL LADDER ───────────────────────────────────────────────────────────
@app.get("/api/model-ladder")
def get_model_ladder():
    """Returns the 5-rung evaluation metrics from evals/metrics.json for test split."""
    metrics_path = BASE_DIR / "evals" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="evals/metrics.json not found")
    with open(metrics_path, "r") as f:
        all_metrics = json.load(f)

    # Filter for test split
    test_metrics = [m for m in all_metrics if m.get("split") == "test"]
    
    # Add standardized notes & rung ordering
    rung_order = ["majority_class", "rule_based", "behavioral_lgbm", "structural_lgbm", "fused_calibrated"]
    ordered_ladder = []
    
    for i, rung_name in enumerate(rung_order, start=1):
        match = next((m for m in test_metrics if m["model"] == rung_name), None)
        if match:
            item = dict(match)
            item["rung_number"] = i
            if rung_name == "majority_class":
                item["notes"] = "Predicts BI for all accounts (trivial baseline)"
            elif rung_name == "rule_based":
                item["notes"] = "Shared-payout threshold heuristics"
            elif rung_name == "behavioral_lgbm":
                item["notes"] = "Single-seed 42 result (Multi-seed range: 27.0% - 95.5%)"
            elif rung_name == "structural_lgbm":
                item["notes"] = "Disagreement/safety detector (Precision: 78.3% - 100.0%)"
            elif rung_name == "fused_calibrated":
                item["notes"] = "Geometric-mean fusion (collapses when P(struct) ~ 0)"
            ordered_ladder.append(item)
            
    return {
        "split": "test",
        "n_test_accounts": 3467,
        "n_true_ac": 198,
        "ladder": ordered_ladder,
        "multi_seed_callout": (
            "Headline behavioral recall (95.45%) is a single-seed (Seed 42) result. "
            "Across seeds 42/43/44, behavioral recall spans 27.03% to 95.45% (mean: 58.63%), "
            "whereas structural metrics remain stable. Behavioral performance is sensitive to "
            "simulator-encoded order distributions (Limitation #3)."
        )
    }


# ── 3. LIVE DECISION & AUDIT TRAIL ───────────────────────────────────────────
@app.get("/api/decision/{account_id}")
def get_decision(account_id: str):
    """
    Runs an account live through the real DecisionEngine + EvidenceGapReasoner.
    Returns full decision, routing lane, sym_KL divergence, probabilities, and audit trail.
    """
    if "idx_lookup" not in CACHE:
        load_artifacts()

    idx_lookup = CACHE["idx_lookup"]
    if account_id not in idx_lookup:
        # Search if account exists in full dataset
        raise HTTPException(
            status_code=404,
            detail=f"Account '{account_id}' not active in evaluation test window (Days 73-90)."
        )

    i = idx_lookup[account_id]
    s_row = CACHE["s_te"].iloc[i]
    b_row = CACHE["b_te"].iloc[i]
    p_s = CACHE["p_struct"][i]
    p_b = CACHE["p_behav"][i]
    p_f = CACHE["p_fused"][i]
    lbl_str = CACHE["test_labels"].iloc[i]["label_str"]
    as_of_ts = CACHE["split_info"]["test_end_ts"]

    n_orders = int(b_row.get("n_orders", 0))
    obs_days = float(b_row.get("account_age_days", 0))

    # Real decision engine evaluation
    engine: DecisionEngine = CACHE["engine"]
    dec_res = engine.decide(
        account_id=account_id,
        p_fused=np.array(p_f),
        p_struct=np.array(p_s),
        p_behav=np.array(p_b),
        observation_days=obs_days,
        n_orders=n_orders,
        as_of_ts=as_of_ts,
    )

    # Consume canonical sym_KL divergence and conflict flag directly from DecisionEngine
    kl_val = round(float(dec_res.sym_kl_divergence), 4)
    conflict_flag = bool(dec_res.evidence_conflict)

    # AI Advisory reasoning
    reasoner: EvidenceGapReasoner = CACHE["reasoner"]
    ai_analysis = reasoner.analyze(
        account_id=account_id,
        struct_feats=s_row,
        behav_feats=b_row,
        p_fused=list(p_f),
        p_struct=list(p_s),
        p_behav=list(p_b),
        conflict_flag=conflict_flag,
        as_of_ts=as_of_ts,
    )

    return {
        "account_id": account_id,
        "true_label": lbl_str,
        "decision": dec_res.decision.value,
        "routing_lane": dec_res.routing_lane.value,
        "sym_kl_divergence": kl_val,
        "evidence_conflict": conflict_flag,
        "probabilities": {
            "p_abusive_behavioral": round(float(p_b[2]), 4),
            "p_abusive_structural": round(float(p_s[2]), 4),
            "p_abusive_fused": round(float(p_f[2]), 4),
            "p_benign_coord_fused": round(float(p_f[1]), 4),
            "p_benign_indep_fused": round(float(p_f[0]), 4),
        },
        "observation": {
            "n_orders": n_orders,
            "account_age_days": obs_days,
            "shared_payout_degree": int(s_row.get("shared_payout_degree", 0)),
            "shared_device_degree": int(s_row.get("shared_device_degree", 0)),
            "shared_ip_degree": int(s_row.get("shared_ip_degree", 0)),
            "referral_degree": int(s_row.get("referral_degree", 0)),
            "burst_score": int(b_row.get("burst_score", 0)),
            "promo_rate": round(float(b_row.get("promo_rate", 0)), 3),
            "return_rate": round(float(b_row.get("return_rate", 0)), 3),
        },
        "ai_advisory": {
            "mode": "mock_deterministic",
            "disclaimer": "Advisory only — mock/deterministic mode verified; does not affect the numeric decision.",
            "conflict_explanation": ai_analysis.get("llm_output", {}).get("conflict_explanation", ""),
            "key_signals": ai_analysis.get("llm_output", {}).get("key_signals", []),
            "analyst_suggestions": ai_analysis.get("llm_output", {}).get("analyst_suggestions", []),
            "qualitative_assessment": ai_analysis.get("llm_output", {}).get("qualitative_assessment", ""),
            "boundary_checks_passed": ai_analysis.get("boundary_valid", True),
        },
        "audit_trail": dec_res.audit_trail,
    }


# ── 4. SAMPLE ACCOUNTS LIST ──────────────────────────────────────────────────
@app.get("/api/sample-accounts")
def get_sample_accounts():
    """Returns curated representative accounts with descriptions for the UI."""
    return CACHE.get("sample_accounts", [])


# ── 5. TRAJECTORY SERIES ─────────────────────────────────────────────────────
@app.get("/api/trajectory/{ring_id}")
def get_trajectory(ring_id: str):
    """
    Returns the 5 lifecycle checkpoints for the specified late-forming ring
    from evals/results/trajectory_results.parquet.
    """
    parquet_path = BASE_DIR / "evals" / "results" / "trajectory_results.parquet"
    if not parquet_path.exists():
        raise HTTPException(status_code=404, detail="trajectory_results.parquet not found")

    df = pd.read_parquet(parquet_path)
    ring_df = df[df["ring_id"] == ring_id]

    if ring_df.empty:
        raise HTTPException(status_code=404, detail=f"Ring '{ring_id}' not found in trajectory dataset.")

    # Group by checkpoint_idx
    checkpoints = []
    for cp_idx in sorted(ring_df["checkpoint_idx"].unique()):
        cp_rows = ring_df[ring_df["checkpoint_idx"] == cp_idx]
        cp_label = cp_rows["checkpoint_label"].iloc[0]
        day = int(cp_rows["checkpoint_day"].iloc[0])
        days_from_start = int(cp_rows["days_from_start"].iloc[0])
        ring_type = cp_rows["ring_type"].iloc[0]
        ring_size = len(cp_rows)

        avg_orders = round(float(cp_rows["n_orders"].mean()), 2)
        p_behav = round(float(cp_rows["p_behav_ac"].mean()), 4)
        p_struct = round(float(cp_rows["p_struct_ac"].mean()), 4)
        p_fused = round(float(cp_rows["p_fused_ac"].mean()), 4)
        sym_kl = round(float(cp_rows["sym_kl_divergence"].mean()), 4)

        dec_counts = cp_rows["decision"].value_counts().to_dict()
        lane_counts = cp_rows["routing_lane"].value_counts().to_dict()
        primary_dec = cp_rows["decision"].mode().iloc[0] if not cp_rows.empty else "WAIT_MONITOR"
        primary_lane = cp_rows["routing_lane"].mode().iloc[0] if not cp_rows.empty else "fused_auto"

        checkpoints.append({
            "checkpoint_num": int(cp_idx),
            "checkpoint_name": cp_label,
            "day": day,
            "day_offset": days_from_start,
            "ring_type": ring_type,
            "ring_size": ring_size,
            "avg_orders": avg_orders,
            "p_behav": p_behav,
            "p_struct": p_struct,
            "p_fused": p_fused,
            "sym_kl": sym_kl,
            "primary_decision": primary_dec,
            "primary_routing_lane": primary_lane,
            "decision_breakdown": {k: int(v) for k, v in dec_counts.items()},
            "lane_breakdown": {k: int(v) for k, v in lane_counts.items()},
        })

    # Summary notes for canonical rings
    trajectory_notes = ""
    if ring_id == "PROMO_001":
        trajectory_notes = (
            "Verified Sequence: ABSTAIN (CP1) -> REVIEW (CP2) -> ACT (CP3) -> ACT (CP4) -> ACT (CP5). "
            "Structural evidence precedes formation burst (negative offset -5d), routing early to REVIEW "
            "before converging to automated ACT as promotional orders accumulate."
        )
    elif ring_id == "REFARM_057":
        trajectory_notes = (
            "Verified Sequence: ABSTAIN (CP1) -> ABSTAIN (CP2) -> REVIEW (CP3) -> REVIEW (CP4) -> REVIEW (CP5). "
            "Unseen referral-farming ring lacks shared payouts (P(struct)=0.00). As behavioral referral velocity climbs, "
            "sym_KL diverges to 9.52, persistently routing to REVIEW without automated ACT resolution."
        )
    elif ring_id == "RETURN_027":
        trajectory_notes = (
            "Verified Sequence: ABSTAIN (CP1) -> ABSTAIN (CP2) -> REVIEW (CP3) -> REVIEW (CP4) -> REVIEW (CP5). "
            "Return abuse ring with moderate structural signal routes to human triage upon return threshold breach."
        )

    return {
        "ring_id": ring_id,
        "ring_type": checkpoints[0]["ring_type"] if checkpoints else "unknown",
        "ring_size": checkpoints[0]["ring_size"] if checkpoints else 0,
        "checkpoints": checkpoints,
        "notes": trajectory_notes,
    }


# ── 6. AVAILABLE RINGS ───────────────────────────────────────────────────────
@app.get("/api/rings")
def get_available_rings():
    """Returns late-forming ring IDs available for trajectory exploration."""
    parquet_path = BASE_DIR / "evals" / "results" / "trajectory_results.parquet"
    if not parquet_path.exists():
        return []
    df = pd.read_parquet(parquet_path)
    ring_meta = df.drop_duplicates("ring_id")[["ring_id", "ring_type", "formation_start_day", "formation_complete_day"]]
    return ring_meta.sort_values("ring_id").to_dict(orient="records")


# ── 7. KL-ROUTING ABLATION ───────────────────────────────────────────────────
@app.get("/api/ablation")
def get_ablation():
    """Returns kl_ablation_results.json as-is."""
    ablation_path = BASE_DIR / "evals" / "results" / "kl_ablation_results.json"
    if not ablation_path.exists():
        raise HTTPException(status_code=404, detail="kl_ablation_results.json not found")
    with open(ablation_path, "r") as f:
        return json.load(f)


# ── 8. PREVALENCE-SHIFT ANALYSIS ─────────────────────────────────────────────
@app.get("/api/prevalence-shift")
def get_prevalence_shift():
    """Returns prevalence_shift_results.json as-is."""
    prev_path = BASE_DIR / "evals" / "results" / "prevalence_shift_results.json"
    if not prev_path.exists():
        raise HTTPException(status_code=404, detail="prevalence_shift_results.json not found")
    with open(prev_path, "r") as f:
        return json.load(f)


# ── 9. MULTI-SEED VARIANCE ───────────────────────────────────────────────────
@app.get("/api/multi-seed")
def get_multiseed():
    """Returns multiseed_results.json as-is."""
    seed_path = BASE_DIR / "evals" / "results" / "multiseed_results.json"
    if not seed_path.exists():
        raise HTTPException(status_code=404, detail="multiseed_results.json not found")
    with open(seed_path, "r") as f:
        return json.load(f)


# ── 10. STAGE 12A ROBUSTNESS ─────────────────────────────────────────────────
@app.get("/api/robustness")
def get_robustness():
    """Returns Stage 12a robustness table data."""
    rob_path = BASE_DIR / "evals" / "results" / "robustness_results.json"
    if not rob_path.exists():
        raise HTTPException(status_code=404, detail="robustness_results.json not found")
    with open(rob_path, "r") as f:
        return json.load(f)


# ── 10a. GNN COMPARISON BASELINE ──────────────────────────────────────────────
@app.get("/api/gnn-comparison")
def get_gnn_comparison():
    """Returns GNN Rung 6 comparison results against structural_lgbm."""
    gnn_path = BASE_DIR / "evals" / "results" / "gnn_comparison_results.json"
    if not gnn_path.exists():
        raise HTTPException(status_code=404, detail="gnn_comparison_results.json not found")
    with open(gnn_path, "r") as f:
        return json.load(f)


# ── 10a2. SCENARIO B GENERALIZATION TEST ─────────────────────────────────────
@app.get("/api/scenario-b")
def get_scenario_b_results():
    """Returns Scenario B (Subscription Platform Trial Abuse) cross-scenario generalization results."""
    scen_path = BASE_DIR / "evals" / "results" / "scenario_b_generalization_results.json"
    if not scen_path.exists():
        raise HTTPException(status_code=404, detail="scenario_b_generalization_results.json not found")
    with open(scen_path, "r") as f:
        return json.load(f)


# ── 10a3. ADVERSARIAL EVASION TEST ───────────────────────────────────────────
@app.get("/api/adversarial-evasion")
def get_adversarial_evasion_results():
    """Returns Adversarial Evasion and Adaptive Attacker stress test results."""
    adv_path = BASE_DIR / "evals" / "results" / "adversarial_results.json"
    if not adv_path.exists():
        raise HTTPException(status_code=404, detail="adversarial_results.json not found")
    with open(adv_path, "r") as f:
        return json.load(f)


# ── 10a4. DYNAMIC COMPOUNDING COST MODEL ─────────────────────────────────────
@app.get("/api/dynamic-cost")
def get_dynamic_cost_results():
    """Returns Time-Dependent Compounding Loss and Break-Even Lag analysis."""
    cost_path = BASE_DIR / "evals" / "results" / "dynamic_cost_results.json"
    if not cost_path.exists():
        raise HTTPException(status_code=404, detail="dynamic_cost_results.json not found")
    with open(cost_path, "r") as f:
        return json.load(f)


# ── 10a5. CAPACITY-CONSTRAINED REVIEW QUEUE ──────────────────────────────────
@app.get("/api/review-queue/capacity")
def get_capacity_constrained_results():
    """Returns Capacity-Constrained Review Queue Triage evaluation results."""
    cap_path = BASE_DIR / "evals" / "results" / "capacity_constrained_results.json"
    if not cap_path.exists():
        raise HTTPException(status_code=404, detail="capacity_constrained_results.json not found")
    with open(cap_path, "r") as f:
        return json.load(f)


# ── 10a6. GRAPH NEIGHBORHOOD & INVESTIGATION WORKSPACE ───────────────────────
@app.get("/api/graph-neighborhood/{account_id}")
def get_graph_neighborhood(account_id: str, max_nodes: int = 25):
    """Returns 1-hop connected graph neighborhood, edge types, and investigation checklist for an account."""
    if not CACHE:
        load_artifacts()

    G = CACHE.get("graph_test")
    if G is None:
        raise HTTPException(status_code=500, detail="Graph not loaded in cache")

    if not G.has_node(account_id):
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found in temporal graph")

    labels_lookup = CACHE["labels_full"]["label"].to_dict() if "labels_full" in CACHE else {}
    sample_acc_ids = {sa["account_id"] for sa in CACHE.get("sample_accounts", [])}
    
    # Extract 1-hop neighbors
    all_neighbors = list(G.neighbors(account_id))
    total_neighbors = len(all_neighbors)
    
    # Sort neighbors by weight descending, then degree
    all_neighbors_sorted = sorted(
        all_neighbors,
        key=lambda n: (G[account_id][n].get("weight", 1), G.degree(n)),
        reverse=True
    )
    
    selected_1_hop = all_neighbors_sorted[:max_nodes - 1]
    is_truncated = total_neighbors > len(selected_1_hop)
    
    selected_nodes = set([account_id] + selected_1_hop)
    sub = G.subgraph(selected_nodes)
    
    # Node list
    idx_lookup = CACHE.get("idx_lookup", {})
    s_te = CACHE.get("s_te")
    b_te = CACHE.get("b_te")
    
    nodes_out = []
    for n in sub.nodes():
        is_center = (str(n) == account_id)
        node_label = labels_lookup.get(str(n), "unknown")
        # Only expose ground truth label for known sample accounts or center account
        exposed_label = node_label if (str(n) in sample_acc_ids or is_center) else "masked_peer"
        
        # Decision if in test split
        node_decision = None
        if str(n) in idx_lookup:
            i = idx_lookup[str(n)]
            obs_d = float(b_te.iloc[i]["account_age_days"]) if b_te is not None else 30.0
            n_ord = int(b_te.iloc[i]["n_orders"]) if b_te is not None else 5
            node_decision = CACHE["engine"].decide(
                account_id=str(n),
                p_fused=CACHE["p_fused"][i],
                p_struct=CACHE["p_struct"][i],
                p_behav=CACHE["p_behav"][i],
                observation_days=obs_d,
                n_orders=n_ord,
                as_of_ts=CACHE["split_info"]["test_end_ts"]
            ).decision.value

        nodes_out.append({
            "id": str(n),
            "is_center": is_center,
            "node_type": G.nodes[n].get("node_type", "account"),
            "degree": int(G.degree(n)),
            "label": exposed_label,
            "decision": node_decision
        })
        
    edges_out = []
    edge_type_counts = {}
    for u, v in sub.edges():
        data = sub[u][v]
        e_types = list(data.get("edge_types", []))
        weight = int(data.get("weight", 1))
        shared_ent = data.get("shared_entity")
        
        # Primary edge type: payout > instrument > device > ip > referral
        primary_type = "other"
        for prio in ["shared_payout", "shared_instrument", "shared_device", "shared_ip", "referral"]:
            if prio in e_types:
                primary_type = prio
                break
        if primary_type == "other" and e_types:
            primary_type = e_types[0]
            
        edge_type_counts[primary_type] = edge_type_counts.get(primary_type, 0) + 1
        
        edges_out.append({
            "source": str(u),
            "target": str(v),
            "edge_types": e_types,
            "primary_type": primary_type,
            "weight": weight,
            "shared_entity": str(shared_ent) if shared_ent else None
        })

    # Generate Investigation Checklist (2-4 concrete next-check steps based on actual evidence)
    checklist = []
    if account_id in idx_lookup:
        i = idx_lookup[account_id]
        s_row = s_te.iloc[i].to_dict() if s_te is not None else {}
        b_row = b_te.iloc[i].to_dict() if b_te is not None else {}
        
        # Check 1: Payout Infrastructure
        if s_row.get("shared_payout_degree", 0) > 0:
            checklist.append({
                "step": "Verify Payout Destination",
                "severity": "CRITICAL",
                "finding": f"Shares payout/bank destination with {int(s_row['shared_payout_degree'])} accounts.",
                "action": "Audit beneficiary bank IFSC/UPI handle for syndicate fund funneling."
            })
        
        # Check 2: Device & Network Multiplicity
        dev_deg = s_row.get("shared_device_degree", 0)
        ip_deg = s_row.get("shared_ip_degree", 0)
        if dev_deg > 0 or ip_deg > 0:
            checklist.append({
                "step": "Device & Network Cluster Inspection",
                "severity": "HIGH" if dev_deg > 0 else "MEDIUM",
                "finding": f"Co-locates on {int(dev_deg)} shared devices and {int(ip_deg)} shared IP subnets.",
                "action": "Inspect Canvas/WebGL hardware fingerprint hashes and residential proxy subnet ASN."
            })
            
        # Check 3: Referral Lineage
        ref_deg = s_row.get("referral_degree", 0)
        n_ref = b_row.get("n_referrals_sent", 0) + b_row.get("n_referrals_received", 0)
        if ref_deg > 0 or n_ref > 0:
            checklist.append({
                "step": "Referral Tree & Voucher Lineage",
                "severity": "HIGH",
                "finding": f"Connected to {int(ref_deg)} accounts via referral links ({int(n_ref)} total referral events).",
                "action": "Check referral bonus redemption timestamps and device overlap across invitees."
            })

        # Check 4: Evidence Disagreement (sym_KL)
        sym_kl = float(CACHE["conflicts"][i]) if "conflicts" in CACHE else 0.0
        if sym_kl > 0.50:
            checklist.append({
                "step": "Model Disagreement Resolution",
                "severity": "MEDIUM",
                "finding": f"Elevated evidence conflict (sym_KL = {sym_kl:.2f} > 0.50 threshold).",
                "action": "Determine if account is an evasive sleeper ring (strong graph, sparse orders) or promo farm."
            })

        # Check 5: Cold-Start / Order History
        n_orders = int(b_row.get("n_orders", 0))
        if n_orders < 2:
            checklist.append({
                "step": "Cold-Start Monitoring",
                "severity": "LOW",
                "finding": f"Only {n_orders} order placed in observation window.",
                "action": "Place in temporary watch queue; re-score automatically on next order placement."
            })

    if not checklist:
        checklist.append({
            "step": "Standard KYC Verification",
            "severity": "LOW",
            "finding": "No high-density structural links or model conflicts detected.",
            "action": "Confirm baseline phone number OTP verification and email domain validity."
        })

    return {
        "account_id": account_id,
        "nodes": nodes_out,
        "edges": edges_out,
        "total_neighbors_count": total_neighbors,
        "is_truncated": is_truncated,
        "truncation_note": f"Displaying top {len(selected_1_hop)} of {total_neighbors} connected neighbors sorted by edge weight" if is_truncated else None,
        "edge_type_counts": edge_type_counts,
        "investigation_checklist": checklist
    }





# ── 10a7. AI SECURITY & PROMPT INJECTION SUITE ────────────────────────────────
@app.get("/api/ai-security")
def get_ai_security_results():
    """Returns AI Security & Prompt Injection Evaluation results."""
    sec_path = BASE_DIR / "evals" / "results" / "ai_security_results.json"
    if not sec_path.exists():
        raise HTTPException(status_code=404, detail="ai_security_results.json not found")
    with open(sec_path, "r") as f:
        return json.load(f)


# ── 11. LIMITATIONS LIST ─────────────────────────────────────────────────────
@app.get("/api/limitations")
def get_limitations():
    """Returns known limitations verbatim matching README Section 9."""
    return [
        {
            "id": 1,
            "title": "Flat FN cost assumption resolved via Symmetric Dynamic Modeling",
            "detail": "While flat FN costs (Rs 2,000) favor behavioral-only (Rs 30,500 vs Rs 149,250), our symmetric compounding exposure model (L(t) = C_0 + alpha * t^1.2) establishes that routing becomes strictly cost-superior once detection lag exceeds 72.5 - 128.8 days for active rings (alpha >= Rs 100/day). See dynamic_cost_results.json."
        },
        {
            "id": 2,
            "title": "Structural signal constraint",
            "detail": "80.8% of true AC test accounts lack strong structural signal due to partial ring observation. Domain property, not a bug. The graph functions as a disagreement detector rather than a strong standalone classifier (F1=0.364)."
        },
        {
            "id": 3,
            "title": "Simulator-encoded patterns & Multi-seed instability",
            "detail": "Behavioral recall varies from 27.03% to 95.45% (mean: 58.63%) across seeds 42, 43, 44 due to synthetic order-timing sensitivity. Top behavioral features partly reflect simulation choices."
        },
        {
            "id": 4,
            "title": "No production signals",
            "detail": "No device fingerprint APIs, IP reputation databases, chargebacks, bank-side signals, KYC age, or SIM swap data. All entities are synthetic IDs."
        },
        {
            "id": 5,
            "title": "Geometric-mean fusion collapses when P(struct) ~ 0",
            "detail": "Fused model is not suitable as a standalone single-score classifier without the routing design, which treats disagreement as a routing signal rather than blending it away."
        },
        {
            "id": 6,
            "title": "Label noise is uniform, not adversarial",
            "detail": "Real label noise is biased toward late-formation and evasive accounts. The 22 noisy labels here are uniformly random -- underestimates real-world evaluation difficulty."
        },
        {
            "id": 7,
            "title": "Simulated LLM Output Validation Scope",
            "detail": "The prompt injection test suite validates the post-generation validator against hypothesized/simulated adversarial outputs rather than verified live model behavior under attack. Live LLM compliance under adversarial prompts remains subject to frontier model alignment boundaries."
        },
        {
            "id": 8,
            "title": "Extreme Signal Sparsity & Cold-Start Limitation (Hand-Crafted Battery Family D)",
            "detail": "Under extreme signal sparsity, the system correctly avoids false positives but experiences a genuine drop in recall: 7/24 Family D accounts (29.2%) represent a genuine detection limitation where adversaries execute low-velocity isolated pairs (TOPO_16) or brand-new cold-start farms with zero entity overlap (TOPO_17); 1/24 accounts (4.2%) reflects the deterministic cold-start gate (TOPO_18, n_orders < 2 -> ABSTAIN) correctly declining to act on single-order accounts per its explicit guardrail design, not a detection failure."
        },
        {
            "id": 9,
            "title": "Gateway Prototype Latency Scope",
            "detail": "All dual-path latency numbers (sync path p99: 9.29 ms, async path p99: 17.28 ms) are prototype design-targets measured in a local single-machine in-memory mock environment processing synthetic test data, not live distributed gateway traffic or remote database network latency."
        },
        {
            "id": 10,
            "title": "Longitudinal Lead-Time Scope & Human-in-the-Loop Quarantine",
            "detail": "The 5.93-day advance warning metric represents organic active-formation detection across 14/19 rings (73.7%). The higher 18.60-day figure applies only to the 5/19 rings with pre-positioned sleeper accounts created before order bursts. QUARANTINE_HOLD is strictly an advisory candidate flag for human-reviewed network holds, not autonomous account enforcement."
        }
    ]


# ── 15. GATEWAY BRIDGE & DUAL-PATH SPECIFICATION ENDPOINTS ───────────────────

@app.get("/api/gateway/spec")
def get_gateway_specification():
    """
    Returns the Dual-Path Production Architecture Specification and Design Targets.
    Explicitly qualified: prototype design-targets in local mock environment.
    """
    qualifier = (
        "Prototype design-target measured in a local single-machine mock environment "
        "(in-memory adapter processing synthetic test data, not live distributed gateway traffic or remote database latency)."
    )
    return {
        "title": "Dual-Path Payment Gateway Architecture Bridge",
        "description": "Architectural contract decoupling in-line payment authorization from out-of-band graph divergence routing.",
        "qualifier": qualifier,
        "design_targets": {
            "sync_path": {
                "name": "In-Line Payment Authorization Path",
                "budget_ms": "< 30.0 ms",
                "scope": "Fast behavioral feature evaluation (order velocity, promo abuse, amount z-score)",
                "actions": ["ALLOW", "CHALLENGE_2FA", "BLOCK"],
                "authority": "Preliminary in-line risk recommendation"
            },
            "async_path": {
                "name": "Near-Line Asynchronous Graph Enrichment & Divergence Routing",
                "budget_ms": "< 500.0 ms",
                "scope": "Multi-relational graph expansion (IP, device, payout, referral) & canonical sym_KL calculation",
                "authority": "Authoritative DecisionEngine routing (REVIEW, ACT, WAIT_MONITOR, ABSTAIN)",
                "conflict_policy": "Preserves both sync and async findings; flags sleeper/burst disagreements for human triage"
            }
        },
        "supported_gateway_schemas": [
            "payment.authorized (Razorpay / Stripe standard schema)",
            "order.created",
            "refund.created",
            "dispute.created"
        ]
    }


@app.post("/api/gateway/simulate-event")
def simulate_gateway_event(payload: Dict[str, Any]):
    """
    Simulates incoming payment event ingestion through the dual-path gateway adapter.
    Executes fast sync authorization followed by async graph divergence enrichment.
    """
    if not CACHE:
        load_artifacts()

    adapter: GatewayEventAdapter = CACHE.get("gateway_adapter")
    if not adapter:
        raise HTTPException(status_code=500, detail="Gateway adapter not initialized.")

    event = GatewayPaymentEvent.from_razorpay_payload(payload)
    account_id = event.account_id

    # Retrieve features from test split or default fallback
    idx_lookup = CACHE.get("idx_lookup", {})
    if account_id in idx_lookup:
        i = idx_lookup[account_id]
        s_feat = CACHE["s_te"].iloc[i]
        b_feat = CACHE["b_te"].iloc[i]
        obs_days = float(b_feat.get("observation_days", 30.0))
        n_orders = int(b_feat.get("n_orders", 5))
    else:
        s_feat = pd.Series({"degree": 0.0, "shared_payout_degree": 0.0})
        b_feat = pd.Series({"promo_rate": 0.0, "order_velocity_1h": 1.0})
        obs_days = 30.0
        n_orders = 5

    # 1. Execute Sync Authorization Path
    sync_resp = adapter.process_sync_authorization(event, b_feat)

    # 2. Execute Async Graph Enrichment Path
    async_resp = adapter.process_async_enrichment(
        event=event,
        sync_response=sync_resp,
        struct_features=s_feat,
        behav_features=b_feat,
        observation_days=obs_days,
        n_orders=n_orders
    )

    return {
        "event_id": event.event_id,
        "account_id": event.account_id,
        "amount_inr": event.amount_inr,
        "event_type": event.event_type.value,
        "sync_authorization": {
            "action": sync_resp.action.value,
            "behavioral_score": sync_resp.behavioral_score,
            "p_behav": sync_resp.p_behav,
            "execution_time_ms": sync_resp.execution_time_ms,
            "rationale": sync_resp.rationale,
            "qualifier": sync_resp.qualifier
        },
        "async_enrichment": {
            "authoritative_decision": async_resp.authoritative_decision.value,
            "routing_lane": async_resp.routing_lane.value,
            "p_struct": async_resp.p_struct,
            "p_fused": async_resp.p_fused,
            "sym_kl_divergence": async_resp.sym_kl_divergence,
            "evidence_conflict": async_resp.evidence_conflict,
            "sync_async_disagreement": async_resp.sync_async_disagreement,
            "disagreement_nature": async_resp.disagreement_nature,
            "execution_time_ms": async_resp.execution_time_ms,
            "qualifier": async_resp.qualifier
        }
    }


@app.get("/api/gateway/benchmark")
def get_gateway_benchmark(n_trials: int = 50):
    """
    Executes local prototype latency benchmark across dual execution paths.
    Explicitly reports p50/p95/p99 with prototype design-target qualifiers.
    """
    if not CACHE:
        load_artifacts()

    adapter: GatewayEventAdapter = CACHE.get("gateway_adapter")
    if not adapter:
        raise HTTPException(status_code=500, detail="Gateway adapter not initialized.")

    # Select representative sample accounts for benchmarking
    idx = CACHE["idx"][:min(10, len(CACHE["idx"]))]
    test_events = []
    for acc in idx:
        i = CACHE["idx_lookup"][acc]
        evt = GatewayPaymentEvent(
            event_id=f"bench_{acc}",
            event_type=GatewayEventType.PAYMENT_AUTHORIZED,
            account_id=acc,
            amount_inr=1200.0,
            currency="INR",
            timestamp=1707776000,
            ip_address="127.0.0.1",
            device_id="DEV_BENCH"
        )
        test_events.append((evt, CACHE["s_te"].iloc[i], CACHE["b_te"].iloc[i]))

    benchmark_data = adapter.benchmark_dual_path(test_events, n_iterations=n_trials)
    return benchmark_data


# ── 16. LONGITUDINAL TEMPORAL ESCALATION ENDPOINTS ───────────────────────────

@app.get("/api/temporal-escalation/summary")
def get_temporal_escalation_summary():
    """
    Returns the population-level longitudinal escalation evaluation across
    all 19 late-forming rings (formation start >= Day 55).
    """
    traj_path = BASE_DIR / "evals" / "results" / "trajectory_results.parquet"
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="trajectory_results.parquet not found.")
    
    df = pd.read_parquet(traj_path)
    policy = LongitudinalEscalationPolicy()
    summary = policy.evaluate_all_rings(df)
    return summary


@app.get("/api/temporal-escalation/ring/{ring_id}")
def get_ring_escalation_trace(ring_id: str):
    """
    Returns the step-by-step state machine trace for a specific late-forming ring.
    """
    traj_path = BASE_DIR / "evals" / "results" / "trajectory_results.parquet"
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="trajectory_results.parquet not found.")
    
    df = pd.read_parquet(traj_path)
    ring_df = df[df["ring_id"] == ring_id]
    if len(ring_df) == 0:
        raise HTTPException(status_code=404, detail=f"Ring '{ring_id}' not found in evaluated late-forming rings.")
    
    policy = LongitudinalEscalationPolicy()
    trace = policy.evaluate_ring_trajectory(ring_df.sort_values("checkpoint_idx"))
    from dataclasses import asdict
    return asdict(trace)


# ── 17. HAND-CRAFTED ADVERSARIAL TOPOLOGY BATTERY ENDPOINTS ─────────────────

@app.get("/api/handcrafted-adversarial/summary")
def get_handcrafted_adversarial_summary():
    """
    Returns the overall summary of the 25-topology out-of-distribution stress battery.
    """
    results_path = BASE_DIR / "evals" / "results" / "handcrafted_adversarial_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="handcrafted_adversarial_results.json not found.")
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Return high-level summary without the full array of per-topology dicts
    return {
        "qualifier": data.get("qualifier"),
        "total_topologies_evaluated": data.get("total_topologies_evaluated"),
        "total_accounts_evaluated": data.get("total_accounts_evaluated"),
        "overall_naive_caught": data.get("overall_naive_caught"),
        "overall_sentinel_caught": data.get("overall_sentinel_caught"),
        "overall_naive_recall_pct": data.get("overall_naive_recall_pct"),
        "overall_sentinel_effective_recall_pct": data.get("overall_sentinel_effective_recall_pct"),
        "total_cases_rescued_by_conflict_review": data.get("total_cases_rescued_by_conflict_review"),
        "family_breakdown": data.get("family_breakdown"),
    }


@app.get("/api/handcrafted-adversarial/topologies")
def get_handcrafted_adversarial_topologies():
    """
    Returns the full list of 25 evaluated topologies and individual routing metrics.
    """
    results_path = BASE_DIR / "evals" / "results" / "handcrafted_adversarial_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="handcrafted_adversarial_results.json not found.")
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/api/handcrafted-adversarial/topology/{topo_id}")
def get_handcrafted_adversarial_topology(topo_id: str):
    """
    Returns metrics for a single specific topology (e.g. TOPO_01_DENSE_CLIQUE_CAMO).
    """
    results_path = BASE_DIR / "evals" / "results" / "handcrafted_adversarial_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="handcrafted_adversarial_results.json not found.")
    
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    topos = data.get("topologies", [])
    matched = [t for t in topos if t.get("topo_id") == topo_id]
    if not matched:
        raise HTTPException(status_code=404, detail=f"Topology '{topo_id}' not found.")
    return matched[0]


# Mount static assets directory if it exists
static_dir = BASE_DIR / "ui" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serves the Single Page Application dashboard."""
    index_path = BASE_DIR / "ui" / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>AbuseRing Sentinel API is running. UI index.html not found.</h1>")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())
