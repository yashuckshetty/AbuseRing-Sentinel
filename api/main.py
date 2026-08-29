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
from decision.decision_engine import DecisionEngine, Decision, RoutingLane
from ai.evidence_reasoner import EvidenceGapReasoner, sym_kl_divergence
from policy.policy_gate import PolicyGate
from models.fused_model import FusedCalibratedClassifier

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
    print("Artifacts loaded successfully.", flush=True)

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


# ── 10b. SAMPLE ACCOUNTS ──────────────────────────────────────────────────────
@app.get("/api/sample-accounts")
def get_sample_accounts():
    """Returns curated sample accounts representing key operational regimes."""
    if not CACHE:
        load_artifacts()
    return CACHE.get("sample_accounts", [])


# ── 11. LIMITATIONS LIST ─────────────────────────────────────────────────────
@app.get("/api/limitations")
def get_limitations():
    """Returns known limitations verbatim matching README Section 9."""
    return [
        {
            "id": 1,
            "title": "Flat FN cost model",
            "detail": "c_fn=Rs2,000 regardless of time-to-detection. Under this assumption, behavioral-only (Rs30,500) dominates routing (Rs1,49,250-Rs1,88,850) on cost. Time-dependent loss modelling is future work."
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
            "title": "Referral-farming unresolved review queue cost",
            "detail": "Unseen referral-farming topology has no automated resolution path within the observed window (sym_KL climbs but never resolves to ACT), creating indefinite human REVIEW queue cost under this design."
        }
    ]


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
