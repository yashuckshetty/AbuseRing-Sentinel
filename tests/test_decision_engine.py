"""
test_decision_engine.py
Real-data tests for decision/decision_engine.py.
All fixtures use actual v2.0 parquet + trained model files.
NO mocks. Assertions are based on the routing architecture requirements.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
# Must be imported at module scope so joblib pickle can find the class
from models.fused_model import FusedCalibratedClassifier  # noqa: F401 -- required for joblib unpickling
from decision.decision_engine import DecisionEngine, Decision, RoutingLane, CostConfig

# ── Real data fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def engine():
    return DecisionEngine(kl_conflict_threshold=0.5)

@pytest.fixture(scope="session")
def e2e_results(engine):
    """Full batch of DecisionResults over v2.0 test split."""
    import joblib, warnings; warnings.filterwarnings("ignore")
    from models.model_suite import FusedCalibratedClassifier
    from features.feature_pipeline import build_temporal_splits, BEHAVIORAL_FEATURES

    events   = pd.read_parquet("data/events.parquet")
    accounts = pd.read_parquet("data/accounts.parquet")
    labels   = pd.read_parquet("data/labels.parquet")
    split    = json.load(open("data/split_info.json"))

    splits = build_temporal_splits(events, accounts, labels, split)
    sp = splits["test"]; idx = sp["labels"].index
    s_te = sp["struct"].reindex(idx).fillna(0)
    b_te = sp["behav"].reindex(idx).fillna(0)
    y_te = sp["labels"]["label"].values

    fused = joblib.load("models/fused_calibrated.pkl")
    p_struct, p_behav, p_fused, _ = fused.predict_proba_sub(s_te, b_te)

    n_orders_arr = b_te["n_orders"].fillna(0).astype(int).values
    obs_days_arr = b_te["account_age_days"].fillna(0).values

    results = engine.decide_batch(
        account_ids=list(idx),
        p_fused_matrix=p_fused, p_struct_matrix=p_struct, p_behav_matrix=p_behav,
        observation_days=obs_days_arr, n_orders_arr=n_orders_arr,
        as_of_ts=split["test_end_ts"],
    )
    return results, y_te

# ── Unit tests: DecisionResult contract ─────────────────────────────────────

def test_decide_returns_correct_fields(engine):
    """Every DecisionResult must carry the new routing fields."""
    result = engine.decide(
        account_id="TEST_ACC",
        p_fused=np.array([0.05, 0.10, 0.85]),
        p_struct=np.array([0.03, 0.05, 0.92]),
        p_behav=np.array([0.03, 0.05, 0.92]),
        observation_days=30.0, n_orders=5, as_of_ts=1707776000,
    )
    assert hasattr(result, "routing_lane"),       "Missing routing_lane"
    assert hasattr(result, "sym_kl_divergence"),  "Missing sym_kl_divergence"
    assert hasattr(result, "kl_conflict_threshold"), "Missing kl_conflict_threshold"
    assert result.kl_conflict_threshold == 0.5,  "KL threshold not stored"
    assert result.sym_kl_divergence >= 0.0,      "KL divergence must be non-negative"

def test_low_conflict_high_confidence_is_act(engine):
    """Low KL + high p_abusive + sufficient orders => ACT in fused lane."""
    # p_struct == p_behav => zero KL conflict
    p = np.array([0.05, 0.10, 0.85])
    result = engine.decide(
        account_id="ACC_ACT", p_fused=p, p_struct=p, p_behav=p,
        observation_days=20.0, n_orders=6, as_of_ts=1707776000,
    )
    assert result.decision == Decision.ACT,           f"Expected ACT, got {result.decision}"
    assert result.routing_lane == RoutingLane.FUSED_AUTO
    assert not result.evidence_conflict

def test_high_kl_conflict_routes_to_review(engine):
    """High KL (struct and behav strongly disagree) => REVIEW lane."""
    p_struct = np.array([0.90, 0.09, 0.01])   # struct says benign
    p_behav  = np.array([0.01, 0.05, 0.94])   # behav says abusive
    p_fused  = np.sqrt(p_struct * p_behav + 1e-9)
    p_fused  = p_fused / p_fused.sum()
    result = engine.decide(
        account_id="ACC_CONFLICT", p_fused=p_fused, p_struct=p_struct, p_behav=p_behav,
        observation_days=25.0, n_orders=4, as_of_ts=1707776000,
    )
    assert result.decision == Decision.REVIEW,       f"Expected REVIEW, got {result.decision}"
    assert result.routing_lane == RoutingLane.CONFLICT_REVIEW
    assert result.evidence_conflict
    assert result.sym_kl_divergence > 0.5

def test_insufficient_orders_abstain(engine):
    """n_orders < MIN_ORDERS_FOR_DECISION => ABSTAIN regardless of p_abusive."""
    p = np.array([0.02, 0.03, 0.95])
    result = engine.decide(
        account_id="ACC_ABSTAIN", p_fused=p, p_struct=p, p_behav=p,
        observation_days=5.0, n_orders=1, as_of_ts=1707776000,
    )
    assert result.decision == Decision.ABSTAIN
    assert result.routing_lane == RoutingLane.ABSTAIN

def test_low_confidence_low_conflict_is_wait(engine):
    """Low p_abusive + low conflict => WAIT_MONITOR, not REVIEW."""
    p = np.array([0.80, 0.15, 0.05])
    result = engine.decide(
        account_id="ACC_WAIT", p_fused=p, p_struct=p, p_behav=p,
        observation_days=10.0, n_orders=3, as_of_ts=1707776000,
    )
    assert result.decision == Decision.WAIT_MONITOR
    assert result.routing_lane == RoutingLane.FUSED_AUTO

def test_audit_trail_contains_required_keys(engine):
    """Audit trail must include routing-specific fields."""
    p = np.array([0.05, 0.10, 0.85])
    result = engine.decide(
        account_id="ACC_AUDIT", p_fused=p, p_struct=p, p_behav=p,
        observation_days=15.0, n_orders=4, as_of_ts=1707776000,
    )
    required = {"as_of_ts","routing_lane","sym_kl_divergence","kl_conflict_threshold",
                "evidence_conflict","p_abusive","e_cost_act","e_cost_review","e_cost_wait",
                "cost_config_note","recall_note","structural_signal_note"}
    missing = required - set(result.audit_trail.keys())
    assert not missing, f"Audit trail missing required keys: {missing}"

def test_recall_note_present_in_audit(engine):
    """The routing recall caveat must appear in every audit trail -- by design."""
    p = np.array([0.05, 0.10, 0.85])
    result = engine.decide(
        account_id="ACC_NOTE", p_fused=p, p_struct=p, p_behav=p,
        observation_days=15.0, n_orders=5, as_of_ts=1707776000,
    )
    note = result.audit_trail.get("recall_note", "")
    assert "direct auto-ACT recall" in note, \
        "recall_note must mention 'direct auto-ACT recall'"
    assert "REVIEW" in note, \
        "recall_note must mention REVIEW routing caveat"

def test_structural_limitation_note_present(engine):
    """Structural signal limitation note must appear in every audit trail."""
    p = np.array([0.05, 0.10, 0.85])
    result = engine.decide(
        account_id="ACC_STRUCT", p_fused=p, p_struct=p, p_behav=p,
        observation_days=15.0, n_orders=5, as_of_ts=1707776000,
    )
    note = result.audit_trail.get("structural_signal_note", "")
    assert "conflict-detection" in note, \
        "structural_signal_note must mention conflict-detection role"
    assert "80.8%" in note, \
        "structural_signal_note must state the 80.8% empirical figure"

def test_kl_threshold_configurable():
    """DecisionEngine must honour a custom kl_conflict_threshold."""
    engine_tight = DecisionEngine(kl_conflict_threshold=0.01)
    p_struct = np.array([0.80, 0.15, 0.05])
    p_behav  = np.array([0.10, 0.10, 0.80])
    p_fused  = np.sqrt(p_struct * p_behav + 1e-9)
    p_fused  = p_fused / p_fused.sum()
    result = engine_tight.decide(
        account_id="ACC_TIGHT", p_fused=p_fused, p_struct=p_struct, p_behav=p_behav,
        observation_days=20.0, n_orders=5, as_of_ts=1707776000,
    )
    assert result.decision == Decision.REVIEW, \
        "Tight KL threshold should route even moderate divergence to REVIEW"
    assert result.kl_conflict_threshold == 0.01

# ── Integration tests: real v2.0 data ───────────────────────────────────────

def test_no_ac_in_wait_monitor(e2e_results):
    """
    At KL=0.5, no true-AC account (with sufficient orders) should end up
    WAIT_MONITOR -- they either auto-ACT (low-conflict) or REVIEW (conflict)
    or ABSTAIN (insufficient orders).
    """
    results, y_te = e2e_results
    ac_wait = [(r, yt) for r, yt in zip(results, y_te)
               if yt == 2 and r.decision == Decision.WAIT_MONITOR]
    assert len(ac_wait) == 0, \
        f"{len(ac_wait)} AC accounts ended up WAIT_MONITOR: {[r.account_id for r,_ in ac_wait[:5]]}"

def test_auto_act_zero_false_positives(e2e_results):
    """Auto-ACT lane must produce zero FP (precision=1.000 on v2.0 test data)."""
    results, y_te = e2e_results
    fp = [(r, yt) for r, yt in zip(results, y_te)
          if r.decision == Decision.ACT and yt != 2]
    assert len(fp) == 0, \
        f"{len(fp)} non-AC accounts auto-ACT'd: {[(r.account_id, yt) for r,yt in fp[:5]]}"

def test_ac_effective_recall_above_80pct(e2e_results):
    """
    Effective recall (ACT + REVIEW, excluding ABSTAIN) must exceed 80% of
    the total AC count. (Target: 82% at KL=0.5, but ABSTAIN-excluded floor is 80%.)
    """
    results, y_te = e2e_results
    n_ac  = (y_te == 2).sum()
    ac_flagged = sum(1 for r, yt in zip(results, y_te)
                     if yt == 2 and r.decision in (Decision.ACT, Decision.REVIEW))
    eff_recall = ac_flagged / n_ac
    print(f"\n  AC flagged (ACT+REVIEW): {ac_flagged}/{n_ac} = {eff_recall:.4f}")
    assert eff_recall >= 0.80, \
        f"Effective recall {eff_recall:.4f} below 0.80 floor"

def test_routing_summary_has_correct_keys(e2e_results, engine):
    """routing_summary must include recall and structural limitation notes."""
    results, _ = e2e_results
    summary = engine.routing_summary(results)
    required = {"n_total","decision_counts","decision_fractions",
                "routing_lane_counts","simulated_review_cost",
                "recall_reporting_note","cost_config_note"}
    missing = required - set(summary.keys())
    assert not missing, f"routing_summary missing keys: {missing}"
    assert "direct auto-ACT recall" in summary["recall_reporting_note"]

def test_abstain_ac_have_low_orders(e2e_results):
    """All ABSTAIN AC accounts must have n_orders < MIN_ORDERS_FOR_DECISION."""
    results, y_te = e2e_results
    engine_ref = DecisionEngine()
    min_orders = engine_ref.MIN_ORDERS_FOR_DECISION
    bad = [(r.account_id, r.n_orders) for r, yt in zip(results, y_te)
           if yt == 2 and r.decision == Decision.ABSTAIN and r.n_orders >= min_orders]
    assert not bad, \
        f"AC accounts ABSTAINed with >= {min_orders} orders: {bad}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])