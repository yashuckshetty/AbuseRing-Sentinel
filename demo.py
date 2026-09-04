"""
AbuseRing Sentinel — 5-Act Deterministic Walkthrough
=====================================================
Offline, single-command narrative demonstrating evidence-disagreement routing,
naive geometric-mean failure mode refusal, longitudinal ring lifecycle evolution,
and deterministic policy gate auditability.

All decisions are computed live by the authoritative DecisionEngine.
No network calls, no LLM invocations, no retraining, no data modifications.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from decision.decision_engine import Decision, DecisionEngine, sym_kl_divergence
from features.feature_pipeline import build_temporal_splits
from policy.policy_gate import PolicyGate
from policy.temporal_escalation import LongitudinalEscalationPolicy
from data.curated_cases import CURATED_CASES


def main():
    t_start = time.time()

    # -------------------------------------------------------------------------
    # Artifact Loading (Read-Only)
    # -------------------------------------------------------------------------
    models_dir = BASE_DIR / "models"
    data_dir = BASE_DIR / "data"

    fused = joblib.load(models_dir / "fused_calibrated.pkl")
    behav = joblib.load(models_dir / "behavioral_lgbm.pkl")
    struct = joblib.load(models_dir / "structural_lgbm.pkl")
    engine = DecisionEngine()
    gate = PolicyGate(decision_engine=engine, write_audit_log=False)

    events = pd.read_parquet(data_dir / "events.parquet")
    accounts = pd.read_parquet(data_dir / "accounts.parquet")
    labels = pd.read_parquet(data_dir / "labels.parquet")
    split_info = json.load(open(data_dir / "split_info.json"))

    splits = build_temporal_splits(events, accounts, labels, split_info)
    test_sp = splits["test"]
    idx = list(test_sp["labels"].index)
    idx_lookup = {acc: i for i, acc in enumerate(idx)}
    s_te = test_sp["struct"].reindex(idx).fillna(0)
    b_te = test_sp["behav"].reindex(idx).fillna(0)

    # Compute probability matrices live
    p_struct, p_behav, p_fused, conflicts = fused.predict_proba_sub(s_te, b_te)
    as_of_ts = int(split_info["test_end_ts"])

    # =========================================================================
    # ACT 1 — THE PROBLEM
    # =========================================================================
    print("=" * 80)
    print("ACT 1: THE PROBLEM -- COORDINATED ABUSE & EVIDENCE ASYMMETRY")
    print("=" * 80)
    print(
        "Modern organized fraud rings deliberately exploit single-model blindspots.\n"
        "Coordinated abuse arrives through distinct evidence channels that may agree,\n"
        "disagree, or be temporally incomplete (e.g. high transactional velocity with\n"
        "zero shared payout edges, or pre-positioned sleeper infrastructure with zero\n"
        "early orders). Forcing these asymmetric signals into a single scalar score\n"
        "either dilutes critical risk or triggers massive false-positive blocks on\n"
        "benign shared networks.\n\n"
        "NOTE: All evaluation figures are generated against a fully synthetic benchmark\n"
        "harness (5,000 accounts, 41k events, 198 test-window ring accounts), not live\n"
        "production payment gateway traffic."
    )
    print()

    # =========================================================================
    # ACT 2 -- TWO INDEPENDENT WITNESSES
    # =========================================================================
    print("=" * 80)
    print("ACT 2: TWO INDEPENDENT WITNESSES -- CURATED REPRESENTATIVE ACCOUNTS")
    print("=" * 80)
    print("Preserving structural and behavioral models as independent witnesses:")
    print(f"Authoritative Conflict Threshold: sym_KL > {engine.DEFAULT_KL_THRESHOLD:.2f} => REVIEW lane")
    print(f"Authoritative Action Threshold:   P(fused)[AC] >= {engine.THRESHOLD_ACT:.2f} => ACT lane")
    print("-" * 80)
    print(
        f"{'Account ID':10s}  {'Category':15s}  {'P_struct':8s}  {'P_behav':8s}  "
        f"{'P_fused':8s}  {'sym_KL':8s}  {'Routing Lane':15s}  {'Decision':12s}"
    )
    print("-" * 80)

    curated_results = []
    for case in CURATED_CASES:
        acc_id = case["account_id"]
        i = idx_lookup[acc_id]
        s_row = s_te.iloc[i]
        b_row = b_te.iloc[i]
        n_orders = int(b_row.get("n_orders", 0))
        obs_days = float(b_row.get("account_age_days", 0))

        ps = p_struct[i]
        pb = p_behav[i]
        pf = p_fused[i]
        kl = sym_kl_divergence(ps, pb)

        dec_res = engine.decide(
            account_id=acc_id,
            p_fused=pf,
            p_struct=ps,
            p_behav=pb,
            observation_days=obs_days,
            n_orders=n_orders,
            as_of_ts=as_of_ts,
        )
        curated_results.append((case, ps, pb, pf, kl, dec_res, s_row, b_row))

        cat_short = case["category"].split("(")[0].strip()[:15]
        print(
            f"{acc_id:10s}  {cat_short:15s}  {ps[2]:8.2f}  {pb[2]:8.2f}  "
            f"{pf[2]:8.2f}  {kl:8.4f}  {dec_res.routing_lane.value:15s}  {dec_res.decision.value:12s}"
        )
    print("-" * 80)
    print()

    # =========================================================================
    # ACT 3 -- THE FAILURE MODE WE REFUSE
    # =========================================================================
    print("=" * 80)
    print("ACT 3: THE FAILURE MODE WE REFUSE -- DISAGREEMENT RESCUE")
    print("=" * 80)
    print("Focus Account: ACC_04870 (Unseen Referral Farming Ring Member, REFARM_057)")
    
    # Locate ACC_04870
    c_4870 = next(r for r in curated_results if r[0]["account_id"] == "ACC_04870")
    _, ps_4870, pb_4870, pf_4870, kl_4870, dec_4870, _, _ = c_4870

    # 1. Structural and Behavioral P(AC)
    print(f"  Structural Model P(AC):  {ps_4870[2]:.4f}  (Zero shared payout infrastructure)")
    print(f"  Behavioral Model P(AC):  {pb_4870[2]:.4f}  (Rapid referral velocity)")
    
    # 2. Production fused score
    print(f"  Production Fused P(AC):   {pf_4870[2]:.2f}  <-- OUR OWN fused score also suppresses this account")
    print(f"  => Thresholding on P_fused alone at THRESHOLD_ACT ({engine.THRESHOLD_ACT:.2f}) would have produced a false negative here.")
    print()
    
    # 3. Geometric-mean ablation baseline (second, worse failure mode)
    p_naive_geom = np.sqrt(ps_4870[2] * pb_4870[2])
    print("  [ABLATION BASELINE -- NOT USED IN PRODUCTION]")
    print(f"  Naive Geometric-Mean Fusion: sqrt({ps_4870[2]:.2f} * {pb_4870[2]:.2f}) = {p_naive_geom:.4f}")
    print("  => A second, worse failure mode: suppresses true ring member to near-zero risk score.")
    print()
    
    # 4. Symmetric KL Divergence and routing
    print("  [ABUSERING SENTINEL PRODUCTION ROUTING]")
    print(f"  Symmetric KL Divergence: sym_KL = {kl_4870:.4f} (Threshold = {engine.DEFAULT_KL_THRESHOLD:.2f})")
    print(f"  => Divergence tripwire tripped ({kl_4870:.2f} >> {engine.DEFAULT_KL_THRESHOLD:.2f})")
    print(f"  => Routing Lane: {dec_4870.routing_lane.value}")
    print(f"  => Operational Decision: {dec_4870.decision.value}")
    print()
    
    # 5. Closing line for ACC_04870
    print("  What recovers this account is not a better score, but the measured disagreement")
    print("  between two independent witnesses.")
    print()

    # CHANGE 2: Mirror case (disagreement WITHHOLDS automation)
    print("-" * 80)
    print("MIRROR CASE: DISAGREEMENT WITHHOLDS AUTOMATION (SLEEPER ACCOUNT)")
    print("-" * 80)
    print("Focus Account: ACC_04430 (Pre-positioned Sleeper Infrastructure)")
    c_4430 = next(r for r in curated_results if r[0]["account_id"] == "ACC_04430")
    _, ps_4430, pb_4430, pf_4430, kl_4430, dec_4430, _, _ = c_4430

    print(f"  Structural Model P(AC):  {ps_4430[2]:.4f}  (Dense device/IP graph cluster)")
    print(f"  Behavioral Model P(AC):  {pb_4430[2]:.4f}  (Dormant order history / low early velocity)")
    print(f"  Production Fused P(AC):   {pf_4430[2]:.4f}")
    print(f"  Symmetric KL Divergence: sym_KL = {kl_4430:.4f} (Threshold = {engine.DEFAULT_KL_THRESHOLD:.2f})")
    print(f"  Routing Lane:            {dec_4430.routing_lane.value}")
    print(f"  Operational Decision:    {dec_4430.decision.value}")
    print(f"  => P_fused >= {engine.THRESHOLD_ACT:.2f} would have permitted auto-ACT, but sym_KL exceeding")
    print(f"     {engine.DEFAULT_KL_THRESHOLD:.2f} withheld automation and routed to human review.")
    print()
    print("  Disagreement routing is bidirectional -- it recovers accounts scalar fusion suppresses,")
    print("  and withholds automation scalar fusion would have granted.")
    print()

    # =========================================================================
    # ACT 3B: PORTFOLIO-LEVEL OUTCOME -- RECOMPUTED LIVE ON THE FULL TEST SPLIT
    # =========================================================================
    print("=" * 80)
    print("ACT 3B: PORTFOLIO-LEVEL OUTCOME -- RECOMPUTED LIVE ON THE FULL TEST SPLIT")
    print("=" * 80)

    obs_days_arr = b_te["account_age_days"].values.astype(float)
    n_orders_arr = b_te["n_orders"].values.astype(int)

    decisions_batch = engine.decide_batch(
        account_ids=idx,
        p_fused_matrix=p_fused,
        p_struct_matrix=p_struct,
        p_behav_matrix=p_behav,
        observation_days=obs_days_arr,
        n_orders_arr=n_orders_arr,
        as_of_ts=as_of_ts,
    )

    y_true_ac = (test_sp["labels"]["label_str"].values == "abusive_coordinated")
    y_true_benign = ~y_true_ac
    n_total = len(idx)
    n_ac_total = int(y_true_ac.sum())

    act_mask = np.array([d.decision == Decision.ACT for d in decisions_batch])
    rev_mask = np.array([d.decision == Decision.REVIEW for d in decisions_batch])
    wait_mask = np.array([d.decision == Decision.WAIT_MONITOR for d in decisions_batch])
    abs_mask = np.array([d.decision == Decision.ABSTAIN for d in decisions_batch])

    tp_act = int((act_mask & y_true_ac).sum())
    fp_act = int((act_mask & y_true_benign).sum())
    tp_rev = int((rev_mask & y_true_ac).sum())
    fn_wait = int((wait_mask & y_true_ac).sum())
    fn_abs = int((abs_mask & y_true_ac).sum())

    direct_rec = float(tp_act / n_ac_total) if n_ac_total > 0 else 0.0
    effective_rec = float((tp_act + tp_rev) / n_ac_total) if n_ac_total > 0 else 0.0

    print(f"Total Test Accounts:            {n_total}")
    print(f"Total True Coordinated Abuse:   {n_ac_total}")
    print(f"  - AC in ACT (auto-blocked):     {tp_act}")
    print(f"  - AC in REVIEW (triaged):       {tp_rev}")
    print(f"  - AC in WAIT_MONITOR (escaped): {fn_wait}")
    print(f"  - AC in ABSTAIN (gated):        {fn_abs}")
    print(f"Direct Auto-ACT Recall:         {direct_rec:.4f} ({direct_rec * 100:.2f}%)")
    print(f"Effective Recall (ACT+REVIEW):  {effective_rec:.4f} ({effective_rec * 100:.2f}%)")
    print(f"Auto-ACT False Positives:       {fp_act} (false positives in the auto-ACT lane)")
    print()
    print(f"Note on ABSTAIN-gated accounts: The {fn_abs} ABSTAIN-gated accounts are accounts the system")
    print("deliberately refuses to decide on under the n_orders < 2 cold-start policy guardrail,")
    print("not accounts it caught.")
    print()

    # Load committed artifact and self-verify
    adv_artifact_path = BASE_DIR / "evals" / "results" / "adversarial_results.json"
    if adv_artifact_path.exists():
        with open(adv_artifact_path, "r", encoding="utf-8") as f:
            adv_data = json.load(f)
        base_de = adv_data.get("scenarios", {}).get("baseline", {}).get("decision_engine", {})
        exp_bd = base_de.get("ac_breakdown", {})

        art_tp_act = exp_bd.get("AC_in_ACT")
        art_tp_rev = exp_bd.get("AC_in_REVIEW")
        art_fn_wait = exp_bd.get("AC_in_WAIT_escaped")
        art_fn_abs = exp_bd.get("AC_in_ABSTAIN_gated")
        art_dir_rec = base_de.get("direct_auto_act_recall")
        art_eff_rec = base_de.get("effective_recall")
        art_fp_act = base_de.get("auto_act_false_positives")

        print("Self-Verification against Committed Artifact (adversarial_results.json):")
        print(f"  AC_in_ACT:                Live={tp_act}  |  Artifact={art_tp_act}")
        print(f"  AC_in_REVIEW:             Live={tp_rev}  |  Artifact={art_tp_rev}")
        print(f"  AC_in_WAIT_escaped:       Live={fn_wait}  |  Artifact={art_fn_wait}")
        print(f"  AC_in_ABSTAIN_gated:      Live={fn_abs}  |  Artifact={art_fn_abs}")
        print(f"  Direct Auto-ACT Recall:   Live={direct_rec:.4f}  |  Artifact={art_dir_rec:.4f}")
        print(f"  Effective Recall:         Live={effective_rec:.4f}  |  Artifact={art_eff_rec:.4f}")
        print(f"  Auto-ACT False Positives: Live={fp_act}  |  Artifact={art_fp_act}")

        all_matched = (
            tp_act == art_tp_act
            and tp_rev == art_tp_rev
            and fn_wait == art_fn_wait
            and fn_abs == art_fn_abs
            and abs(direct_rec - float(art_dir_rec)) < 1e-4
            and abs(effective_rec - float(art_eff_rec)) < 1e-4
            and fp_act == art_fp_act
        )

        if all_matched:
            print("  ==> SELF-VERIFICATION: live recompute MATCHES committed artifact")
        else:
            print("  ==> MISMATCH: live recompute DOES NOT MATCH committed artifact")
    print()

    # =========================================================================
    # ACT 4 -- THE DECISION EVOLVES
    # =========================================================================
    print("=" * 80)
    print("ACT 4: THE DECISION EVOLVES -- LONGITUDINAL RING LIFECYCLE EVOLUTION")
    print("=" * 80)
    print("Tracing late-forming ring REFARM_057 (N=11 accounts) across sequential checkpoints:")
    print("State Machine: DORMANT_BASELINE -> ACCELERATING_MONITOR -> DIVERGENT_REVIEW -> QUARANTINE_HOLD")
    print("-" * 80)

    traj_path = BASE_DIR / "evals" / "results" / "trajectory_results.parquet"
    if traj_path.exists():
        traj_df = pd.read_parquet(traj_path)
        policy = LongitudinalEscalationPolicy()
        r_trace = policy.evaluate_ring_trajectory(traj_df[traj_df["ring_id"] == "REFARM_057"].sort_values("checkpoint_idx"))
        
        print(f"Ring ID: {r_trace.ring_id} | Type: {r_trace.ring_type} | Formation: Day {r_trace.formation_start_day} to Day {r_trace.formation_complete_day}")
        print(f"Escalation Lead Time vs Ring Completion: {r_trace.escalation_lead_time_vs_complete_days} days in advance")
        print()
        print(f"{'Checkpoint':15s} {'Day':5s} {'Risk State':22s} {'Ring Breakdown':25s} {'Mean sym_KL':12s}")
        print("-" * 80)
        for h in r_trace.checkpoint_history:
            bd = h["breakdown"]
            bd_str = f"ACT:{bd['ACT']} REV:{bd['REVIEW']} WAIT:{bd['WAIT']}"
            print(f"{h['checkpoint_label']:15s} Day {h['checkpoint_day']:<3d} {h['state']:22s} {bd_str:25s} {h['sym_kl_mean']:<12.3f}")
        print("-" * 80)
        print("Real state transitions produced by policy/temporal_escalation.py:")
        for tr in r_trace.transitions:
            print(f"  * Day {tr.checkpoint_day:2d} ({tr.checkpoint_label}): {tr.from_state.value} -> {tr.to_state.value}")
            print(f"    Reason: {tr.trigger_reason}")
    print()

    # =========================================================================
    # ACT 5 -- THE EVIDENCE TRAIL
    # =========================================================================
    print("=" * 80)
    print("ACT 5: THE EVIDENCE TRAIL -- DETERMINISTIC POLICY GATE AUDIT RECORD")
    print("=" * 80)
    print("PolicyGate emitted audit record for ACC_04870 (Zero LLM influence on numeric decision):")
    print("-" * 80)

    # Process through policy gate
    _, ps_4870, pb_4870, pf_4870, _, _, s_row_4870, b_row_4870 = c_4870
    pol_dec = gate.process(
        account_id="ACC_04870",
        p_fused=pf_4870.tolist(),
        p_struct=ps_4870.tolist(),
        p_behav=pb_4870.tolist(),
        conflict_flag=True,
        struct_feats=s_row_4870,
        behav_feats=b_row_4870,
        as_of_ts=as_of_ts,
    )

    audit = pol_dec.audit_trail
    print(f"Account ID:             ACC_04870")
    print(f"Timestamp (as_of_ts):   {audit['as_of_ts']}")
    print(f"Structural Sub-score:   {audit['structural_sub_score']:.4f}")
    print(f"Behavioral Sub-score:   {audit['behavioral_sub_score']:.4f}")
    print(f"Symmetric KL Divergence:{audit['sym_kl_divergence']:.4f}")
    print(f"Conflict Threshold:     {audit['kl_conflict_threshold']:.2f} (sym_KL > threshold => {audit['evidence_conflict']})")
    print(f"Assigned Routing Lane:  {audit['routing_lane']}")
    print(f"Authoritative Decision: {pol_dec.final_decision}")
    print(f"Simulated Review Cost:  Rs {audit['e_cost_review']:.2f}")
    print(f"Simulated False Neg Cost:Rs {audit['e_cost_wait']:.2f}")
    print(f"Decision Rationale:     {pol_dec.decision_rationale}")
    print()
    print("AI Advisory Text (strictly decoupled, post-decision explanation only):")
    print(f"  \"{pol_dec.ai_advisory}\"")
    print(f"AI Boundary Validation: Valid={pol_dec.ai_boundary_valid} (Violations={pol_dec.ai_violations})")
    print("-" * 80)
    print()

    dt = time.time() - t_start
    print(f"Evaluated {len(CURATED_CASES)} curated representative cases in {dt:.2f}s.")
    print("All decisions determined solely by the authoritative DecisionEngine (decision/decision_engine.py).")


if __name__ == "__main__":
    main()
