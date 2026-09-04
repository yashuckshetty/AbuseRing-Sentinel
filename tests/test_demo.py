"""
Tests for demo.py deterministic walkthrough script.
===================================================
Verifies demo.py executes cleanly offline and outputs all 6 curated cases
with their expected decisions.
"""

import sys
import subprocess
from pathlib import Path

from data.curated_cases import CURATED_CASES

BASE_DIR = Path(__file__).resolve().parent.parent


def test_demo_script_execution():
    """Run demo.py in subprocess, assert exit code 0 and all curated decisions present."""
    cmd = [sys.executable, str(BASE_DIR / "demo.py")]
    res = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert res.returncode == 0, f"demo.py failed with return code {res.returncode}:\n{res.stderr}"

    output = res.stdout
    assert "ACT 1: THE PROBLEM" in output
    assert "ACT 2: TWO INDEPENDENT WITNESSES" in output
    assert "ACT 3: THE FAILURE MODE WE REFUSE" in output
    assert "ACT 3B: PORTFOLIO-LEVEL OUTCOME" in output
    assert "ACT 4: THE DECISION EVOLVES" in output
    assert "ACT 5: THE EVIDENCE TRAIL" in output
    assert "SELF-VERIFICATION: live recompute MATCHES committed artifact" in output

    # Assert all 6 curated accounts appear with their expected decisions
    for case in CURATED_CASES:
        acc_id = case["account_id"]
        exp_dec = case["expected_decision"]
        assert acc_id in output, f"Account {acc_id} missing from demo output"
        assert exp_dec in output, f"Decision {exp_dec} missing from demo output"


def test_live_recompute_matches_committed_artifact():
    """Verify live recomputed test split metrics match adversarial_results.json baseline exactly."""
    import json
    import joblib
    import numpy as np
    import pandas as pd
    from decision.decision_engine import Decision, DecisionEngine
    from features.feature_pipeline import build_temporal_splits

    models_dir = BASE_DIR / "models"
    data_dir = BASE_DIR / "data"

    fused = joblib.load(models_dir / "fused_calibrated.pkl")
    engine = DecisionEngine()

    events = pd.read_parquet(data_dir / "events.parquet")
    accounts = pd.read_parquet(data_dir / "accounts.parquet")
    labels = pd.read_parquet(data_dir / "labels.parquet")
    split_info = json.load(open(data_dir / "split_info.json"))

    splits = build_temporal_splits(events, accounts, labels, split_info)
    test_sp = splits["test"]
    idx = list(test_sp["labels"].index)
    s_te = test_sp["struct"].reindex(idx).fillna(0)
    b_te = test_sp["behav"].reindex(idx).fillna(0)

    p_struct, p_behav, p_fused, _ = fused.predict_proba_sub(s_te, b_te)
    as_of_ts = int(split_info["test_end_ts"])
    obs_days = b_te["account_age_days"].values.astype(float)
    n_orders_arr = b_te["n_orders"].values.astype(int)

    decisions = engine.decide_batch(
        account_ids=idx,
        p_fused_matrix=p_fused,
        p_struct_matrix=p_struct,
        p_behav_matrix=p_behav,
        observation_days=obs_days,
        n_orders_arr=n_orders_arr,
        as_of_ts=as_of_ts,
    )

    y_true_ac = (test_sp["labels"]["label_str"].values == "abusive_coordinated")
    y_true_benign = ~y_true_ac

    act_mask = np.array([d.decision == Decision.ACT for d in decisions])
    rev_mask = np.array([d.decision == Decision.REVIEW for d in decisions])
    wait_mask = np.array([d.decision == Decision.WAIT_MONITOR for d in decisions])
    abs_mask = np.array([d.decision == Decision.ABSTAIN for d in decisions])

    tp_act = int((act_mask & y_true_ac).sum())
    fp_act = int((act_mask & y_true_benign).sum())
    tp_rev = int((rev_mask & y_true_ac).sum())
    fn_wait = int((wait_mask & y_true_ac).sum())
    fn_abs = int((abs_mask & y_true_ac).sum())

    with open(BASE_DIR / "evals" / "results" / "adversarial_results.json", "r", encoding="utf-8") as f:
        art = json.load(f)["scenarios"]["baseline"]["decision_engine"]

    assert tp_act == art["ac_breakdown"]["AC_in_ACT"]
    assert tp_rev == art["ac_breakdown"]["AC_in_REVIEW"]
    assert fn_wait == art["ac_breakdown"]["AC_in_WAIT_escaped"]
    assert fn_abs == art["ac_breakdown"]["AC_in_ABSTAIN_gated"]
    assert fp_act == art["auto_act_false_positives"]

