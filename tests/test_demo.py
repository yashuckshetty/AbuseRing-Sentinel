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
    assert "ACT 4: THE DECISION EVOLVES" in output
    assert "ACT 5: THE EVIDENCE TRAIL" in output

    # Assert all 6 curated accounts appear with their expected decisions
    for case in CURATED_CASES:
        acc_id = case["account_id"]
        exp_dec = case["expected_decision"]
        assert acc_id in output, f"Account {acc_id} missing from demo output"
        assert exp_dec in output, f"Decision {exp_dec} missing from demo output"
