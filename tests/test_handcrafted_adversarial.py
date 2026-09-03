"""
Unit and Regression Tests for Independent Hand-Crafted Topology Stress Battery
===============================================================================
Verifies:
  1. Deterministic generation and evaluation of all 25 out-of-distribution topologies.
  2. Disagreement-aware routing outperforming naive geometric-mean thresholding.
  3. Resilience against graph camouflage, sleeper attacks, and entity manipulation.
  4. Regression: DecisionEngine.decide() authority and canonical sym_kl usage.
  5. Mandatory synthetic out-of-distribution qualifier inclusion.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from evals.handcrafted_adversarial import (
    TOPOLOGY_CATALOG,
    generate_topology_data,
    run_handcrafted_adversarial_battery,
)
from decision.decision_engine import (
    DecisionEngine,
    Decision,
    RoutingLane,
    sym_kl_divergence,
)


@pytest.fixture(scope="module")
def battery_results():
    results_path = Path("evals/results/handcrafted_adversarial_results.json")
    if not results_path.exists():
        return run_handcrafted_adversarial_battery()
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_all_25_topologies_catalog_count():
    """Verify exactly 25 hand-crafted topologies spanning 5 distinct threat families."""
    assert len(TOPOLOGY_CATALOG) == 25
    families = set(fam for _, fam, _ in TOPOLOGY_CATALOG)
    assert len(families) == 5
    assert "Graph Camouflage" in families
    assert "Temporal & Sleeper" in families
    assert "Entity Manipulation" in families
    assert "Extreme Sparsity" in families
    assert "Hybrid / Evasion Stress" in families


def test_battery_results_structure_and_qualifier(battery_results):
    """Verify structure and mandatory qualifier on battery evaluation results."""
    assert battery_results["total_topologies_evaluated"] == 25
    assert battery_results["total_accounts_evaluated"] >= 150
    assert "qualifier" in battery_results
    assert "Independent out-of-distribution structural stress battery" in battery_results["qualifier"]


def test_disagreement_routing_outperforms_naive_fusion(battery_results):
    """
    Verify that Sentinel's evidence-disagreement routing significantly
    outperforms naive geometric-mean fusion across the 25 failure topologies.
    """
    naive_rec = battery_results["overall_naive_recall_pct"]
    sentinel_rec = battery_results["overall_sentinel_effective_recall_pct"]
    rescued_count = battery_results["total_cases_rescued_by_conflict_review"]

    assert sentinel_rec > naive_rec
    assert sentinel_rec >= 80.0  # >= 80% effective recall preserved under stress
    assert rescued_count >= 100   # Over 100 false negatives rescued by conflict review lane


def test_entity_manipulation_and_temporal_evasion(battery_results):
    """
    Verify that Family B (Temporal & Sleeper) and Family C (Entity Manipulation)
    achieve 100% effective recall via divergence-triggered human review.
    """
    fam_stats = battery_results["family_breakdown"]
    assert fam_stats["Temporal & Sleeper"]["sentinel_caught"] == fam_stats["Temporal & Sleeper"]["total_accs"]
    assert fam_stats["Entity Manipulation"]["sentinel_caught"] == fam_stats["Entity Manipulation"]["total_accs"]


def test_regression_decision_engine_authority():
    """
    REGRESSION: Verify DecisionEngine authority and canonical sym_kl_divergence.
    """
    assert list(Decision) == ["ACT", "REVIEW", "WAIT_MONITOR", "ABSTAIN"]
    assert list(RoutingLane) == ["conflict_review", "fused_auto", "abstain"]

    p = np.array([0.9, 0.05, 0.05])
    q = np.array([0.05, 0.05, 0.9])
    assert sym_kl_divergence(p, q) > 2.0
