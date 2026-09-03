"""
Tests for Graph Neighborhood Subgraph Visualizer and Investigation Workspace.
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_ACCOUNTS = [
    "ACC_03653",
    "ACC_04870",
    "ACC_04430",
    "ACC_04295",
    "ACC_00505",
    "ACC_04987",
]

CANONICAL_EDGE_TYPES = {
    "shared_device",
    "shared_ip",
    "shared_instrument",
    "shared_payout",
    "referral",
}

def test_graph_neighborhood_sample_accounts_structure():
    """Verify /api/graph-neighborhood/{account_id} returns valid structure for all sample accounts."""
    for acc_id in SAMPLE_ACCOUNTS:
        resp = client.get(f"/api/graph-neighborhood/{acc_id}")
        assert resp.status_code == 200, f"Failed for {acc_id}: {resp.text}"
        data = resp.json()
        
        # Check root fields
        assert data["account_id"] == acc_id
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
        assert isinstance(data["total_neighbors_count"], int)
        assert isinstance(data["is_truncated"], bool)
        assert isinstance(data["edge_type_counts"], dict)
        assert isinstance(data["investigation_checklist"], list)
        assert len(data["investigation_checklist"]) >= 1

        # Check nodes
        center_nodes = [n for n in data["nodes"] if n["is_center"]]
        assert len(center_nodes) == 1
        assert center_nodes[0]["id"] == acc_id

        for n in data["nodes"]:
            assert "id" in n
            assert "node_type" in n
            assert "degree" in n
            assert "label" in n
            assert "decision" in n

        # Check edges
        for e in data["edges"]:
            assert "source" in e
            assert "target" in e
            assert "edge_types" in e
            assert "primary_type" in e
            assert "weight" in e
            assert e["weight"] >= 1
            for et in e["edge_types"]:
                assert et in CANONICAL_EDGE_TYPES

def test_graph_neighborhood_truncation_cap():
    """Verify dense accounts like ACC_04430 cap nodes properly and report truncation."""
    resp = client.get("/api/graph-neighborhood/ACC_04430?max_nodes=25")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_neighbors_count"] > 25
    assert data["is_truncated"] is True
    assert len(data["nodes"]) <= 25
    assert data["truncation_note"] is not None

def test_graph_neighborhood_investigation_checklist_content():
    """Verify investigation checklist items contain required operational fields and valid severity."""
    valid_severities = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
    for acc_id in SAMPLE_ACCOUNTS:
        resp = client.get(f"/api/graph-neighborhood/{acc_id}")
        data = resp.json()
        for item in data["investigation_checklist"]:
            assert "step" in item
            assert item["severity"] in valid_severities
            assert "finding" in item
            assert "action" in item
            assert len(item["action"]) > 0

def test_graph_neighborhood_nonexistent_account_404():
    """Verify 404 is returned for non-existent accounts."""
    resp = client.get("/api/graph-neighborhood/ACC_NONEXISTENT_99999")
    assert resp.status_code == 404
