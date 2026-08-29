"""Temporal Leakage Test Suite - STAGE 5 Required Gate"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.temporal_graph import build_graph_as_of, extract_account_structural_features

@pytest.fixture(scope="module")
def data():
    events = pd.read_parquet("data/events.parquet")
    accounts = pd.read_parquet("data/accounts.parquet")
    labels = pd.read_parquet("data/labels.parquet")
    split = json.load(open("data/split_info.json"))
    return {"events": events, "accounts": accounts, "labels": labels, "split": split}

def test_no_future_events_in_graph(data):
    events = data["events"]; split = data["split"]
    as_of_ts = split["train_end_ts"]
    G = build_graph_as_of(events, as_of_ts)
    past_orders = events[(events["timestamp"] <= as_of_ts) &
                          (events["event_type"] == "order_placed")][["account_id","payout_id"]].dropna()
    past_payout_to_accounts = past_orders.groupby("payout_id")["account_id"].apply(set).to_dict()
    for u, v, edge_data in G.edges(data=True):
        if "shared_payout" not in edge_data.get("edge_types", set()):
            continue
        found = any(u in acc_set and v in acc_set for acc_set in past_payout_to_accounts.values())
        assert found, f"Edge ({u}, {v}) has shared_payout but no matching past payout event - leakage."
    print(f"\n[PASS] No future payout edges in graph as-of ts={as_of_ts}")

def test_graph_stores_cutoff_timestamp(data):
    events = data["events"]; split = data["split"]
    ts = split["train_end_ts"]
    G = build_graph_as_of(events, ts)
    assert G.graph["as_of_ts"] == ts
    print(f"\n[PASS] Graph.as_of_ts = {G.graph['as_of_ts']}")

def test_future_graph_has_more_edges(data):
    events = data["events"]; split = data["split"]
    G_train = build_graph_as_of(events, split["train_end_ts"])
    G_test = build_graph_as_of(events, split["test_end_ts"])
    assert G_test.number_of_edges() >= G_train.number_of_edges(), (
        f"Test graph ({G_test.number_of_edges()} edges) < train graph ({G_train.number_of_edges()} edges).")
    print(f"\n[PASS] Train edges={G_train.number_of_edges()}, Test edges={G_test.number_of_edges()} (monotone)")

def test_label_not_in_graph_or_features(data):
    events = data["events"]; split = data["split"]
    G = build_graph_as_of(events, split["train_end_ts"])
    forbidden = {"abusive_coordinated","benign_coordinated","benign_independent","abusive","benign","ring_id","label"}
    for node, attrs in G.nodes(data=True):
        for key, val in attrs.items():
            assert key not in forbidden, f"Node {node} has forbidden attr key '{key}'"
            if isinstance(val, str):
                assert val not in forbidden, f"Node {node} attr '{key}'='{val}' is a label string"
    print("\n[PASS] No label strings found in graph node attributes")

def _build_graph_with_injected_leakage(events_df, as_of_ts):
    import networkx as nx
    G = nx.Graph(); G.graph["as_of_ts"] = as_of_ts
    all_orders = events_df[events_df["event_type"] == "order_placed"].copy()
    for payout_id, grp in all_orders.groupby("payout_id"):
        accs = list(grp["account_id"].unique())
        for i in range(len(accs)):
            for j in range(i+1, len(accs)):
                G.add_edge(accs[i], accs[j], edge_types={"shared_payout"}, weight=1, shared_entity=None)
    return G

def _detect_leakage(G, events_df, as_of_ts):
    past_orders = events_df[(events_df["timestamp"] <= as_of_ts) &
                             (events_df["event_type"] == "order_placed")][["account_id","payout_id"]].dropna()
    past_payout_to_accounts = past_orders.groupby("payout_id")["account_id"].apply(set).to_dict()
    for u, v, edge_data in G.edges(data=True):
        if "shared_payout" not in edge_data.get("edge_types", set()):
            continue
        found = any(u in acc_set and v in acc_set for acc_set in past_payout_to_accounts.values())
        if not found:
            return True
    return False

def test_broken_builder_catches_leakage(data):
    events = data["events"]; split = data["split"]
    as_of_ts = split["train_end_ts"]
    future_events = events[events["timestamp"] > as_of_ts]
    if len(future_events) == 0:
        pytest.skip("No future events to inject")
    broken_G = _build_graph_with_injected_leakage(events, as_of_ts)
    leakage_detected = _detect_leakage(broken_G, events, as_of_ts)
    assert leakage_detected, "CRITICAL: Leakage detector failed to catch the broken builder."
    print("\n[PASS] Broken builder WAS detected (leakage detector works correctly)")

def test_structural_features_monotone_with_time(data):
    events = data["events"]; split = data["split"]; accounts = data["accounts"]
    sample_accs = accounts["account_id"].iloc[:50].tolist()
    G_train = build_graph_as_of(events, split["train_end_ts"])
    G_test = build_graph_as_of(events, split["test_end_ts"])
    feats_train = extract_account_structural_features(G_train, sample_accs)
    feats_test = extract_account_structural_features(G_test, sample_accs)
    train_deg = feats_train.set_index("account_id")["shared_payout_degree"]
    test_deg = feats_test.set_index("account_id")["shared_payout_degree"]
    violations = (train_deg > test_deg).sum()
    assert violations == 0, (f"{violations} accounts have HIGHER shared_payout_degree at train time than test time.")
    print(f"\n[PASS] Structural features are monotone for all {len(sample_accs)} sampled accounts")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
