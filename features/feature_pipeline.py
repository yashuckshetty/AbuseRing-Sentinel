"""AbuseRing Sentinel - Feature Engineering Pipeline"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))
from graph.temporal_graph import (build_graph_as_of, extract_account_structural_features,
                                   extract_account_behavioral_features)

LABEL_MAP = {"benign_independent": 0, "benign_coordinated": 1, "abusive_coordinated": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
STRUCTURAL_FEATURES = ["degree","shared_device_degree","shared_ip_degree","shared_payout_degree",
    "shared_instrument_degree","referral_degree","clustering_coeff","connected_component_size",
    "max_edge_weight","mean_edge_weight","multi_signal_edges"]
BEHAVIORAL_FEATURES = ["n_orders","n_sessions","n_returns","return_rate","promo_rate","has_promo",
    "mean_order_amount","std_order_amount","max_order_amount","order_days_active","mean_daily_orders",
    "account_age_days","first_order_age_days","burst_score","n_referrals_sent","n_referrals_received"]
ALL_FEATURES = STRUCTURAL_FEATURES + BEHAVIORAL_FEATURES

def build_feature_matrix(events_df, accounts_df, labels_df, as_of_ts, account_ids=None):
    if account_ids is None:
        account_ids = accounts_df["account_id"].tolist()
    G = build_graph_as_of(events_df, as_of_ts)
    struct_df = extract_account_structural_features(G, account_ids).set_index("account_id")
    behav_df = extract_account_behavioral_features(events_df, as_of_ts, account_ids, accounts_df).set_index("account_id")
    label_lookup = labels_df.set_index("account_id")["label"]
    labels_encoded = pd.DataFrame({"account_id": account_ids,
        "label_str": [label_lookup.get(acc, "benign_independent") for acc in account_ids]})
    labels_encoded["label"] = labels_encoded["label_str"].map(LABEL_MAP)
    labels_encoded = labels_encoded.set_index("account_id")
    return struct_df, behav_df, labels_encoded

def build_temporal_splits(events_df, accounts_df, labels_df, split_info):
    train_end = split_info["train_end_ts"]
    val_end = split_info["val_end_ts"]
    test_end = split_info["test_end_ts"]
    results = {}
    for split_name, as_of_ts, start_ts in [
        ("train", train_end, split_info["sim_start_ts"]),
        ("val", val_end, train_end),
        ("test", test_end, val_end),
    ]:
        print(f"\n--- Building {split_name} split (as_of_ts={as_of_ts}) ---")
        window_events = events_df[(events_df["timestamp"] > start_ts) &
                                   (events_df["timestamp"] <= as_of_ts) &
                                   (events_df["event_type"] == "order_placed")]
        active_accs = window_events["account_id"].unique().tolist()
        print(f"  Active accounts in window: {len(active_accs)}")
        if len(active_accs) == 0:
            print(f"  WARNING: No active accounts in {split_name} window!")
            continue
        struct_df, behav_df, labels_df_split = build_feature_matrix(
            events_df, accounts_df, labels_df, as_of_ts, active_accs)
        label_counts = labels_df_split["label_str"].value_counts()
        print(f"  Label distribution: {label_counts.to_dict()}")
        results[split_name] = {"struct": struct_df, "behav": behav_df, "labels": labels_df_split,
                                "as_of_ts": as_of_ts, "n_accounts": len(active_accs)}
    return results
