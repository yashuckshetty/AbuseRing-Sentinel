"""
AbuseRing Sentinel - Temporal Graph Builder
All features computed as-of a specified timestamp (no future leakage).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import networkx as nx
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

def build_graph_as_of(events_df, as_of_ts, include_edge_types=None):
    if include_edge_types is None:
        include_edge_types = ["shared_device","shared_ip","shared_instrument","shared_payout","referral"]
    past = events_df[events_df["timestamp"] <= as_of_ts].copy()
    G = nx.Graph()
    for acc in past["account_id"].dropna().unique():
        G.add_node(acc, node_type="account")
    order_evts = past[past["event_type"] == "order_placed"].copy()

    def _add_shared_edges(node_type, entity_col, edge_type):
        if edge_type not in include_edge_types:
            return
        sub = order_evts[["account_id", entity_col]].dropna()
        grouped = sub.groupby(entity_col)["account_id"].apply(set)
        for entity_id, acc_set in grouped.items():
            acc_list = list(acc_set)
            if len(acc_list) < 2:
                continue
            entity_node = str(entity_id)
            if not G.has_node(entity_node):
                G.add_node(entity_node, node_type=node_type)
            for i in range(len(acc_list)):
                for j in range(i+1, len(acc_list)):
                    a, b = acc_list[i], acc_list[j]
                    if G.has_edge(a, b):
                        G[a][b]["edge_types"].add(edge_type)
                        G[a][b]["weight"] += 1
                    else:
                        G.add_edge(a, b, edge_types={edge_type}, weight=1, shared_entity=entity_node)

    _add_shared_edges("device", "device_id", "shared_device")
    _add_shared_edges("ip", "ip_id", "shared_ip")
    _add_shared_edges("instrument", "instrument_id", "shared_instrument")
    _add_shared_edges("payout", "payout_id", "shared_payout")

    if "referral" in include_edge_types:
        ref_evts = past[past["event_type"] == "referral"].dropna(subset=["account_id","referrer_id"])
        for _, row in ref_evts.iterrows():
            a, b = row["account_id"], row["referrer_id"]
            if a == b: continue
            for node in [a, b]:
                if not G.has_node(node):
                    G.add_node(node, node_type="account")
            if G.has_edge(a, b):
                G[a][b]["edge_types"].add("referral")
                G[a][b]["weight"] += 1
            else:
                G.add_edge(a, b, edge_types={"referral"}, weight=1, shared_entity=None)

    G.graph["as_of_ts"] = as_of_ts
    G.graph["n_events_used"] = len(past)
    return G

def extract_account_structural_features(G, account_ids):
    rows = []
    clustering = nx.clustering(G)
    comp_map = {}
    for comp in nx.connected_components(G):
        sz = len([n for n in comp if G.nodes[n].get("node_type") == "account"])
        for n in comp:
            comp_map[n] = sz
    for acc in account_ids:
        if acc not in G:
            rows.append({"account_id": acc, "degree": 0, "shared_device_degree": 0,
                         "shared_ip_degree": 0, "shared_payout_degree": 0,
                         "shared_instrument_degree": 0, "referral_degree": 0,
                         "clustering_coeff": 0.0, "connected_component_size": 1,
                         "max_edge_weight": 0, "mean_edge_weight": 0.0, "multi_signal_edges": 0})
            continue
        acc_neighbors = [n for n in G.neighbors(acc) if G.nodes.get(n, {}).get("node_type") == "account"]
        sd = si = sp = sinstr = ref = multi = 0
        weights = []
        for nb in acc_neighbors:
            edge = G[acc][nb]
            et = edge.get("edge_types", set())
            w = edge.get("weight", 1)
            weights.append(w)
            if "shared_device" in et: sd += 1
            if "shared_ip" in et: si += 1
            if "shared_payout" in et: sp += 1
            if "shared_instrument" in et: sinstr += 1
            if "referral" in et: ref += 1
            if len(et) >= 2: multi += 1
        rows.append({"account_id": acc, "degree": len(acc_neighbors),
                     "shared_device_degree": sd, "shared_ip_degree": si,
                     "shared_payout_degree": sp, "shared_instrument_degree": sinstr,
                     "referral_degree": ref, "clustering_coeff": clustering.get(acc, 0.0),
                     "connected_component_size": comp_map.get(acc, 1),
                     "max_edge_weight": max(weights) if weights else 0,
                     "mean_edge_weight": float(np.mean(weights)) if weights else 0.0,
                     "multi_signal_edges": multi})
    return pd.DataFrame(rows)

def extract_account_behavioral_features(events_df, as_of_ts, account_ids, accounts_df):
    """
    Vectorized behavioral feature extraction.
    Replaces per-account pandas filter loop (O(n_accs * n_events)) with
    groupby aggregations (O(n_events)) -- critical for 5K accounts / 40K events.
    """
    past = events_df[events_df["timestamp"] <= as_of_ts].copy()
    acc_created = accounts_df.set_index("account_id")["created_ts"]
    SECONDS_PER_DAY = 86400
    acc_set = set(account_ids)

    # Filter to relevant accounts only
    past_acc = past[past["account_id"].isin(acc_set)]

    orders   = past_acc[past_acc["event_type"] == "order_placed"].copy()
    returns  = past_acc[past_acc["event_type"] == "order_returned"]
    sessions = past_acc[past_acc["event_type"] == "session_start"]

    # refs_sent: referrer_id in account_ids
    refs_sent_all = past[past["event_type"] == "referral"].dropna(subset=["referrer_id"])
    refs_sent_all = refs_sent_all[refs_sent_all["referrer_id"].isin(acc_set)]
    refs_recv     = past_acc[past_acc["event_type"] == "referral"]

    # Aggregate per account
    order_grp   = orders.groupby("account_id")
    n_orders_s  = order_grp.size().rename("n_orders")
    amounts_agg = order_grp["amount"].agg(["mean", "std", "max"]).rename(
        columns={"mean": "mean_order_amount", "std": "std_order_amount", "max": "max_order_amount"})
    promo_agg   = order_grp["promo_code"].apply(lambda x: x.notna().sum()).rename("promo_orders")
    first_ts    = order_grp["timestamp"].min().rename("first_order_ts")

    n_returns_s  = returns.groupby("account_id").size().rename("n_returns")
    n_sessions_s = sessions.groupby("account_id")["session_id"].nunique().rename("n_sessions")
    n_refs_sent  = refs_sent_all.groupby("referrer_id").size().rename("n_referrals_sent")
    n_refs_recv  = refs_recv.groupby("account_id").size().rename("n_referrals_received")

    # Per-account burst score
    def _burst(ts_series):
        ts = sorted(ts_series.values)
        if len(ts) < 2:
            return len(ts)
        window_sec = 48 * 3600; lo = 0; best = 1
        for hi in range(len(ts)):
            while ts[hi] - ts[lo] > window_sec:
                lo += 1
            best = max(best, hi - lo + 1)
        return best

    if len(orders) > 0:
        burst_s = order_grp["timestamp"].apply(_burst).rename("burst_score")
        
        def _order_days(grp):
            created = acc_created.get(grp.name, as_of_ts)
            return grp["timestamp"].apply(lambda t: (t - created) // SECONDS_PER_DAY).nunique()
        order_days_s = order_grp.apply(_order_days, include_groups=False)
        if isinstance(order_days_s, pd.DataFrame):
            order_days_s = order_days_s.squeeze()
        order_days_s = order_days_s.rename("order_days_active")
    else:
        burst_s = pd.Series(dtype=int, name="burst_score", index=pd.Index([], name="account_id"))
        order_days_s = pd.Series(dtype=int, name="order_days_active", index=pd.Index([], name="account_id"))

    # Assemble final DataFrame
    base = pd.DataFrame({"account_id": account_ids}).set_index("account_id")
    for s in [n_orders_s, amounts_agg, promo_agg, first_ts, burst_s, order_days_s,
               n_returns_s, n_sessions_s, n_refs_recv]:
        base = base.join(s, how="left")
    base = base.join(n_refs_sent, how="left")

    base["n_orders"]             = base["n_orders"].fillna(0).astype(int)
    base["n_returns"]            = base["n_returns"].fillna(0).astype(int)
    base["n_sessions"]           = base["n_sessions"].fillna(0).astype(int)
    base["promo_orders"]         = base["promo_orders"].fillna(0).astype(int)
    base["n_referrals_sent"]     = base["n_referrals_sent"].fillna(0).astype(int)
    base["n_referrals_received"] = base["n_referrals_received"].fillna(0).astype(int)
    base["burst_score"]          = base["burst_score"].fillna(0).astype(int)
    base["order_days_active"]    = base["order_days_active"].fillna(0).astype(int)
    base["mean_order_amount"]    = base["mean_order_amount"].fillna(0.0)
    base["std_order_amount"]     = base["std_order_amount"].fillna(0.0)
    base["max_order_amount"]     = base["max_order_amount"].fillna(0.0)

    # Derived features
    base["return_rate"]  = base["n_returns"] / base["n_orders"].clip(lower=1)
    base["promo_rate"]   = base["promo_orders"] / base["n_orders"].clip(lower=1)
    base["has_promo"]    = (base["promo_orders"] > 0).astype(int)
    base["mean_daily_orders"] = base["n_orders"] / base["order_days_active"].clip(lower=1)

    # Age features (vectorized)
    created_series = pd.Series(
        [acc_created.get(acc, as_of_ts) for acc in base.index],
        index=base.index, name="created_ts"
    )
    base["account_age_days"]    = ((as_of_ts - created_series) / SECONDS_PER_DAY).clip(lower=0)
    base["first_order_age_days"] = base["account_age_days"]  # default if no orders
    has_orders = base["first_order_ts"].notna()
    base.loc[has_orders, "first_order_age_days"] = (
        (base.loc[has_orders, "first_order_ts"] - created_series[has_orders]) / SECONDS_PER_DAY
    ).clip(lower=0)

    base = base.drop(columns=["promo_orders", "first_order_ts"], errors="ignore")
    return base.reset_index()
