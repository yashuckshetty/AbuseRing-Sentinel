import sys; sys.path.insert(0, ".")
import json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import networkx as nx
from graph.temporal_graph import build_graph_as_of
from features.feature_pipeline import STRUCTURAL_FEATURES, LABEL_MAP, build_feature_matrix
import joblib

EDGE_WEIGHTS = {"shared_payout": 5.0, "shared_instrument": 4.0,
                "shared_device": 3.0, "shared_ip": 2.0, "referral": 1.0}

events   = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels   = pd.read_parquet("data/labels.parquet")
split    = json.load(open("data/split_info.json"))
test_end = split["test_end_ts"]

print("=== DIAGNOSTIC 2: Edge weight threshold vs giant component ===")
G_raw = build_graph_as_of(events, test_end)
acct_nodes = [n for n,d in G_raw.nodes(data=True) if d.get("node_type")=="account"]
comps_raw = list(nx.connected_components(G_raw.subgraph(acct_nodes)))
sizes_raw = sorted([len(c) for c in comps_raw], reverse=True)
print(f"BEFORE threshold: {G_raw.number_of_edges()} edges, "
      f"{len(comps_raw)} components, largest={sizes_raw[0]}, "
      f"second={sizes_raw[1] if len(sizes_raw)>1 else 0}")

# Apply semantic edge weights and filter low-weight edges
for u, v, d in G_raw.edges(data=True):
    weight = sum(EDGE_WEIGHTS.get(et, 1.0) for et in d.get("edge_types", set()))
    d["semantic_weight"] = weight

# Try thresholds
for threshold in [2.0, 3.0, 5.0]:
    G_thresh = nx.Graph()
    for u, v, d in G_raw.edges(data=True):
        if d.get("semantic_weight", 0) >= threshold:
            G_thresh.add_edge(u, v, **d)
    for n in acct_nodes:
        if not G_thresh.has_node(n):
            G_thresh.add_node(n, node_type="account")
    G_thresh_acct = G_thresh.subgraph(acct_nodes)
    comps = list(nx.connected_components(G_thresh_acct))
    sizes = sorted([len(c) for c in comps], reverse=True)
    print(f"AFTER threshold>={threshold}: {G_thresh.number_of_edges()} edges, "
          f"{len(comps)} components, largest={sizes[0]}, "
          f"second={sizes[1] if len(sizes)>1 else 0}, "
          f"singleton={sum(1 for s in sizes if s==1)}")

# Now rebuild structural features with threshold=5.0 (payout-only edges)
# and retrain structural LightGBM -- report F1 change
print("\n=== RETRAINING structural_lgbm with semantic weight threshold >= 5.0 ===")
from graph.temporal_graph import extract_account_structural_features, extract_account_behavioral_features

def build_struct_with_threshold(events, as_of_ts, active_accs, threshold):
    G = build_graph_as_of(events, as_of_ts)
    # Apply semantic weight and prune
    edges_to_remove = []
    for u, v, d in G.edges(data=True):
        w = sum(EDGE_WEIGHTS.get(et, 1.0) for et in d.get("edge_types", set()))
        d["semantic_weight"] = w
        if w < threshold:
            edges_to_remove.append((u, v))
    G.remove_edges_from(edges_to_remove)
    return extract_account_structural_features(G, active_accs)

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import lightgbm as lgb

label_map_str = labels.set_index("account_id")["label_true"].to_dict()

def get_split_accs(start_ts, end_ts):
    w = events[(events["timestamp"]>start_ts)&(events["timestamp"]<=end_ts)&
               (events["event_type"]=="order_placed")]
    return w["account_id"].unique().tolist()

train_accs = get_split_accs(split["sim_start_ts"], split["train_end_ts"])
test_accs  = get_split_accs(split["val_end_ts"],   split["test_end_ts"])

LABEL_MAP = {"benign_independent":0,"benign_coordinated":1,"abusive_coordinated":2}

for thresh in [3.0, 5.0]:
    s_tr = build_struct_with_threshold(events, split["train_end_ts"], train_accs, thresh).set_index("account_id")
    s_te = build_struct_with_threshold(events, split["test_end_ts"],  test_accs,  thresh).set_index("account_id")
    y_tr = np.array([LABEL_MAP[label_map_str.get(a,"benign_independent")] for a in train_accs])
    y_te = np.array([LABEL_MAP[label_map_str.get(a,"benign_independent")] for a in test_accs])

    clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, class_weight="balanced", random_state=42,
        verbose=-1, objective="multiclass", num_class=3)
    clf.fit(s_tr[STRUCTURAL_FEATURES].fillna(0), y_tr)
    pred = clf.predict(s_te[STRUCTURAL_FEATURES].fillna(0))
    proba = clf.predict_proba(s_te[STRUCTURAL_FEATURES].fillna(0))
    f1_a = f1_score(y_te, pred, average=None, zero_division=0)[2]
    auc = roc_auc_score(label_binarize(y_te,[0,1,2]),proba,multi_class="ovr",average="macro")

    # Component sizes
    G_t = build_graph_as_of(events, split["test_end_ts"])
    for u,v,d in G_t.edges(data=True):
        w = sum(EDGE_WEIGHTS.get(et,1.0) for et in d.get("edge_types",set()))
        d["semantic_weight"] = w
    G_t.remove_edges_from([(u,v) for u,v,d in G_t.edges(data=True) if d.get("semantic_weight",0)<thresh])
    acct_n = [n for n,d in G_t.nodes(data=True) if d.get("node_type")=="account"]
    comps = sorted([len(c) for c in nx.connected_components(G_t.subgraph(acct_n))], reverse=True)
    print(f"threshold>={thresh}: F1-abusive={f1_a:.4f} AUC={auc:.4f} "
          f"largest_component={comps[0]} singleton={sum(1 for s in comps if s==1)}")