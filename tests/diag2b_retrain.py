import sys; sys.path.insert(0, ".")
import json, warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
from graph.temporal_graph import build_graph_as_of, extract_account_structural_features
from features.feature_pipeline import STRUCTURAL_FEATURES
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import lightgbm as lgb

EDGE_WEIGHTS = {"shared_payout":5.0,"shared_instrument":4.0,
                "shared_device":3.0,"shared_ip":2.0,"referral":1.0}
LABEL_MAP = {"benign_independent":0,"benign_coordinated":1,"abusive_coordinated":2}

events   = pd.read_parquet("data/events.parquet")
labels   = pd.read_parquet("data/labels.parquet")
split    = json.load(open("data/split_info.json"))
label_map = labels.set_index("account_id")["label_true"].to_dict()

def get_split_accs(start_ts, end_ts):
    w = events[(events["timestamp"]>start_ts)&(events["timestamp"]<=end_ts)&
               (events["event_type"]=="order_placed")]
    return w["account_id"].unique().tolist()

train_accs = get_split_accs(split["sim_start_ts"], split["train_end_ts"])
test_accs  = get_split_accs(split["val_end_ts"],   split["test_end_ts"])

def build_thresholded_struct(events, as_of_ts, accs, threshold):
    G = build_graph_as_of(events, as_of_ts)
    to_remove = []
    for u,v,d in G.edges(data=True):
        sem_w = sum(EDGE_WEIGHTS.get(et,1.0) for et in d.get("edge_types",set()))
        d["semantic_weight"] = sem_w
        if sem_w < threshold:
            to_remove.append((u,v))
    G.remove_edges_from(to_remove)
    return extract_account_structural_features(G, accs)

print("=== RETRAINING structural_lgbm at weight thresholds ===")
for thresh in [3.0, 5.0]:
    s_tr = build_thresholded_struct(events, split["train_end_ts"], train_accs, thresh).set_index("account_id")
    s_te = build_thresholded_struct(events, split["test_end_ts"],  test_accs,  thresh).set_index("account_id")
    y_tr = np.array([LABEL_MAP[label_map.get(a,"benign_independent")] for a in train_accs])
    y_te = np.array([LABEL_MAP[label_map.get(a,"benign_independent")] for a in test_accs])

    clf = lgb.LGBMClassifier(n_estimators=300, max_depth=6, num_leaves=31,
        learning_rate=0.05, class_weight="balanced", random_state=42,
        verbose=-1, objective="multiclass", num_class=3)
    clf.fit(s_tr[STRUCTURAL_FEATURES].fillna(0), y_tr)
    pred  = clf.predict(s_te[STRUCTURAL_FEATURES].fillna(0))
    proba = clf.predict_proba(s_te[STRUCTURAL_FEATURES].fillna(0))
    f1_a  = f1_score(y_te, pred, average=None, zero_division=0)[2]
    yb    = label_binarize(y_te, classes=[0,1,2])
    auc   = roc_auc_score(yb, proba, multi_class="ovr", average="macro")
    rec_a = float(((pred==2)&(y_te==2)).sum()) / max((y_te==2).sum(), 1)
    prec_a = float(((pred==2)&(y_te==2)).sum()) / max((pred==2).sum(), 1)
    fn = int(((pred!=2)&(y_te==2)).sum()); fp = int(((pred==2)&(y_te!=2)).sum())
    print(f"threshold>={thresh}: F1-abusive={f1_a:.4f} Prec={prec_a:.4f} "
          f"Rec={rec_a:.4f} AUC={auc:.4f} FP={fp} FN={fn}")