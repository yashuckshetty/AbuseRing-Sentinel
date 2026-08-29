import sys
sys.path.insert(0, ".")
import joblib, json
import pandas as pd, numpy as np
from features.feature_pipeline import BEHAVIORAL_FEATURES, STRUCTURAL_FEATURES

print("=== DIAGNOSTIC 1: Behavioral LightGBM Feature Importances ===")
model = joblib.load("models/behavioral_lgbm.pkl")
importances = model.feature_importances_
feat_imp = sorted(zip(BEHAVIORAL_FEATURES, importances), key=lambda x: -x[1])
total = sum(importances)
print(f"{'Feature':<25} {'Importance':>12} {'% of total':>12}")
print("-" * 52)
for f, imp in feat_imp[:10]:
    print(f"{f:<25} {imp:>12.4f} {100*imp/total:>11.1f}%")

promo_rank = next(i for i,(f,_) in enumerate(feat_imp) if f=="promo_rate") + 1
promo_imp  = next(imp for f,imp in feat_imp if f=="promo_rate")
print(f"\npromo_rate rank: #{promo_rank} / {len(BEHAVIORAL_FEATURES)}, share: {100*promo_imp/total:.1f}%")

events = pd.read_parquet("data/events.parquet")
labels = pd.read_parquet("data/labels.parquet")
label_map = labels.set_index("account_id")["label_true"].to_dict()
orders = events[events["event_type"] == "order_placed"]
promo_per_acc = orders.groupby("account_id").apply(
    lambda g: g["promo_code"].notna().mean()).rename("promo_rate").to_frame()
promo_per_acc["label"] = promo_per_acc.index.map(label_map)

print("\n=== PROMO RATE DISTRIBUTIONS BY CLASS ===")
for cls in ["benign_independent","benign_coordinated","abusive_coordinated"]:
    sub = promo_per_acc[promo_per_acc["label"]==cls]["promo_rate"]
    print(f"{cls}: mean={sub.mean():.3f} median={sub.median():.3f} "
          f"p95={sub.quantile(0.95):.3f} pct>0.5={100*(sub>0.5).mean():.1f}%")

# Depth-1 tree on promo_rate alone against test split
split = json.load(open("data/split_info.json"))
test_orders = orders[(orders["timestamp"]>split["val_end_ts"]) &
                     (orders["timestamp"]<=split["test_end_ts"])]
test_promo = test_orders.groupby("account_id").apply(
    lambda g: g["promo_code"].notna().mean()).rename("promo_rate").to_frame()
test_promo["y"] = test_promo.index.map(label_map).map(
    {"benign_independent":0,"benign_coordinated":1,"abusive_coordinated":2})
test_promo = test_promo.dropna(subset=["y"])
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import f1_score
clf = DecisionTreeClassifier(max_depth=1, random_state=42)
clf.fit(test_promo[["promo_rate"]], test_promo["y"])
pred = clf.predict(test_promo[["promo_rate"]])
f1_a = f1_score(test_promo["y"], pred, average=None, zero_division=0)[2]
print(f"\n=== DEPTH-1 TREE ON PROMO_RATE ALONE ===")
print(export_text(clf, feature_names=["promo_rate"]))
print(f"F1-abusive from promo_rate alone: {f1_a:.4f}")
print("VERDICT: If F1 > 0.70, behavioral recall is a simulator artifact.")