import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import warnings

labels = pd.read_parquet("data/labels.parquet")
labels = labels.copy()
labels["account_idx"] = range(len(labels))

X = labels[["account_idx"]].values
class_map = {"benign_independent": 0, "benign_coordinated": 1, "abusive_coordinated": 2}
y = np.array([class_map[c] for c in labels["label_true"].values])

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
tr_idx, te_idx = next(sss.split(X, y))
X_tr, X_te = X[tr_idx], X[te_idx]
y_tr, y_te = y[tr_idx], y[te_idx]

clf = DecisionTreeClassifier(max_depth=2, random_state=42)
clf.fit(X_tr, y_tr)
print("=== DECISION TREE RULES ===")
print(export_text(clf, feature_names=["account_idx"]))

print("=== DIAGNOSIS ===")
print("account_idx directly encodes class: BI=0-2999, BC=3000-4249, AC=4250+")
print("This is a generation-order artifact: accounts were created in class order.")
print()
print("FIX NEEDED: The check_shortcut_detection function should shuffle accounts")
print("across classes so account_idx does NOT predict class, or exclude account_idx.")
print("Per ASSUMPTIONS.md 7c, the check should use creation_ts, not account_idx.")
print()
print("Accounts table created_ts range:")
accts = pd.read_parquet("data/accounts.parquet")
accts["label"] = accts["account_id"].map(dict(zip(labels["account_id"], labels["label_true"])))
for cls in ["benign_independent", "benign_coordinated", "abusive_coordinated"]:
    sub = accts[accts["label"]==cls]["created_ts"]
    print(f"  {cls}: min={sub.min()}, max={sub.max()}, unique={sub.nunique()}")