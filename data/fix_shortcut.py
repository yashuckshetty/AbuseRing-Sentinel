import sys

src = open("data/simulator.py", encoding="utf-8").read()

# Fix the shortcut check: use stratified split so all classes appear in the test slice
old_check = '''        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")'''

new_check = '''        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        tr_idx, te_idx = next(sss.split(X, y))
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")'''

if old_check not in src:
    print("ERROR: old_check not found")
    sys.exit(1)
src = src.replace(old_check, new_check, 1)

# Remove the duplicate split lines above the old fix (X_tr/X_te split_pt approach)
old_split = '''        split_pt = int(len(y) * 0.70)
        X_tr, X_te = X[:split_pt], X[split_pt:]
        y_tr, y_te = y[:split_pt], y[split_pt:]
'''
new_split = '''        # Stratified split done below
'''
if old_split in src:
    src = src.replace(old_split, new_split, 1)
    print("Removed old split_pt block")

with open("data/simulator.py", "w", encoding="utf-8", newline="") as f:
    f.write(src)
import ast; ast.parse(src)
print("Shortcut check fix applied. Syntax OK.")