import sys, io, warnings, traceback
sys.path.insert(0, ".")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Capture the exact warning with full traceback including line number
warnings.filterwarnings("error", category=FutureWarning)

import pandas as pd, json
from features.feature_pipeline import build_temporal_splits, STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES

events   = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels   = pd.read_parquet("data/labels.parquet")
split    = json.load(open("data/split_info.json"))

print("Building train split - watching for FutureWarning...")
try:
    splits = build_temporal_splits(events, accounts, labels, split)
    print("No FutureWarning raised.")
except FutureWarning as e:
    tb = traceback.format_exc()
    print(f"FutureWarning caught:\n{e}")
    print(f"\nTraceback:\n{tb}")
    
    # Now show the exact offending line
    import re
    match = re.search(r'File "(.+?)", line (\d+)', tb)
    if match:
        fpath, lineno = match.group(1), int(match.group(2))
        print(f"\nOffending file: {fpath}, line {lineno}")
        try:
            with open(fpath) as f:
                lines = f.readlines()
            start = max(0, lineno-4)
            end = min(len(lines), lineno+3)
            for i, l in enumerate(lines[start:end], start=start+1):
                marker = ">>>" if i == lineno else "   "
                print(f"  {marker} {i:4d}: {l}", end="")
        except Exception as read_err:
            print(f"Could not read file: {read_err}")