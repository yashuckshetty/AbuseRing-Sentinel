import pandas as pd, json, sys

events   = pd.read_parquet("data/events.parquet")
accounts = pd.read_parquet("data/accounts.parquet")
labels   = pd.read_parquet("data/labels.parquet")
split    = json.load(open("data/split_info.json"))

print("=== SCHEMA CHECK ===")
print("labels columns:", list(labels.columns))
print("events columns:", list(events.columns))
print("accounts columns:", list(accounts.columns))

print()
print("=== FIELD EXISTENCE CHECKS ===")
print("labels has 'label'?",        "label" in labels.columns)
print("labels has 'label_true'?",   "label_true" in labels.columns)
print("labels has 'label_observed'?","label_observed" in labels.columns)
print("labels has 'partial_signal'?","partial_signal" in labels.columns)
print("labels has 'counterfactual_subset'?","counterfactual_subset" in labels.columns)
print("events has 'referrer_id'?",  "referrer_id" in events.columns)
print("events has 'ring_id'?",      "ring_id" in events.columns)

print()
print("=== FEATURE PIPELINE schema mismatch detection ===")
# L27 in feature_pipeline.py: label_lookup = labels_df.set_index("account_id")["label"]
# This will raise KeyError if "label" column is absent
if "label" not in labels.columns:
    print("MISMATCH: feature_pipeline.py L27 reads labels_df['label'] -- column NOT PRESENT")
    print("  Present columns:", [c for c in labels.columns if "label" in c])
else:
    print("OK: 'label' column present (alias added in save_outputs)")
    print("  Sample values:", labels["label"].value_counts().to_dict())

print()
print("=== EVENT TYPE DISTRIBUTION ===")
print(events["event_type"].value_counts().to_dict())
print()
print("=== NULL CHECK: key columns ===")
for col in ["payout_id","device_id","ip_id","instrument_id","referrer_id"]:
    pct = events[col].isna().mean()
    print(f"  {col}: {pct*100:.1f}% null")