import pandas as pd, numpy as np

events = pd.read_parquet("data/events.parquet")
labels = pd.read_parquet("data/labels.parquet")
label_map = labels.set_index("account_id")["label_true"].to_dict()

orders = events[(events["event_type"] == "order_placed")].dropna(subset=["payout_id"]).copy()
orders["label"] = orders["account_id"].map(label_map)

# Payout co-sharing degree per account
payout_acc = orders.groupby("payout_id")["account_id"].apply(set)
acc_payout_deg = {}
for pid, accs in payout_acc.items():
    accs = list(accs)
    deg = len(accs)
    for a in accs:
        acc_payout_deg[a] = acc_payout_deg.get(a, 0) + deg

rows = [{"account_id": a, "payout_degree": d, "label": label_map.get(a, "?")}
        for a, d in acc_payout_deg.items()]
deg_df = pd.DataFrame(rows)

print("=== PAYOUT DEGREE DISTRIBUTIONS (actual v2.0 data) ===")
for cls in ["benign_independent", "benign_coordinated", "abusive_coordinated"]:
    sub = deg_df[deg_df["label"] == cls]["payout_degree"]
    print(f"{cls}:")
    print(f"  n={len(sub)}, mean={sub.mean():.2f}, median={sub.median():.1f}, "
          f"p75={sub.quantile(0.75):.1f}, p95={sub.quantile(0.95):.1f}, max={sub.max()}")

print()
print("=== PAYOUT POOL OVERLAP ===")
ac_pays = set(orders[orders["label"]=="abusive_coordinated"]["payout_id"].unique())
bi_pays = set(orders[orders["label"]=="benign_independent"]["payout_id"].unique())
bc_pays = set(orders[orders["label"]=="benign_coordinated"]["payout_id"].unique())
print(f"AC-exclusive payouts:    {len(ac_pays - bi_pays - bc_pays)}")
print(f"AC + BI overlap:         {len(ac_pays & bi_pays)}  (contamination indicator)")
print(f"AC + BC overlap:         {len(ac_pays & bc_pays)}")

bi_orders = orders[orders["label"]=="benign_independent"]
tier1 = bi_orders["payout_id"].str.startswith("PAY_SOLO_").sum()
tier2 = (~bi_orders["payout_id"].str.startswith("PAY_SOLO_")).sum()
print(f"BI Tier1 solo orders:    {tier1} ({100*tier1/len(bi_orders):.1f}%)")
print(f"BI Tier2 shared orders:  {tier2} ({100*tier2/len(bi_orders):.1f}%)")