"""Payout signal validation script - run after simulator to check signal strength."""
import pandas as pd, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

events = pd.read_parquet("data/events.parquet")
labels = pd.read_parquet("data/labels.parquet")
orders = events[events["event_type"] == "order_placed"].dropna(subset=["payout_id"])
label_map = labels.set_index("account_id")["label"]
orders["label"] = orders["account_id"].map(label_map)

payout_label_sets = orders.groupby("payout_id")["label"].apply(set)
payout_acc_counts = orders.groupby("payout_id")["account_id"].nunique()

ac_payouts = set(orders[orders["label"] == "abusive_coordinated"]["payout_id"].unique())
bc_payouts = set(orders[orders["label"] == "benign_coordinated"]["payout_id"].unique())
bi_payouts = set(orders[orders["label"] == "benign_independent"]["payout_id"].unique())

print(f"AC-exclusive payouts:  {len(ac_payouts - bc_payouts - bi_payouts)} / {len(ac_payouts)}")
print(f"BC-exclusive payouts:  {len(bc_payouts - ac_payouts - bi_payouts)} / {len(bc_payouts)}")
print(f"BI-exclusive payouts:  {len(bi_payouts - ac_payouts - bc_payouts)} / {len(bi_payouts)}")
print(f"AC+BI overlap payouts: {len(ac_payouts & bi_payouts)}")
print(f"AC+BC overlap payouts: {len(ac_payouts & bc_payouts)}")

print("\nRing payout sharing (abusive_coordinated):")
ac_orders = orders[orders["label"] == "abusive_coordinated"]
ac_payout_sharing = ac_orders.groupby("payout_id")["account_id"].nunique()
ac_shared = ac_payout_sharing[ac_payout_sharing >= 2]
print(ac_shared.describe())

print("\nBI payout sharing:")
bi_orders = orders[orders["label"] == "benign_independent"]
bi_payout_sharing = bi_orders.groupby("payout_id")["account_id"].nunique()
bi_shared = bi_payout_sharing[bi_payout_sharing >= 1]
print(bi_shared.describe())
bi_unique = (bi_payout_sharing == 1).sum()
print(f"BI accs with uniquely-theirs payout: {bi_unique} / {len(bi_payout_sharing)}")
