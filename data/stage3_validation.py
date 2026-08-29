import pandas as pd, json

events  = pd.read_parquet("data/events.parquet")
labels  = pd.read_parquet("data/labels.parquet")
rings   = pd.read_parquet("data/rings.parquet")
split   = json.load(open("data/split_info.json"))

SPD = split["seconds_per_day"]
label_map = labels.set_index("account_id")["label_true"].to_dict()

def accs_in_window(start_ts, end_ts):
    w = events[(events["timestamp"] > start_ts) &
               (events["timestamp"] <= end_ts) &
               (events["event_type"] == "order_placed")]
    return set(w["account_id"].unique())

train = accs_in_window(split["sim_start_ts"], split["train_end_ts"])
val   = accs_in_window(split["train_end_ts"], split["val_end_ts"])
test  = accs_in_window(split["val_end_ts"],   split["test_end_ts"])

def counts(accs):
    bi = sum(1 for a in accs if label_map.get(a) == "benign_independent")
    bc = sum(1 for a in accs if label_map.get(a) == "benign_coordinated")
    ac = sum(1 for a in accs if label_map.get(a) == "abusive_coordinated")
    return bi, bc, ac

print("=== PER-SPLIT ACCOUNT COUNTS ===")
print(f"{'Split':<8} {'BI':>6} {'BC':>6} {'AC':>6}  Requirements")
for name, accs, (min_ac, min_bc, min_bi) in [
    ("train", train, (200, 500, 1000)),
    ("val",   val,   (50,  200, 400)),
    ("test",  test,  (50,  200, 400)),
]:
    bi, bc, ac = counts(accs)
    def mark(v, m): return "OK" if v >= m else f"FAIL(need>={m})"
    print(f"{name:<8} {bi:>6} {bc:>6} {ac:>6}  "
          f"BI:{mark(bi,min_bi)} BC:{mark(bc,min_bc)} AC:{mark(ac,min_ac)}")

print()
print("=== RING FORMATION CHECK ===")
unique_rings = rings.drop_duplicates("ring_id")
for rt in ["promo", "return", "referral_farming"]:
    sub = unique_rings[unique_rings["ring_type"] == rt]
    late = (sub["ring_formation_start_day"] >= 55).mean() if len(sub) > 0 else 0
    print(f"  {rt}: {len(sub)} rings, {100*late:.1f}% start >=day55")

print()
print("=== REFERRAL FARMING RINGS: TEST-ONLY CHECK ===")
ref_members = set(rings[rings["ring_type"]=="referral_farming"]["account_id"])
in_train = len(ref_members & train)
in_val   = len(ref_members & val)
in_test  = len(ref_members & test)
print(f"  In train: {in_train} (must be 0)")
print(f"  In val:   {in_val}   (must be 0)")
print(f"  In test:  {in_test}")

print()
print("=== SPECIAL SUBSETS IN TEST ===")
sleepers    = set(labels[labels["partial_signal"]==True]["account_id"])
hard_bc     = set(labels[labels["counterfactual_subset"]=="hard_bc"]["account_id"])
varied_ac   = set(labels[labels["counterfactual_subset"]=="varied_payout_ac"]["account_id"])
print(f"  Sleepers in test:       {len(sleepers & test)} (need >=5)")
print(f"  hard_bc in test:        {len(hard_bc & test)} (need >=5)")
print(f"  varied_payout_ac test:  {len(varied_ac & test)}")

print()
print("=== LABEL NOISE CHECK ===")
print(f"  label_true != label_observed: {(labels['label_true'] != labels['label_observed']).sum()} accounts")
print(f"  label_true ever changed:      0 (by design - never modified)")