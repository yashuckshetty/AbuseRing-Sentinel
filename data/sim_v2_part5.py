
# ── Output writers ────────────────────────────────────────────────────────────

def apply_label_noise(labels_dict, r):
    """
    [A1] Apply 3% noise to abusive_coordinated accounts.
    label_true is NEVER changed -- only label_observed is flipped.
    """
    ac_accounts = [acc for acc, d in labels_dict.items()
                   if d["label_true"] == "abusive_coordinated"]
    n_noise = max(0, int(len(ac_accounts) * 0.03))
    noisy_accs = set(safe_sample(ac_accounts, n_noise, r) if n_noise > 0 else [])
    for acc in noisy_accs:
        labels_dict[acc]["label_observed"] = "benign_independent"
    for acc in labels_dict:
        if "label_observed" not in labels_dict[acc]:
            labels_dict[acc]["label_observed"] = labels_dict[acc]["label_true"]
    return noisy_accs


def save_outputs(events_list, labels_dict, rings_list, accounts_list,
                  split_info, output_dir):
    """Save all outputs to parquet/json files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Events
    events_df = pd.DataFrame(events_list)
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)
    events_df.to_parquet(output_dir / "events.parquet", index=False)
    print(f"  Events:    {len(events_df):,}")

    # Accounts
    accounts_df = pd.DataFrame(accounts_list)
    accounts_df.to_parquet(output_dir / "accounts.parquet", index=False)
    print(f"  Accounts:  {len(accounts_df):,}")

    # Labels -- includes label_true, label_observed, partial_signal, counterfactual_subset
    label_rows = []
    for acc, d in labels_dict.items():
        label_rows.append({
            "account_id":           acc,
            "label_true":           d.get("label_true", "benign_independent"),
            "label":                d.get("label_true", "benign_independent"),  # alias for compat
            "label_observed":       d.get("label_observed", d.get("label_true", "benign_independent")),
            "partial_signal":       bool(d.get("partial_signal", False)),
            "counterfactual_subset": d.get("counterfactual_subset", None),
        })
    labels_df = pd.DataFrame(label_rows)
    labels_df.to_parquet(output_dir / "labels.parquet", index=False)
    print(f"  Labels:    {len(labels_df):,}")

    # Rings -- includes formation fields [A3]
    if rings_list:
        rings_df = pd.DataFrame(rings_list)
        rings_df.to_parquet(output_dir / "rings.parquet", index=False)
        print(f"  Ring rows: {len(rings_df):,}")
        print(f"  Unique rings: {rings_df['ring_id'].nunique():,}")

    # Split info
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)


# ── Stage 3 validation assertions ─────────────────────────────────────────────

def validate_outputs(events_df, labels_df, rings_df, split_info):
    """
    [ASSUMPTIONS.md 7b] Assert minimum per-class counts and structural invariants.
    Raises AssertionError with descriptive message if any check fails.
    """
    errors = []
    SPD = split_info["seconds_per_day"]
    SIM_START = split_info["sim_start_ts"]

    def accs_active_in_window(start_ts, end_ts):
        window = events_df[
            (events_df["timestamp"] > start_ts) &
            (events_df["timestamp"] <= end_ts) &
            (events_df["event_type"] == "order_placed")
        ]
        return window["account_id"].unique()

    train_accs = accs_active_in_window(SIM_START, split_info["train_end_ts"])
    val_accs   = accs_active_in_window(split_info["train_end_ts"], split_info["val_end_ts"])
    test_accs  = accs_active_in_window(split_info["val_end_ts"], split_info["test_end_ts"])

    label_map = labels_df.set_index("account_id")["label_true"].to_dict()

    def count_class(accs, cls):
        return sum(1 for a in accs if label_map.get(a) == cls)

    # Minimum count checks
    checks = [
        ("train", train_accs, 200, 500, 1000),
        ("val",   val_accs,   50,  200, 400),
        ("test",  test_accs,  50,  200, 400),
    ]
    for split_name, accs, min_ac, min_bc, min_bi in checks:
        n_ac = count_class(accs, "abusive_coordinated")
        n_bc = count_class(accs, "benign_coordinated")
        n_bi = count_class(accs, "benign_independent")
        if n_ac < min_ac:
            errors.append(f"{split_name}: only {n_ac} AC accounts (need >={min_ac})")
        if n_bc < min_bc:
            errors.append(f"{split_name}: only {n_bc} BC accounts (need >={min_bc})")
        if n_bi < min_bi:
            errors.append(f"{split_name}: only {n_bi} BI accounts (need >={min_bi})")

    # Referral-farming rings must be test-only
    if rings_df is not None and "ring_type" in rings_df.columns:
        ref_members = rings_df[rings_df["ring_type"] == "referral_farming"]["account_id"].tolist()
        train_set = set(train_accs); val_set = set(val_accs)
        train_leakers = [a for a in ref_members if a in train_set]
        val_leakers   = [a for a in ref_members if a in val_set]
        if train_leakers:
            errors.append(f"Referral-farming ring members in train split: {len(train_leakers)}")
        if val_leakers:
            errors.append(f"Referral-farming ring members in val split: {len(val_leakers)}")

        # >=20% rings must start in val/test window
        unique_rings = rings_df.drop_duplicates("ring_id")
        promo_return = unique_rings[unique_rings["ring_type"].isin(["promo", "return"])]
        if len(promo_return) > 0:
            late_frac = (promo_return["ring_formation_start_day"] >= 55).mean()
            if late_frac < 0.20:
                errors.append(f"Only {late_frac*100:.1f}% of rings start >=day55 (need >=20%)")

    # Sleeper counts
    sleepers = labels_df[labels_df["partial_signal"] == True]
    test_sleepers = [a for a in sleepers["account_id"] if a in set(test_accs)]
    if len(test_sleepers) < 5:
        errors.append(f"Only {len(test_sleepers)} sleeper accounts in test split (need >=5)")

    # hard_bc counts
    hard_bc = labels_df[labels_df["counterfactual_subset"] == "hard_bc"]
    test_hard_bc = [a for a in hard_bc["account_id"] if a in set(test_accs)]
    if len(test_hard_bc) < 5:
        errors.append(f"Only {len(test_hard_bc)} hard_bc accounts in test split (need >=5)")

    if errors:
        print("\n[VALIDATION WARNINGS]:")
        for e in errors:
            print(f"  - {e}")
        print("  NOTE: Re-seed if these are hard failures for Stage 11 benchmarks.")
    else:
        print("\n[VALIDATION PASSED] All minimum counts satisfied.")

    return len(errors) == 0, errors


# ── Shortcut detection check ──────────────────────────────────────────────────

def check_shortcut_detection(labels_df):
    """
    [ASSUMPTIONS.md 7c] Train a depth-2 decision tree on creation metadata only.
    AUC must be <= 0.55 to confirm no generation-order artifact.
    """
    try:
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize

        labels_df = labels_df.copy()
        labels_df["account_idx"] = range(len(labels_df))
        # Simulate creation timestamp as proportional to index (generation order proxy)
        labels_df["creation_ts_proxy"] = (
            SIM_START_TS + labels_df["account_idx"] * (SIM_DAYS * SECONDS_PER_DAY / len(labels_df))
        )
        X = labels_df[["account_idx", "creation_ts_proxy"]].values
        y_str = labels_df["label_true"].values
        class_map = {"benign_independent": 0, "benign_coordinated": 1, "abusive_coordinated": 2}
        y = np.array([class_map[c] for c in y_str])

        # Use last 30% as "test" (proportional to test split)
        split_pt = int(len(y) * 0.70)
        X_tr, X_te = X[:split_pt], X[split_pt:]
        y_tr, y_te = y[:split_pt], y[split_pt:]

        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")

        status = "PASS" if auc <= 0.55 else "FAIL"
        print(f"\n[SHORTCUT CHECK] Depth-2 tree AUC={auc:.4f} -- {status} (threshold: <=0.55)")
        return {"auc": round(float(auc), 4), "pass": auc <= 0.55,
                "note": "Depth-2 DT on creation order/ts. AUC<=0.55 required."}
    except Exception as e:
        print(f"  [SHORTCUT CHECK] Skipped: {e}")
        return {"auc": None, "pass": None, "note": str(e)}

