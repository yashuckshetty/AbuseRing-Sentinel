
# ── Main entry point ──────────────────────────────────────────────────────────

def generate_dataset(
    seed=DEFAULT_SEED,
    abuse_prevalence=None,
    output_dir=OUTPUT_DIR,
    counterfactual_hard_bc=True,
    counterfactual_varied_payout=True,
    n_accounts=N_ACCOUNTS,
):
    """
    Generate the full AbuseRing Sentinel synthetic dataset.

    Parameters
    ----------
    seed : int
        RNG seed. Deterministic.
    abuse_prevalence : dict or None
        {"benign_independent": f, "benign_coordinated": f, "abusive_coordinated": f}
        Must sum to 1.0. Default: 0.60/0.25/0.15.
    output_dir : str or Path
        Directory to write parquet files and split_info.json.
    counterfactual_hard_bc : bool
        If True, generate hard_bc counterfactual subset (15% of BC family groups).
    counterfactual_varied_payout : bool
        If True, generate varied_payout_ac counterfactual (10% of AC rings).
    n_accounts : int
        Total account count.
    """
    if abuse_prevalence is None:
        abuse_prevalence = DEFAULT_PREVALENCE.copy()

    total = sum(abuse_prevalence.values())
    assert abs(total - 1.0) < 1e-6, f"abuse_prevalence must sum to 1.0, got {total}"

    r = make_rng(seed)
    print(f"=== AbuseRing Sentinel Simulator v2.0 ===")
    print(f"Seed: {seed} | Output: {output_dir}")
    print(f"Prevalence: {abuse_prevalence}")

    # Account counts
    n_bi = int(n_accounts * abuse_prevalence["benign_independent"])
    n_bc = int(n_accounts * abuse_prevalence["benign_coordinated"])
    n_ac = n_accounts - n_bi - n_bc
    print(f"Accounts: {n_bi} BI | {n_bc} BC | {n_ac} AC")

    # AC subtype split: 50% promo, 30% return, 20% referral-farming
    n_ac_promo  = int(n_ac * 0.50)
    n_ac_return = int(n_ac * 0.30)
    n_ac_ref    = n_ac - n_ac_promo - n_ac_return

    # Ring counts (rough: assume avg ring size 8 for promo, 6 for return, 15 for refarm)
    n_promo_rings  = max(1, n_ac_promo  // 8)
    n_return_rings = max(1, n_ac_return // 6)
    n_ref_rings    = max(1, n_ac_ref    // 15)
    print(f"Rings: {n_promo_rings} promo, {n_return_rings} return, {n_ref_rings} referral-farming")

    # Account ID lists
    bi_accounts    = [f"ACC_{i:05d}" for i in range(0, n_bi)]
    bc_accounts    = [f"ACC_{i:05d}" for i in range(n_bi, n_bi + n_bc)]
    ac_promo_accs  = [f"ACC_{i:05d}" for i in range(n_bi + n_bc, n_bi + n_bc + n_ac_promo)]
    ac_return_accs = [f"ACC_{i:05d}" for i in range(n_bi + n_bc + n_ac_promo,
                                                       n_bi + n_bc + n_ac_promo + n_ac_return)]
    ac_ref_accs    = [f"ACC_{i:05d}" for i in range(n_bi + n_bc + n_ac_promo + n_ac_return,
                                                      n_accounts)]

    # Build entity pools
    pools = build_entity_pools(n_accounts, abuse_prevalence, seed + 1)

    # Shared mutable state
    events_out  = []
    rings_out   = []
    labels_dict = {acc: {} for acc in bi_accounts + bc_accounts +
                   ac_promo_accs + ac_return_accs + ac_ref_accs}

    # Accounts table (created_ts = SIM_START_TS for all, realistic enough)
    accounts_list = [
        {"account_id": acc, "created_ts": SIM_START_TS + int(r.integers(0, 3 * SECONDS_PER_DAY))}
        for acc in (bi_accounts + bc_accounts + ac_promo_accs + ac_return_accs + ac_ref_accs)
    ]

    # Generate BI
    order_ctr, session_ctr = generate_bi_accounts(
        bi_accounts, pools, r, events_out, labels_dict
    )

    # Generate BC
    generate_bc_accounts(
        bc_accounts, pools, r, events_out, labels_dict,
        counterfactual_hard_bc=counterfactual_hard_bc
    )

    # Generate AC promo + return rings
    ac_promo_return = list(r.permutation(ac_promo_accs + ac_return_accs))
    # Re-split after permutation to preserve approximate ratio
    n_pr = len(ac_promo_accs)
    ring_id_counter = generate_promo_return_rings(
        ac_accs_promo=ac_promo_accs,
        ac_accs_return=ac_return_accs,
        pools=pools, r=r,
        events_out=events_out, labels_out=labels_dict, rings_out=rings_out,
        order_ctr=order_ctr, session_ctr=session_ctr,
        n_promo_rings=n_promo_rings, n_return_rings=n_return_rings,
        counterfactual_varied_payout=counterfactual_varied_payout,
    )

    # Generate referral-farming rings (test-only)
    generate_referral_farming_rings(
        ac_ref_accs, pools, r, events_out, labels_dict,
        rings_out, order_ctr, session_ctr, ring_id_counter,
    )

    # [A1] Apply label noise (only to label_observed, never label_true)
    noisy_accs = apply_label_noise(labels_dict, r)
    print(f"Label noise applied: {len(noisy_accs)} accounts have label_observed != label_true")

    # Split info
    split_info = {
        "sim_start_ts": SIM_START_TS,
        "train_end_ts": SIM_START_TS + TRAIN_END_DAY * SECONDS_PER_DAY,
        "val_end_ts":   SIM_START_TS + VAL_END_DAY * SECONDS_PER_DAY,
        "test_end_ts":  SIM_START_TS + SIM_DAYS * SECONDS_PER_DAY,
        "train_end_day": TRAIN_END_DAY,
        "val_end_day":   VAL_END_DAY,
        "seconds_per_day": SECONDS_PER_DAY,
        "seed": seed,
        "n_accounts": n_accounts,
        "abuse_prevalence": abuse_prevalence,
        "counterfactual_hard_bc": counterfactual_hard_bc,
        "counterfactual_varied_payout": counterfactual_varied_payout,
        "_note": "Temporal split -- no random splits permitted.",
    }

    # Save outputs
    print("\nBuilding DataFrames and saving...")
    events_df  = None  # built inside save_outputs
    rings_df   = pd.DataFrame(rings_out) if rings_out else pd.DataFrame()
    save_outputs(events_out, labels_dict, rings_out, accounts_list, split_info, output_dir)

    # Reload for validation
    events_df  = pd.read_parquet(Path(output_dir) / "events.parquet")
    labels_df  = pd.read_parquet(Path(output_dir) / "labels.parquet")
    rings_df2  = pd.read_parquet(Path(output_dir) / "rings.parquet") if rings_out else pd.DataFrame()

    print("\nRunning Stage 3 validation assertions...")
    passed, errors = validate_outputs(events_df, labels_df, rings_df2, split_info)

    # Shortcut detection check [A7c]
    shortcut = check_shortcut_detection(labels_df)

    # Summary
    print(f"\n=== Done ===")
    print(f"  Events:         {len(events_df):,}")
    print(f"  Accounts:       {len(labels_df):,}")
    print(f"  Label breakdown: {labels_df['label_true'].value_counts().to_dict()}")
    print(f"  Sleepers:        {labels_df['partial_signal'].sum()}")
    print(f"  hard_bc:         {(labels_df['counterfactual_subset'] == 'hard_bc').sum()}")
    print(f"  varied_payout_ac: {(labels_df['counterfactual_subset'] == 'varied_payout_ac').sum()}")
    print(f"  Unique rings:    {rings_df2['ring_id'].nunique() if len(rings_df2) > 0 else 0}")
    if len(rings_df2) > 0:
        late = (rings_df2.drop_duplicates('ring_id')['ring_formation_start_day'] >= 55).mean()
        print(f"  Rings starting >=day55: {late*100:.1f}%")
    print(f"  Shortcut check: AUC={shortcut.get('auc')} | Pass={shortcut.get('pass')}")
    print(f"  Validation: {'PASSED' if passed else 'WARNINGS -- see above'}")

    return {"passed": passed, "errors": errors, "shortcut": shortcut}


# ── Shim for old generate_promo_return_rings call signature ──────────────────

def generate_promo_return_rings(ac_accs_promo, ac_accs_return, pools, r,
                                 events_out, labels_out, rings_out,
                                 order_ctr, session_ctr,
                                 n_promo_rings, n_return_rings,
                                 counterfactual_varied_payout=True):
    """Wrapper calling internal implementation with correct arg order."""
    # Partition into groups
    promo_groups  = partition_into_groups(ac_accs_promo, 5, 25, r)
    return_groups = partition_into_groups(ac_accs_return, 3, 12, r)
    all_groups    = promo_groups + return_groups
    group_types   = (["promo"] * len(promo_groups) + ["return"] * len(return_groups))
    formation_starts = _sample_ring_formation_days(len(all_groups), r,
                                                    ensure_late_fraction=0.20,
                                                    ring_type="promo_return")
    n_varied = max(0, int(len(all_groups) * 0.10))
    varied_idx = set(r.choice(len(all_groups), size=n_varied, replace=False).tolist()
                     ) if n_varied > 0 and counterfactual_varied_payout else set()

    ac_pay_pool  = pools["ac_payout_pool"]
    dev_pool     = pools["device_ids"][int(N_DEVICES * 0.40):]
    ip_pool      = pools["ip_ids"][int(N_IPS * 0.30):]
    instr_pool   = pools["instr_ids"][int(N_INSTRUMENTS * 0.40):]

    ring_id_counter = [0]

    for g_idx, (grp, ring_type, start_day) in enumerate(
            zip(all_groups, group_types, formation_starts)):
        ring_id_counter[0] += 1
        ring_id = f"{'PROMO' if ring_type == 'promo' else 'RETURN'}_{ring_id_counter[0]:03d}"

        is_varied = g_idx in varied_idx
        n_shared_dev = int(r.integers(1, 4))
        n_shared_ip  = int(r.integers(1, 6))
        n_shared_pay = int(r.integers(1, 4))
        ring_devs  = safe_sample(dev_pool, n_shared_dev, r, replace=True)
        ring_ips   = safe_sample(ip_pool, n_shared_ip, r, replace=True)
        ring_pays  = safe_sample(ac_pay_pool, n_shared_pay, r, replace=True)
        ring_instrs = safe_sample(instr_pool, int(r.integers(1, 4)), r, replace=True)
        promo_code = f"PROMO_RING_{ring_id_counter[0]:03d}" if ring_type == "promo" else None

        n_sleepers = max(0, int(len(grp) * 0.10)) if ring_type == "promo" else 0
        sleeper_set = set(grp[-n_sleepers:]) if n_sleepers > 0 else set()
        complete_day = min(start_day + int(r.integers(3, 20)), SIM_DAYS)

        for m_idx, acc in enumerate(grp):
            is_sleeper = acc in sleeper_set
            labels_out[acc]["label_true"]     = "abusive_coordinated"
            labels_out[acc]["partial_signal"]  = is_sleeper
            labels_out[acc]["counterfactual_subset"] = (
                "varied_payout_ac" if is_varied else None)

            member_pay = (f"PAY_VARIED_{ring_id_counter[0]:03d}_{m_idx:02d}"
                          if is_varied else r.choice(ring_pays))

            stagger_hours = float(np.clip(r.normal(12, 6), 2, 48))
            burst_ts = (SIM_START_TS + (start_day - 1) * SECONDS_PER_DAY
                        + int(stagger_hours * 3600) * m_idx)

            rings_out.append({
                "ring_id": ring_id, "ring_type": ring_type, "account_id": acc,
                "burst_day": start_day, "ring_formation_start_day": start_day,
                "ring_formation_complete_day": complete_day,
                "is_sleeper": is_sleeper, "is_varied_payout": is_varied,
            })

            n_ring_orders = int(r.integers(2, 8))
            n_indep_orders = int(r.integers(3, 12)) if is_sleeper else 0
            alt_instr = r.choice(ring_instrs)

            for j in range(n_ring_orders):
                order_day = max(1, min(SIM_DAYS, start_day + j))
                order_ts  = ts_in_day(r, order_day)
                use_promo = promo_code is not None and r.random() < 0.90
                order_ctr[0] += 1; session_ctr[0] += 1
                events_out.append({
                    "account_id": acc, "session_id": new_session_id(session_ctr),
                    "device_id": r.choice(ring_devs), "ip_id": r.choice(ring_ips),
                    "instrument_id": alt_instr, "payout_id": member_pay,
                    "order_id": new_order_id(order_ctr), "amount": max(50.0, float(r.lognormal(5.8, 0.3))),
                    "promo_code": promo_code if use_promo else None,
                    "event_type": "order_placed", "timestamp": order_ts,
                    "ring_id": ring_id, "referrer_id": None,
                })
                if ring_type == "return" and r.random() < 0.80:
                    ret_day = min(SIM_DAYS, order_day + int(r.integers(5, 16)))
                    order_ctr[0] += 1; session_ctr[0] += 1
                    events_out.append({
                        "account_id": acc, "session_id": new_session_id(session_ctr),
                        "device_id": r.choice(ring_devs), "ip_id": r.choice(ring_ips),
                        "instrument_id": alt_instr, "payout_id": member_pay,
                        "order_id": new_order_id(order_ctr), "amount": -max(50.0, float(r.lognormal(5.8, 0.3))),
                        "promo_code": None, "event_type": "order_returned",
                        "timestamp": ts_in_day(r, ret_day),
                        "ring_id": ring_id, "referrer_id": None,
                    })

            for _ in range(n_indep_orders):
                day = int(r.integers(1, SIM_DAYS + 1))
                order_ctr[0] += 1; session_ctr[0] += 1
                events_out.append({
                    "account_id": acc, "session_id": new_session_id(session_ctr),
                    "device_id": r.choice(pools["device_ids"][:int(N_DEVICES*0.60)]),
                    "ip_id": r.choice(pools["ip_ids"][:int(N_IPS*0.60)]),
                    "instrument_id": r.choice(pools["instr_ids"][:int(N_INSTRUMENTS*0.60)]),
                    "payout_id": member_pay, "order_id": new_order_id(order_ctr),
                    "amount": max(50.0, float(r.lognormal(5.5, 1.0))),
                    "promo_code": None, "event_type": "order_placed",
                    "timestamp": ts_in_day(r, day), "ring_id": None, "referrer_id": None,
                })

            if m_idx > 0 and r.random() > RING_EDGE_UNOBSERVED_RATE:
                ref_ts = burst_ts - int(r.integers(3600, 86400))
                events_out.append({
                    "account_id": acc, "session_id": new_session_id(session_ctr),
                    "device_id": r.choice(ring_devs), "ip_id": r.choice(ring_ips),
                    "instrument_id": None, "payout_id": None, "order_id": None,
                    "amount": None, "promo_code": None, "event_type": "referral",
                    "timestamp": ref_ts, "ring_id": ring_id, "referrer_id": grp[m_idx - 1],
                })

    return ring_id_counter


if __name__ == "__main__":
    result = generate_dataset(seed=DEFAULT_SEED)
    sys.exit(0 if result["passed"] else 1)
