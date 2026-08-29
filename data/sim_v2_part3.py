
# ── BC account generator ──────────────────────────────────────────────────────

def generate_bc_accounts(bc_accounts, pools, r, events_out, labels_out,
                          counterfactual_hard_bc=True):
    """
    Generate benign-coordinated account events.
    Two subtypes: family groups (70%) and office groups (30%).
    [A6a] hard_bc: 15% of family groups get one shared payout (still label_true=BC).
    """
    print(f"Generating {len(bc_accounts)} benign-coordinated events...")

    dev_pool   = pools["device_ids"][:int(N_DEVICES * 0.70)]
    ip_pool    = pools["ip_ids"][:int(N_IPS * 0.70)]
    instr_pool = pools["instr_ids"][:int(N_INSTRUMENTS * 0.70)]
    bc_pay_pool = pools["bc_payout_pool"]    # [900:1260] band

    n_bc = len(bc_accounts)
    n_family = int(n_bc * 0.70)
    n_office = n_bc - n_family

    family_accs = bc_accounts[:n_family]
    office_accs = bc_accounts[n_family:]

    # Partition into groups
    family_groups = partition_into_groups(family_accs, 3, 8, r)
    office_groups = partition_into_groups(office_accs, 5, 20, r)

    order_ctr = [1_000_000]; session_ctr = [1_000_000]

    # --- Family groups ---
    for grp_idx, grp in enumerate(family_groups):
        # Shared infra for this family
        n_shared_dev   = int(r.integers(1, 3))
        n_shared_ip    = int(r.integers(1, 3))
        n_shared_instr = int(r.integers(1, 4))
        shared_devs   = safe_sample(dev_pool, n_shared_dev, r, replace=True)
        shared_ips    = safe_sample(ip_pool, n_shared_ip, r, replace=True)
        shared_instrs = safe_sample(instr_pool, n_shared_instr, r, replace=True)
        base_amount   = float(r.lognormal(5.8, 0.5))

        # [A6a] 15% of family groups get one shared payout (hard counterfactual)
        is_hard_bc = (counterfactual_hard_bc and
                      grp_idx < max(1, int(len(family_groups) * 0.15)))
        shared_payout_for_hard_bc = r.choice(bc_pay_pool) if is_hard_bc else None

        # Referral chain: each member refers the next (gifted referral)
        for i, acc in enumerate(grp):
            labels_out[acc]["label_true"]    = "benign_coordinated"
            labels_out[acc]["partial_signal"] = False
            labels_out[acc]["counterfactual_subset"] = "hard_bc" if is_hard_bc else None

            # Each member has their own payout (or shared one if hard_bc)
            own_payout = r.choice(bc_pay_pool)
            payout_id = shared_payout_for_hard_bc if is_hard_bc else own_payout

            n_orders = int(r.integers(3, 20))
            referrer = grp[i - 1] if i > 0 else None

            for j in range(n_orders):
                day = int(r.integers(1, SIM_DAYS + 1))
                ts  = ts_in_day(r, day, hour_bias="evening")
                dev   = r.choice(shared_devs)
                ip    = r.choice(shared_ips)
                instr = r.choice(shared_instrs)
                order_ctr[0] += 1; session_ctr[0] += 1
                oid = new_order_id(order_ctr); sid = new_session_id(session_ctr)
                amount = max(50.0, float(r.normal(base_amount, base_amount * 0.3)))
                # 15% chance of coincidental promo use (realistic label noise)
                pc = None
                if r.random() < 0.15:
                    pc = f"PROMO_FAM_{int(r.integers(1, 20))}"
                events_out.append({
                    "account_id": acc, "session_id": sid, "device_id": dev,
                    "ip_id": ip, "instrument_id": instr, "payout_id": payout_id,
                    "order_id": oid, "amount": amount, "promo_code": pc,
                    "event_type": "order_placed", "timestamp": ts,
                    "ring_id": None, "referrer_id": None,
                })
                # First order: referral event from previous member
                if j == 0 and referrer:
                    ref_ts = ts - int(r.integers(60, 3600))
                    events_out.append({
                        "account_id": acc, "session_id": new_session_id(session_ctr),
                        "device_id": dev, "ip_id": ip, "instrument_id": None,
                        "payout_id": None, "order_id": None, "amount": None,
                        "promo_code": None, "event_type": "referral",
                        "timestamp": ref_ts, "ring_id": None, "referrer_id": referrer,
                    })

    # --- Office groups ---
    for grp in office_groups:
        # Shared IPs only (corporate NAT); independent devices, instruments, payouts
        n_shared_ip = int(r.integers(1, 4))
        shared_ips  = safe_sample(ip_pool[:int(N_IPS * 0.40)], n_shared_ip, r, replace=True)

        for acc in grp:
            labels_out[acc]["label_true"]    = "benign_coordinated"
            labels_out[acc]["partial_signal"] = False
            labels_out[acc]["counterfactual_subset"] = None

            own_dev   = r.choice(dev_pool)
            own_instr = r.choice(instr_pool)
            own_pay   = r.choice(bc_pay_pool)

            n_orders = int(r.integers(2, 12))
            for _ in range(n_orders):
                day = int(r.integers(1, SIM_DAYS + 1))
                ts  = ts_in_day(r, day, hour_bias="business")
                ip  = r.choice(shared_ips)
                order_ctr[0] += 1; session_ctr[0] += 1
                oid = new_order_id(order_ctr); sid = new_session_id(session_ctr)
                amount = max(50.0, float(r.lognormal(5.5, 1.0)))
                events_out.append({
                    "account_id": acc, "session_id": sid, "device_id": own_dev,
                    "ip_id": ip, "instrument_id": own_instr, "payout_id": own_pay,
                    "order_id": oid, "amount": amount, "promo_code": None,
                    "event_type": "order_placed", "timestamp": ts,
                    "ring_id": None, "referrer_id": None,
                })

    return order_ctr, session_ctr

