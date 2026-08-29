
# ── AC ring generators ────────────────────────────────────────────────────────

def _sample_ring_formation_days(n_rings, r, ensure_late_fraction=0.20,
                                 ring_type="promo_return", max_start_day=75):
    """
    [A3] Sample ring_formation_start_day for n_rings rings.
    Ensures >= ensure_late_fraction of rings start in val/test window (day >= 55).
    For referral-farming rings, start day is fixed in test window (>= 73).
    """
    if ring_type == "referral":
        # All referral-farming rings start in test window
        start_days = [int(r.integers(73, max(74, SIM_DAYS - 8 + 1))) for _ in range(n_rings)]
        return start_days

    # For promo/return: ensure >= 20% start at day >= 55
    n_late = max(int(np.ceil(n_rings * ensure_late_fraction)), 1 if n_rings > 0 else 0)
    n_early = n_rings - n_late

    early_starts = [int(r.integers(1, 55)) for _ in range(n_early)]
    late_starts  = [int(r.integers(55, max_start_day + 1)) for _ in range(n_late)]

    all_starts = early_starts + late_starts
    r.shuffle(all_starts)
    return all_starts


def generate_promo_return_rings(ac_accounts, pools, r, events_out, labels_out,
                                 rings_out, order_ctr, session_ctr,
                                 n_promo_rings, n_return_rings,
                                 counterfactual_varied_payout=True):
    """
    [A3,A2,A6b] Generate promo-abuse and return-abuse rings.
    Promo ring accounts: 50% of AC
    Return ring accounts: 30% of AC (referral-farming gets remaining 20%)
    """
    ac_pay_pool = pools["ac_payout_pool"]    # upper 20% of payouts [1440:1800]
    dev_pool    = pools["device_ids"][int(N_DEVICES * 0.40):]   # upper band
    ip_pool     = pools["ip_ids"][int(N_IPS * 0.30):]           # upper band
    instr_pool  = pools["instr_ids"][int(N_INSTRUMENTS * 0.40):]

    total_rings = n_promo_rings + n_return_rings

    # Partition ac_accounts into promo and return ring pools
    # (caller ensures correct fractions)
    n_promo_accs  = int(len(ac_accounts) * (n_promo_rings / max(total_rings, 1)))
    promo_accs    = ac_accounts[:n_promo_accs]
    return_accs   = ac_accounts[n_promo_accs:]

    promo_groups  = partition_into_groups(promo_accs, 5, 25, r)
    return_groups = partition_into_groups(return_accs, 3, 12, r)

    # [A3] Formation days -- ensure >= 20% start late
    all_groups    = promo_groups + return_groups
    group_types   = (["promo"] * len(promo_groups) + ["return"] * len(return_groups))
    formation_starts = _sample_ring_formation_days(len(all_groups), r,
                                                    ensure_late_fraction=0.20,
                                                    ring_type="promo_return")

    # [A6b] varied_payout_ac: 10% of rings get per-member unique payouts
    n_varied = max(0, int(len(all_groups) * 0.10))
    varied_ring_indices = set(
        r.choice(len(all_groups), size=n_varied, replace=False).tolist()
    ) if n_varied > 0 else set()

    ring_id_counter = [0]

    for g_idx, (grp, ring_type, start_day) in enumerate(
            zip(all_groups, group_types, formation_starts)):
        ring_id_counter[0] += 1
        ring_id = f"{'PROMO' if ring_type == 'promo' else 'RETURN'}_{ring_id_counter[0]:03d}"

        is_varied = g_idx in varied_ring_indices

        # Shared ring infra
        n_shared_dev   = int(r.integers(1, 4))
        n_shared_ip    = int(r.integers(1, 6))
        n_shared_pay   = int(r.integers(1, 4))
        ring_devs      = safe_sample(dev_pool, n_shared_dev, r, replace=True)
        ring_ips       = safe_sample(ip_pool, n_shared_ip, r, replace=True)
        ring_pays      = safe_sample(ac_pay_pool, n_shared_pay, r, replace=True)
        ring_instrs    = safe_sample(instr_pool, int(r.integers(1, 4)), r, replace=True)

        # Ring promo code (same for all members)
        promo_code = f"PROMO_RING_{ring_id_counter[0]:03d}" if ring_type == "promo" else None

        # [A2] 10% of members in promo rings are sleeper accounts
        n_sleepers = max(0, int(len(grp) * 0.10)) if ring_type == "promo" else 0
        sleeper_set = set(grp[-n_sleepers:]) if n_sleepers > 0 else set()

        complete_day = min(start_day + int(r.integers(3, 20)), SIM_DAYS)

        for m_idx, acc in enumerate(grp):
            is_sleeper = acc in sleeper_set
            is_varied_ac = is_varied

            labels_out[acc]["label_true"]     = "abusive_coordinated"
            labels_out[acc]["partial_signal"]  = is_sleeper
            labels_out[acc]["counterfactual_subset"] = (
                "varied_payout_ac" if is_varied_ac else None
            )

            # [A6b] varied_payout_ac: each member gets unique payout
            if is_varied_ac:
                member_pay = f"PAY_VARIED_{ring_id_counter[0]:03d}_{m_idx:02d}"
            else:
                member_pay = r.choice(ring_pays)

            # Activation: staggered within burst window (2-48h offsets from start_day)
            stagger_hours = float(np.clip(r.normal(12, 6), 2, 48))
            burst_ts = (SIM_START_TS + (start_day - 1) * SECONDS_PER_DAY
                        + int(stagger_hours * 3600) * m_idx)

            rings_out.append({
                "ring_id": ring_id, "ring_type": ring_type, "account_id": acc,
                "burst_day": start_day,
                "ring_formation_start_day": start_day,
                "ring_formation_complete_day": complete_day,
                "is_sleeper": is_sleeper,
                "is_varied_payout": is_varied_ac,
            })

            # Events: ring burst orders
            n_ring_orders = int(r.integers(2, 8))
            # Sleepers also have independent orders
            n_indep_orders = int(r.integers(3, 12)) if is_sleeper else 0

            alt_instr = r.choice(ring_instrs) if ring_type == "return" else r.choice(ring_instrs)

            for j in range(n_ring_orders):
                order_ts = burst_ts + j * int(r.integers(300, 7200))
                order_day = max(1, min(SIM_DAYS, (order_ts - SIM_START_TS) // SECONDS_PER_DAY + 1))
                order_ts_clamped = ts_in_day(r, order_day)

                use_promo = (promo_code is not None and r.random() < 0.90)
                order_ctr[0] += 1; session_ctr[0] += 1
                oid = new_order_id(order_ctr); sid = new_session_id(session_ctr)
                amount = max(50.0, float(r.lognormal(5.8, 0.3)))

                events_out.append({
                    "account_id": acc, "session_id": sid,
                    "device_id": r.choice(ring_devs), "ip_id": r.choice(ring_ips),
                    "instrument_id": alt_instr, "payout_id": member_pay,
                    "order_id": oid, "amount": amount,
                    "promo_code": promo_code if use_promo else None,
                    "event_type": "order_placed", "timestamp": order_ts_clamped,
                    "ring_id": ring_id, "referrer_id": None,
                })

                # Return-abuse: add return event 5-15 days later
                if ring_type == "return" and r.random() < 0.80:
                    ret_day = min(SIM_DAYS, order_day + int(r.integers(5, 16)))
                    ret_ts  = ts_in_day(r, ret_day)
                    order_ctr[0] += 1; session_ctr[0] += 1
                    events_out.append({
                        "account_id": acc, "session_id": new_session_id(session_ctr),
                        "device_id": r.choice(ring_devs), "ip_id": r.choice(ring_ips),
                        "instrument_id": alt_instr, "payout_id": member_pay,
                        "order_id": new_order_id(order_ctr), "amount": -amount,
                        "promo_code": None, "event_type": "order_returned",
                        "timestamp": ret_ts, "ring_id": ring_id, "referrer_id": None,
                    })

            # Sleeper independent orders (spread randomly, NOT in burst window)
            for _ in range(n_indep_orders):
                day = int(r.integers(1, SIM_DAYS + 1))
                ts  = ts_in_day(r, day)
                order_ctr[0] += 1; session_ctr[0] += 1
                oid = new_order_id(order_ctr); sid = new_session_id(session_ctr)
                events_out.append({
                    "account_id": acc, "session_id": sid,
                    "device_id": r.choice(pools["device_ids"][:int(N_DEVICES * 0.60)]),
                    "ip_id": r.choice(pools["ip_ids"][:int(N_IPS * 0.60)]),
                    "instrument_id": r.choice(pools["instr_ids"][:int(N_INSTRUMENTS * 0.60)]),
                    "payout_id": member_pay,
                    "order_id": oid, "amount": max(50.0, float(r.lognormal(5.5, 1.0))),
                    "promo_code": None, "event_type": "order_placed",
                    "timestamp": ts, "ring_id": None, "referrer_id": None,
                })

            # Intra-ring referral chain (dense)
            if m_idx > 0 and r.random() > RING_EDGE_UNOBSERVED_RATE:
                ref_acc = grp[m_idx - 1]
                ref_ts  = burst_ts - int(r.integers(3600, 86400))
                events_out.append({
                    "account_id": acc, "session_id": new_session_id(session_ctr),
                    "device_id": r.choice(ring_devs), "ip_id": r.choice(ring_ips),
                    "instrument_id": None, "payout_id": None, "order_id": None,
                    "amount": None, "promo_code": None, "event_type": "referral",
                    "timestamp": ref_ts, "ring_id": ring_id, "referrer_id": ref_acc,
                })

    return ring_id_counter


def generate_referral_farming_rings(ref_accounts, pools, r, events_out,
                                     labels_out, rings_out, order_ctr,
                                     session_ctr, ring_id_counter):
    """
    [A5] Referral-farming rings -- TEST WINDOW ONLY (day >= 73).
    Dense referral chains, minimal shared payment infrastructure.
    Structurally distinct: high referral_degree, low shared_payout_degree.
    """
    print(f"Generating {len(ref_accounts)} referral-farming ring events (test window only)...")

    # Use general device/IP pools (low sharing by design)
    dev_pool   = pools["device_ids"]
    ip_pool    = pools["ip_ids"]
    instr_pool = pools["instr_ids"]

    ref_groups = partition_into_groups(ref_accounts, 8, 30, r)

    for grp in ref_groups:
        ring_id_counter[0] += 1
        ring_id = f"REFARM_{ring_id_counter[0]:03d}"

        # [A3] Formation starts in test window (day 73+)
        start_day    = int(r.integers(73, max(74, SIM_DAYS - 7)))
        complete_day = min(SIM_DAYS, start_day + int(r.integers(5, 18)))

        # Minimal shared infra (each member uses own device/IP)
        # Referral chain: member i refers member i+1
        for m_idx, acc in enumerate(grp):
            labels_out[acc]["label_true"]     = "abusive_coordinated"
            labels_out[acc]["partial_signal"]  = False
            labels_out[acc]["counterfactual_subset"] = None

            # Each member uses own unique payout (no structural payout sharing)
            own_pay   = f"PAY_REFARM_{ring_id_counter[0]:03d}_{m_idx:02d}"
            own_dev   = r.choice(dev_pool)
            own_ip    = r.choice(ip_pool)
            own_instr = r.choice(instr_pool)

            rings_out.append({
                "ring_id": ring_id, "ring_type": "referral_farming", "account_id": acc,
                "burst_day": start_day,
                "ring_formation_start_day": start_day,
                "ring_formation_complete_day": complete_day,
                "is_sleeper": False, "is_varied_payout": False,
            })

            # Events: mostly referrals (the abuse pattern), few orders
            n_orders = int(r.integers(1, 5))
            for j in range(n_orders):
                day = int(r.integers(start_day, SIM_DAYS + 1))
                ts  = ts_in_day(r, day)
                order_ctr[0] += 1; session_ctr[0] += 1
                oid = new_order_id(order_ctr); sid = new_session_id(session_ctr)
                events_out.append({
                    "account_id": acc, "session_id": sid,
                    "device_id": own_dev, "ip_id": own_ip,
                    "instrument_id": own_instr, "payout_id": own_pay,
                    "order_id": oid, "amount": max(50.0, float(r.lognormal(5.0, 0.8))),
                    "promo_code": None, "event_type": "order_placed",
                    "timestamp": ts, "ring_id": ring_id, "referrer_id": None,
                })

            # Dense referral chain: each member refers next (core abuse pattern)
            if m_idx > 0:
                referrer = grp[m_idx - 1]
                ref_ts   = ts_in_day(r, start_day)
                events_out.append({
                    "account_id": acc, "session_id": new_session_id(session_ctr),
                    "device_id": own_dev, "ip_id": own_ip,
                    "instrument_id": None, "payout_id": None, "order_id": None,
                    "amount": None, "promo_code": None, "event_type": "referral",
                    "timestamp": ref_ts, "ring_id": ring_id, "referrer_id": referrer,
                })

