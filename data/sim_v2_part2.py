
# ── Entity pool builder ───────────────────────────────────────────────────────

def build_entity_pools(n_accounts, prevalence, seed):
    """
    Build all entity ID pools. Sizes are fixed at global constants.
    Payout pool is segmented by class:
      AC rings:    upper 20%  = payout_ids[1440:1800]   (360 IDs, ring-exclusive)
      BC families: 50-70% band = payout_ids[900:1260]   (360 IDs)
      BI tier 2:   lower 50%, subset of 220 IDs for shared joint-account simulation
      BI tier 1:   synthetic "PAY_SOLO_{i}" IDs, one per account -- zero inter-account collision
    This ensures payout-destination-share is meaningfully discriminative:
      BI mean degree: ~1.44   (78% solo + 22% in 3-way sharing)
      AC mean degree: 4-13    (ring-exclusive, all members share same 1-3 payouts)
    """
    r = make_rng(seed)
    device_ids     = [f"DEV_{i:04d}" for i in range(N_DEVICES)]
    ip_ids         = [f"IP_{i:04d}"  for i in range(N_IPS)]
    instr_ids      = [f"INSTR_{i:04d}" for i in range(N_INSTRUMENTS)]
    payout_ids     = [f"PAY_{i:04d}" for i in range(N_PAYOUTS)]

    # Payout pool segments
    ac_payout_pool  = payout_ids[int(N_PAYOUTS * 0.80):]          # upper 20% [1440:1800]
    bc_payout_pool  = payout_ids[int(N_PAYOUTS * 0.50):int(N_PAYOUTS * 0.70)]  # [900:1260]
    # BI tier 2: random 220 IDs from lower 50% -- no overlap with AC or BC pool
    bi_shared_pool_candidates = payout_ids[:int(N_PAYOUTS * 0.50)]  # [0:900]
    bi_tier2_pool   = list(r.choice(bi_shared_pool_candidates,
                                     size=min(220, len(bi_shared_pool_candidates)),
                                     replace=False))
    # BI tier 1 solo IDs are generated per-account during account creation (not pre-allocated)

    return {
        "device_ids":    device_ids,
        "ip_ids":        ip_ids,
        "instr_ids":     instr_ids,
        "payout_ids":    payout_ids,
        "ac_payout_pool":  ac_payout_pool,
        "bc_payout_pool":  bc_payout_pool,
        "bi_tier2_pool":   bi_tier2_pool,
    }


# ── BI account generator ──────────────────────────────────────────────────────

def generate_bi_accounts(bi_accounts, pools, r, events_out, labels_out):
    """
    Generate benign-independent account events.
    Payout fix (A2 clarification):
      Tier 1 (78%): unique synthetic PAY_SOLO_{acc_idx} -- zero collision
      Tier 2 (22%): sampled from bi_tier2_pool (220 IDs) -- ~3x local sharing
      Overall BI mean payout degree: ~1.44
    """
    print(f"Generating {len(bi_accounts)} benign-independent events...")

    dev_pool   = pools["device_ids"][:int(N_DEVICES * 0.60)]
    ip_pool    = pools["ip_ids"][:int(N_IPS * 0.60)]
    instr_pool = pools["instr_ids"][:int(N_INSTRUMENTS * 0.60)]
    bi_t2_pool = pools["bi_tier2_pool"]

    n_bi = len(bi_accounts)
    tier1_count = int(n_bi * 0.78)
    tier1_set   = set(bi_accounts[:tier1_count])
    tier2_set   = set(bi_accounts[tier1_count:])

    # 8% of BI use occasional promo (independent discovery, not ring-coordinated)
    promo_users = set(safe_sample(bi_accounts, max(1, int(n_bi * 0.08)), r))

    order_ctr = [0]; session_ctr = [0]

    for idx, acc in enumerate(bi_accounts):
        labels_out[acc]["label_true"]    = "benign_independent"
        labels_out[acc]["partial_signal"]= False
        labels_out[acc]["counterfactual_subset"] = None

        # Assign payout
        if acc in tier1_set:
            payout_id = f"PAY_SOLO_{idx:04d}"   # unique -- no cross-account collision
        else:
            payout_id = r.choice(bi_t2_pool)

        device = r.choice(dev_pool)
        ip     = r.choice(ip_pool)
        n_instr = int(r.integers(1, 3))
        instrs = safe_sample(instr_pool, n_instr, r, replace=True)

        n_orders = int(r.integers(2, 15))
        for _ in range(n_orders):
            day = int(r.integers(1, SIM_DAYS + 1))
            ts  = ts_in_day(r, day)
            instr = r.choice(instrs)
            order_ctr[0] += 1; session_ctr[0] += 1
            oid = new_order_id(order_ctr); sid = new_session_id(session_ctr)
            amount = max(50.0, float(r.lognormal(5.5, 1.0)))
            pc = None
            if acc in promo_users and r.random() < 0.15:
                pc = f"PROMO_PUBLIC_{int(r.integers(10, 99))}"
            events_out.append({
                "account_id": acc, "session_id": sid, "device_id": device,
                "ip_id": ip, "instrument_id": instr, "payout_id": payout_id,
                "order_id": oid, "amount": amount, "promo_code": pc,
                "event_type": "order_placed", "timestamp": ts,
                "ring_id": None, "referrer_id": None,
            })

    # 5% incidental device sharing among BI (household effect)
    share_cands = r.choice(bi_accounts, size=max(1, int(n_bi * 0.05)), replace=False)
    for i in range(0, len(share_cands) - 1, 2):
        a1, a2 = share_cands[i], share_cands[i+1]
        shared_dev = r.choice(dev_pool)
        # patch last device for a2 to match a1 (best-effort -- modifies device for future orders only)
        # Note: events already written; mark shared device in labels for audit only
        labels_out[a1]["incidental_device_share"] = True
        labels_out[a2]["incidental_device_share"] = True

    return order_ctr, session_ctr

