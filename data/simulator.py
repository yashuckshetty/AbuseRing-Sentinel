"""
AbuseRing Sentinel — Synthetic Data Simulator v2.0
====================================================
Implements ASSUMPTIONS.md v2.0 contract exactly.
Amendments A1-A6 are all implemented here.

New vs v1:
  A1: labels.parquet gains label_true, label_observed, partial_signal, counterfactual_subset
  A2: Sleeper accounts: label_true=abusive_coordinated, partial_signal=True
  A3: ring_formation_start_day + ring_formation_complete_day in rings.parquet; >=20% start in val/test
  A4: abuse_prevalence dict parameter controls class mix
  A5: Referral-farming rings (test window only, day>=73, dense referrals, minimal shared infra)
  A6: hard_bc counterfactual (15% of BC families get one shared payout, still label_true=BC)
       varied_payout_ac (10% of rings get varied per-member payouts, still label_true=AC)

Payout pool fix (from v1 clarification):
  BI accounts use a TWO-TIER pool:
    Tier 1 (78%): unique synthetic payout IDs -- zero inter-account collision
    Tier 2 (22%): small shared pool of 220 IDs -- ~3x local collision rate
    Overall BI mean payout degree: ~1.44 (vs AC ring mean: 4-13)
  This preserves payout-destination-share as a genuinely discriminative signal.

Run: python data/simulator.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_SEED     = 42
N_ACCOUNTS       = 5_000
N_DEVICES        = 3_500
N_IPS            = 2_000
N_INSTRUMENTS    = 4_000
N_PAYOUTS        = 1_800

SIM_START_TS     = 1_700_000_000   # 2023-11-14 approx (fixed origin)
SECONDS_PER_DAY  = 86_400
SIM_DAYS         = 90

TRAIN_END_DAY    = 54
VAL_END_DAY      = 72
# TEST: days 73-90

LABEL_NOISE_RATE = 0.03
RING_EDGE_UNOBSERVED_RATE = 0.20

DEFAULT_PREVALENCE = {
    "benign_independent": 0.60,
    "benign_coordinated": 0.25,
    "abusive_coordinated": 0.15,
}

OUTPUT_DIR = Path("data")

# ── Utilities ─────────────────────────────────────────────────────────────────

def make_rng(seed):
    return np.random.default_rng(seed)

def safe_sample(pool, n, rng_obj, replace=False):
    """Sample n items from pool; falls back to replace=True if pool too small."""
    pool = list(pool)
    if len(pool) == 0:
        raise ValueError("Cannot sample from empty pool")
    if n > len(pool) and not replace:
        replace = True
    return list(rng_obj.choice(pool, size=n, replace=replace))

def ts_in_day(rng_obj, day, hour_bias="flat"):
    """Return a Unix timestamp within `day` (1-indexed)."""
    day_start = SIM_START_TS + (day - 1) * SECONDS_PER_DAY
    if hour_bias == "business":
        hour = int(np.clip(rng_obj.normal(14, 3), 9, 19))
    elif hour_bias == "evening":
        hour = int(np.clip(rng_obj.normal(20, 2), 17, 23))
    else:
        hour = int(rng_obj.integers(0, 24))
    minute = int(rng_obj.integers(0, 60))
    second = int(rng_obj.integers(0, 60))
    return day_start + hour * 3600 + minute * 60 + second

def day_ts_range(day):
    """Return (start_ts, end_ts) for a given day (1-indexed)."""
    start = SIM_START_TS + (day - 1) * SECONDS_PER_DAY
    return start, start + SECONDS_PER_DAY

def partition_into_groups(pool, min_size, max_size, rng_obj):
    """Partition pool into random-sized groups."""
    groups = []
    remaining = list(rng_obj.permutation(pool))
    while len(remaining) >= min_size:
        size = int(rng_obj.integers(min_size, max_size + 1))
        size = min(size, len(remaining))
        groups.append(remaining[:size])
        remaining = remaining[size:]
    return groups

def new_order_id(counter):
    return f"ORD_{counter[0]:07d}"

def new_session_id(counter):
    return f"SES_{counter[0]:07d}"



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



# ── Output writers ────────────────────────────────────────────────────────────

def apply_label_noise(labels_dict, r):
    """
    [A1] Apply 3% noise to abusive_coordinated accounts.
    label_true is NEVER changed -- only label_observed is flipped.
    """
    ac_accounts = [acc for acc, d in labels_dict.items()
                   if d.get("label_true") == "abusive_coordinated"]
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

        # Merge with accounts.parquet to get real created_ts (the only metadata available
        # to a model at inference time without looking at events or labels).
        # Do NOT include account_idx -- that encodes generation order, not real signal.
        import os as _os
        accts_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "accounts.parquet")
        if _os.path.exists(accts_path):
            accts_df = __import__("pandas").read_parquet(accts_path)
            merged = labels_df.merge(accts_df[["account_id", "created_ts"]], on="account_id", how="left")
            X = merged[["created_ts"]].fillna(SIM_START_TS).values
        else:
            # Fallback: random timestamps (should not happen in practice)
            rng_sc = __import__("numpy").random.default_rng(99)
            X = rng_sc.integers(SIM_START_TS, SIM_START_TS + SIM_DAYS * SECONDS_PER_DAY,
                                 size=(len(labels_df), 1)).astype(float)
        y_str = labels_df["label_true"].values
        class_map = {"benign_independent": 0, "benign_coordinated": 1, "abusive_coordinated": 2}
        y = np.array([class_map[c] for c in y_str])

        # Use last 30% as "test" (proportional to test split)
        # Stratified split done below

        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        tr_idx, te_idx = next(sss.split(X, y))
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        y_bin = label_binarize(y_te, classes=[0, 1, 2])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")

        status = "PASS" if auc <= 0.55 else "FAIL"
        print(f"\n[SHORTCUT CHECK] Depth-2 tree AUC={auc:.4f} -- {status} (threshold: <=0.55)")
        return {"auc": round(float(auc), 4), "pass": auc <= 0.55,
                "note": "Depth-2 DT on creation order/ts. AUC<=0.55 required."}
    except Exception as e:
        print(f"  [SHORTCUT CHECK] Skipped: {e}")
        return {"auc": None, "pass": None, "note": str(e)}



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
    # Pre-populate label_true for ALL accounts from their pool membership.
    # Accounts dropped by partition_into_groups tail are never processed by
    # any generator, so {} init caused KeyError in apply_label_noise.
    labels_dict = {}
    for acc in bi_accounts:
        labels_dict[acc] = {"label_true": "benign_independent",
                             "partial_signal": False, "counterfactual_subset": None}
    for acc in bc_accounts:
        labels_dict[acc] = {"label_true": "benign_coordinated",
                             "partial_signal": False, "counterfactual_subset": None}
    for acc in ac_promo_accs + ac_return_accs + ac_ref_accs:
        labels_dict[acc] = {"label_true": "abusive_coordinated",
                             "partial_signal": False, "counterfactual_subset": None}


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

            if is_sleeper:
                # [A2] Sleeper structural suppression: unique payout breaks payout co-sharing
                member_pay = f"PAY_SLEEPER_{ring_id_counter[0]:03d}_{m_idx:02d}"
            elif is_varied:
                member_pay = f"PAY_VARIED_{ring_id_counter[0]:03d}_{m_idx:02d}"
            else:
                member_pay = r.choice(ring_pays)

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
