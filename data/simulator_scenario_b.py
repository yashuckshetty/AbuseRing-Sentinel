"""
AbuseRing Sentinel — Scenario B Simulator: Subscription Platform Trial Abuse
=============================================================================
Context: SaaS / Streaming Recurring-Billing Platform.
Abuse Pattern: Virtual card recycling, disposable trial farming (1-order burst, zero promo, zero returns).
Benign Coordinated: Corporate multi-seat billing and shared corporate payment instruments.
Benign Independent: Standard individual subscribers.

Generates:
  - data/scenario_b/accounts.parquet
  - data/scenario_b/events.parquet
  - data/scenario_b/labels.parquet
  - data/scenario_b/rings.parquet
  - data/scenario_b/split_info.json
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_SEED = 101
N_ACCOUNTS = 1_800
SIM_START_TS = 1_700_000_000
SECONDS_PER_DAY = 86_400
SIM_DAYS = 90
TRAIN_END_DAY = 54
VAL_END_DAY = 72

OUTPUT_DIR = Path("data/scenario_b")

def safe_sample(pool, n, rng_obj):
    pool = list(pool)
    if n > len(pool):
        return list(rng_obj.choice(pool, size=n, replace=True))
    return list(rng_obj.choice(pool, size=n, replace=False))

def generate_scenario_b(seed=DEFAULT_SEED, output_dir=OUTPUT_DIR):
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_bi = int(N_ACCOUNTS * 0.60)   # 1080
    n_bc = int(N_ACCOUNTS * 0.25)   # 450
    n_ac = N_ACCOUNTS - n_bi - n_bc  # 270
    
    account_ids = [f"ACC_B_{i:05d}" for i in range(N_ACCOUNTS)]
    
    bi_accs = account_ids[:n_bi]
    bc_accs = account_ids[n_bi:n_bi + n_bc]
    ac_accs = account_ids[n_bi + n_bc:]
    
    # Entity pools
    devices = [f"DEV_B_{i:05d}" for i in range(1500)]
    ips = [f"IP_B_{i:05d}" for i in range(1000)]
    instruments = [f"CARD_B_{i:05d}" for i in range(1200)]
    payouts = [f"PAY_B_{i:05d}" for i in range(800)]
    
    accounts_rows = []
    events_rows = []
    labels_rows = []
    rings_rows = []
    
    order_id_counter = 1
    session_id_counter = 1
    
    # ── 1. BENIGN INDEPENDENT (Standard Subscribers) ──────────────────────────
    # Normal recurring activity: signup, 1 trial order, steady monthly subscription renewals, distinct card/device/ip.
    for acc in bi_accs:
        created_day = int(rng.integers(1, 80))
        created_ts = SIM_START_TS + (created_day - 1) * SECONDS_PER_DAY + int(rng.integers(0, 86400))
        
        dev = rng.choice(devices[:900])
        ip = rng.choice(ips[:600])
        card = rng.choice(instruments[:800])
        pay = rng.choice(payouts[:500])
        
        accounts_rows.append({
            "account_id": acc,
            "created_ts": created_ts,
            "account_status": "active",
            "country": "US",
            "email_domain": rng.choice(["gmail.com", "yahoo.com", "outlook.com", "icloud.com"]),
        })
        labels_rows.append({
            "account_id": acc,
            "label": "benign_independent",
            "label_str": "benign_independent",
            "is_sleeper": False,
            "counterfactual_subset": "none"
        })
        
        # Events: session starts + subscription renewals (every 30 days)
        n_renewals = int((SIM_DAYS - created_day) // 30) + 1
        for r in range(n_renewals):
            evt_day = created_day + r * 30 + int(rng.integers(0, 3))
            if evt_day > SIM_DAYS:
                break
            evt_ts = SIM_START_TS + (evt_day - 1) * SECONDS_PER_DAY + int(rng.integers(0, 86400))
            
            # Session
            events_rows.append({
                "event_id": f"EVT_B_{len(events_rows)+1:08d}",
                "account_id": acc,
                "event_type": "session_start",
                "timestamp": evt_ts - 300,
                "session_id": f"SES_B_{session_id_counter:07d}",
                "device_id": dev,
                "ip_id": ip,
                "instrument_id": card,
                "payout_id": pay,
                "amount": None,
                "promo_code": None,
                "referrer_id": None
            })
            session_id_counter += 1
            
            # Subscription Charge / Trial
            events_rows.append({
                "event_id": f"EVT_B_{len(events_rows)+1:08d}",
                "account_id": acc,
                "event_type": "order_placed",
                "timestamp": evt_ts,
                "session_id": f"SES_B_{session_id_counter-1:07d}",
                "device_id": dev,
                "ip_id": ip,
                "instrument_id": card,
                "payout_id": pay,
                "amount": 0.0 if r == 0 else float(rng.choice([14.99, 29.99, 49.99])),
                "promo_code": None, # Non-promo context
                "referrer_id": None
            })
            order_id_counter += 1

    # ── 2. BENIGN COORDINATED (Corporate Multi-Seat Accounts) ───────────────────
    # 20 corporate clusters (15-25 members each) sharing corporate billing cards & IP ranges, but on distinct employee devices with steady usage.
    bc_groups = np.array_split(bc_accs, 20)
    for g_idx, grp in enumerate(bc_groups):
        corp_card = instruments[800 + g_idx]
        corp_pay = payouts[500 + g_idx]
        corp_ip = ips[600 + g_idx]
        group_start_day = int(rng.integers(5, 45))
        
        for acc in grp:
            created_day = group_start_day + int(rng.integers(0, 10))
            created_ts = SIM_START_TS + (created_day - 1) * SECONDS_PER_DAY + int(rng.integers(0, 86400))
            dev = rng.choice(devices[900:1200]) # Distinct employee device
            
            accounts_rows.append({
                "account_id": acc,
                "created_ts": created_ts,
                "account_status": "active",
                "country": "US",
                "email_domain": f"enterprise{g_idx}.com",
            })
            labels_rows.append({
                "account_id": acc,
                "label": "benign_coordinated",
                "label_str": "benign_coordinated",
                "is_sleeper": False,
                "counterfactual_subset": "hard_bc" if g_idx < 4 else "corporate_multiseat"
            })
            
            # Legitimate ongoing monthly usage
            n_charges = int((SIM_DAYS - created_day) // 30) + 1
            for r in range(n_charges):
                evt_day = created_day + r * 30 + int(rng.integers(0, 4))
                if evt_day > SIM_DAYS:
                    break
                evt_ts = SIM_START_TS + (evt_day - 1) * SECONDS_PER_DAY + int(rng.integers(0, 86400))
                
                events_rows.append({
                    "event_id": f"EVT_B_{len(events_rows)+1:08d}",
                    "account_id": acc,
                    "event_type": "session_start",
                    "timestamp": evt_ts - 120,
                    "session_id": f"SES_B_{session_id_counter:07d}",
                    "device_id": dev,
                    "ip_id": corp_ip,
                    "instrument_id": corp_card,
                    "payout_id": corp_pay,
                    "amount": None,
                    "promo_code": None,
                    "referrer_id": None
                })
                session_id_counter += 1
                
                events_rows.append({
                    "event_id": f"EVT_B_{len(events_rows)+1:08d}",
                    "account_id": acc,
                    "event_type": "order_placed",
                    "timestamp": evt_ts,
                    "session_id": f"SES_B_{session_id_counter-1:07d}",
                    "device_id": dev,
                    "ip_id": corp_ip,
                    "instrument_id": corp_card,
                    "payout_id": corp_pay,
                    "amount": 29.99,
                    "promo_code": None,
                    "referrer_id": None
                })
                order_id_counter += 1

    # ── 3. ABUSIVE COORDINATED (Trial Abuse & Card Recycling Rings) ─────────────
    # 15 trial-farming rings (12-25 disposable accounts per ring).
    # Ring members share virtual cards / payout endpoints and device clusters, claim 1 trial order immediately, then go silent.
    ac_groups = np.array_split(ac_accs, 15)
    for r_idx, ring in enumerate(ac_groups):
        ring_id = f"RING_B_{r_idx+1:03d}"
        shared_vcc = instruments[1000 + r_idx]
        shared_pay = payouts[600 + r_idx]
        ring_dev_pool = safe_sample(devices[1200:1500], 3, rng)
        ring_ip_pool = safe_sample(ips[800:1000], 3, rng)
        
        start_day = int(rng.integers(10, 75))
        complete_day = min(start_day + int(rng.integers(2, 6)), SIM_DAYS)
        
        is_sleeper_ring = (r_idx == 0)
        
        for acc in ring:
            acc_day = int(rng.integers(start_day, complete_day + 1))
            created_ts = SIM_START_TS + (acc_day - 1) * SECONDS_PER_DAY + int(rng.integers(0, 86400))
            
            dev = rng.choice(ring_dev_pool)
            ip = rng.choice(ring_ip_pool)
            
            accounts_rows.append({
                "account_id": acc,
                "created_ts": created_ts,
                "account_status": "active",
                "country": "US",
                "email_domain": rng.choice(["tempmail.com", "guerrillamail.com", "10minutemail.com"]),
            })
            
            is_sleeper = is_sleeper_ring and (acc == ring[0])
            labels_rows.append({
                "account_id": acc,
                "label": "abusive_coordinated",
                "label_str": "abusive_coordinated",
                "is_sleeper": is_sleeper,
                "counterfactual_subset": "none"
            })
            rings_rows.append({
                "ring_id": ring_id,
                "account_id": acc,
                "ring_type": "trial_abuse_vcc",
                "formation_start_day": start_day,
                "formation_complete_day": complete_day,
                "is_sleeper": is_sleeper
            })
            
            # Abusive signature: Fast 1-order trial claim burst (amount=0.0 or 1.0 auth hold), then abandonment.
            burst_time = created_ts + int(rng.integers(60, 600))
            
            events_rows.append({
                "event_id": f"EVT_B_{len(events_rows)+1:08d}",
                "account_id": acc,
                "event_type": "session_start",
                "timestamp": burst_time - 30,
                "session_id": f"SES_B_{session_id_counter:07d}",
                "device_id": dev,
                "ip_id": ip,
                "instrument_id": shared_vcc,
                "payout_id": shared_pay,
                "amount": None,
                "promo_code": None,
                "referrer_id": None
            })
            session_id_counter += 1
            
            events_rows.append({
                "event_id": f"EVT_B_{len(events_rows)+1:08d}",
                "account_id": acc,
                "event_type": "order_placed",
                "timestamp": burst_time,
                "session_id": f"SES_B_{session_id_counter-1:07d}",
                "device_id": dev,
                "ip_id": ip,
                "instrument_id": shared_vcc,
                "payout_id": shared_pay,
                "amount": 0.0, # Free trial claim
                "promo_code": None, # Zero promo code used
                "referrer_id": None
            })
            order_id_counter += 1

    accounts_df = pd.DataFrame(accounts_rows)
    events_df = pd.DataFrame(events_rows).sort_values("timestamp").reset_index(drop=True)
    labels_df = pd.DataFrame(labels_rows)
    rings_df = pd.DataFrame(rings_rows)
    
    split_info = {
        "sim_start_ts": SIM_START_TS,
        "train_end_ts": SIM_START_TS + TRAIN_END_DAY * SECONDS_PER_DAY,
        "val_end_ts": SIM_START_TS + VAL_END_DAY * SECONDS_PER_DAY,
        "test_end_ts": SIM_START_TS + SIM_DAYS * SECONDS_PER_DAY,
        "n_accounts": N_ACCOUNTS,
        "scenario": "B_subscription_trial_abuse",
        "description": "Subscription platform with trial abuse via virtual cards and corporate multi-seat billing"
    }
    
    accounts_df.to_parquet(output_dir / "accounts.parquet", index=False)
    events_df.to_parquet(output_dir / "events.parquet", index=False)
    labels_df.to_parquet(output_dir / "labels.parquet", index=False)
    rings_df.to_parquet(output_dir / "rings.parquet", index=False)
    
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
        
    print(f"Scenario B generated successfully in {output_dir}:")
    print(f"  Accounts: {len(accounts_df)}")
    print(f"  Events:   {len(events_df)}")
    print(f"  Labels:   {labels_df['label_str'].value_counts().to_dict()}")
    print(f"  Rings:    {rings_df['ring_id'].nunique()} rings ({len(rings_df)} accounts)")

if __name__ == "__main__":
    generate_scenario_b()
