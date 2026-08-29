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

