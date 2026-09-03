"""
AbuseRing Sentinel — Independent Hand-Crafted Topology Stress Battery
======================================================================
Evaluates 25 deterministic, out-of-distribution structural failure topologies
constructed completely independently of data/simulator.py.

Evaluates how the canonical DecisionEngine and evidence-disagreement routing
resist structural edge cases, topological camouflage, bipartite referral trees,
extreme sparsity, and adversarial entity manipulation.

5 THREAT FAMILIES (5 Topologies Each = 25 Total):
  Family A: Graph Camouflage & Topological Noise Injection (TOPO_01 - TOPO_05)
  Family B: Temporal & Sleeper Attack Patterns (TOPO_06 - TOPO_10)
  Family C: Entity & Identifier Manipulation (TOPO_11 - TOPO_15)
  Family D: Extreme Graph Sparsity & Cold-Start (TOPO_16 - TOPO_20)
  Family E: Hybrid / Evasion Stress Topologies (TOPO_21 - TOPO_25)

Outputs:
  evals/results/handcrafted_adversarial_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.temporal_graph import (
    build_graph_as_of,
    extract_account_structural_features,
    extract_account_behavioral_features,
)
from features.feature_pipeline import (
    STRUCTURAL_FEATURES,
    BEHAVIORAL_FEATURES,
)
from decision.decision_engine import (
    DecisionEngine,
    Decision,
    RoutingLane,
    DecisionResult,
    sym_kl_divergence,
)

RESULTS_DIR = Path("evals/results")
MODELS_DIR = Path("models")


def generate_topology_data(topo_id: str) -> Tuple[pd.DataFrame, pd.DataFrame, int, List[str]]:
    """
    Deterministically generates events_df, accounts_df, as_of_ts, and target_acc_ids
    for each of the 25 out-of-distribution structural topologies.
    """
    np.random.seed(42 + int(topo_id.split("_")[1]))
    base_ts = 1704067200 # Day 0: 2024-01-01
    as_of_ts = base_ts + 80 * 86400 # Day 80

    accounts = []
    events = []

    def _add_acc(acc_id, created_day):
        c_ts = base_ts + created_day * 86400
        accounts.append({"account_id": acc_id, "created_ts": c_ts, "created_at": c_ts})

    def _add_order(evt_id, acc_id, day, amount=500.0, dev="DEV_1", ip="IP_1", p_out="PO_1", instr="INS_1", promo=None):
        ts = base_ts + day * 86400 + int(np.random.randint(100, 3600))
        events.append({
            "event_id": evt_id,
            "timestamp": ts,
            "event_type": "order_placed",
            "account_id": acc_id,
            "amount": amount,
            "device_id": dev,
            "ip_id": ip,
            "payout_id": p_out,
            "instrument_id": instr,
            "promo_code": promo,
            "order_status": "completed"
        })

    def _add_return(evt_id, acc_id, day):
        ts = base_ts + day * 86400 + 7200
        events.append({
            "event_id": evt_id,
            "timestamp": ts,
            "event_type": "order_returned",
            "account_id": acc_id,
            "amount": 500.0,
            "device_id": "DEV_RET",
            "ip_id": "IP_RET",
            "payout_id": None,
            "instrument_id": None,
            "promo_code": None,
            "order_status": "returned"
        })

    def _add_referral(evt_id, referee_id, referrer_id, day):
        ts = base_ts + day * 86400 + 1000
        events.append({
            "event_id": evt_id,
            "timestamp": ts,
            "event_type": "referral",
            "account_id": referee_id,
            "referrer_id": referrer_id,
            "amount": 0.0,
            "device_id": None,
            "ip_id": None,
            "payout_id": None,
            "instrument_id": None,
            "promo_code": "REF_PROMO",
            "order_status": "completed"
        })

    # ==========================================
    # FAMILY A: Graph Camouflage & Noise Injection
    # ==========================================
    if topo_id == "TOPO_01_DENSE_CLIQUE_CAMO":
        # 6 ring accounts form clique on shared payout, but make orders on benign high-degree merchant device
        target_accs = [f"ACC_T01_{i}" for i in range(6)]
        for a in target_accs:
            _add_acc(a, 60)
            for d in range(65, 75):
                _add_order(f"EVT_T01_{a}_{d}", a, d, amount=1200.0, dev="DEV_BENIGN_HUB", ip=f"IP_CLEAN_{a}", p_out="PO_SHARED_CLIQUE", promo="RING_PROMO")

    elif topo_id == "TOPO_02_STAR_DISPERSION":
        # Hub account connects to 15 satellite accounts via referrals with 1-2 orders each
        hub = "ACC_T02_HUB"
        _add_acc(hub, 50)
        target_accs = [hub] + [f"ACC_T02_SAT_{i}" for i in range(15)]
        for sat in target_accs[1:]:
            _add_acc(sat, 70)
            _add_referral(f"REF_T02_{sat}", sat, hub, 71)
            _add_order(f"EVT_T02_{sat}_1", sat, 72, amount=800.0, dev=f"DEV_SAT_{sat}", ip="IP_SAT", p_out=f"PO_{sat}", promo="REF_BONUS")
            _add_order(f"EVT_T02_{sat}_2", sat, 74, amount=850.0, dev=f"DEV_SAT_{sat}", ip="IP_SAT", p_out=f"PO_{sat}", promo="REF_BONUS")

    elif topo_id == "TOPO_03_BRIDGE_ISOLATION":
        # Two 4-node cliques connected by a single bridge account
        target_accs = [f"ACC_T03_L_{i}" for i in range(4)] + [f"ACC_T03_R_{i}" for i in range(4)]
        for a in target_accs[:4]:
            _add_acc(a, 55)
            for d in range(60, 64):
                _add_order(f"EVT_T03_{a}_{d}", a, d, dev="DEV_CLUSTER_L", ip="IP_L", p_out="PO_L", promo="PROMO_A")
        for a in target_accs[4:]:
            _add_acc(a, 55)
            for d in range(60, 64):
                _add_order(f"EVT_T03_{a}_{d}", a, d, dev="DEV_CLUSTER_R", ip="IP_R", p_out="PO_R", promo="PROMO_A")
        # Bridge link
        _add_referral("REF_T03_BRIDGE", target_accs[3], target_accs[4], 65)

    elif topo_id == "TOPO_04_CYCLE_CHAIN_DILUTION":
        # 8 accounts in a directed referral cycle with high synthetic benign volume
        target_accs = [f"ACC_T04_{i}" for i in range(8)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 50)
            next_a = target_accs[(i + 1) % 8]
            _add_referral(f"REF_T04_{a}", next_a, a, 55)
            for d in range(58, 66):
                _add_order(f"EVT_T04_{a}_{d}", a, d, amount=300.0, dev=f"DEV_T04_{i}", ip=f"IP_T04_{i}", p_out=f"PO_T04_{i}")

    elif topo_id == "TOPO_05_RANDOM_AFFILIATE_INJECTION":
        # 6 ring members injecting random legitimate affiliate codes to disrupt graph clustering
        target_accs = [f"ACC_T05_{i}" for i in range(6)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 60)
            for d in range(65, 71):
                # Shared device and payout, but random affiliate / promo
                _add_order(f"EVT_T05_{a}_{d}", a, d, amount=900.0, dev="DEV_SHARED_T05", ip=f"IP_NOISE_{i}", p_out="PO_SHARED_T05", promo=f"LEGIT_AFFILIATE_{d%3}")

    # ==========================================
    # FAMILY B: Temporal & Sleeper Attack Patterns
    # ==========================================
    elif topo_id == "TOPO_06_PROLONGED_SLEEPER_BURST":
        # Accounts created on Day 5, dormant until Day 78, then 6 orders in 1 day
        target_accs = [f"ACC_T06_{i}" for i in range(5)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 5) # 75 days old!
            for d in [78, 79]:
                for k in range(3):
                    _add_order(f"EVT_T06_{a}_{d}_{k}", a, d, amount=1500.0, dev="DEV_SLEEPER_T06", ip=f"IP_SLP_{i}", p_out="PO_SLEEPER_T06", promo="SLEEPER_BURST")

    elif topo_id == "TOPO_07_MICRO_VELOCITY_STAGGER":
        # 6 accounts firing staggered orders exactly every 2 days to evade 24h/48h burst windows
        target_accs = [f"ACC_T07_{i}" for i in range(6)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 50)
            for d in range(52 + i%2, 75, 4):
                _add_order(f"EVT_T07_{a}_{d}", a, d, amount=650.0, dev=f"DEV_STAGGER_{i}", ip="IP_SHARED_VPN", p_out="PO_SHARED_STAGGER", promo="MICRO_PROMO")

    elif topo_id == "TOPO_08_BURST_THEN_ABANDON":
        # 5 accounts execute 5 rapid promo orders on Day 60, then zero activity for 20 days
        target_accs = [f"ACC_T08_{i}" for i in range(5)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 60)
            for k in range(5):
                _add_order(f"EVT_T08_{a}_{k}", a, 60, amount=400.0, dev="DEV_BURST_T08", ip=f"IP_B_{i}", p_out="PO_BURST_T08", promo="FLASH_50")

    elif topo_id == "TOPO_09_SLOW_BURN_REFERRAL":
        # Referral farming spreading across 40 days at 1 order per 8 days
        target_accs = [f"ACC_T09_{i}" for i in range(6)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 30 + i*5)
            if i > 0:
                _add_referral(f"REF_T09_{a}", a, target_accs[0], 31 + i*5)
            for d in range(40 + i*5, 78, 8):
                _add_order(f"EVT_T09_{a}_{d}", a, d, amount=500.0, dev=f"DEV_SLOW_{i}", ip=f"IP_SLOW_{i}", p_out=f"PO_SLOW_{i}", promo="SLOW_REF")

    elif topo_id == "TOPO_10_LATE_DEVICE_COLLISION":
        # 5 accounts on clean distinct IPs/devices that collide on a single shared device on their 5th order
        target_accs = [f"ACC_T10_{i}" for i in range(5)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 60)
            for d in range(62, 66):
                _add_order(f"EVT_T10_{a}_{d}", a, d, dev=f"DEV_CLEAN_{i}", ip=f"IP_CLEAN_{i}", p_out=f"PO_CLEAN_{i}")
            # Collide on 5th order
            _add_order(f"EVT_T10_{a}_COLLIDE", a, 70, dev="DEV_COLLIDE_SHARED", ip="IP_SHARED", p_out="PO_SHARED", promo="COLLIDE_PROMO")

    # ==========================================
    # FAMILY C: Entity & Identifier Manipulation
    # ==========================================
    elif topo_id == "TOPO_11_PAYOUT_ROTATION":
        # Ring sharing 1 bank account rotated across 8 distinct fake account IDs
        target_accs = [f"ACC_T11_{i}" for i in range(8)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 65)
            for d in range(68, 74):
                _add_order(f"EVT_T11_{a}_{d}", a, d, amount=1100.0, dev=f"DEV_ROT_{i}", ip=f"IP_ROT_{i}", p_out="PO_SHARED_BANK_999", promo="PAYOUT_FARM")

    elif topo_id == "TOPO_12_DEVICE_FINGERPRINT_CHURN":
        # Single fraudster spinning up new device ID per transaction on same IP and payout
        target_accs = [f"ACC_T12_{i}" for i in range(5)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 60)
            for k in range(4):
                _add_order(f"EVT_T12_{a}_{k}", a, 62 + k*2, amount=800.0, dev=f"DEV_CHURN_{a}_{k}", ip="IP_STATIC_RESIDENTIAL", p_out="PO_FINGERPRINT_CHURN", promo="DISCOUNT_20")

    elif topo_id == "TOPO_13_SHARED_IP_SUBNET_HOPPING":
        # Accounts operating on adjacent IP subnets with identical device and payout
        target_accs = [f"ACC_T13_{i}" for i in range(6)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 62)
            for d in range(65, 71):
                _add_order(f"EVT_T13_{a}_{d}", a, d, dev="DEV_SUBNET_SHARED", ip=f"IP_SUBNET_192_168_1_{10+i}", p_out="PO_SUBNET_SHARED", promo="SUBNET_PROMO")

    elif topo_id == "TOPO_14_COLLUDING_MERCHANT_DISPUTE":
        # Accounts buying from colluding merchant and filing return/dispute claims
        target_accs = [f"ACC_T14_{i}" for i in range(5)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 55)
            for d in range(58, 65):
                _add_order(f"EVT_T14_{a}_{d}", a, d, amount=2500.0, dev=f"DEV_M_{i}", ip=f"IP_M_{i}", p_out=f"PO_M_{i}")
                if d in [60, 64]:
                    _add_return(f"RET_T14_{a}_{d}", a, d)

    elif topo_id == "TOPO_15_CIRCULAR_PAYOUT_RECYCLING":
        # Triangular payout recycling between 3 pairs of accounts
        target_accs = [f"ACC_T15_{i}" for i in range(6)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 60)
            p_id = f"PO_CYCLE_{i // 2}"
            for d in range(64, 72):
                _add_order(f"EVT_T15_{a}_{d}", a, d, amount=750.0, dev=f"DEV_CYC_{i}", ip=f"IP_CYC_{i}", p_out=p_id, promo="RECYCLE_PROMO")

    # ==========================================
    # FAMILY D: Extreme Graph Sparsity & Cold-Start
    # ==========================================
    elif topo_id == "TOPO_16_ISOLATED_PAIR_COLLUSION":
        # Exactly 2 accounts sharing 1 device, zero other edges anywhere
        target_accs = ["ACC_T16_A", "ACC_T16_B"]
        for a in target_accs:
            _add_acc(a, 70)
            for d in range(72, 77):
                _add_order(f"EVT_T16_{a}_{d}", a, d, amount=600.0, dev="DEV_ISOLATED_PAIR", ip=f"IP_ISO_{a}", p_out=f"PO_ISO_{a}")

    elif topo_id == "TOPO_17_COLD_START_PROMO_FARM":
        # 5 brand-new accounts (created Day 78, 2 orders each) sharing promo code, zero prior graph
        target_accs = [f"ACC_T17_{i}" for i in range(5)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 78)
            _add_order(f"EVT_T17_{a}_1", a, 79, amount=999.0, dev=f"DEV_CS_{i}", ip=f"IP_CS_{i}", p_out=f"PO_CS_{i}", promo="SUPER_NEW_USER")
            _add_order(f"EVT_T17_{a}_2", a, 80, amount=999.0, dev=f"DEV_CS_{i}", ip=f"IP_CS_{i}", p_out=f"PO_CS_{i}", promo="SUPER_NEW_USER")

    elif topo_id == "TOPO_18_HIGH_VALUE_SINGLETON":
        # Single account with exactly n=1 order attempting massive promo abuse (tests ABSTAIN discipline)
        target_accs = ["ACC_T18_SINGLETON"]
        _add_acc("ACC_T18_SINGLETON", 79)
        _add_order("EVT_T18_1", "ACC_T18_SINGLETON", 80, amount=5000.0, dev="DEV_SINGLE", ip="IP_SINGLE", p_out="PO_SINGLE", promo="WHALE_EXPLOIT")

    elif topo_id == "TOPO_19_ASYMMETRIC_BIPARTITE":
        # 10 accounts connected to 2 shared devices with no direct account-to-account links
        target_accs = [f"ACC_T19_{i}" for i in range(10)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 65)
            dev_id = "DEV_BIPARTITE_A" if i < 5 else "DEV_BIPARTITE_B"
            for d in range(68, 74):
                _add_order(f"EVT_T19_{a}_{d}", a, d, amount=700.0, dev=dev_id, ip=f"IP_BIP_{i}", p_out=f"PO_BIP_{i}", promo="BIPARTITE_PROMO")

    elif topo_id == "TOPO_20_ZERO_STRUCTURAL_SIGNAL":
        # 6 coordinated accounts with 100% anonymized/isolated entities (pure structural blindness stress)
        target_accs = [f"ACC_T20_{i}" for i in range(6)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 60)
            for d in range(65, 73):
                _add_order(f"EVT_T20_{a}_{d}", a, d, amount=1200.0, dev=f"DEV_ANON_{i}_{d}", ip=f"IP_ANON_{i}_{d}", p_out=f"PO_ANON_{i}", promo="ZERO_STRUCT_PROMO")

    # ==========================================
    # FAMILY E: Hybrid / Evasion Stress Topologies
    # ==========================================
    elif topo_id == "TOPO_21_BEHAVIORAL_MIMICRY":
        # Ring matching benign transaction velocity perfectly (1 order / 10 days) but sharing payout & device
        target_accs = [f"ACC_T21_{i}" for i in range(6)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 20)
            for d in [30, 45, 60, 75]:
                _add_order(f"EVT_T21_{a}_{d}", a, d, amount=350.0, dev="DEV_MIMIC_SHARED", ip=f"IP_MIMIC_{i}", p_out="PO_MIMIC_SHARED")

    elif topo_id == "TOPO_22_ASYMMETRIC_DISAGREEMENT":
        # High structural density (shared device, IP, payout, instruments) with low behavioral velocity
        target_accs = [f"ACC_T22_{i}" for i in range(8)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 40)
            for d in [50, 70]:
                _add_order(f"EVT_T22_{a}_{d}", a, d, amount=200.0, dev="DEV_DENSE_CORE", ip="IP_DENSE_CORE", p_out="PO_DENSE_CORE", instr="INS_DENSE_CORE")

    elif topo_id == "TOPO_23_SPLIT_PAYOUT_TRIANGLE":
        # 3 accounts splitting payouts in alternating combinations
        target_accs = ["ACC_T23_A", "ACC_T23_B", "ACC_T23_C"]
        for a in target_accs:
            _add_acc(a, 58)
        # Pairwise shared payouts
        _add_order("EVT_T23_A_1", "ACC_T23_A", 62, dev="DEV_T23_A", ip="IP_T23_A", p_out="PO_AB")
        _add_order("EVT_T23_B_1", "ACC_T23_B", 62, dev="DEV_T23_B", ip="IP_T23_B", p_out="PO_AB")
        _add_order("EVT_T23_B_2", "ACC_T23_B", 66, dev="DEV_T23_B", ip="IP_T23_B", p_out="PO_BC")
        _add_order("EVT_T23_C_1", "ACC_T23_C", 66, dev="DEV_T23_C", ip="IP_T23_C", p_out="PO_BC")
        _add_order("EVT_T23_C_2", "ACC_T23_C", 70, dev="DEV_T23_C", ip="IP_T23_C", p_out="PO_CA")
        _add_order("EVT_T23_A_2", "ACC_T23_A", 70, dev="DEV_T23_A", ip="IP_T23_A", p_out="PO_CA")

    elif topo_id == "TOPO_24_SYBIL_REFERRAL_TREE":
        # 4-level deep sybil tree with 1 root, 2 children, 4 grandchildren, 8 great-grandchildren
        root = "ACC_T24_ROOT"
        _add_acc(root, 40)
        target_accs = [root]
        level1 = [f"ACC_T24_L1_{i}" for i in range(2)]
        level2 = [f"ACC_T24_L2_{i}" for i in range(4)]
        level3 = [f"ACC_T24_L3_{i}" for i in range(8)]
        target_accs.extend(level1 + level2 + level3)

        for i, a in enumerate(level1):
            _add_acc(a, 50)
            _add_referral(f"REF_L1_{a}", a, root, 51)
            _add_order(f"EVT_L1_{a}", a, 52, dev=f"DEV_T24_{a}", ip="IP_T24", p_out="PO_TREE", promo="SYBIL_PROMO")
        for i, a in enumerate(level2):
            _add_acc(a, 60)
            _add_referral(f"REF_L2_{a}", a, level1[i//2], 61)
            _add_order(f"EVT_L2_{a}", a, 62, dev=f"DEV_T24_{a}", ip="IP_T24", p_out="PO_TREE", promo="SYBIL_PROMO")
        for i, a in enumerate(level3):
            _add_acc(a, 70)
            _add_referral(f"REF_L3_{a}", a, level2[i//2], 71)
            _add_order(f"EVT_L3_{a}", a, 72, dev=f"DEV_T24_{a}", ip="IP_T24", p_out="PO_TREE", promo="SYBIL_PROMO")

    elif topo_id == "TOPO_25_ADVERSARIAL_BORDERLINE":
        # Calibrated directly on the decision boundary (moderate velocity + moderate structural degree)
        target_accs = [f"ACC_T25_{i}" for i in range(5)]
        for i, a in enumerate(target_accs):
            _add_acc(a, 55)
            for d in [60, 68, 76]:
                _add_order(f"EVT_T25_{a}_{d}", a, d, amount=450.0, dev="DEV_BORDERLINE", ip=f"IP_B_{i}", p_out=f"PO_B_{i%2}", promo="BORDER_PROMO")

    else:
        raise ValueError(f"Unknown topology ID: {topo_id}")

    events_df = pd.DataFrame(events)
    # Ensure all required standard event columns are present
    required_cols = [
        "event_id", "timestamp", "event_type", "account_id", "referrer_id",
        "amount", "device_id", "ip_id", "payout_id", "instrument_id",
        "promo_code", "order_status", "session_id"
    ]
    for col in required_cols:
        if col not in events_df.columns:
            events_df[col] = None

    accounts_df = pd.DataFrame(accounts)
    return events_df, accounts_df, as_of_ts, target_accs


TOPOLOGY_CATALOG = [
    # Family A: Graph Camouflage
    ("TOPO_01_DENSE_CLIQUE_CAMO", "Graph Camouflage", "Dense 6-node ring masking via high-degree benign merchant device"),
    ("TOPO_02_STAR_DISPERSION", "Graph Camouflage", "Star referral topology: central mule with 15 synthetic satellite accounts"),
    ("TOPO_03_BRIDGE_ISOLATION", "Graph Camouflage", "Two dense 4-node cliques joined solely by a single weak referral bridge"),
    ("TOPO_04_CYCLE_CHAIN_DILUTION", "Graph Camouflage", "8-node referral cycle with synthetic low-amount volume dilution"),
    ("TOPO_05_RANDOM_AFFILIATE_INJECTION", "Graph Camouflage", "Ring members injecting random legitimate affiliate codes to disrupt graph clustering"),

    # Family B: Temporal & Sleeper Attack Patterns
    ("TOPO_06_PROLONGED_SLEEPER_BURST", "Temporal & Sleeper", "Accounts dormant for 75 days executing burst promo orders in <24 hours"),
    ("TOPO_07_MICRO_VELOCITY_STAGGER", "Temporal & Sleeper", "Micro-velocity staggering: orders spaced 4 days apart to evade burst windows"),
    ("TOPO_08_BURST_THEN_ABANDON", "Temporal & Sleeper", "5 rapid promo orders on Day 60 followed by 20+ days total dormancy"),
    ("TOPO_09_SLOW_BURN_REFERRAL", "Temporal & Sleeper", "Slow-burn referral chain spreading across 40 days at 1 order per 8 days"),
    ("TOPO_10_LATE_DEVICE_COLLISION", "Temporal & Sleeper", "Clean accounts that collide on a shared device ID only on 5th transaction"),

    # Family C: Entity & Identifier Manipulation
    ("TOPO_11_PAYOUT_ROTATION", "Entity Manipulation", "8 fake accounts rotating across 1 shared bank account"),
    ("TOPO_12_DEVICE_FINGERPRINT_CHURN", "Entity Manipulation", "Attacker spoofing fresh device IDs per order on static IP & payout"),
    ("TOPO_13_SHARED_IP_SUBNET_HOPPING", "Entity Manipulation", "Adjacent IP subnet hopping (192.168.1.X) with shared device & payout"),
    ("TOPO_14_COLLUDING_MERCHANT_DISPUTE", "Entity Manipulation", "Colluding merchant self-generating orders and filing claim chargebacks"),
    ("TOPO_15_CIRCULAR_PAYOUT_RECYCLING", "Entity Manipulation", "Circular payout routing across 3 pairs of accounts"),

    # Family D: Extreme Graph Sparsity & Cold-Start
    ("TOPO_16_ISOLATED_PAIR_COLLUSION", "Extreme Sparsity", "Isolated 2-node pair sharing 1 device with zero background graph links"),
    ("TOPO_17_COLD_START_PROMO_FARM", "Extreme Sparsity", "5 brand-new accounts (2 orders each) sharing promo code with 0 prior edges"),
    ("TOPO_18_HIGH_VALUE_SINGLETON", "Extreme Sparsity", "Single n=1 whale account attempting promo exploit (tests ABSTAIN discipline)"),
    ("TOPO_19_ASYMMETRIC_BIPARTITE", "Extreme Sparsity", "10 accounts connected to 2 shared devices with no direct peer edges"),
    ("TOPO_20_ZERO_STRUCTURAL_SIGNAL", "Extreme Sparsity", "Coordinated behavioral ring with 100% anonymized/isolated entities"),

    # Family E: Hybrid / Evasion Stress Topologies
    ("TOPO_21_BEHAVIORAL_MIMICRY", "Hybrid / Evasion Stress", "Ring matching benign transaction velocity perfectly with shared payout & device"),
    ("TOPO_22_ASYMMETRIC_DISAGREEMENT", "Hybrid / Evasion Stress", "High structural density (deg>15) with low behavioral velocity (P_behav < 0.10)"),
    ("TOPO_23_SPLIT_PAYOUT_TRIANGLE", "Hybrid / Evasion Stress", "Triangular payout splitting across 3 accounts"),
    ("TOPO_24_SYBIL_REFERRAL_TREE", "Hybrid / Evasion Stress", "4-level deep sybil tree with geometric referral cascade"),
    ("TOPO_25_ADVERSARIAL_BORDERLINE", "Hybrid / Evasion Stress", "Ring calibrated exactly to the boundary thresholds (sym_KL ~ 0.49, P_fused ~ 0.48)"),
]


def run_handcrafted_adversarial_battery() -> Dict[str, Any]:
    """
    Executes the full 25-topology stress battery through the canonical pipeline.
    """
    print("=" * 80)
    print("ABUSERING SENTINEL — INDEPENDENT HAND-CRAFTED TOPOLOGY STRESS BATTERY")
    print("=" * 80)

    # 1. Load models
    fused_model = joblib.load(MODELS_DIR / "fused_calibrated.pkl")
    engine = DecisionEngine(kl_conflict_threshold=0.5)

    results = []
    family_stats: Dict[str, Dict[str, int]] = {}

    total_naive_caught = 0
    total_sentinel_caught = 0
    total_accounts = 0

    for topo_id, family, desc in TOPOLOGY_CATALOG:
        ev_df, acc_df, as_of_ts, target_accs = generate_topology_data(topo_id)

        # Build graph and extract features
        G = build_graph_as_of(ev_df, as_of_ts)
        struct_df = extract_account_structural_features(G, target_accs).set_index("account_id")
        behav_df = extract_account_behavioral_features(ev_df, as_of_ts, target_accs, acc_df).set_index("account_id")

        s_mat = struct_df[STRUCTURAL_FEATURES].reindex(target_accs).fillna(0.0)
        b_mat = behav_df[BEHAVIORAL_FEATURES].reindex(target_accs).fillna(0.0)

        p_struct, p_behav, p_fused, conflicts = fused_model.predict_proba_sub(s_mat, b_mat)

        n_orders_arr = b_mat["n_orders"].fillna(0).astype(int).values
        obs_days_arr = b_mat["account_age_days"].fillna(0).values

        dec_results = engine.decide_batch(
            account_ids=target_accs,
            p_fused_matrix=p_fused,
            p_struct_matrix=p_struct,
            p_behav_matrix=p_behav,
            observation_days=obs_days_arr,
            n_orders_arr=n_orders_arr,
            as_of_ts=as_of_ts
        )

        n_accs = len(target_accs)
        total_accounts += n_accs

        n_act = sum(1 for d in dec_results if d.decision == Decision.ACT)
        n_rev = sum(1 for d in dec_results if d.decision == Decision.REVIEW)
        n_wait = sum(1 for d in dec_results if d.decision == Decision.WAIT_MONITOR)
        n_abs = sum(1 for d in dec_results if d.decision == Decision.ABSTAIN)

        n_conflict_lane = sum(1 for d in dec_results if d.routing_lane == RoutingLane.CONFLICT_REVIEW)
        n_fused_lane = sum(1 for d in dec_results if d.routing_lane == RoutingLane.FUSED_AUTO)

        mean_sym_kl = float(np.mean([d.sym_kl_divergence for d in dec_results]))
        mean_p_struct_ac = float(p_struct[:, 2].mean())
        mean_p_behav_ac = float(p_behav[:, 2].mean())
        mean_p_fused_ac = float(p_fused[:, 2].mean())

        # Naive baseline comparison: geometric mean > 0.50 -> ACT, else drop/abstain
        naive_act = sum(1 for p in p_fused[:, 2] if p >= 0.50)
        # Sentinel effective caught: ACT + REVIEW (disagreement routing rescues missed cases)
        sentinel_caught = n_act + n_rev

        total_naive_caught += naive_act
        total_sentinel_caught += sentinel_caught

        if family not in family_stats:
            family_stats[family] = {"total_accs": 0, "naive_caught": 0, "sentinel_caught": 0, "review_rescued": 0}
        family_stats[family]["total_accs"] += n_accs
        family_stats[family]["naive_caught"] += naive_act
        family_stats[family]["sentinel_caught"] += sentinel_caught
        family_stats[family]["review_rescued"] += n_rev

        topo_res = {
            "topo_id": topo_id,
            "family": family,
            "description": desc,
            "n_accounts": n_accs,
            "mean_p_struct_ac": round(mean_p_struct_ac, 4),
            "mean_p_behav_ac": round(mean_p_behav_ac, 4),
            "mean_p_fused_ac": round(mean_p_fused_ac, 4),
            "mean_sym_kl": round(mean_sym_kl, 4),
            "decision_breakdown": {
                "ACT": n_act,
                "REVIEW": n_rev,
                "WAIT_MONITOR": n_wait,
                "ABSTAIN": n_abs
            },
            "routing_lane_breakdown": {
                "conflict_review": n_conflict_lane,
                "fused_auto": n_fused_lane,
                "abstain": n_abs
            },
            "naive_geometric_caught": naive_act,
            "sentinel_effective_caught": sentinel_caught,
            "conflict_review_rescue_count": n_rev
        }
        results.append(topo_res)
        print(f"[{topo_id}] {family} | Accs={n_accs} | ACT={n_act}, REV={n_rev}, WAIT={n_wait}, ABS={n_abs} | sym_KL={mean_sym_kl:.2f} | Naive={naive_act} vs Sentinel={sentinel_caught}")

    summary = {
        "qualifier": (
            "Independent out-of-distribution structural stress battery evaluated on N=25 deterministic "
            "failure topologies constructed without data/simulator.py; tests topological edge cases and routing robustness."
        ),
        "total_topologies_evaluated": len(results),
        "total_accounts_evaluated": total_accounts,
        "overall_naive_caught": total_naive_caught,
        "overall_sentinel_caught": total_sentinel_caught,
        "overall_naive_recall_pct": round(total_naive_caught / max(1, total_accounts) * 100, 2),
        "overall_sentinel_effective_recall_pct": round(total_sentinel_caught / max(1, total_accounts) * 100, 2),
        "total_cases_rescued_by_conflict_review": sum(r["conflict_review_rescue_count"] for r in results),
        "family_breakdown": family_stats,
        "topologies": results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "handcrafted_adversarial_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"OVERALL RESULTS: {summary['overall_sentinel_effective_recall_pct']}% Effective Caught vs {summary['overall_naive_recall_pct']}% Naive Fusion")
    print(f"Conflict Review Rescued: {summary['total_cases_rescued_by_conflict_review']} accounts from silent false negative drop")
    print(f"Results saved to {out_path}")
    print("=" * 80)
    return summary


if __name__ == "__main__":
    run_handcrafted_adversarial_battery()
