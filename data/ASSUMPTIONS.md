# Simulator Assumptions — AbuseRing Sentinel

This file documents every generation assumption in the synthetic simulator.
Honest limitations are marked **[SIMPLIFIED]** or **[UNREALISTIC]**.
This document must be updated whenever the simulator changes.

> **VERSION**: 2.0 — Updated 2026-08-28 to incorporate required amendments A1-A6.
> Changes from v1.0 are marked **[NEW-A1]** through **[NEW-A6]**.

---

## 1. Entity Counts (Fixed Seed = 42)

| Entity Type          | Count   | Notes |
|----------------------|---------|-------|
| Accounts             | 5,000   | mix determined by abuse_prevalence parameter (default 60/25/15%) |
| Devices              | 3,500   | accounts share devices at realistic rates per class |
| IP addresses         | 2,000   | includes subnet-level grouping (last octet masked) |
| Payment instruments  | 4,000   | cards/UPI VPAs; shared within rings and families |
| Payout destinations  | 1,800   | bank accounts / wallets; key signal for promo-abuse rings |
| Orders               | ~72,000 | distributed across 90-day timeline |
| Sessions             | ~55,000 | ~1-3 sessions per order, with retries |

---

## 2. Timeline

- **Simulation window**: 90 days (Days 1-90)
- **Temporal split**: Train = Days 1-54, Validation = Days 55-72, Test = Days 73-90
  - Rationale: 60/20/20 split by time, not random.
- **Event granularity**: second-level timestamps (Unix epoch)

**[SIMPLIFIED]**: Real payment systems have intra-day seasonality (lunch peaks, midnight
fraud spikes). The simulator uses a flat Poisson arrival process with a mild diurnal
multiplier (business hours x2), which does not capture true seasonality complexity.

---

## 3. Ground-Truth Classes

### [NEW-A1] Dual Label Fields

- **label_true** -- deterministic ground truth, stored in labels.parquet. Used ONLY in /evals.
  NEVER read by /features or /models. Enforced by test_leakage.py::test_label_not_in_graph_or_features.
- **label_observed** -- label_true with 3% uniform noise applied (see Section 4).
  Any historical-label-dependent evaluation step must use label_observed, not label_true.
- The feature pipeline and model suite are FULLY UNSUPERVISED with respect to labels.

### [NEW-A4] Prevalence Parameter

The account mix is controlled by abuse_prevalence dict parameter:

  abuse_prevalence = {
      "benign_independent": 0.60,
      "benign_coordinated": 0.25,
      "abusive_coordinated": 0.15,
  }

Default: 60/25/15%. Pre-built alternate regimes for Stage 11:
- Low:  {BI: 0.90, BC: 0.07, AC: 0.03} -- realistic production prevalence (~3% abusive)
- High: {BI: 0.40, BC: 0.30, AC: 0.30} -- stress test for high-fraud regimes

Counts of rings, ring members, and all entity pools scale proportionally.

### 3a. Benign-Independent (~60% of accounts, configurable)

- Each account operates with its own device, IP, and payment instrument.
- Occasional natural sharing: ~5% chance of sharing a device with one other benign-independent
  account (e.g., household member who is NOT coordinated with them).
- Payout destinations: randomly sampled from lower 50% of the standard payout pool.
  This creates ~3.3x natural collision rate (~3 BI accounts per payout ID), which is
  realistic and makes payout-degree a noisy-not-trivial signal.
- 8% of BI accounts occasionally use a public promo code (independent discovery, <=15% of orders).
- Orders are random, no referral links.

**[SIMPLIFIED]**: Real benign-independent users occasionally have repeated purchase patterns
(subscriptions, loyalty programs) not modeled here.

### 3b. Benign-Coordinated (~25% of accounts, configurable)

This is the **hard negative** class -- the core challenge. Two subtypes:

**Family groups** (70% of benign-coordinated):
- Groups of 3-8 accounts sharing 1-2 devices, 1-2 IP addresses, 1-3 payment instruments.
- Order amounts similar (family budget correlation, sigma=0.3 of mean).
- Payout destinations: DIFFERENT per member (no shared cashout) -- sampled from 50-70% of payout pool.
- Timing: orders spread across day/evening, no artificial staggering.
- Referral links between family members exist (gifted referral).

**Office/shared-infra groups** (30% of benign-coordinated):
- Groups of 5-20 accounts sharing 1-3 IP subnets (corporate NAT), 0-1 shared devices.
- Payment instruments are INDEPENDENT, payout destinations INDEPENDENT.
- Timing: clustered in business hours.
- No referral links.

### [NEW-A6a] Counterfactual subset: hard_bc

- 15% of benign-coordinated family groups are generated with ONE additional shared payout
  destination (simulating a joint bank account or shared UPI ID).
- label_true remains benign_coordinated for all members.
- Flagged as counterfactual_subset = "hard_bc" in labels.parquet.
- Reportable separately in Stage 11 under hard_bc_fp_rate.
- Rationale: tests whether model over-triggers on shared payout alone vs. joint signal.

**[UNREALISTIC]**: 15% of family groups make a coupon redemption coincidentally (genuine label noise).

### 3c. Abusive-Coordinated (~15% of accounts, configurable)

Three ring subtypes -- [NEW-A5] adds a third (referral-farming, test-only).

### [NEW-A3] Ring Formation Timeline

Ring formation is distributed across the full 90-day window.
Each ring has two new fields in rings.parquet:
- ring_formation_start_day: day the first member account is created/activated (Day 1-75)
- ring_formation_complete_day: day the last member completes their ring activity pattern

**REQUIRED MINIMUM**: >=20% of rings (across all subtypes) must have
ring_formation_start_day >= 55 (formation explicitly starts in val or test window).
Enforced by Stage 3 validation with re-seed logic.

Rationale: Without this, the loss-vs-detection-lag curve is meaningless -- rings that
fully form in train time are trivially detectable at any evaluation checkpoint.

**Promo-abuse rings** (50% of abusive-coordinated accounts):
- Ring size: 5-25 accounts.
- ring_formation_start_day sampled uniformly in [Day 1, Day 75].
- Pattern: coordinated activation window, staggered 2-48 hours.
- Shared signals: 1-3 devices, 1-5 IPs (VPN-simulated), 1-3 payout destinations.
  Payout IDs sampled from upper 20% of payout pool (exclusive high-sharing zone).
- Coupon/promo usage: 80-100% of ring members redeem same promo code.
- Referral links: dense intra-ring chain.
- Sleeper accounts (10% of ring members): [NEW-A2]
  - label_true = abusive_coordinated
  - Additional flag: partial_signal = True in labels.parquet
  - Share payout destination but also have independent orders (lower behavioral score)
  - Reportable separately in Stage 11 under sleeper_recall and sleeper_f1

**Return-abuse rings** (30% of abusive-coordinated accounts):
- Ring size: 3-12 accounts.
- ring_formation_start_day sampled uniformly in [Day 1, Day 75].
- Pattern: buy expensive items, return them with different payment method for cash equivalent.
- Shared signals: 1-2 devices, 1-3 IPs, shared payout destination (upper 20% pool).
- No systematic promo usage. Referral links: sparse (1-2 per ring).

### [NEW-A5] Referral-farming rings (TEST WINDOW ONLY)

- 20% of abusive-coordinated accounts.
- ring_formation_start_day >= 73 (ONLY in test window: Days 73-90).
- Ring size: 8-30 accounts.
- Structurally DISTINCT: dense referral chains (every member refers the next),
  but MINIMAL shared payment infrastructure (low device/IP/payout sharing).
- Shared signals: referral edges only. Each member uses own payout destination.
- Abuse signal: coordinated referral bonus extraction.
- Purpose: held-out generalization test for Stage 11.
  A model trained only on promo/return patterns should degrade on this subtype.
  Benchmark MUST report referral_ring_recall separately.

### [NEW-A6b] Counterfactual subset: varied_payout_ac

- 10% of promo/return rings have deliberately varied (non-shared) payout destinations.
  Each member uses a unique payout ID (no cross-member sharing on this signal).
- label_true remains abusive_coordinated.
- Flagged as counterfactual_subset = "varied_payout_ac" in labels.parquet.
- Reportable separately in Stage 11 under low_signal_ac_recall.
- Rationale: tests detection through behavioral + timing signals alone when
  the strongest structural signal (shared payout) is absent.

**[SIMPLIFIED]**: Real rings use SIM swapping, IMEI spoofing, emulator farms -- not modeled.
**[SIMPLIFIED]**: Ring activation uses truncated Gaussian stagger (sigma=12h), not adversarial optimization.

---

## 4. Noise and Label Uncertainty

- **label_observed noise**: 3% of abusive_coordinated accounts have label_observed flipped to
  benign_independent. label_true is NEVER modified. [NEW-A1]
- **Partial ring observation**: 20% of ring edges are unobserved (encrypted channels).
- labels.parquet columns: account_id | label_true | label_observed | partial_signal | counterfactual_subset
- /features and /models must NOT read label_true. Tests enforce this.

**[UNREALISTIC]**: Real label noise is not uniformly random -- it is biased toward
accounts that avoid detection. The simulator uses uniform noise.

---

## 5. Feature Generation Assumptions

- **Behavioral features**: order frequency, amount statistics, promo usage rate, return rate,
  session statistics. Computed as-of timestamp T (no future leakage).
- **Structural features**: degree centrality, clustering coefficient, shared-destination count.
  Computed as-of timestamp T (no future leakage).

**[SIMPLIFIED]**: Real features would include device fingerprint richness, geolocation
consistency, typing cadence, etc. None of these are modeled.

---

## 6. Cost Model Assumptions (ALL SIMULATED)

| Parameter | Simulated Value | Justification |
|-----------|----------------|---------------|
| c_false_positive | Rs500/account | estimated friction/support cost |
| c_false_negative | Rs2000/account | estimated promo value stolen |
| c_review | Rs150/account | analyst time cost |
| c_wait | Rs50/day/account | accrued daily loss |
| discount_rate | 0.0 | no time-discounting |

**[SIMULATED]**: All cost constants are illustrative, NOT from actual Razorpay data.
Parameterized in data/cost_config.json.

---

## 7. Notes for Future Stages (Not Yet Implemented)

### 7a. Edge Weight Order (Stage 5 -- Graph Builder)

Fixed weighting priority for edge types when building the entity graph:

  payout-destination-share > payment-instrument-share > device-share > IP-subnet-share > referral-link
         5.0                          4.0                   3.0             2.0               1.0

Rationale: payout-destination sharing has highest specificity for coordinated cashout.
Referral links have lowest weight (easily manufactured).

### 7b. Stage 3 Validation Requirements

Stage 3 must assert minimum per-class counts in Train/Val/Test:
- Train: >=200 AC, >=500 BC, >=1000 BI
- Val:   >=50 AC, >=200 BC, >=400 BI
- Test:  >=50 AC (including >=5 referral-farming members), >=200 BC, >=400 BI
- Sleeper subset: >=20 sleeper accounts in test split
- Hard-BC subset: >=30 hard-BC accounts in test split
- Referral-farming rings: ALL members must fall ONLY in test split
- Ring formation: >=20% of rings with ring_formation_start_day >= 55

Re-seed and regenerate if minimums not met (try seeds 42-52).

### 7c. Stage 3 Shortcut-Detection Check

A depth-2 decision tree trained ONLY on:
- account_creation_order (row index)
- account_id (as integer)
- raw_creation_timestamp

must NOT predict label_true above near-chance (AUC <= 0.55) on test split.
If it exceeds this threshold, the simulator has a generation-order artifact.
This check's result MUST be reported in the README.

---

## 8. What This Simulator Does NOT Model

- Real device fingerprint APIs (DeviceAtlas, ThreatMetrix)
- Real IP reputation databases
- Velocity rules from a production rule engine
- Chargeback data or bank-side signals
- Account age from external KYC
- Any PII -- all entities are synthetic IDs
- SIM swapping, IMEI spoofing, emulator farms
- Adversarially-optimized ring timing patterns

---

## 9. Running Known Limitations (validated against v2.0 test data)

These limitations are stated plainly rather than implied. Each was identified
by running the real pipeline against v2.0 simulator output.

1. **Structural evidence contributes primarily to conflict-detection, not direct prediction.**
   80.8% of true AC accounts in the test window lack strong standalone structural
   signal (shared_payout_degree, multi_signal_edges). This is consistent with
   the ~20% partial-ring-observation constraint in Section 3.4
   (`ring_formation_start_day >= val_start_day` for ≥20% of rings). It is a
   property of the problem domain -- rings that are partially formed as of the
   evaluation checkpoint have not yet generated observable payout co-sharing --
   and should not be treated as a modelling bug or a failure of the graph pipeline.

2. **Routing recall must always be reported as two numbers, never one.**
   Effective recall at KL=0.5 is 82% (38 auto-ACT + 124 human-REVIEW out of 198 AC);
   the remaining 18.2% (36 accounts) are ABSTAIN'd due to insufficient evidence
   (n_orders < 2). Direct auto-ACT recall is 19.2%. Never state "100% recall"
   without the ABSTAIN and routing caveats in the same sentence.

3. **Cost model assumes flat per-account FN cost; it does not model compounding
   loss from undetected rings over time.** Under this assumption, routing-based
   100%-ceiling-recall is not cost-justified versus behavioral-only detection.
   Full corrected cost breakdown (SIMULATED, `c_fn=Rs2000`, `c_review=Rs150`):
   - Behavioral-only: Rs30,500 (TP=189, FP=25, FN=9; no ABSTAIN gate)
   - Routing KL=0.5, ABSTAIN=FN (worst case): Rs1,88,850
     (779 reviews × Rs150 + 36 ABSTAIN'd AC × Rs2,000)
   - Routing KL=0.5, ABSTAIN=wait-cost (best case): Rs1,49,250
     (779 reviews × Rs150 + 36 × 18 days × Rs50)
   With ABSTAIN cost included, the ABSTAIN-FN component alone (Rs72,000) already
   exceeds behavioral total (Rs30,500), so at any c_review >= 0, routing is never
   cost-competitive with behavioral-alone under this cost model. The Rs37/account
   break-even figure from an earlier analysis was incorrect -- it excluded ABSTAIN.
   A more complete cost model incorporating time-to-detection-dependent loss growth
   (e.g., a missed ring keeps recruiting members or draining promo budget the longer
   it goes undetected) is identified as future work, not implemented here.

4. **promo_rate is not the primary behavioral discriminator** (rank 6 of 16, 6.7%
   importance). The top features are order-timing and amount statistics
   (first_order_age_days, account_age_days, mean/std/max_order_amount). These
   reflect real ring coordination patterns but also partly encode simulator
   generation choices (tighter lognormal amount distributions for ring members).

5. **Component size is degenerate in this graph.** Incidental BI household
   device-sharing and BC family IP-sharing chains merge nearly all accounts
   into one giant component (~4,987 nodes). component_size is not a useful
   discriminator. The reliable structural discriminators are shared_payout_degree
   (directional: AC > BI) and multi_signal_edges (AC ~563× BI).

6. **The geometric-mean fusion formula collapses to near-zero** when p_struct ≈ 0
   regardless of p_behav. This is mathematically correct given the formula but
   operationally unsuitable as a single-score threshold: 80.8% of true AC accounts
   have p_struct median = 0.000, making p_fused median = 0.095 even when
   p_behav median = 0.989. The routing design addresses this by making conflict
   detection the primary use of p_struct, not probability blending.

---

*Last updated: 2026-08-28 (v2.0 -- amendments A1-A6 + Stage notes 7a-7c + Section 9 limitations incorporated, cost-model framing corrected).*
*Must be updated if simulator parameters change.*
