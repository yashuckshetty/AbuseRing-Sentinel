# AbuseRing Sentinel

A **simulated** abuse-ring detection system for a payments platform context.
Detects coordinated abusive accounts (AC rings) that exploit promotional codes
through shared payout destinations, devices, and IPs, using a behavioural x
structural x AI evidence fusion pipeline with a cost-aware decision layer.

> **All data is fully synthetic.** No real transaction, account, or PII data is
> used anywhere. All cost figures are illustrative assumptions. See
> `data/ASSUMPTIONS.md` for the full contract.

---

## Contents

- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Architecture overview](#architecture-overview)
- [Data layer (Stage 2)](#data-layer-stage-2)
- [Feature pipeline (Stage 3)](#feature-pipeline-stage-3)
- [Model ladder (Stage 4)](#model-ladder-stage-4)
- [Decision methodology (Stage 5)](#decision-methodology-stage-5)
- [AI evidence layer](#ai-evidence-layer)
- [Policy gate](#policy-gate)
- [Test suite](#test-suite)
- [Robustness Results (Stage 12a)](#robustness-results-stage-12a)
- [Trajectory Evaluation (Evolving-Risk Dynamics)](#trajectory-evaluation-evolving-risk-dynamics)
- [KL-Routing Ablation Study](#kl-routing-ablation-study)
- [Prevalence-Shift Sensitivity Analysis](#prevalence-shift-sensitivity-analysis)
- [Multi-Seed Robustness (Seeds 42, 43, 44)](#multi-seed-robustness-seeds-42-43-44)
- [Known limitations](#known-limitations)

---

## Project structure

```
AbuseRing Sentinel/
+-- data/
|   +-- simulator.py          # Synthetic dataset generator (v2.0)
|   +-- ASSUMPTIONS.md        # Full data contract and limitations
|   +-- cost_config.json      # Simulated cost constants
|   +-- events.parquet        # ~41k synthetic events
|   +-- accounts.parquet      # 5,000 synthetic accounts
|   +-- labels.parquet        # label_true / label_observed / metadata
|   +-- rings.parquet         # Ring membership ground truth
|   +-- split_info.json       # Train/val/test temporal boundaries
+-- graph/
|   +-- temporal_graph.py     # As-of-T graph construction + feature extraction
+-- features/
|   +-- feature_pipeline.py   # Structural + behavioural feature matrices
+-- models/
|   +-- model_suite.py        # Rung 1-5 model ladder + FusedCalibratedClassifier
+-- decision/
|   +-- decision_engine.py    # KL-routing decision engine (v2.0)
+-- ai/
|   +-- evidence_reasoner.py  # LLM evidence gap reasoning (boundary-enforced)
+-- policy/
|   +-- policy_gate.py        # Final policy application + audit trail
+-- evals/
|   +-- metrics.json          # Stored evaluation results per model per split
+-- tests/                    # 49 tests (real data, no mocks for stage tests)
+-- conftest.py
```

---

## Quickstart

```bash
# 1. Generate synthetic dataset (v2.0, ~2 min)
python data/simulator.py

# 2. Train the model ladder and save artefacts
python models/model_suite.py

# 3. Run the full test suite (49 tests)
python -m pytest tests/ -v

# 4. Run a single decision end-to-end
python -c "
import sys; sys.path.insert(0, '.')
from decision.decision_engine import DecisionEngine
import numpy as np
engine = DecisionEngine(kl_conflict_threshold=0.5)
result = engine.decide(
    account_id='ACC_001',
    p_fused=np.array([0.05, 0.10, 0.85]),
    p_struct=np.array([0.03, 0.05, 0.92]),
    p_behav=np.array([0.03, 0.05, 0.92]),
    observation_days=30.0, n_orders=5, as_of_ts=1707776000,
)
print(result.decision, result.rationale)
"
```

**Dependencies:** `pandas`, `numpy`, `networkx`, `lightgbm`, `scikit-learn`,
`joblib`, `pyarrow` (for parquet), `pytest`.

---

## Architecture overview

```
events.parquet
accounts.parquet  -->  temporal_graph.py  -->  structural features (Rung 4)
rings.parquet                                        |
labels.parquet                                       v
               -->  feature_pipeline.py  -->  behavioural features (Rung 3)
                                                     |
                         +---------------------------+---------------------------+
                         |             model_suite.py                           |
                         |   Rung 1: majority_class (baseline)                  |
                         |   Rung 2: rule_based                                 |
                         |   Rung 3: behavioral_lgbm                            |
                         |   Rung 4: structural_lgbm                            |
                         |   Rung 5: fused_calibrated (geometric mean)          |
                         +---------------------------+---------------------------+
                                                     |
                                                     v
                                       decision_engine.py  (KL-routing)
                                         |-- abstain lane         --> ABSTAIN (n_orders < 2)
                                         |-- conflict_review lane --> REVIEW  (sym_KL > 0.5)
                                         |-- fused_auto lane      --> ACT     (p_fused >= 0.70)
                                         |                        --> WAIT_MONITOR (p_fused < 0.70)
                                                     |
                                       evidence_reasoner.py  (LLM boundary)
                                                     |
                                            policy_gate.py  (audit trail)
```

### Architectural Intent vs. Demonstrated Behavior

To maintain rigorous scientific honesty, this document distinguishes between what is empirically demonstrated and what represents architectural intent:

- **Demonstrated Behavior (Single-Checkpoint Evaluation)**: The multi-modal feature ladder, symmetric KL-divergence conflict routing, 0% auto-ACT false positive rate, boundary-tested AI advisory layer, and cost-benefit trade-offs are empirically proven at a fixed time cutoff ($T = \text{Day 90}$) across $3,467$ active test accounts.
- **Architectural Intent (Evolving Risk Process)**: The overarching design thesis—that payment abuse risk is an evolving trajectory rather than a static snapshot—is an architectural premise. While graph features use as-of-$T$ temporal windowing, tracking how decisions transition across ring formation days is evaluated explicitly in the [Trajectory Evaluation](#trajectory-evaluation) section.

### Decision Engine Routing Lanes vs. Operational Decisions

The decision engine cleanly separates the routing mechanism from the operational decision:

| Routing Lane (`routing_lane`) | Criterion | Operational Decision (`decision`) | Operational Action |
|---|---|---|---|
| `abstain` | $n\_\text{orders} < 2$ | `ABSTAIN` | Defer scoring; await additional transaction history |
| `conflict_review` | $\text{sym\_KL}(p_{\text{struct}}, p_{\text{behav}}) > 0.50$ | `REVIEW` | Route to human fraud analyst with AI evidence narrative |
| `fused_auto` | Low conflict ($\text{sym\_KL} \le 0.50$) & $p_{\text{fused}}[\text{AC}] \ge 0.70$ | `ACT` | Automated mitigation (0 FP demonstrated) |
| `fused_auto` | Low conflict ($\text{sym\_KL} \le 0.50$) & $p_{\text{fused}}[\text{AC}] < 0.70$ | `WAIT_MONITOR` | Baseline passive monitoring; no intervention |

---

## Data layer (Stage 2)

The synthetic dataset is generated by `data/simulator.py` against the contract
in `data/ASSUMPTIONS.md` (v2.0).

### Account classes

| Class | N | Description |
|---|---|---|
| benign_independent (BI) | 3,000 | Normal users; household device/IP sharing only |
| benign_coordinated (BC) | 1,500 | Family/friend groups; legitimate coordination |
| abusive_coordinated (AC) | 500 | Abuse rings; shared payouts, devices, IPs, referrals |

### Temporal splits (time-based, no random shuffle)

| Split | Days | Approx accounts active |
|---|---|---|
| Train | 1-54 | 4,656 |
| Val | 55-72 | 3,519 |
| Test | 73-90 | 3,467 |

Accounts appear in multiple windows because they generate events across the
full 90-day simulation. Per-window feature vectors use strict as-of-T cutoffs
to prevent leakage.

### Label design

Each account carries two labels:
- `label_true`: ground truth, never exposed to `/features` or `/models`
- `label_observed`: what operations would see (22 AC accounts have
  `label_observed != label_true` due to simulated label noise)

Label leakage is enforced by `tests/test_leakage.py`.

### Shortcut-detection check

A depth-2 decision tree trained on `created_ts` alone achieves AUC=0.474 on
the test split -- no exploitable temporal shortcut from account creation timestamp.

---

## Feature pipeline (Stage 3)

`features/feature_pipeline.py` constructs two feature families per account
per temporal split.

### Structural features (16 features, from temporal graph)

| Feature | What it measures |
|---|---|
| shared_payout_degree | Distinct accounts sharing a payout destination |
| multi_signal_edges | Edges carrying >=2 signal types simultaneously |
| shared_ip_degree | Co-IP neighbours |
| shared_device_degree | Co-device neighbours |
| referral_degree | Referral connections |
| community_size | Louvain community membership size |
| component_size | Connected component size (NOTE: degenerate -- see limitations) |
| clustering_coeff | Local graph clustering |
| + 8 more | Velocity and edge-type-specific counts |

### Behavioural features (16 features)

n_orders, mean/std/max_order_amount, promo_rate, burst_score,
n_referrals_sent, referral_conversion_rate, order_days_active,
account_age_days, first_order_age_days, n_distinct_devices/ips/payouts,
return_rate, order_velocity.

Top-5 by LightGBM importance: first_order_age_days (15.8%), account_age_days
(13.6%), mean_order_amount (13.5%), std_order_amount (13.3%),
max_order_amount (12.7%). promo_rate ranks 6th at 6.7%. Behavioral recall is
driven by order-timing/amount patterns, not promo rate alone.

---

## Model ladder (Stage 4)

`models/model_suite.py` trains five rungs against real v2.0 data.
All single-model headline metrics in the ladder table below are evaluated on the **test split** (days 73-90) under **Seed 42** against `label_true`.

> [!WARNING]
> **Multi-Seed Behavioral Variance Caveat:**
> The headline behavioral recall ($95.45\%$) and F1 ($0.917$) are single-seed (Seed 42) results. Multi-seed evaluation across seeds 42, 43, and 44 reveals that **behavioral recall varies dramatically from $27.03\%$ to $95.45\%$ (mean: $58.63\%$)**, while structural metrics remain comparatively stable (Recall: $17.84\% - 30.59\%$, Precision: $78.33\% - 100\%$). This instability must be read as an overarching caveat on every behavioral-only claim in this document: behavioral features are tightly coupled to specific synthetic order-timing distributions (connecting directly to the *Simulator-encoded patterns* limitation in Section 9), making standalone behavioral performance brittle across random realizations. In contrast, the KL-routing architecture achieves robust effective recall ($78.38\% - 84.02\%$, mean $81.40\%$) and invariant zero auto-ACT false positives across all seeds.

| Rung | Model | Prec (AC) | Recall (AC) | F1 (AC) | Notes |
|---|---|---|---|---|---|
| 1 | majority_class | 0.000 | 0.000 | 0.000 | Predicts BI for all |
| 2 | rule_based | 0.065 | 0.182 | 0.095 | Shared-payout threshold rules |
| 3 | behavioral_lgbm | **0.883** | **0.955** | **0.917** | Single-seed 42 result (Multi-seed range: 27.0% - 95.5%) |
| 4 | structural_lgbm | 0.783 | 0.237 | 0.364 | Multi-seed stable (Precision: 78.3% - 100.0%) |
| 5 | fused_calibrated | 1.000 | 0.308 | 0.471 | Geometric-mean fusion; collapses when $p_{\text{struct}} \approx 0$ |

**Structural model (Rung 4) — Honest empirical finding:**
The structural model is **not a strong standalone detector** ($F_1 = 0.364$, Recall $= 23.7\%$, despite an AUC of $0.907$). In the evaluation window, $80.8\%$ of true AC accounts lack dense structural links because rings are still forming or operate without direct payout co-sharing. Rather than masking this, the pipeline treats it as a core architectural insight: graph features are valuable not as standalone classifiers, but as an **evidence-disagreement detector**, a **safe-routing safeguard** against false positives, and a **generalization anchor for unseen abuse topologies** (such as `referral_farming`).

**Fused model (Rung 5) — Failure mode of single-score blending:**
The naive geometric-mean fusion $\sqrt{p_{\text{struct}} \times p_{\text{behav}}}$ collapses toward $0$ whenever $p_{\text{struct}} \approx 0$, regardless of how elevated $p_{\text{behav}}$ is. Across $80.8\%$ of true AC accounts: $p_{\text{struct}}$ median $= 0.000$, $p_{\text{behav}}$ median $= 0.989 \implies p_{\text{fused}}$ median $= 0.095$. As a single-score ranking model, Rung 5 achieves $100\%$ precision but suffers catastrophic recall degradation ($30.8\%$). This failure mode directly motivates the KL-divergence routing engine in Stage 5, which treats disagreement as a routing signal rather than blending it away into a degraded scalar score.

---

## Decision methodology (Stage 5)

### Leading finding: behavioral-only is cost-superior under the current cost model

This finding is presented first because it demonstrates cost-aware decisioning
applied to the project's own design choices -- not just to external detection
decisions. This is the core thesis in action.

**The cost model assumes flat per-account FN cost (Rs2,000), regardless of
how long an account goes undetected.** Under this assumption:

| Strategy | FP cost | FN / Abstain cost | Review cost | Total (SIMULATED) | Recall |
|---|---|---|---|---|---|
| Behavioral-only (Rung 3) | Rs12,500 | Rs18,000 (9 FN) | Rs0 | **Rs30,500** | 95.5% direct |
| Routing KL=0.5 (ABSTAIN=wait, best case) | Rs0 | Rs32,400 (36 x 18d x Rs50) | Rs1,16,850 | **Rs1,49,250** | 82% effective |
| Routing KL=0.5 (ABSTAIN=FN, worst case) | Rs0 | Rs72,000 (36 x Rs2,000) | Rs1,16,850 | **Rs1,88,850** | 82% effective |

All costs are SIMULATED. See `data/cost_config.json`.

**Behavioral-only is unambiguously cheaper (4-6x lower cost)** under this model.

Routing would only be cost-justified if undetected AC rings impose compounding
or tail losses that grow with time -- e.g., a missed ring keeps recruiting
members or draining promo budget the longer it goes undetected. The current
flat-FN cost model does not capture that. Modelling time-to-detection-dependent
loss growth is explicitly identified as future work.

> Cost-model limitation stated plainly, not papered over: the routing engine
> is implemented because the cost model is incomplete, the escalation path is
> principled, and the conflict signal from p_struct is most useful for routing
> -- not for blending into a score that collapses.

### Routing strategy

`decision/decision_engine.py` uses symmetric KL divergence between the
structural and behavioural probability vectors to route each account:

```
ABSTAIN:       n_orders < 2
               --> Insufficient evidence; defer to next scoring pass

REVIEW:        sym_KL(p_struct, p_behav) > 0.5
               --> Models disagree; human adjudication required
               --> 62.6% (124/198) of true AC accounts route here (expected:
                  partial-ring-observation in test window)

ACT:           low-conflict AND p_fused[AC] >= 0.70
               --> 19.2% (38/198) of true AC accounts; 0 false positives in auto-ACT lane

WAIT_MONITOR:  low-conflict AND p_fused[AC] < 0.70
               --> 0 true AC accounts end here (correct)
```

**Routing recall -- always stated as two numbers:**
82% effective recall via routing: direct auto-ACT recall 19.2% (38/198),
remaining 62.6% (124/198) correctly routed to human REVIEW.
The 18.2% (36/198) who ABSTAIN have n_orders=1 -- not detectable by any
evidence-based system in this test window.

**KL threshold is configurable** (default=0.5, set at construction):
```python
engine = DecisionEngine(kl_conflict_threshold=0.5)
```

Every `DecisionResult` carries `routing_lane`, `sym_kl_divergence`,
`evidence_conflict`, `kl_conflict_threshold`, and a full audit trail including
cost estimates, recall caveat, and structural signal limitation note.

### Structural signal: role and constraint

Structural evidence contributes primarily to **evidence-disagreement detection**, **safe routing**, and **generalization across unseen topologies** (`referral_farming`), rather than raw standalone predictive power.
- $80.8\%$ of true AC accounts in the test window lack dense standalone structural connections at the time of scoring (consistent with partial ring observation and in-flight formation).
- In isolation, the structural model achieves $F_1 = 0.364$ (Recall $= 23.7\%$).
- However, when combined with behavioral scoring via symmetric KL-divergence, structural evidence flags $100\%$ of evaluable unseen referral rings into the human REVIEW queue and prevents false auto-ACTs on benign family coordination (`hard_bc` auto-ACT $\text{FP} = 0.00\%$).
- The graph's primary operational role is identifying **where model uncertainty requires human adjudication**, rather than serving as an autonomous blocking heuristic.

---

## AI evidence layer

`ai/evidence_reasoner.py` provides LLM-assisted evidence gap reasoning for accounts routed to the human REVIEW lane.

**Verification Mode (Labeled Fact)**:
- **Verified against: Deterministic Mock Mode only** (tests and pipelines run in an offline/deterministic test harness without active external `GEMINI_API_KEY` dependencies).
- **Live API Pathway**: Fully implemented via Google GenAI SDK (`gemini-1.5-flash`) with structured JSON schema output and strict runtime validation.

Strict boundary contracts enforced and validated on all outputs:
- **No risk scores**: The LLM may not produce, modify, or suggest numeric probability scores
- **No fabricated IDs**: Output is pattern-matched via regex and validated against payload sets for account IDs (`ACC_`), ring IDs (`PROMO_`, `RETURN_`, `REFARM_`), and entity IDs (`DEV_`, `IP_`, `PAY_`, `INSTR_`)
- **No forbidden actions**: Strictly rejects operational action recommendations (`block`, `terminate`, `suspend`, `ban`)
- **Evidence-only narrative**: All prose is framed objectively as investigative evidence for human analyst judgment

---

## Policy gate

`policy/policy_gate.py` applies the final deterministic policy layer:

- Enforces minimum evidence thresholds before ACT decisions are emitted
- Attaches the full audit trail to every output (decision, sub-scores,
  KL divergence, cost estimates, cost model notes)
- AI advisory does not modify the numeric decision; it adds evidence narrative only

---

## Test suite

50 tests across 5 files. All stage integration tests load real v2.0 parquet
output -- no mocked fixtures for data-level assertions.

| File | Tests | Coverage |
|---|---|---|
| test_leakage.py | 6 | Temporal leakage: no future events in graph, label isolation, feature monotonicity |
| test_feature_pipeline.py | 15 | Real-data feature assertions: AC > BI on structural/behavioural features; split integrity |
| test_decision_engine.py | 14 | Routing logic (unit) + real-data integration: 0 FP in auto-ACT, >=80% effective recall, ABSTAIN gate |
| test_ai_boundary.py | 8 | LLM boundary contracts: no scores, no fabrication, no forbidden actions, entity hallucination checks |
| test_policy_gate.py | 7 | Policy gate logic + audit trail emission |

```bash
python -m pytest tests/ -v
# Expected: 50 passed
```

---

## Robustness Results (Stage 12a)

The full model and KL-routing decision pipeline was stress-tested across 5 labeled counterfactual and adversarial subsets in the $v2.0$ test split ($N=3,467$ accounts):

| Stress Test Scenario | Subset & Filter | N | Auto-ACT | REVIEW | WAIT | ABSTAIN | Effective Recall / FP Rate | Finding & Failure Mode |
|---|---|---|---|---|---|---|---|---|
| **1. Unseen Ring Structure** | `referral_farming` (test-window only) | 143 | 0 (0.0%) | 107 (74.8%) | 0 (0.0%) | 36 (25.2%) | **74.8%** (100% of evaluable) | $P(\text{struct})=0.00$ vs $P(\text{behav})=0.95 \implies \text{sym\_KL}=9.84$. Extreme conflict prevents incorrect auto-ACT and cleanly routes all evaluable members to REVIEW. |
| **2. Sparse Evidence** | Sleeper accounts (`is_sleeper=True`) | 19 | 10 (52.6%) | 9 (47.4%) | 0 (0.0%) | 0 (0.0%) | **100.0%** (19/19) | Legitimate early behavior lowers $P(\text{behav})$ to 0.66, but structural ring link ($P(\text{struct})=1.00$) triggers conflict routing ($47.4\%$ REVIEW) or auto-ACT ($52.6\%$). 0 missed. |
| **3. Benign-Dense Stress** | `hard_bc` (family + shared payout) | 101 | 0 (0.0%) | 13 (12.9%) | 88 (87.1%) | 0 (0.0%) | **0.0% Auto-ACT FP** (12.9% review) | Even with an injected shared payout destination, benign behavior ($P(\text{behav})=0.02$) prevents false auto-ACT ($0.0\%$ FP). Review queue rate drops to 12.9% vs 31.2% general BC. |
| **4. Low-Signal Abuse** | `varied_payout_ac` (no shared payout) | 2 (test) / 67 (total) | 2 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **100.0%** (Caveat: test $N=2$ is small; full pop $N=67$ is $100\%$ recall) | `varied_payout_ac` was generated across all splits (Days 1–75); only 2 were active in test window. Across full dataset ($N=67$), all 67 are detected via behavioral burst/timing signatures ($P(\text{behav})=1.00$). |
| **5. Cold-Start** | $n\_\text{orders} < 2$ (all classes) | 36 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 36 (100.0%) | **100.0% Gated** | Deterministic evidence gate enforces ABSTAIN on insufficient observation ($100\%$ gated), protecting against premature automated actions. |

---

## Trajectory Evaluation (Evolving-Risk Dynamics)

To empirically test the core thesis—that abuse ring risk is an **evolving temporal process** rather than a static snapshot—the 5-rung model suite and DecisionEngine were evaluated longitudinally across **19 late-forming rings** whose formation began in the validation or test windows ($\text{formation\_start\_day} \ge 55$, representing $30.2\%$ of all 63 simulated rings).

Each ring was scored at **5 distinct checkpoints** along its lifecycle using strict as-of-$T$ feature and graph reconstruction:
1. $T_1 = \max(1, \text{start\_day} - 5)$ (Pre-formation baseline)
2. $T_2 = \text{start\_day}$ (Ring creation)
3. $T_3 = \text{int}((\text{start\_day} + \text{complete\_day}) / 2)$ (Mid-formation)
4. $T_4 = \text{complete\_day}$ (Full infrastructure connected)
5. $T_5 = \min(90, \text{complete\_day} + 5)$ (Post-formation operational activity)

Full dataset saved to `/evals/results/trajectory_results.parquet` ($1,370$ account-checkpoint records).

> **Methodological & Split Verification**:
> - **Zero Training Contamination**: All models (`behavioral_lgbm`, `structural_lgbm`, `fused_calibrated`) were fit strictly on the **train split (Days 1–54)**.
> - **Validation/Test Trajectory Overlap**: Checkpoints for rings with formation start $\ge \text{Day 55}$ fall inside the validation (Days 55–72) or test (Days 73–90) periods. Models were not fit or calibrated on these evaluation checkpoints.
> - **Negative Time-to-First-Review ($-5.0\text{d}$ for Promo Rings)**: Negative offsets reflect accounts whose pre-existing structural links (shared devices, IPs, or payment instruments established during legitimate pre-abuse usage) pre-date the ring's defined `formation_start_day`. This is consistent with the sleeper and infrastructure pre-positioning mechanism (amendment A2).
> - **Referral-Farming Cost Implication (Open Problem)**: Referral-farming rings have no automated resolution path within the observed window ($\text{sym\_KL}$ climbs to $\approx 9.76$ but never resolves to ACT due to lack of shared payouts). Consequently, referral-farming accounts generate indefinite human REVIEW-queue cost under this design. This connects directly to the flat-FN cost limitation (Decision methodology, Stage 5) and is highlighted as an open systems challenge.

### Aggregate Longitudinal Findings ($N=19$ Rings)

| Metric | Result | Operational Meaning |
|---|---|---|
| **Evolving Decision Rate** | **100.0%** (19/19 rings) | Every ring's decision profile changed across checkpoints; no ring remained static. |
| **Escalation Coverage** | **100.0%** (19/19 rings) | **0.0%** of rings remained permanently trapped in `WAIT_MONITOR` or `ABSTAIN`. |
| **Review Queue Escalation** | **89.5%** (17/19 rings) | 17 rings triggered human escalation via `conflict_review` ($\text{sym\_KL} > 0.50$). |
| **Auto-ACT Escalation** | **47.4%** (9/19 rings) | 9 rings reached convergent high confidence ($P \ge 0.70, \text{sym\_KL} \le 0.50$) for automated action. |
| **Mean Time-to-First-REVIEW** | **+2.1 days** from start | Early disagreement triggers rapid human escalation (range: $-5.0\text{d}$ to $+9.0\text{d}$; see note on negative offsets). |
| **Mean Time-to-First-ACT** | **+5.4 days** from start | Direct auto-mitigation occurs once shared payouts and transactions converge (range: $+1.0\text{d}$ to $+8.0\text{d}$). |

### Lifecycle Progression by Ring Topology

| Topology | N | % Reaching REVIEW | % Reaching ACT | Time to Review | Time to ACT | $\text{sym\_KL}$ Progression ($T_1 \to T_3 \to T_5$) | Lifecycle Behavior |
|---|---|---|---|---|---|---|---|
| **Promo Rings** | 5 | 100.0% | 100.0% | $-5.0\text{d}$ | $+6.6\text{d}$ | $3.409 \to 0.358 \to 0.094$ | **Conflict $\to$ Convergence**: Pre-existing shared devices trigger early review; heavy promo volume then aligns behavioral & structural models into auto-ACT. |
| **Referral Farming** | 8 | 100.0% | 0.0% | $+5.6\text{d}$ | N/A | $3.288 \to 9.594 \to 9.758$ | **Persistent Escalation (Indefinite Review Cost)**: High behavioral velocity without shared payouts causes $\text{sym\_KL}$ to climb to $9.76$, routing all active members safely to human review. |
| **Return Abuse** | 6 | 66.7% | 66.7% | $+4.0\text{d}$ | $+4.0\text{d}$ | $3.345 \to 2.342 \to 3.011$ | **Bimodal**: Rings with shared payout destinations transition to auto-ACT; isolated return abusers remain in human review. |

---

### Concrete Ring Trajectories (Verbatim Checkpoint Records)

#### 1. Promo Ring: `PROMO_001` (14 members, Formation: Day 58 $\to$ Day 63)
Demonstrates the **Conflict $\to$ Convergence** transition as shared device links precede order volume:
- **Checkpoint 1 (Pre-Start, Day 53, offset $-5\text{d}$)**: `ABSTAIN: 13, REVIEW: 1`
  - *Metrics*: Avg Orders $= 0.4$, $P(\text{struct}) = 0.071$, $P(\text{behav}) = 0.005$, $\text{sym\_KL} = 5.205$
- **Checkpoint 2 (Start, Day 58, offset $+0\text{d}$)**: `ABSTAIN: 13, REVIEW: 1`
  - *Metrics*: Avg Orders $= 1.4$, $P(\text{struct}) = 0.976$, $P(\text{behav}) = 0.082$, $\text{sym\_KL} = 6.027$
- **Checkpoint 3 (Midpoint, Day 60, offset $+2\text{d}$)**: `ACT: 13, REVIEW: 1`
  - *Metrics*: Avg Orders $= 3.2$, $P(\text{struct}) = 1.000$, $P(\text{behav}) = 0.927$, $\text{sym\_KL} = 0.664$
- **Checkpoint 4 (Complete, Day 63, offset $+5\text{d}$)**: `ACT: 13, REVIEW: 1`
  - *Metrics*: Avg Orders $= 5.0$, $P(\text{struct}) = 1.000$, $P(\text{behav}) = 0.931$, $\text{sym\_KL} = 0.593$
- **Checkpoint 5 (Post-Comp, Day 68, offset $+10\text{d}$)**: `ACT: 14`
  - *Metrics*: Avg Orders $= 5.1$, $P(\text{struct}) = 1.000$, $P(\text{behav}) = 0.998$, $P(\text{fused}) = 1.000$, $\text{sym\_KL} = 0.005$

#### 2. Referral Farming: `REFARM_057` (11 members, Formation: Day 74 $\to$ Day 84)
Demonstrates **Persistent Disagreement Escalation** on an unseen topological structure:
- **Checkpoint 1 (Pre-Start, Day 69, offset $-5\text{d}$)**: `ABSTAIN: 11`
  - *Metrics*: Avg Orders $= 0.0$, $P(\text{struct}) = 0.000$, $P(\text{behav}) = 0.003$, $\text{sym\_KL} = 3.288$
- **Checkpoint 2 (Start, Day 74, offset $+0\text{d}$)**: `ABSTAIN: 11`
  - *Metrics*: Avg Orders $= 0.1$, $P(\text{struct}) = 0.000$, $P(\text{behav}) = 0.788$, $\text{sym\_KL} = 9.124$
- **Checkpoint 3 (Midpoint, Day 79, offset $+5\text{d}$)**: `ABSTAIN: 8, REVIEW: 3`
  - *Metrics*: Avg Orders $= 0.9$, $P(\text{struct}) = 0.000$, $P(\text{behav}) = 0.857$, $\text{sym\_KL} = 9.373$
- **Checkpoint 4 (Complete, Day 84, offset $+10\text{d}$)**: `ABSTAIN: 6, REVIEW: 5`
  - *Metrics*: Avg Orders $= 1.5$, $P(\text{struct}) = 0.000$, $P(\text{behav}) = 0.922$, $\text{sym\_KL} = 9.502$
- **Checkpoint 5 (Post-Comp, Day 89, offset $+15\text{d}$)**: `REVIEW: 7, ABSTAIN: 4`
  - *Metrics*: Avg Orders $= 2.3$, $P(\text{struct}) = 0.000$, $P(\text{behav}) = 0.938$, $P(\text{fused}) = 0.061$, $\text{sym\_KL} = 9.523$

#### 3. Return Abuse Ring: `RETURN_027` (5 members, Formation: Day 67 $\to$ Day 74)
Demonstrates clean transition from cold-start to human review as return velocity builds:
- **Checkpoint 1 (Pre-Start, Day 62, offset $-5\text{d}$)**: `ABSTAIN: 5` ($0$ orders)
- **Checkpoint 2 (Start, Day 67, offset $+0\text{d}$)**: `ABSTAIN: 5` (Avg Orders $= 1.0$, $P(\text{behav}) = 0.551$)
- **Checkpoint 3 (Midpoint, Day 70, offset $+3\text{d}$)**: `REVIEW: 5` (Avg Orders $= 3.0$, $P(\text{behav}) = 0.999$, $P(\text{struct}) = 0.111$, $\text{sym\_KL} = 4.930$)
- **Checkpoint 4 (Complete, Day 74, offset $+7\text{d}$)**: `REVIEW: 5` (Avg Orders $= 3.4$, $P(\text{behav}) = 1.000$, $\text{sym\_KL} = 6.580$)
- **Checkpoint 5 (Post-Comp, Day 79, offset $+12\text{d}$)**: `REVIEW: 5` (Avg Orders $= 3.4$, $P(\text{behav}) = 1.000$, $\text{sym\_KL} = 7.199$)

---

## KL-Routing Ablation Study

To directly test whether the disagreement-aware routing architecture adds value over a simpler threshold baseline, we evaluated an ablated non-routing decision policy against the identical test split ($N=3,467$, True $\text{AC}=198$):
- **Ablation Policy**: $\text{ACT if } p_{\text{fused}}[\text{AC}] \ge 0.70 \text{ else WAIT\_MONITOR}$; with the standard $n\_\text{orders} < 2 \implies \text{ABSTAIN}$ gate (no `REVIEW` lane).

Raw outputs saved to `/evals/results/kl_ablation_results.json`.

### Full Test Split Comparison ($N=3,467$, True $\text{AC}=198$)

| Metric | KL-Routing Engine (Proposed) | Threshold-Only Ablation | Difference / Impact |
|---|---|---|---|
| **Direct Auto-ACT Precision** | **100.00%** (38/38) | **100.00%** (55/55) | 0 FP in both |
| **Direct Auto-ACT Recall** | 19.19% (38/198) | 27.78% (55/198) | +8.59% auto-acted |
| **REVIEW Queue Volume** | 779 accounts (124 true AC) | 0 (No review lane) | Eliminated human triage |
| **True AC in WAIT_MONITOR (Missed FN)** | **0** (0.00%) | **107** (54.04%) | **107 AC accounts missed** |
| **True AC in ABSTAIN** | 36 (18.18%) | 36 (18.18%) | Same cold-start gate |
| **Effective Recall (ACT + REVIEW)** | **81.82%** (162/198) | **27.78%** (55/198) | **-54.04% recall drop** |
| **Total Simulated Cost (Best Case)** | **Rs 1,49,250** | **Rs 2,46,400** | **+Rs 97,150 (+65.1% cost penalty)** |
| **Total Simulated Cost (Worst Case)** | **Rs 1,88,850** | **Rs 2,86,000** | **+Rs 97,150 (+51.4% cost penalty)** |

### Robustness Subsets Comparison

| Subset | N | KL-Routing Decisions | Threshold Ablation Decisions | Impact of Disagreement Routing |
|---|---|---|---|---|
| **Referral Farming (Unseen Structure)** | 143 | `REVIEW`: 107, `ABSTAIN`: 36 | `WAIT_MONITOR`: 107, `ABSTAIN`: 36 | **100% of evaluable referral-farming accounts are missed** under threshold ablation because $p_{\text{fused}}$ collapses ($p_{\text{struct}}=0$). KL-routing catches 100%. |
| **Sleeper Accounts (Sparse Evidence)** | 19 | `ACT`: 10, `REVIEW`: 9 | `ACT`: 19 | Threshold ablation forces all 19 to auto-ACT; KL-routing routes 47.4% to human review due to behavioral velocity discrepancy. |
| **Hard BC (Benign Family + Shared Payout)** | 101 | `WAIT`: 88, `REVIEW`: 13 | `WAIT`: 101 | **0 Auto-ACT FP in both.** Under KL-routing, 13 accounts are routed to `REVIEW` due to structural disagreement from the injected payout edge ($p_{\text{struct}}=0.98, p_{\text{behav}}=0.00$), preventing any automated false positives. |

---

## Prevalence-Shift Sensitivity Analysis

To test robustness against realistic shifts in base abuse rates, the already-trained baseline models and DecisionEngine were evaluated across three prevalence regimes without retraining. Raw data saved to `/evals/results/prevalence_shift_results.json`.

| Prevalence Regime | Test Accounts | True AC (Active %) | Auto-ACT Precision | Auto-ACT Recall | Effective Recall | Auto-ACT FP | Routing Cost (Best) | Behavioral-Only Cost |
|---|---|---|---|---|---|---|---|---|
| **Low Prevalence (~3% AC)** | 3,838 | 67 (1.8%) | **100.00%** | 52.24% | **92.54%** | **0** | Rs 1,31,100 | Rs 22,000 |
| **Baseline Prevalence (~15% AC)** | 3,467 | 198 (5.7%) | **100.00%** | 19.19% | **81.82%** | **0** | Rs 1,49,250 | Rs 30,500 |
| **High Prevalence (~30% AC)** | 3,155 | 406 (12.9%) | **100.00%** | 15.76% | **79.56%** | **0** | Rs 2,10,450 | Rs 71,000 |

**Key Invariance**: Across a 7x swing in active test abuse prevalence (1.8% to 12.9%), the auto-ACT lane maintained **0.00% False Positives (100% precision)** with stable effective recall ($79.6\% - 92.5\%$).

---

## Multi-Seed Robustness (Seeds 42, 43, 44)

The full end-to-end pipeline (synthetic generation, feature engineering, 5-rung model training, and decision engine) was executed across 3 independent random seeds. Raw outputs saved to `/evals/results/multiseed_results.json`.

| Metric | Mean across Seeds (42, 43, 44) | Range [Min - Max] |
|---|---|---|
| **Behavioral Model F1 (AC)** | 0.67 | [0.41 - 0.92] |
| **Behavioral Model Precision** | 90.40% | [88.32% - 93.60%] |
| **Behavioral Model Recall** | 58.63% | [27.03% - 95.45%] |
| **Structural Model F1 (AC)** | 0.38 | [0.30 - 0.47] |
| **Structural Model Precision** | 92.78% | [78.33% - 100.00%] |
| **Structural Model Recall** | 24.06% | [17.84% - 30.59%] |
| **Fused Model Precision (AC)** | 100.00% | [100.00% - 100.00%] |
| **Fused Model Recall (AC)** | 36.60% | [20.54% - 58.45%] |
| **Auto-ACT Precision** | **100.00%** | [100.00% - 100.00%] |
| **Auto-ACT Recall** | 18.10% | [11.35% - 23.74%] |
| **Effective Recall via Routing** | **81.40%** | [78.38% - 84.02%] |
| **Auto-ACT False Positives** | **0.00** | [0 - 0] |
| **Behavioral-Only Total Cost** | Rs 1,70,500.00 | [Rs 30,500.00 - Rs 2,73,000.00] |
| **Routing Total Cost (Best Case)** | Rs 2,19,200.00 | [Rs 1,49,250.00 - Rs 2,86,650.00] |
| **Routing Total Cost (Worst Case)** | Rs 2,59,900.00 | [Rs 1,88,850.00 - Rs 3,25,150.00] |

---

## Known limitations

Full enumeration maintained in `data/ASSUMPTIONS.md Section 9`. Key items:

1. **Flat FN cost model.** c_fn=Rs2,000 regardless of time-to-detection.
   Under this assumption, behavioral-only (Rs30,500) dominates routing
   (Rs1,49,250-Rs1,88,850) on cost. Time-dependent loss modelling is future work.

2. **Structural signal constraint.** 80.8% of true AC test accounts lack strong
   structural signal due to partial ring observation. Domain property, not a bug.

3. **Simulator-encoded patterns.** Top behavioural features partly reflect
   simulation choices (tighter amount distributions for ring members). Real-world
   discriminative power of order-amount features is unvalidated.

4. **No production signals.** No device fingerprint APIs, IP reputation databases,
   chargebacks, bank-side signals, KYC age, or SIM swap data. All entities are
   synthetic IDs.

5. **Geometric-mean fusion collapses when p_struct ~ 0.** Fused model not
   suitable as a standalone single-score classifier without the routing design.

6. **Label noise is uniform, not adversarial.** Real label noise is biased
   toward late-formation and evasive accounts. The 22 noisy labels here are
   uniformly random -- underestimates real-world evaluation difficulty.

---

*All cost figures, detection rates, and model metrics are from a fully synthetic
simulation. See `data/ASSUMPTIONS.md`.*