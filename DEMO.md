# AbuseRing Sentinel — Demo Presentation Guide

Deterministic evaluation scripts and video presentation guide for AbuseRing Sentinel.

---

## 1. 5-Minute Pitch Video Script

| Time | On Screen | What You Say |
|---|---|---|
| **0:00–0:30** | Opening slide / Title banner | "Organized payment fraud rings exploit single-model blindspots. Coordinated abuse arrives through asymmetric channels: high order velocity with zero shared payout edges, or pre-positioned sleepers with mature infrastructure but zero early orders. On our held-out test window of 198 coordinated abuse accounts, Sentinel auto-actions 38 accounts with exactly 0 false positives in the auto-ACT lane, and safely routes 124 more to human review via evidence divergence. Our core claim is not that every ring is autonomously blocked, but that no enforcement decision is automated unless independent evidence channels agree. All evaluation numbers are measured on a synthetic benchmark harness, not live Razorpay production traffic." |
| **0:30–1:10** | Terminal: `python demo.py` (Acts 1 & 2) | "Running `python demo.py`. In Act 2, we evaluate 6 representative accounts live through our calibrated LightGBM models. Notice how the four distinct operational decisions emerge: concordant abuse in `ACC_04295` triggers direct auto-ACT; benign family coordination in `ACC_03653` and independent shoppers in `ACC_00505` resolve to WAIT_MONITOR without human review; and cold-start `ACC_04987` is deterministically held in ABSTAIN because it has fewer than 2 orders." |
| **1:10–2:20** | Terminal: `demo.py` (Act 3) | "Act 3 demonstrates the core architectural failure mode we refuse. Look at `ACC_04870`, a referral farming ring member. Structural graph model gives it 0.00 risk because it shares no payout edges; behavioral model gives it 0.94 due to rapid referral velocity. If we used naive geometric-mean fusion `sqrt(0.00 * 0.94)`, the fused score collapses to 0.0007—silently clearing a true attacker as a false negative. Sentinel instead measures symmetric KL divergence. Here, sym_KL is 10.22, far exceeding our 0.50 conflict threshold. The system flags the disagreement and routes the account into the conflict_review lane for human adjudication." |
| **2:20–3:00** | Terminal: `demo.py` (Act 4) | "In Act 4, risk evolves over time. Tracing referral ring `REFARM_057` across checkpoints from Day 69 to Day 89: as velocity accelerates at Day 74, it enters ACCELERATING_MONITOR; as evidence divergence triggers at Day 79, it escalates to DIVERGENT_REVIEW; and at Day 84, as multiple connected accounts enter review, it flags QUARANTINE_HOLD—providing 5 days of advance warning before ring completion." |
| **3:00–3:30** | Terminal: `demo.py` (Act 5) | "Act 5 shows our deterministic audit record. Every decision is computed strictly by `DecisionEngine` before any LLM advisory is generated. The cost model, symmetric KL, thresholds, and audit rationale are fully logged. The AI explanation is appended strictly as decoupled advisory text for human analysts." |
| **3:30–4:15** | Browser: Dashboard at `localhost:8000` | "Opening the dashboard: In the Top Bar, we see our 5-rung model ladder progressing from baseline to calibrated fusion (`evals/metrics.json`). Selecting `ACC_04870` in Curated Cases displays the dual-radar evidence asymmetry and the live 1-hop graph neighborhood. In the Dual-Path Gateway tab, our local benchmark demonstrates sub-7ms in-line decisions and sub-24ms near-line graph enrichment (`evals/results/gateway_latency_results.json`)." |
| **4:15–4:45** | Slide / README Architecture | "We state our boundaries plainly: Sentinel is designed as a reference specification to integrate with a Razorpay-like payment webhook gateway; it is evaluated on synthetic test data rather than live merchant traffic. Under extreme signal sparsity (Battery Family D), it safely abstains rather than firing false blocks." |
| **4:45–5:00** | Terminal: `python evals/reproduce_all.py` | "Every metric in this repository is 100% reproducible. Running `python evals/reproduce_all.py` executes all 17 pipeline steps from a clean slate deterministically. Thank you." |

---

## 2. Dashboard Click-Path (`ui/index.html`)

Execute `run_all.bat` (or `python -m uvicorn api.main:app --port 8000`) and navigate to `http://localhost:8000`:

1. **System Health & Model Ladder (Top Header)**
   - Click **/api/health** status badge: confirms 17/17 artifacts loaded, status `healthy` ([`api/main.py`](api/main.py)).
   - Inspect **5-Rung Model Ladder**:
     - Rung 1 (Behavioral LGBM): F1 Macro 0.7431, Test AC Recall 95.45% ([`evals/metrics.json`](evals/metrics.json)).
     - Rung 2 (Structural LGBM): F1 Macro 0.5401, Test AC Recall 23.74% ([`evals/metrics.json`](evals/metrics.json)).
     - Rung 5 (Calibrated Fused): 0 False Positives in auto-ACT lane, 81.82% effective recall via routing (162/198 AC accounts: 38 auto-ACT + 124 REVIEW) ([`evals/metrics.json`](evals/metrics.json)).

2. **Interactive Decision & Evidence Disagreement Inspector (Main Panel)**
   - Select **`ACC_04870` (Referral Farming)** from Curated Accounts dropdown:
     - Verified Metrics: P(struct)[AC] = `0.00`, P(behav)[AC] = `0.94`, sym_KL = `10.22` > `0.50` ([`data/curated_cases.py`](data/curated_cases.py)).
     - Routing Output: Lane `conflict_review`, Decision `REVIEW`, Human Review Rationale displayed live ([`/api/decision/ACC_04870`](api/main.py)).
   - Select **`ACC_04295` (Promo Abuse)**:
     - Verified Metrics: P(struct)[AC] = `1.00`, P(behav)[AC] = `1.00`, sym_KL = `0.0010` < `0.50` ([`data/curated_cases.py`](data/curated_cases.py)).
     - Routing Output: Lane `fused_auto`, Decision `ACT` ([`/api/decision/ACC_04295`](api/main.py)).
   - Select **`ACC_04987` (Cold-Start Guardrail)**:
     - Verified Metrics: $n_{\text{orders}} = 1 < 2$, Decision `ABSTAIN` ([`/api/decision/ACC_04987`](api/main.py)).

3. **Temporal Escalation & Lead Time (Longitudinal Tab)**
   - View Ring **`REFARM_057`** lifecycle:
     - Formation: Day 74 to Day 84.
     - State progression: `DORMANT_BASELINE` (Day 69) $\to$ `ACCELERATING_MONITOR` (Day 74) $\to$ `DIVERGENT_REVIEW` (Day 79) $\to$ `QUARANTINE_HOLD` (Day 84) ([`evals/results/trajectory_results.parquet`](evals/results/trajectory_results.parquet)).
     - Advance Warning Lead Time: `5 days` before ring completion ([`policy/temporal_escalation.py`](policy/temporal_escalation.py)).

4. **Dual-Path Gateway Performance (Gateway Tab)**
   - Synchronous Path (p50: `3.664 ms`, p95: `4.510 ms`, p99: `6.578 ms` vs `< 30.0 ms` budget) ([`evals/results/gateway_latency_results.json`](evals/results/gateway_latency_results.json)).
   - Asynchronous Path (p50: `12.372 ms`, p95: `14.693 ms`, p99: `23.743 ms` vs `< 500.0 ms` budget) ([`evals/results/gateway_latency_results.json`](evals/results/gateway_latency_results.json)).

5. **Stress Test Battery & Out-of-Distribution Transfer (Evals Tab)**
   - Hand-Crafted Battery: `85.19%` effective recall (138/162 accounts caught) across 25 adversarial topologies vs `54.32%` naive fusion ([`evals/results/handcrafted_adversarial_results.json`](evals/results/handcrafted_adversarial_results.json)).
