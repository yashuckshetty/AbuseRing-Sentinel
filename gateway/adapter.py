"""
AbuseRing Sentinel — Gateway Adapter & Dual-Path Architecture Specification
==========================================================================
Architectural bridge mapping standard payment gateway events (e.g. Razorpay/Stripe schemas)
into AbuseRing Sentinel's dual-path fraud risk infrastructure.

DESIGN SPECIFICATION & PROTOTYPE CONTRACT:
  1. Synchronous In-Line Path (Design Target <30ms):
     Fast behavioral evaluation for authorization-time decisioning (ALLOW / CHALLENGE / BLOCK).
  2. Asynchronous Near-Line Path (Design Target <500ms):
     Out-of-band graph enrichment, multi-relational topology expansion, and canonical
     Symmetric KL Divergence routing (REVIEW lane dispatch) via DecisionEngine.

MANDATORY MEASUREMENT QUALIFIER:
  All latency measurements, execution timings, and throughput figures produced by this
  module represent prototype design-targets measured in a local single-machine mock
  environment (in-memory adapter processing synthetic test data, not live distributed
  gateway traffic or remote database latency).
"""

from __future__ import annotations

import hmac
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Canonical imports from protected baseline — SOLE SOURCE OF TRUTH
from decision.decision_engine import (
    DecisionEngine,
    Decision,
    RoutingLane,
    DecisionResult,
    sym_kl_divergence,  # Single canonical implementation
)


class GatewayEventType(str, Enum):
    PAYMENT_AUTHORIZED = "payment.authorized"
    PAYMENT_FAILED     = "payment.failed"
    ORDER_CREATED      = "order.created"
    REFUND_CREATED     = "refund.created"
    DISPUTE_CREATED    = "dispute.created"


class SyncAction(str, Enum):
    """
    Preliminary in-line authorization recommendations (<30ms design target).
    
    CRITICAL DISTINCTION:
      SyncAction values (ALLOW, CHALLENGE_2FA, BLOCK) are preliminary, non-authoritative
      in-line recommendations for the payment gateway authorization loop. They are
      distinct from and NOT mapped 1:1 onto the canonical Decision enum
      (ACT, REVIEW, WAIT_MONITOR, ABSTAIN), which remains the sole operational authority
      evaluated exclusively via the asynchronous path through DecisionEngine.decide().
    """
    ALLOW           = "ALLOW"
    CHALLENGE_2FA   = "CHALLENGE_2FA"
    BLOCK           = "BLOCK"


@dataclass
class GatewayPaymentEvent:
    """Normalized representation of a standard payment gateway event."""
    event_id: str
    event_type: GatewayEventType
    account_id: str
    amount_inr: float
    currency: str
    timestamp: int
    ip_address: str
    device_id: str
    payout_id: Optional[str] = None
    referral_code_used: Optional[str] = None
    is_promo_applied: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_razorpay_payload(cls, payload: Dict[str, Any]) -> "GatewayPaymentEvent":
        """
        Map a standard Razorpay-compatible webhook payload to normalized gateway event.
        Standard Razorpay entity schema: payload['payload']['payment']['entity']
        """
        event_type_str = payload.get("event", "payment.authorized")
        event_id = payload.get("id", f"evt_{int(time.time()*1000)}")
        
        # Extract payment entity
        p_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        if not p_entity:
            # Fallback if flat payload provided
            p_entity = payload

        account_id = str(p_entity.get("notes", {}).get("account_id") or p_entity.get("account_id") or "ACC_UNKNOWN")
        amount_raw = float(p_entity.get("amount", 0.0))
        # Razorpay amounts are in paise (1 INR = 100 paise)
        amount_inr = amount_raw / 100.0 if amount_raw > 1000 and "notes" in p_entity else amount_raw

        return cls(
            event_id=event_id,
            event_type=GatewayEventType(event_type_str) if event_type_str in [e.value for e in GatewayEventType] else GatewayEventType.PAYMENT_AUTHORIZED,
            account_id=account_id,
            amount_inr=amount_inr,
            currency=p_entity.get("currency", "INR"),
            timestamp=int(p_entity.get("created_at") or time.time()),
            ip_address=str(p_entity.get("ip") or p_entity.get("notes", {}).get("ip_address") or "0.0.0.0"),
            device_id=str(p_entity.get("notes", {}).get("device_id") or "DEV_UNKNOWN"),
            payout_id=p_entity.get("notes", {}).get("payout_id"),
            referral_code_used=p_entity.get("notes", {}).get("referral_code"),
            is_promo_applied=bool(p_entity.get("notes", {}).get("is_promo", False)),
            metadata=p_entity.get("notes", {})
        )


@dataclass
class SyncAuthorizationResponse:
    """In-line fast path response (<30ms design target)."""
    event_id: str
    account_id: str
    action: SyncAction
    behavioral_score: float
    p_behav: List[float]
    execution_time_ms: float
    qualifier: str = "Prototype design-target measured in a local single-machine mock environment"
    rationale: str = ""


@dataclass
class AsyncEnrichmentResponse:
    """Near-line asynchronous graph enrichment and divergence routing response (<500ms design target)."""
    event_id: str
    account_id: str
    authoritative_decision: Decision
    routing_lane: RoutingLane
    p_struct: List[float]
    p_behav: List[float]
    p_fused: List[float]
    sym_kl_divergence: float
    evidence_conflict: bool
    sync_async_disagreement: bool
    disagreement_nature: str
    execution_time_ms: float
    qualifier: str = "Prototype design-target measured in a local single-machine mock environment"
    audit_trail: Dict[str, Any] = field(default_factory=dict)


class GatewayEventAdapter:
    """
    Production Gateway Bridge Adapter.
    
    Coordinates the dual-path execution flow:
      1. process_sync_authorization(): In-line behavioral scoring within <30ms design budget.
      2. process_async_enrichment(): Near-line graph expansion & canonical DecisionEngine routing.
    """

    LATENCY_QUALIFIER = (
        "Prototype design-target measured in a local single-machine mock environment "
        "(in-memory adapter processing synthetic test data, not live distributed gateway traffic or remote database latency)."
    )

    def __init__(
        self,
        decision_engine: DecisionEngine,
        behavioral_model: Any,
        structural_model: Any,
        fused_model: Any,
        webhook_secret: Optional[str] = "whsec_test_mock_secret_key_12345"
    ):
        self.decision_engine = decision_engine
        self.behavioral_model = behavioral_model
        self.structural_model = structural_model
        self.fused_model = fused_model
        self.webhook_secret = webhook_secret
        self._processed_events: Dict[str, Any] = {}

    def verify_webhook_signature(self, payload_bytes: bytes, signature_header: str) -> bool:
        """
        Verify standard HMAC-SHA256 signature from gateway webhook header.
        Matches Razorpay / Stripe webhook signature verification contract.
        """
        if not self.webhook_secret or not signature_header:
            return False
        expected_sig = hmac.new(
            self.webhook_secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected_sig, signature_header)

    def process_sync_authorization(
        self,
        event: GatewayPaymentEvent,
        behav_features: pd.Series
    ) -> SyncAuthorizationResponse:
        """
        Synchronous in-line evaluation path (<30ms design target).
        Uses exclusively fast behavioral features available at transaction time.
        
        NOTE: Returns preliminary SyncAction (ALLOW / CHALLENGE_2FA / BLOCK), which
        does not override the authoritative DecisionEngine Decision.
        """
        t0 = time.perf_counter()

        # Format behavioral feature vector
        X_behav = pd.DataFrame([behav_features]).fillna(0)
        
        if hasattr(self.behavioral_model, "predict_proba"):
            p_behav_arr = self.behavioral_model.predict_proba(X_behav)[0]
        else:
            p_behav_arr = np.array([0.9, 0.05, 0.05])
            
        p_ac = float(p_behav_arr[2]) if len(p_behav_arr) > 2 else 0.0

        # Fast in-line policy heuristics (design-target sandbox)
        if p_ac >= 0.85:
            action = SyncAction.BLOCK
            rationale = f"High immediate behavioral risk (P_ac={p_ac:.3f} >= 0.85); fast transaction decline."
        elif p_ac >= 0.40:
            action = SyncAction.CHALLENGE_2FA
            rationale = f"Moderate behavioral risk (P_ac={p_ac:.3f}); step-up 2FA verification triggered."
        else:
            action = SyncAction.ALLOW
            rationale = f"Low immediate behavioral risk (P_ac={p_ac:.3f} < 0.40); authorized for payment execution."

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return SyncAuthorizationResponse(
            event_id=event.event_id,
            account_id=event.account_id,
            action=action,
            behavioral_score=p_ac,
            p_behav=[float(x) for x in p_behav_arr],
            execution_time_ms=round(elapsed_ms, 3),
            qualifier=self.LATENCY_QUALIFIER,
            rationale=rationale
        )

    def process_async_enrichment(
        self,
        event: GatewayPaymentEvent,
        sync_response: SyncAuthorizationResponse,
        struct_features: pd.Series,
        behav_features: pd.Series,
        observation_days: float = 30.0,
        n_orders: int = 5,
        as_of_ts: int = 1707776000
    ) -> AsyncEnrichmentResponse:
        """
        Asynchronous near-line graph enrichment & divergence routing (<500ms design target).
        
        CRITICAL ARCHITECTURAL CONSTRAINTS:
          1. Uses the authoritative DecisionEngine.decide() — does NOT bypass it.
          2. Uses canonical sym_kl_divergence from decision.decision_engine.
          3. Uses canonical fused_model.predict_proba_sub — no duplicate manual fallback.
          4. If async routing conflicts with sync action, preserves BOTH and flags disagreement.
        """
        t0 = time.perf_counter()

        X_struct = pd.DataFrame([struct_features]).fillna(0)
        X_behav = pd.DataFrame([behav_features]).fillna(0)

        # 1. Structural model evaluation
        if hasattr(self.structural_model, "predict_proba"):
            p_struct_arr = self.structural_model.predict_proba(X_struct)[0]
        else:
            p_struct_arr = np.array([0.9, 0.05, 0.05])

        p_behav_arr = np.array(sync_response.p_behav)

        # 2. Canonical Fused model computation (using models.fused_model interface)
        if hasattr(self.fused_model, "predict_proba_sub"):
            _, _, p_fused_mat, _ = self.fused_model.predict_proba_sub(X_struct, X_behav)
            p_fused_arr = p_fused_mat[0]
        elif hasattr(self.fused_model, "predict_proba"):
            p_fused_arr = self.fused_model.predict_proba(X_behav)[0]
        else:
            raise TypeError("fused_model must implement predict_proba_sub or predict_proba")

        # 3. Canonical DecisionEngine evaluation (Sole Operational Authority)
        dec_res: DecisionResult = self.decision_engine.decide(
            account_id=event.account_id,
            p_fused=p_fused_arr,
            p_struct=p_struct_arr,
            p_behav=p_behav_arr,
            observation_days=observation_days,
            n_orders=n_orders,
            as_of_ts=as_of_ts
        )

        # 4. Conflict & Disagreement Analysis between Sync and Async
        sync_async_disagreement = False
        disagreement_nature = "Agreement: Sync action and async decision are consistent."

        if sync_response.action == SyncAction.ALLOW and dec_res.decision in [Decision.REVIEW, Decision.ACT]:
            sync_async_disagreement = True
            disagreement_nature = (
                f"Sleeper / Graph Conflict: In-line sync authorized payment (ALLOW), but async graph "
                f"expansion revealed structural risk routing to {dec_res.decision.value} "
                f"(sym_KL={dec_res.sym_kl_divergence:.3f}). Account flagged for near-line human triage."
            )
        elif sync_response.action == SyncAction.BLOCK and dec_res.decision in [Decision.WAIT_MONITOR, Decision.ABSTAIN]:
            sync_async_disagreement = True
            disagreement_nature = (
                f"Benign Burst Conflict: In-line sync blocked on velocity spike, but async topology confirmed "
                f"zero shared abuse infrastructure (routed to {dec_res.decision.value})."
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Record event in idempotency table
        self._processed_events[event.event_id] = {
            "account_id": event.account_id,
            "decision": dec_res.decision.value,
            "lane": dec_res.routing_lane.value,
            "processed_at": time.time()
        }

        return AsyncEnrichmentResponse(
            event_id=event.event_id,
            account_id=event.account_id,
            authoritative_decision=dec_res.decision,
            routing_lane=dec_res.routing_lane,
            p_struct=[float(x) for x in p_struct_arr],
            p_behav=[float(x) for x in p_behav_arr],
            p_fused=[float(x) for x in p_fused_arr],
            sym_kl_divergence=dec_res.sym_kl_divergence,
            evidence_conflict=dec_res.evidence_conflict,
            sync_async_disagreement=sync_async_disagreement,
            disagreement_nature=disagreement_nature,
            execution_time_ms=round(elapsed_ms, 3),
            qualifier=self.LATENCY_QUALIFIER,
            audit_trail=dec_res.audit_trail
        )

    def benchmark_dual_path(
        self,
        test_events: List[Tuple[GatewayPaymentEvent, pd.Series, pd.Series]],
        n_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark prototype in-memory execution latency across sync and async paths.
        Explicitly reports p50, p95, p99 with mandatory prototype qualifiers.
        """
        sync_times = []
        async_times = []

        for i in range(n_iterations):
            idx = i % len(test_events)
            evt, s_feat, b_feat = test_events[idx]
            
            # Sync timing
            t0 = time.perf_counter()
            sync_resp = self.process_sync_authorization(evt, b_feat)
            sync_times.append((time.perf_counter() - t0) * 1000.0)

            # Async timing
            t1 = time.perf_counter()
            _ = self.process_async_enrichment(evt, sync_resp, s_feat, b_feat)
            async_times.append((time.perf_counter() - t1) * 1000.0)

        sync_arr = np.array(sync_times)
        async_arr = np.array(async_times)

        return {
            "qualifier": self.LATENCY_QUALIFIER,
            "environment": "Single-machine local in-memory prototype",
            "n_trials": n_iterations,
            "sync_path": {
                "design_target_ms": "< 30.0 ms",
                "measured_p50_ms": round(float(np.percentile(sync_arr, 50)), 3),
                "measured_p95_ms": round(float(np.percentile(sync_arr, 95)), 3),
                "measured_p99_ms": round(float(np.percentile(sync_arr, 99)), 3),
                "measured_mean_ms": round(float(np.mean(sync_arr)), 3),
                "meets_design_target": bool(np.percentile(sync_arr, 99) < 30.0)
            },
            "async_path": {
                "design_target_ms": "< 500.0 ms",
                "measured_p50_ms": round(float(np.percentile(async_arr, 50)), 3),
                "measured_p95_ms": round(float(np.percentile(async_arr, 95)), 3),
                "measured_p99_ms": round(float(np.percentile(async_arr, 99)), 3),
                "measured_mean_ms": round(float(np.mean(async_arr)), 3),
                "meets_design_target": bool(np.percentile(async_arr, 99) < 500.0)
            }
        }
