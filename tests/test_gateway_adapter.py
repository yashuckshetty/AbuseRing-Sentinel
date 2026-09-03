"""
Unit and Regression Tests for Gateway Adapter & Dual-Path Architecture Bridge
=============================================================================
Verifies:
  1. Gateway payload ingestion (Razorpay/Stripe schemas).
  2. HMAC-SHA256 signature verification.
  3. Synchronous in-line evaluation vs. Asynchronous near-line enrichment.
  4. Preservation of sync/async evidence disagreement (no silent overwriting).
  5. Regression: Canonical sym_kl_divergence from decision_engine is used (no duplicates).
  6. Regression: DecisionEngine.decide() is the sole authoritative decision maker.
  7. Mandatory prototype latency disclaimer on all timing artifacts.
"""

import inspect
import json
import time
import pytest
import numpy as np
import pandas as pd

from gateway.adapter import (
    GatewayEventAdapter,
    GatewayPaymentEvent,
    GatewayEventType,
    SyncAction,
    SyncAuthorizationResponse,
    AsyncEnrichmentResponse,
)
from decision.decision_engine import (
    DecisionEngine,
    Decision,
    RoutingLane,
    DecisionResult,
    sym_kl_divergence as canonical_sym_kl,
)


class MockModel:
    """Deterministic mock classifier returning specified probability distributions."""
    def __init__(self, p_vec):
        self.p_vec = np.array(p_vec)

    def predict_proba(self, X):
        return np.tile(self.p_vec, (len(X), 1))


class MockFusedModel:
    """Mock fused model implementing canonical predict_proba_sub."""
    def __init__(self, struct_model, behav_model):
        self.struct_model = struct_model
        self.behav_model = behav_model

    def predict_proba_sub(self, X_struct, X_behav):
        p_s = self.struct_model.predict_proba(X_struct)
        p_b = self.behav_model.predict_proba(X_behav)
        geom = np.sqrt(np.clip(p_s, 1e-9, 1.0) * np.clip(p_b, 1e-9, 1.0))
        p_f = geom / geom.sum(axis=1, keepdims=True)
        return p_s, p_b, p_f, np.zeros(len(p_s), dtype=bool)


@pytest.fixture
def mock_gateway_adapter():
    engine = DecisionEngine(kl_conflict_threshold=0.5)
    behav_model = MockModel([0.05, 0.05, 0.90])  # High behavioral abuse
    struct_model = MockModel([0.90, 0.05, 0.05]) # Low structural signal (classic conflict)
    fused_model = MockFusedModel(struct_model, behav_model)
    
    return GatewayEventAdapter(
        decision_engine=engine,
        behavioral_model=behav_model,
        structural_model=struct_model,
        fused_model=fused_model,
        webhook_secret="whsec_test_secret_abc123"
    )


def test_razorpay_payload_parsing():
    """Verify standard Razorpay webhook schema parsing."""
    raw_payload = {
        "entity": "event",
        "account_id": "acc_razorpay_live_01",
        "event": "payment.authorized",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_O9rK12345678",
                    "amount": 250000,  # 2500.00 INR
                    "currency": "INR",
                    "status": "authorized",
                    "ip": "103.21.244.2",
                    "created_at": 1707776000,
                    "notes": {
                        "account_id": "ACC_04870",
                        "device_id": "DEV_ANDROID_882",
                        "is_promo": True,
                        "referral_code": "REF_991"
                    }
                }
            }
        }
    }

    event = GatewayPaymentEvent.from_razorpay_payload(raw_payload)
    assert event.account_id == "ACC_04870"
    assert event.amount_inr == 2500.0
    assert event.ip_address == "103.21.244.2"
    assert event.device_id == "DEV_ANDROID_882"
    assert event.is_promo_applied is True
    assert event.event_type == GatewayEventType.PAYMENT_AUTHORIZED


def test_webhook_signature_verification(mock_gateway_adapter):
    """Verify HMAC-SHA256 signature verification."""
    payload_str = json.dumps({"test": "data"}).encode("utf-8")
    import hmac, hashlib
    valid_sig = hmac.new(b"whsec_test_secret_abc123", payload_str, hashlib.sha256).hexdigest()
    invalid_sig = "invalid_signature_hex_string"

    assert mock_gateway_adapter.verify_webhook_signature(payload_str, valid_sig) is True
    assert mock_gateway_adapter.verify_webhook_signature(payload_str, invalid_sig) is False
    assert mock_gateway_adapter.verify_webhook_signature(payload_str, "") is False


def test_dual_path_execution_and_conflict_preservation(mock_gateway_adapter):
    """
    Verify the dual-path execution flow:
      1. Sync in-line path evaluates behavioral signal.
      2. Async near-line path performs graph expansion & canonical routing.
      3. Disagreement between sync and async is explicitly preserved and explained.
    """
    event = GatewayPaymentEvent(
        event_id="evt_test_001",
        event_type=GatewayEventType.PAYMENT_AUTHORIZED,
        account_id="ACC_04870",
        amount_inr=1500.0,
        currency="INR",
        timestamp=1707776000,
        ip_address="192.168.1.1",
        device_id="DEV_001"
    )

    # Benign behavioral features for sync path (P_ac < 0.40 -> ALLOW)
    mock_gateway_adapter.behavioral_model = MockModel([0.85, 0.10, 0.05])
    # Severe structural abuse signal discovered later in graph expansion (P_ac = 0.95)
    mock_gateway_adapter.structural_model = MockModel([0.02, 0.03, 0.95])

    b_feat = pd.Series({"promo_rate": 0.0, "order_velocity_1h": 1.0})
    s_feat = pd.Series({"shared_payout_degree": 4.0, "degree": 12.0})

    # Step 1: Sync authorization
    sync_resp = mock_gateway_adapter.process_sync_authorization(event, b_feat)
    assert sync_resp.action == SyncAction.ALLOW
    assert sync_resp.execution_time_ms > 0
    assert "Prototype design-target" in sync_resp.qualifier

    # Step 2: Async enrichment
    async_resp = mock_gateway_adapter.process_async_enrichment(
        event=event,
        sync_response=sync_resp,
        struct_features=s_feat,
        behav_features=b_feat
    )

    # Must invoke DecisionEngine -> severe structural vs benign behavioral triggers sym_KL > 0.50 -> REVIEW
    assert async_resp.authoritative_decision in [Decision.REVIEW, Decision.ACT]
    assert async_resp.evidence_conflict is True
    assert async_resp.sym_kl_divergence > 0.50
    assert async_resp.sync_async_disagreement is True
    assert "Sleeper / Graph Conflict" in async_resp.disagreement_nature
    assert "Prototype design-target" in async_resp.qualifier


def test_regression_canonical_sym_kl_used():
    """
    REGRESSION TEST: Verify that gateway.adapter does NOT define its own
    sym_kl_divergence function, and strictly imports canonical sym_kl_divergence
    from decision.decision_engine.
    """
    import gateway.adapter as ga
    
    # Ensure sym_kl_divergence in gateway.adapter IS identical to canonical
    assert ga.sym_kl_divergence is canonical_sym_kl

    # Ensure no local 'def sym_kl_divergence' exists inside gateway/adapter.py
    src = inspect.getsource(ga)
    assert "def sym_kl_divergence(" not in src, "gateway/adapter.py must not define a duplicate sym_kl_divergence function!"


def test_regression_decision_engine_authority():
    """
    REGRESSION TEST: Verify that async enrichment relies strictly on
    DecisionEngine.decide() and cannot bypass it.
    """
    class SpyEngine(DecisionEngine):
        def __init__(self):
            super().__init__()
            self.decide_called = False

        def decide(self, *args, **kwargs):
            self.decide_called = True
            return super().decide(*args, **kwargs)

    spy = SpyEngine()
    bm = MockModel([0.8, 0.1, 0.1])
    sm = MockModel([0.8, 0.1, 0.1])
    fm = MockFusedModel(sm, bm)
    adapter = GatewayEventAdapter(
        decision_engine=spy,
        behavioral_model=bm,
        structural_model=sm,
        fused_model=fm
    )

    event = GatewayPaymentEvent(
        event_id="evt_spy_01",
        event_type=GatewayEventType.PAYMENT_AUTHORIZED,
        account_id="ACC_SPY",
        amount_inr=100.0,
        currency="INR",
        timestamp=1707776000,
        ip_address="1.1.1.1",
        device_id="DEV_SPY"
    )

    sync_resp = adapter.process_sync_authorization(event, pd.Series({"velocity": 1.0}))
    _ = adapter.process_async_enrichment(
        event=event,
        sync_response=sync_resp,
        struct_features=pd.Series({"deg": 0.0}),
        behav_features=pd.Series({"velocity": 1.0})
    )

    assert spy.decide_called is True, "Async enrichment failed to call authoritative DecisionEngine.decide()!"


def test_benchmark_dual_path_reporting(mock_gateway_adapter):
    """Verify prototype benchmark reporting structure and required qualifiers."""
    event = GatewayPaymentEvent(
        event_id="evt_bench_01",
        event_type=GatewayEventType.PAYMENT_AUTHORIZED,
        account_id="ACC_BENCH",
        amount_inr=500.0,
        currency="INR",
        timestamp=1707776000,
        ip_address="127.0.0.1",
        device_id="DEV_BENCH"
    )
    b_feat = pd.Series({"f1": 0.1, "f2": 0.2})
    s_feat = pd.Series({"s1": 0.5, "s2": 0.0})

    results = mock_gateway_adapter.benchmark_dual_path([(event, s_feat, b_feat)], n_iterations=20)

    assert "qualifier" in results
    assert "Prototype design-target" in results["qualifier"]
    assert "sync_path" in results
    assert "async_path" in results
    assert results["sync_path"]["measured_p50_ms"] >= 0.0
    assert results["async_path"]["measured_p50_ms"] >= 0.0
    assert "< 30.0 ms" in results["sync_path"]["design_target_ms"]
    assert "< 500.0 ms" in results["async_path"]["design_target_ms"]


def test_fused_model_interface_enforcement():
    """Verify that fused_model must implement predict_proba_sub or predict_proba."""
    engine = DecisionEngine(kl_conflict_threshold=0.5)
    adapter = GatewayEventAdapter(
        decision_engine=engine,
        behavioral_model=MockModel([0.8, 0.1, 0.1]),
        structural_model=MockModel([0.8, 0.1, 0.1]),
        fused_model="invalid_string_not_a_model"
    )
    event = GatewayPaymentEvent(
        event_id="evt_invalid_fused",
        event_type=GatewayEventType.PAYMENT_AUTHORIZED,
        account_id="ACC_INV",
        amount_inr=100.0,
        currency="INR",
        timestamp=1707776000,
        ip_address="1.1.1.1",
        device_id="DEV_INV"
    )
    sync_resp = adapter.process_sync_authorization(event, pd.Series({"velocity": 1.0}))
    with pytest.raises(TypeError, match="fused_model must implement"):
        adapter.process_async_enrichment(
            event=event,
            sync_response=sync_resp,
            struct_features=pd.Series({"deg": 0.0}),
            behav_features=pd.Series({"velocity": 1.0})
        )

