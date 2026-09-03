"""
AbuseRing Sentinel — Capacity-Constrained Review Queue Engine & Triage Policy
==============================================================================
Provides priority ranking algorithms for human review queues operating under
finite daily review capacity constraints (K cases/day).

Strategies:
  1. FIFO: Unprioritized baseline (chronological / observation order)
  2. RANDOM_SHUFFLE: Uninformative neutral baseline (random draw)
  3. TIME_OF_FLAGGING: Chronological queue based on initial order timestamp
  4. SCORE_DESC (Recommended): Highest fused abuse probability P_fused(AC)
  5. EXPOSURE_WEIGHTED (Recommended): P_fused(AC) * sqrt(Total Order Exposure)
  6. VAR_FINANCIAL: Linear Value-at-Risk (P_fused(AC) * Total Order Amount)
  7. CONFLICT_AWARE: P_fused(AC) * (1 + log(1 + sym_KL)) * sqrt(Total Order Exposure)
     (Evaluated to test within-queue KL utility; finding: KL is vital for routing decisions,
      but exposure-weighted / score-descending ranking performs identically or better in-queue).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
import numpy as np

class TriageStrategy(str, Enum):
    FIFO = "fifo"
    RANDOM_SHUFFLE = "random_shuffle"
    TIME_OF_FLAGGING = "time_of_flagging"
    SCORE_DESC = "score_desc"
    VAR_FINANCIAL = "var_financial"
    EXPOSURE_WEIGHTED = "exposure_weighted"
    CONFLICT_AWARE = "conflict_aware"

@dataclass
class QueueItem:
    account_id: str
    p_abusive: float
    p_benign_coord: float
    p_benign_indep: float
    p_struct_ac: float
    p_behav_ac: float
    sym_kl_divergence: float
    n_orders: int
    total_order_amount: float
    true_label: Optional[str] = None  # For evaluation audit
    flag_timestamp: float = 0.0  # Exact timestamp when flagged into review

    @property
    def is_true_ac(self) -> bool:
        return self.true_label == "abusive_coordinated"

class ReviewQueueEngine:
    """Triage ranking engine for review queue management under capacity limits."""

    @staticmethod
    def calculate_priority_score(item: QueueItem, strategy: TriageStrategy) -> float:
        if strategy in (TriageStrategy.FIFO, TriageStrategy.RANDOM_SHUFFLE, TriageStrategy.TIME_OF_FLAGGING):
            return 0.0
        elif strategy == TriageStrategy.SCORE_DESC:
            return float(item.p_abusive)
        elif strategy == TriageStrategy.VAR_FINANCIAL:
            return float(item.p_abusive * item.total_order_amount)
        elif strategy == TriageStrategy.EXPOSURE_WEIGHTED:
            # Ablation variant: P_fused * sqrt(Amount) without sym_KL term
            return float(item.p_abusive * np.sqrt(max(1.0, item.total_order_amount)))
        elif strategy == TriageStrategy.CONFLICT_AWARE:
            # Multi-signal triage: fraud confidence * divergence severity * sqrt(financial exposure)
            kl_weight = float(1.0 + np.log1p(max(0.0, item.sym_kl_divergence)))
            amount_weight = float(np.sqrt(max(1.0, item.total_order_amount)))
            return float(item.p_abusive * kl_weight * amount_weight)
        else:
            raise ValueError(f"Unknown triage strategy: {strategy}")

    @classmethod
    def rank_queue(
        cls,
        items: List[QueueItem],
        strategy: TriageStrategy,
        random_seed: int = 42
    ) -> List[QueueItem]:
        """Ranks a list of review queue items according to the specified triage policy."""
        if strategy == TriageStrategy.FIFO:
            return list(items)  # Preserves natural list arrival order
        elif strategy == TriageStrategy.RANDOM_SHUFFLE:
            rng = np.random.RandomState(random_seed)
            perm = rng.permutation(len(items))
            return [items[i] for i in perm]
        elif strategy == TriageStrategy.TIME_OF_FLAGGING:
            # Chronological order of first triggering event
            return sorted(items, key=lambda it: float(it.flag_timestamp or 0.0))
        
        # Sort descending by priority score (break ties by flag_timestamp)
        scored_items = [
            (cls.calculate_priority_score(it, strategy), -float(it.flag_timestamp or 0.0), it)
            for it in items
        ]
        scored_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [it for _, _, it in scored_items]

    @classmethod
    def evaluate_capacity_limit(
        cls,
        items: List[QueueItem],
        capacity_limit: int,
        strategy: TriageStrategy,
        auto_act_tp: int = 38,
        total_true_ac: int = 198,
        c_review_unit: float = 150.0
    ) -> Dict:
        """Evaluates operational metrics for a queue evaluated up to capacity_limit."""
        ranked = cls.rank_queue(items, strategy)
        n_total_queue = len(items)
        n_reviewed = min(capacity_limit, n_total_queue)
        
        reviewed_items = ranked[:n_reviewed]
        unreviewed_items = ranked[n_reviewed:]

        # True AC in reviewed vs unreviewed
        reviewed_ac = [it for it in reviewed_items if it.is_true_ac]
        unreviewed_ac = [it for it in unreviewed_items if it.is_true_ac]

        tp_review = len(reviewed_ac)
        total_effective_tp = auto_act_tp + tp_review
        effective_recall = float(total_effective_tp / total_true_ac) if total_true_ac > 0 else 0.0
        precision_at_k = float(tp_review / n_reviewed) if n_reviewed > 0 else 0.0

        # Financial exposure
        prevented_fraud_amount = sum(it.total_order_amount for it in reviewed_ac)
        missed_fraud_amount = sum(it.total_order_amount for it in unreviewed_ac)
        review_cost = n_reviewed * c_review_unit

        return {
            "capacity_limit": capacity_limit,
            "strategy": strategy.value,
            "queue_size_total": n_total_queue,
            "accounts_reviewed": n_reviewed,
            "unreviewed_deferred": n_total_queue - n_reviewed,
            "true_ac_in_queue_total": sum(1 for it in items if it.is_true_ac),
            "true_ac_captured_in_review": tp_review,
            "true_ac_unreviewed_missed": len(unreviewed_ac),
            "precision_at_k": round(precision_at_k, 4),
            "retained_effective_recall": round(effective_recall, 4),
            "prevented_fraud_exposure_rs": round(prevented_fraud_amount, 2),
            "missed_fraud_exposure_rs": round(missed_fraud_amount, 2),
            "review_cost_rs": round(review_cost, 2)
        }
