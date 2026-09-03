"""
AbuseRing Sentinel — Dynamic Cost & Time-Dependent Exposure Model
==================================================================
Extends the baseline static cost matrix to support time-dependent compounding
false negative losses:
  L(t) = C_0 + alpha * t^gamma

Where:
  - C_0: Baseline transaction loss (Rs 2,000)
  - alpha: Illustrative daily exposure compounding coefficient (e.g. Rs 100.0/day)
  - gamma: Illustrative compounding acceleration factor (e.g. 1.2)
  - t: Detection lag in days before an undetected ring account is mitigated.

Note: All parameters are explicitly illustrative assumptions for sensitivity analysis.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

@dataclass
class DynamicCostConfig:
    c_false_positive: float = 500.0
    c_false_negative_base: float = 2000.0
    c_review: float = 150.0
    c_wait_per_day: float = 50.0
    alpha_compounding_per_day: float = 100.0  # Illustrative exposure growth rate
    gamma_acceleration: float = 1.2           # Illustrative non-linear compounding factor

    @classmethod
    def from_file(cls, path: str = "data/cost_config.json") -> "DynamicCostConfig":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            c_false_positive=float(d.get("c_false_positive", 500.0)),
            c_false_negative_base=float(d.get("c_false_negative", 2000.0)),
            c_review=float(d.get("c_review", 150.0)),
            c_wait_per_day=float(d.get("c_wait_per_day", 50.0)),
            alpha_compounding_per_day=float(d.get("alpha_compounding_per_day", 100.0)),
            gamma_acceleration=float(d.get("gamma_acceleration", 1.2))
        )

    def calculate_fn_loss(self, lag_days: float) -> float:
        """Calculates compounding false negative exposure for an undetected ring account after t days."""
        # t <= 0 floor returns exact baseline c_false_negative_base (avoids 0^gamma edge case at t=0)
        if lag_days <= 0:
            return self.c_false_negative_base
        return float(self.c_false_negative_base + self.alpha_compounding_per_day * (lag_days ** self.gamma_acceleration))

    def evaluate_static_cost(self, n_fp: int, n_fn: int, n_review: int = 0, n_wait_days: int = 0) -> float:
        """Calculates baseline flat static financial cost."""
        return float(
            n_fp * self.c_false_positive +
            n_fn * self.c_false_negative_base +
            n_review * self.c_review +
            n_wait_days * self.c_wait_per_day
        )

    def evaluate_dynamic_cost(
        self,
        n_fp: int,
        fn_lag_days: List[float],
        n_review: int = 0,
        n_wait_days: int = 0
    ) -> float:
        """Calculates dynamic total cost accounting for per-account detection lag."""
        fp_cost = n_fp * self.c_false_positive
        review_cost = n_review * self.c_review
        wait_cost = n_wait_days * self.c_wait_per_day
        fn_cost = sum(self.calculate_fn_loss(t) for t in fn_lag_days)
        return float(fp_cost + review_cost + wait_cost + fn_cost)
