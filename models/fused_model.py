"""
FusedCalibratedClassifier definition module.
Dedicated module to guarantee clean, invariant pickle serialization across all entry points.
"""

import numpy as np
import lightgbm as lgb
from features.feature_pipeline import STRUCTURAL_FEATURES, BEHAVIORAL_FEATURES

SEED = 42

def build_lgbm_model(num_class=3, seed=SEED):
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        objective="multiclass",
        num_class=num_class,
    )


class FusedCalibratedClassifier:
    name = "fused_calibrated"

    def __init__(self, conflict_kl_threshold=0.3):
        self.struct_model = build_lgbm_model()
        self.behav_model = build_lgbm_model()
        self.conflict_kl_threshold = conflict_kl_threshold

    def fit(self, struct_X, behav_X, y):
        self.struct_model.fit(struct_X[STRUCTURAL_FEATURES], y)
        self.behav_model.fit(behav_X[BEHAVIORAL_FEATURES], y)
        self.classes_ = np.arange(3)
        return self

    def predict_proba_sub(self, struct_X, behav_X):
        eps = 1e-9
        p_struct = self.struct_model.predict_proba(struct_X[STRUCTURAL_FEATURES].fillna(0))
        p_behav = self.behav_model.predict_proba(behav_X[BEHAVIORAL_FEATURES].fillna(0))
        p_fused = np.sqrt(p_struct * p_behav + eps)
        p_fused = p_fused / p_fused.sum(axis=1, keepdims=True)

        def kl_div(p, q):
            p = np.clip(p, eps, 1)
            q = np.clip(q, eps, 1)
            return np.sum(p * np.log(p / q), axis=1)

        symmetric_kl = (kl_div(p_struct, p_behav) + kl_div(p_behav, p_struct)) / 2
        conflicts = symmetric_kl > self.conflict_kl_threshold
        return p_struct, p_behav, p_fused, conflicts

    def predict_proba(self, X_combined):
        struct_X = X_combined[STRUCTURAL_FEATURES]
        behav_X = X_combined[BEHAVIORAL_FEATURES]
        _, _, p_fused, _ = self.predict_proba_sub(struct_X, behav_X)
        return p_fused

    def predict(self, X_combined):
        return np.argmax(self.predict_proba(X_combined), axis=1)
