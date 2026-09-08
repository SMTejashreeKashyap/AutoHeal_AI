"""Autonomous retraining engine: trains debiased candidate models and evaluates canary promotion gates."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from autoheal.config import PromotionConfig
from autoheal.core.events import RemediationStep
from autoheal.core.registry import ModelArtifact, ModelRegistry


class SelfHealingRetrainer:
    """Retrains models using cleansed data and fair sample weights, evaluating canary criteria."""

    def __init__(self, registry: ModelRegistry, config: Optional[PromotionConfig] = None):
        self.registry = registry
        self.config = config or PromotionConfig()

    def _build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """Construct ColumnTransformer for numerical scaling and categorical one-hot encoding."""
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))

        return ColumnTransformer(transformers=transformers)

    def evaluate_model(
        self,
        model: Any,
        preprocessor: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        group_thresholds: Optional[Dict[str, Dict[Any, float]]] = None,
        sensitive_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Compute standard predictive metrics."""
        y_test_arr = np.asarray(y_test).ravel()
        X_trans = preprocessor.transform(X_test)
        
        # Determine probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_trans)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_trans)
            probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs = model.predict(X_trans)

        # Apply group thresholds if present
        if group_thresholds and sensitive_df is not None:
            preds = np.zeros(len(probs), dtype=int)
            for i in range(len(probs)):
                rec_thresh = 0.50
                for sens_col, t_map in group_thresholds.items():
                    if sens_col in sensitive_df.columns:
                        v = sensitive_df.iloc[i][sens_col]
                        if v in t_map:
                            rec_thresh = t_map[v]
                            break
                preds[i] = int(probs[i] >= rec_thresh)
        else:
            preds = (probs >= 0.50).astype(int)

        acc = float(accuracy_score(y_test_arr, preds))
        prec = float(precision_score(y_test_arr, preds, zero_division=0))
        rec = float(recall_score(y_test_arr, preds, zero_division=0))
        f1 = float(f1_score(y_test_arr, preds, zero_division=0))
        try:
            auc = float(roc_auc_score(y_test_arr, probs))
        except Exception:
            auc = 0.50

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
        }

    def train_candidate(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        model_type: str = "gradient_boosting",
        group_thresholds: Optional[Dict[str, Dict[Any, float]]] = None,
        val_sensitive_df: Optional[pd.DataFrame] = None
    ) -> Tuple[ModelArtifact, Dict[str, float]]:
        """Fit preprocessor and candidate model on clean training data and validate."""
        preprocessor = self._build_preprocessor(X_train)
        X_train_trans = preprocessor.fit_transform(X_train)

        # Select model algorithm
        if model_type == "random_forest":
            clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
        elif model_type == "logistic_regression":
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:  # default gradient_boosting
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

        # Train model with fair sample weights if provided
        if sample_weights is not None:
            clf.fit(X_train_trans, y_train, sample_weight=sample_weights)
        else:
            clf.fit(X_train_trans, y_train)

        # Evaluate candidate metrics
        metrics = self.evaluate_model(
            clf,
            preprocessor,
            X_val,
            y_val,
            group_thresholds=group_thresholds,
            sensitive_df=val_sensitive_df
        )

        version_id = f"v{len(self.registry._models) + 1}_{model_type}_{datetime.now().strftime('%H%M%S')}"
        artifact = ModelArtifact(
            version=version_id,
            model=clf,
            model_type=model_type,
            feature_names=X_train.columns.tolist(),
            target_name="target",
            preprocessor=preprocessor,
            group_thresholds=group_thresholds or {},
            metrics=metrics,
            status="CANDIDATE",
            training_data_signature={"rows": len(X_train), "features": len(X_train.columns)},
        )

        return artifact, metrics

    def canary_evaluation(
        self,
        challenger: ModelArtifact,
        champion: Optional[ModelArtifact],
        challenger_fairness: Dict[str, Any],
        champion_fairness: Optional[Dict[str, Any]] = None,
        champion_current_metrics: Optional[Dict[str, float]] = None,
        targeted_bias_attributes: Optional[List[str]] = None
    ) -> Tuple[bool, str, RemediationStep]:
        """Verify whether challenger passes canary criteria to be promoted over current champion."""
        passed = True
        reasons = []

        # 1. Performance check (F1 score tolerance)
        ch_f1 = challenger.metrics.get("f1", 0.0)
        ch_acc = challenger.metrics.get("accuracy", 0.0)

        # Baseline comparison: compare against degraded champion on current data if available
        baseline_f1 = 0.0
        if champion_current_metrics and "f1" in champion_current_metrics:
            baseline_f1 = champion_current_metrics["f1"]
        elif champion and "f1" in champion.metrics:
            baseline_f1 = champion.metrics["f1"]

        if baseline_f1 > 0:
            min_allowed_f1 = baseline_f1 - self.config.f1_tolerance
            if ch_f1 < min_allowed_f1:
                passed = False
                reasons.append(f"Challenger F1 ({ch_f1:.3f}) fell below minimum required ({min_allowed_f1:.3f}).")
            else:
                reasons.append(f"Challenger F1 ({ch_f1:.3f}) meets threshold (vs Baseline {baseline_f1:.3f}).")
        else:
            if ch_acc < self.config.min_accuracy_threshold:
                passed = False
                reasons.append(f"Challenger Accuracy ({ch_acc:.3f}) below baseline threshold {self.config.min_accuracy_threshold:.3f}.")
            else:
                reasons.append(f"Challenger Accuracy ({ch_acc:.3f}) meets baseline threshold.")

        # 2. Fairness checks
        if self.config.enforce_fairness_in_promotion:
            targets = targeted_bias_attributes or list(challenger_fairness.keys())
            for col, f_data in challenger_fairness.items():
                di = f_data.get("disparate_impact", 1.0)
                if col in targets:
                    # Enforce four-fifths rule on targeted debiased attributes
                    if di < 0.80:
                        passed = False
                        reasons.append(f"Challenger failed Disparate Impact gate on targeted '{col}' ({di:.2f} < 0.80).")
                    else:
                        reasons.append(f"Challenger passed Disparate Impact gate on '{col}' ({di:.2f} >= 0.80).")
                else:
                    # Non-targeted attributes must not experience severe regression
                    prev_di = champion_current_metrics.get(f"di_{col}", 0.80) if champion_current_metrics else 0.80
                    if di < max(0.60, prev_di - 0.20):
                        passed = False
                        reasons.append(f"Challenger regressed on secondary attribute '{col}' ({di:.2f}).")
                    else:
                        reasons.append(f"Challenger maintained fairness on secondary '{col}' ({di:.2f}).")

        summary_reason = " | ".join(reasons)
        step = RemediationStep(
            step_id="STEP_CANARY_EVALUATION",
            target=challenger.version,
            action_type="CANARY_EVALUATION",
            description="Evaluated challenger against canary safety and fairness gates.",
            parameters={"passed": passed, "reasons": reasons},
            status="EXECUTED" if passed else "FAILED",
            impact_summary="Passed all performance and fairness gates." if passed else f"Promotion blocked: {summary_reason}",
        )

        return passed, summary_reason, step
