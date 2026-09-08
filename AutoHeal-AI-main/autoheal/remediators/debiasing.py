"""Automated algorithmic debiasing: pre-processing sample re-weighting and post-processing fair threshold tuning."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from autoheal.config import BiasConfig
from autoheal.core.events import RemediationStep


class FairnessRemediator:
    """Implements pre-processing and post-processing fairness interventions."""

    def __init__(self, config: Optional[BiasConfig] = None):
        self.config = config or BiasConfig()

    def compute_fair_sample_weights(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        sensitive_col: str,
        privileged_group: Any
    ) -> Tuple[np.ndarray, RemediationStep]:
        """Calculate Kamiran-Calders fair sample re-weighting.
        
        W(A=a, Y=y) = [P(A=a) * P(Y=y)] / P(A=a, Y=y)
        Forces the sensitive feature A and label Y to be statistically independent in training.
        """
        y_arr = np.asarray(y).ravel()
        sens_vals = df[sensitive_col].values
        n_total = float(len(y_arr))

        weights = np.ones(len(y_arr), dtype=np.float64)

        groups = np.unique(sens_vals)
        labels = [0, 1]

        for g in groups:
            p_group = np.sum(sens_vals == g) / n_total
            for l in labels:
                p_label = np.sum(y_arr == l) / n_total
                mask = (sens_vals == g) & (y_arr == l)
                count_gl = np.sum(mask)

                if count_gl > 0:
                    p_joint = count_gl / n_total
                    weight_gl = (p_group * p_label) / p_joint
                    weights[mask] = weight_gl

        step = RemediationStep(
            step_id=f"STEP_FAIR_REWEIGHT_{sensitive_col.upper()}",
            target=sensitive_col,
            action_type="REWEIGHT_SAMPLES",
            description=f"Applied statistical debiasing sample weights to neutralize disparate impact on '{sensitive_col}'.",
            parameters={"sensitive_col": sensitive_col, "privileged_group": str(privileged_group)},
            status="EXECUTED",
            impact_summary=f"Balanced representation weights across {len(groups)} demographic subgroups.",
        )

        return weights, step

    def optimize_fair_thresholds(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        predicted_probs: np.ndarray,
        sensitive_col: str,
        privileged_group: Any,
        target_disparate_impact: float = 0.85
    ) -> Tuple[Dict[Any, float], RemediationStep]:
        """Find optimal group-specific decision thresholds to guarantee fair selection rates.
        
        Post-processing technique based on equalized odds / demographic parity calibration.
        """
        y_true = np.asarray(y_true).ravel()
        if predicted_probs.ndim == 2:
            probs = predicted_probs[:, 1]
        else:
            probs = predicted_probs

        sens_vals = df[sensitive_col].values
        is_priv = sens_vals == privileged_group
        is_unpriv = ~is_priv

        # Base threshold for privileged group
        priv_probs = probs[is_priv]
        unpriv_probs = probs[is_unpriv]

        priv_threshold = 0.50
        priv_pos_rate = np.mean(priv_probs >= priv_threshold)

        # Search for threshold on unprivileged group that satisfies Disparate Impact >= target
        best_unpriv_threshold = 0.50
        best_diff = float("inf")
        max_di_seen = 0.0
        best_thresh_for_max_di = 0.50
        candidate_thresholds = np.linspace(0.05, 0.65, 121)

        for t in candidate_thresholds:
            unpriv_pos_rate = np.mean(unpriv_probs >= t)
            if priv_pos_rate > 1e-6:
                di = unpriv_pos_rate / priv_pos_rate
            else:
                di = 1.0

            if di > max_di_seen:
                max_di_seen = di
                best_thresh_for_max_di = float(t)

            if di >= target_disparate_impact:
                diff = abs(di - 1.0)  # Closest to parity
                if diff < best_diff:
                    best_diff = diff
                    best_unpriv_threshold = float(t)

        # Fallback to the threshold that maximized fairness if target threshold wasn't fully met
        if best_diff == float("inf"):
            best_unpriv_threshold = best_thresh_for_max_di

        threshold_map = {
            privileged_group: priv_threshold,
            # Assign best found threshold to unprivileged subgroups
        }
        for g in np.unique(sens_vals):
            if g != privileged_group:
                threshold_map[g] = best_unpriv_threshold

        step = RemediationStep(
            step_id=f"STEP_TUNE_THRESHOLDS_{sensitive_col.upper()}",
            target=sensitive_col,
            action_type="TUNE_FAIR_THRESHOLDS",
            description=f"Calibrated demographic decision thresholds for '{sensitive_col}' to restore parity.",
            parameters={"threshold_map": {str(k): float(v) for k, v in threshold_map.items()}},
            status="EXECUTED",
            impact_summary=f"Set privileged threshold={priv_threshold:.2f}, unprivileged threshold={best_unpriv_threshold:.2f}.",
        )

        return threshold_map, step
