"""Algorithmic bias and fairness auditing module: detects Disparate Impact, Demographic Parity, and Equalized Odds violations."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

from autoheal.config import BiasConfig
from autoheal.core.events import DiagnosticFinding, FindingSeverity, IssueType


class BiasDetector:
    """Evaluates fairness metrics across sensitive protected attributes."""

    def __init__(self, config: Optional[BiasConfig] = None):
        self.config = config or BiasConfig()

    def evaluate_fairness(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_col: str,
        privileged_group: Any
    ) -> Dict[str, Any]:
        """Compute detailed fairness metrics for a specific sensitive attribute.
        
        Args:
            df: DataFrame containing the sensitive attribute column.
            y_true: Ground-truth binary labels (1 = favorable, 0 = unfavorable).
            y_pred: Model predicted binary labels.
            sensitive_col: Name of protected column (e.g., 'gender').
            privileged_group: Value representing privileged class (e.g., 'Male').
        """
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        sens_vals = df[sensitive_col].values

        is_priv = sens_vals == privileged_group
        is_unpriv = ~is_priv

        priv_count = int(np.sum(is_priv))
        unpriv_count = int(np.sum(is_unpriv))

        if priv_count == 0 or unpriv_count == 0:
            return {"error": f"Insufficient representation for comparison in '{sensitive_col}'"}

        # Positive selection rates P(Y_hat = 1)
        priv_pos_rate = float(np.mean(y_pred[is_priv] == 1))
        unpriv_pos_rate = float(np.mean(y_pred[is_unpriv] == 1))

        # 1. Disparate Impact
        if priv_pos_rate > 1e-6:
            disparate_impact = unpriv_pos_rate / priv_pos_rate
        else:
            disparate_impact = 1.0 if unpriv_pos_rate == 0 else 2.0

        # 2. Demographic Parity Difference
        demographic_parity_diff = abs(priv_pos_rate - unpriv_pos_rate)

        # 3. Equal Opportunity & Equalized Odds (TPR and FPR comparisons)
        # Privileged TPR & FPR
        priv_pos_true = y_true[is_priv] == 1
        priv_neg_true = y_true[is_priv] == 0

        priv_tpr = float(np.mean(y_pred[is_priv][priv_pos_true] == 1)) if np.sum(priv_pos_true) > 0 else 0.0
        priv_fpr = float(np.mean(y_pred[is_priv][priv_neg_true] == 1)) if np.sum(priv_neg_true) > 0 else 0.0

        # Unprivileged TPR & FPR
        unpriv_pos_true = y_true[is_unpriv] == 1
        unpriv_neg_true = y_true[is_unpriv] == 0

        unpriv_tpr = float(np.mean(y_pred[is_unpriv][unpriv_pos_true] == 1)) if np.sum(unpriv_pos_true) > 0 else 0.0
        unpriv_fpr = float(np.mean(y_pred[is_unpriv][unpriv_neg_true] == 1)) if np.sum(unpriv_neg_true) > 0 else 0.0

        tpr_diff = abs(priv_tpr - unpriv_tpr)
        fpr_diff = abs(priv_fpr - unpriv_fpr)
        equalized_odds_max_diff = max(tpr_diff, fpr_diff)

        return {
            "sensitive_col": sensitive_col,
            "privileged_group": str(privileged_group),
            "priv_count": priv_count,
            "unpriv_count": unpriv_count,
            "priv_selection_rate": priv_pos_rate,
            "unpriv_selection_rate": unpriv_pos_rate,
            "disparate_impact": disparate_impact,
            "demographic_parity_diff": demographic_parity_diff,
            "priv_tpr": priv_tpr,
            "unpriv_tpr": unpriv_tpr,
            "tpr_diff": tpr_diff,
            "priv_fpr": priv_fpr,
            "unpriv_fpr": unpriv_fpr,
            "fpr_diff": fpr_diff,
            "equalized_odds_max_diff": equalized_odds_max_diff,
        }

    def detect_bias(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Tuple[List[DiagnosticFinding], Dict[str, Dict[str, Any]]]:
        """Audit all configured sensitive attributes and flag fairness violations."""
        findings: List[DiagnosticFinding] = []
        full_metrics: Dict[str, Dict[str, Any]] = {}

        for sens_col in self.config.sensitive_features:
            if sens_col not in df.columns:
                continue

            priv_group = self.config.privileged_groups.get(sens_col)
            if priv_group is None:
                # Default to mode if not specified
                priv_group = df[sens_col].mode()[0]

            metrics = self.evaluate_fairness(df, y_true, y_pred, sens_col, priv_group)
            if "error" in metrics:
                continue

            full_metrics[sens_col] = metrics
            di = metrics["disparate_impact"]
            dpd = metrics["demographic_parity_diff"]
            eq_diff = metrics["equalized_odds_max_diff"]

            # 1. Model Prediction Disparate Impact check (Four-Fifths Rule)
            if di < self.config.disparate_impact_min:
                severity = FindingSeverity.CRITICAL if di < 0.70 else FindingSeverity.WARNING
                findings.append(
                    DiagnosticFinding(
                        issue_type=IssueType.DISPARATE_IMPACT,
                        severity=severity,
                        title=f"Disparate impact bias on '{sens_col}'",
                        description=(
                            f"Model favorable decision rate for unprivileged '{sens_col}' is only {di:.1%} "
                            f"of privileged group '{priv_group}' (Violates 80% rule: threshold >= {self.config.disparate_impact_min})."
                        ),
                        affected_target=sens_col,
                        metric_name="disparate_impact",
                        metric_value=di,
                        threshold_value=self.config.disparate_impact_min,
                        metadata=metrics,
                    )
                )

            # 2. Ground-Truth / Historical Label Bias check
            if y_true is not None:
                is_priv_arr = df[sens_col].values == priv_group
                true_priv_pos = float(np.mean(y_true[is_priv_arr] == 1)) if np.sum(is_priv_arr) > 0 else 0.0
                true_unpriv_pos = float(np.mean(y_true[~is_priv_arr] == 1)) if np.sum(~is_priv_arr) > 0 else 0.0
                true_di = (true_unpriv_pos / true_priv_pos) if true_priv_pos > 1e-6 else 1.0
                metrics["ground_truth_disparate_impact"] = true_di

                if true_di < self.config.disparate_impact_min:
                    findings.append(
                        DiagnosticFinding(
                            issue_type=IssueType.DISPARATE_IMPACT,
                            severity=FindingSeverity.CRITICAL if true_di < 0.70 else FindingSeverity.WARNING,
                            title=f"Historical ground-truth bias in '{sens_col}' labels",
                            description=(
                                f"Historical favorable rate for unprivileged '{sens_col}' is only {true_di:.1%} "
                                f"of privileged group '{priv_group}' (Violates 80% rule: threshold >= {self.config.disparate_impact_min})."
                            ),
                            affected_target=sens_col,
                            metric_name="ground_truth_disparate_impact",
                            metric_value=true_di,
                            threshold_value=self.config.disparate_impact_min,
                            metadata=metrics,
                        )
                    )

            # 2. Demographic Parity Difference check
            if dpd > self.config.demographic_parity_max_diff:
                findings.append(
                    DiagnosticFinding(
                        issue_type=IssueType.DEMOGRAPHIC_PARITY,
                        severity=FindingSeverity.WARNING,
                        title=f"Demographic parity disparity on '{sens_col}'",
                        description=(
                            f"Selection rate gap of {dpd:.1%} across '{sens_col}' subgroups "
                            f"(threshold <= {self.config.demographic_parity_max_diff:.1%})."
                        ),
                        affected_target=sens_col,
                        metric_name="demographic_parity_diff",
                        metric_value=dpd,
                        threshold_value=self.config.demographic_parity_max_diff,
                        metadata=metrics,
                    )
                )

            # 3. Equalized Odds / Error Rate disparity check
            if eq_diff > self.config.equalized_odds_max_diff:
                findings.append(
                    DiagnosticFinding(
                        issue_type=IssueType.EQUALIZED_ODDS,
                        severity=FindingSeverity.WARNING,
                        title=f"Unequal error rates (Equalized Odds) on '{sens_col}'",
                        description=(
                            f"Disparity of {eq_diff:.1%} in TPR/FPR across '{sens_col}' subgroups "
                            f"(threshold <= {self.config.equalized_odds_max_diff:.1%})."
                        ),
                        affected_target=sens_col,
                        metric_name="equalized_odds_max_diff",
                        metric_value=eq_diff,
                        threshold_value=self.config.equalized_odds_max_diff,
                        metadata=metrics,
                    )
                )

        return findings, full_metrics
