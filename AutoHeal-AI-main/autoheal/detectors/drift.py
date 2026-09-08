"""Covariate and concept drift detection using KS-test, PSI, Wasserstein distance, and Page-Hinkley."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats

from autoheal.config import DriftConfig
from autoheal.core.events import DiagnosticFinding, FindingSeverity, IssueType


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_buckets: int = 10,
    epsilon: float = 1e-4
) -> float:
    """Calculate the Population Stability Index (PSI) between reference and current samples."""
    ref_clean = reference[~np.isnan(reference)]
    curr_clean = current[~np.isnan(current)]
    
    if len(ref_clean) < 10 or len(curr_clean) < 10:
        return 0.0

    # Determine quantile bins from reference data
    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        bin_edges = np.percentile(ref_clean, percentiles)
        # Ensure distinct bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    except Exception:
        return 0.0

    # Count frequencies
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    curr_counts, _ = np.histogram(curr_clean, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (len(ref_clean) + epsilon * len(ref_counts))
    curr_pct = (curr_counts + epsilon) / (len(curr_clean) + epsilon * len(curr_counts))

    # PSI formula: sum((Actual - Expected) * ln(Actual / Expected))
    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(max(0.0, psi_val))


class PageHinkleyConceptDrift:
    """Page-Hinkley test for detecting change-points in streaming loss or error rates."""

    def __init__(self, delta: float = 0.005, threshold: float = 25.0):
        self.delta = delta
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.count = 0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = float("inf")

    def update(self, error_val: float) -> bool:
        """Update with a new error observation (e.g. 0 for correct, 1 for error).
        
        Returns True if concept drift is detected.
        """
        self.count += 1
        self.mean += (error_val - self.mean) / self.count
        self.cumulative_sum += (error_val - self.mean - self.delta)
        if self.cumulative_sum < self.min_cumulative_sum:
            self.min_cumulative_sum = self.cumulative_sum

        ph_stat = self.cumulative_sum - self.min_cumulative_sum
        return ph_stat > self.threshold


class DriftDetector:
    """Detects statistical covariate drift in features and concept drift in model predictions."""

    def __init__(self, config: Optional[DriftConfig] = None):
        self.config = config or DriftConfig()

    def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[List[DiagnosticFinding], Dict[str, Dict[str, Any]]]:
        """Test for covariate drift across continuous and categorical features."""
        findings: List[DiagnosticFinding] = []
        drift_metrics: Dict[str, Dict[str, Any]] = {}

        if feature_columns is None:
            feature_columns = [
                c for c in reference_df.columns
                if c in current_df.columns and c not in ["target", "label", "id"]
            ]

        for col in feature_columns:
            ref_col = reference_df[col].dropna()
            curr_col = current_df[col].dropna()

            if len(ref_col) < 10 or len(curr_col) < 10:
                continue

            # 1. Continuous Numerical Feature Analysis
            if pd.api.types.is_numeric_dtype(ref_col) and ref_col.nunique() > 5:
                # Kolmogorov-Smirnov 2-sample test
                ks_stat, ks_pval = stats.ks_2samp(ref_col.values, curr_col.values)
                # Population Stability Index (PSI)
                psi_val = calculate_psi(ref_col.values, curr_col.values)
                # Wasserstein distance
                ref_std = ref_col.std() or 1.0
                w_dist = stats.wasserstein_distance(ref_col.values, curr_col.values)
                w_rel = float(w_dist / ref_std)

                drift_detected = (ks_pval < self.config.ks_p_value_threshold) or (psi_val >= self.config.psi_warning_threshold)

                drift_metrics[col] = {
                    "type": "numerical",
                    "ks_statistic": float(ks_stat),
                    "ks_p_value": float(ks_pval),
                    "psi": psi_val,
                    "wasserstein_relative": w_rel,
                    "drift_detected": drift_detected,
                }

                if drift_detected:
                    is_critical = psi_val >= self.config.psi_critical_threshold or ks_pval < 1e-4
                    severity = FindingSeverity.CRITICAL if is_critical else FindingSeverity.WARNING
                    findings.append(
                        DiagnosticFinding(
                            issue_type=IssueType.COVARIATE_DRIFT,
                            severity=severity,
                            title=f"Significant covariate drift detected in '{col}'",
                            description=(
                                f"Feature '{col}' distribution drifted: KS p-value = {ks_pval:.2e} "
                                f"(stat={ks_stat:.3f}), PSI = {psi_val:.3f} "
                                f"(threshold >= {self.config.psi_warning_threshold})."
                            ),
                            affected_target=col,
                            metric_name="psi",
                            metric_value=psi_val,
                            threshold_value=self.config.psi_warning_threshold,
                            metadata=drift_metrics[col],
                        )
                    )

            # 2. Categorical Feature Analysis
            else:
                # Value counts frequency alignment
                all_cats = list(set(ref_col.unique()).union(set(curr_col.unique())))
                ref_counts = ref_col.value_counts(normalize=True).reindex(all_cats, fill_value=1e-5).values
                curr_counts = curr_col.value_counts(normalize=True).reindex(all_cats, fill_value=1e-5).values

                # Categorical PSI
                cat_psi = float(np.sum((curr_counts - ref_counts) * np.log(curr_counts / ref_counts)))
                cat_psi = max(0.0, cat_psi)

                drift_detected = cat_psi >= self.config.psi_warning_threshold
                drift_metrics[col] = {
                    "type": "categorical",
                    "psi": cat_psi,
                    "drift_detected": drift_detected,
                }

                if drift_detected:
                    severity = FindingSeverity.CRITICAL if cat_psi >= self.config.psi_critical_threshold else FindingSeverity.WARNING
                    findings.append(
                        DiagnosticFinding(
                            issue_type=IssueType.COVARIATE_DRIFT,
                            severity=severity,
                            title=f"Categorical distribution drift in '{col}'",
                            description=f"Categorical shift in '{col}': PSI = {cat_psi:.3f} (threshold >= {self.config.psi_warning_threshold}).",
                            affected_target=col,
                            metric_name="psi",
                            metric_value=cat_psi,
                            threshold_value=self.config.psi_warning_threshold,
                            metadata=drift_metrics[col],
                        )
                    )

        return findings, drift_metrics

    def detect_prediction_drift(
        self,
        ref_predictions: np.ndarray,
        curr_predictions: np.ndarray
    ) -> Tuple[List[DiagnosticFinding], Dict[str, Any]]:
        """Detect drift in model output predictions (soft probabilities and binary outputs)."""
        findings: List[DiagnosticFinding] = []
        metrics: Dict[str, Any] = {}

        ref_preds = np.asarray(ref_predictions).ravel()
        curr_preds = np.asarray(curr_predictions).ravel()

        if len(ref_preds) < 10 or len(curr_preds) < 10:
            return findings, metrics

        psi_val = calculate_psi(ref_preds, curr_preds)
        ks_stat, ks_pval = stats.ks_2samp(ref_preds, curr_preds)

        drift_detected = psi_val >= self.config.psi_warning_threshold or ks_pval < self.config.ks_p_value_threshold

        metrics = {
            "prediction_psi": psi_val,
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_pval),
            "drift_detected": drift_detected,
        }

        if drift_detected:
            severity = FindingSeverity.CRITICAL if psi_val >= self.config.psi_critical_threshold else FindingSeverity.WARNING
            findings.append(
                DiagnosticFinding(
                    issue_type=IssueType.CONCEPT_DRIFT,
                    severity=severity,
                    title="Model prediction distribution drift detected",
                    description=(
                        f"Output prediction distribution shifted significantly: PSI = {psi_val:.3f}, "
                        f"KS p-value = {ks_pval:.2e}."
                    ),
                    affected_target="model_predictions",
                    metric_name="prediction_psi",
                    metric_value=psi_val,
                    threshold_value=self.config.psi_warning_threshold,
                    metadata=metrics,
                )
            )

        return findings, metrics
