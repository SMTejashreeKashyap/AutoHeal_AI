"""Data error detection module: identifies missing values, schema violations, outliers, and noisy labels."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest

from autoheal.config import DataErrorConfig
from autoheal.core.events import DiagnosticFinding, FindingSeverity, IssueType


class DataErrorDetector:
    """Detects missingness anomalies, domain/schema corruptions, outliers, and label noise."""

    def __init__(self, config: Optional[DataErrorConfig] = None):
        self.config = config or DataErrorConfig()

    def detect_missing_values(self, df: pd.DataFrame) -> Tuple[List[DiagnosticFinding], Dict[str, float]]:
        """Identify columns with abnormal rates of missing/null values."""
        findings: List[DiagnosticFinding] = []
        missing_stats: Dict[str, float] = {}
        total_rows = len(df)
        if total_rows == 0:
            return findings, missing_stats

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_rate = float(missing_count / total_rows)
            missing_stats[col] = missing_rate

            if missing_rate >= self.config.missing_rate_critical_threshold:
                findings.append(
                    DiagnosticFinding(
                        issue_type=IssueType.MISSING_VALUES,
                        severity=FindingSeverity.CRITICAL,
                        title=f"Critical missingness in column '{col}'",
                        description=f"Column '{col}' is missing {missing_rate:.1%} of its records ({missing_count}/{total_rows}).",
                        affected_target=col,
                        metric_name="missing_rate",
                        metric_value=missing_rate,
                        threshold_value=self.config.missing_rate_critical_threshold,
                        metadata={"missing_count": int(missing_count), "total_rows": total_rows},
                    )
                )
            elif missing_rate >= self.config.missing_rate_alert_threshold:
                findings.append(
                    DiagnosticFinding(
                        issue_type=IssueType.MISSING_VALUES,
                        severity=FindingSeverity.WARNING,
                        title=f"Elevated missingness in column '{col}'",
                        description=f"Column '{col}' has {missing_rate:.1%} missing records ({missing_count}/{total_rows}).",
                        affected_target=col,
                        metric_name="missing_rate",
                        metric_value=missing_rate,
                        threshold_value=self.config.missing_rate_alert_threshold,
                        metadata={"missing_count": int(missing_count), "total_rows": total_rows},
                    )
                )

        return findings, missing_stats

    def detect_schema_and_domain_violations(
        self,
        df: pd.DataFrame,
        domain_rules: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Tuple[List[DiagnosticFinding], Dict[str, int]]:
        """Detect invalid domains such as negative values where forbidden or values outside sensible bounds."""
        findings: List[DiagnosticFinding] = []
        violation_counts: Dict[str, int] = {}

        if domain_rules is None:
            # Default sensible rules for credit/risk datasets
            domain_rules = {
                "income": {"min": 0.0, "max": 10_000_000.0},
                "debt_to_income": {"min": 0.0, "max": 10.0},
                "credit_score": {"min": 300.0, "max": 850.0},
                "loan_amount": {"min": 100.0, "max": 5_000_000.0},
                "employment_length": {"min": 0.0, "max": 70.0},
            }

        for col, rules in domain_rules.items():
            if col not in df.columns:
                continue

            series = pd.to_numeric(df[col], errors="coerce")
            violations = pd.Series(False, index=df.index)

            if "min" in rules:
                violations |= series < rules["min"]
            if "max" in rules:
                violations |= series > rules["max"]

            # Also check type coercion failures (NaNs created from non-numeric text)
            type_errors = df[col].notnull() & series.isnull()
            violations |= type_errors

            v_count = int(violations.sum())
            violation_counts[col] = v_count

            if v_count > 0:
                v_rate = float(v_count / len(df))
                findings.append(
                    DiagnosticFinding(
                        issue_type=IssueType.SCHEMA_VIOLATION,
                        severity=FindingSeverity.CRITICAL if v_rate > 0.02 else FindingSeverity.WARNING,
                        title=f"Domain bounds violation in column '{col}'",
                        description=f"{v_count} records ({v_rate:.1%}) violate domain constraints (expected {rules.get('min')} to {rules.get('max')}).",
                        affected_target=col,
                        metric_name="domain_violation_count",
                        metric_value=float(v_count),
                        threshold_value=0.0,
                        metadata={"violation_count": v_count, "violation_rate": v_rate, "rules": rules},
                    )
                )

        return findings, violation_counts

    def detect_outliers(
        self,
        df: pd.DataFrame,
        numerical_columns: Optional[List[str]] = None
    ) -> Tuple[List[DiagnosticFinding], Dict[str, np.ndarray]]:
        """Identify anomalous outlier samples via IQR or Isolation Forest."""
        findings: List[DiagnosticFinding] = []
        outlier_masks: Dict[str, np.ndarray] = {}

        if numerical_columns is None:
            numerical_columns = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in ["target", "label", "id"]
            ]

        for col in numerical_columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 10:
                continue

            if self.config.outlier_method == "iqr":
                q25, q75 = series.quantile(0.25), series.quantile(0.75)
                iqr = q75 - q25
                if iqr == 0:
                    iqr = 1e-6
                lower_bound = q25 - (self.config.outlier_iqr_multiplier * iqr)
                upper_bound = q75 + (self.config.outlier_iqr_multiplier * iqr)
                mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            elif self.config.outlier_method == "zscore":
                mean = series.mean()
                std = series.std() or 1e-6
                z = (df[col] - mean).abs() / std
                mask = z > self.config.outlier_zscore_threshold
            else:  # isolation_forest
                iso = IsolationForest(
                    contamination=self.config.outlier_contamination,
                    random_state=42
                )
                preds = iso.fit_predict(df[[col]].fillna(series.median()))
                mask = preds == -1

            mask_arr = mask.fillna(False).values
            outlier_masks[col] = mask_arr
            outlier_count = int(mask_arr.sum())
            outlier_rate = float(outlier_count / len(df))

            if outlier_rate >= self.config.outlier_rate_alert_threshold:
                findings.append(
                    DiagnosticFinding(
                        issue_type=IssueType.OUTLIERS,
                        severity=FindingSeverity.WARNING if outlier_rate < 0.08 else FindingSeverity.CRITICAL,
                        title=f"Excessive statistical outliers in '{col}'",
                        description=f"Identified {outlier_count} anomalous outliers ({outlier_rate:.1%}) exceeding {self.config.outlier_method.upper()} bounds.",
                        affected_target=col,
                        metric_name="outlier_rate",
                        metric_value=outlier_rate,
                        threshold_value=self.config.outlier_rate_alert_threshold,
                        metadata={
                            "outlier_count": outlier_count,
                            "method": self.config.outlier_method,
                        },
                    )
                )

        return findings, outlier_masks

    def detect_label_noise(
        self,
        y_true: np.ndarray,
        predicted_proba: np.ndarray
    ) -> Tuple[List[DiagnosticFinding], np.ndarray]:
        """Detect label corruption using confident learning principles.
        
        Flags samples where the model's confidence in the opposite class strongly contradicts
        the provided ground-truth label.
        """
        findings: List[DiagnosticFinding] = []
        y_true = np.asarray(y_true).ravel()
        
        # P(Class=1)
        if predicted_proba.ndim == 2:
            p1 = predicted_proba[:, 1]
        else:
            p1 = predicted_proba

        # If true label is 0, but model says P(Class=1) >= threshold -> label noise suspected
        # If true label is 1, but model says P(Class=0) >= threshold -> label noise suspected
        thresh = self.config.label_noise_confidence_threshold
        suspected_noise = ((y_true == 0) & (p1 >= thresh)) | ((y_true == 1) & ((1.0 - p1) >= thresh))
        
        noisy_count = int(suspected_noise.sum())
        total = len(y_true)
        noise_rate = float(noisy_count / total) if total > 0 else 0.0

        if noise_rate >= self.config.label_noise_rate_alert_threshold:
            findings.append(
                DiagnosticFinding(
                    issue_type=IssueType.LABEL_NOISE,
                    severity=FindingSeverity.CRITICAL if noise_rate > 0.10 else FindingSeverity.WARNING,
                    title="Suspected noisy or flipped labels detected",
                    description=f"Detected {noisy_count} samples ({noise_rate:.1%}) where high-confidence predictions sharply contradict ground-truth labels.",
                    affected_target="target",
                    metric_name="label_noise_rate",
                    metric_value=noise_rate,
                    threshold_value=self.config.label_noise_rate_alert_threshold,
                    metadata={"noisy_count": noisy_count, "total_samples": total},
                )
            )

        return findings, suspected_noise

    def run_all(
        self,
        df: pd.DataFrame,
        numerical_columns: Optional[List[str]] = None,
        domain_rules: Optional[Dict[str, Dict[str, Any]]] = None,
        y_true: Optional[np.ndarray] = None,
        predicted_proba: Optional[np.ndarray] = None
    ) -> List[DiagnosticFinding]:
        """Run all data error detection routines and aggregate findings."""
        all_findings: List[DiagnosticFinding] = []
        
        # 1. Missing values
        f_missing, _ = self.detect_missing_values(df)
        all_findings.extend(f_missing)
        
        # 2. Schema and domain bounds
        f_schema, _ = self.detect_schema_and_domain_violations(df, domain_rules)
        all_findings.extend(f_schema)
        
        # 3. Outliers
        f_outliers, _ = self.detect_outliers(df, numerical_columns)
        all_findings.extend(f_outliers)
        
        # 4. Label noise if model probabilities provided
        if y_true is not None and predicted_proba is not None:
            f_labels, _ = self.detect_label_noise(y_true, predicted_proba)
            all_findings.extend(f_labels)
            
        return all_findings
