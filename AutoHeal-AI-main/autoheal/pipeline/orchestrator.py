"""Autonomous Self-Healing Feedback Pipeline Orchestrator."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split

from autoheal.config import AutoHealConfig
from autoheal.core.events import (
    DiagnosticFinding,
    RemediationStep,
    SelfHealingReport,
    MetricComparison,
    IssueType,
    FindingSeverity,
)
from autoheal.core.registry import ModelRegistry, ModelArtifact
from autoheal.core.reporter import SelfHealingReporter
from autoheal.detectors.data_errors import DataErrorDetector
from autoheal.detectors.bias import BiasDetector
from autoheal.detectors.drift import DriftDetector
from autoheal.remediators.cleaner import AutomatedDataCleaner
from autoheal.remediators.debiasing import FairnessRemediator
from autoheal.remediators.retrainer import SelfHealingRetrainer


class SelfHealingOrchestrator:
    """End-to-end controller orchestrating continuous diagnosis, automated remediation,

    model retraining, and canary verification.
    """

    def __init__(self, config: Optional[AutoHealConfig] = None):
        self.config = config or AutoHealConfig()
        self.registry = ModelRegistry()
        
        # Initialize components
        self.data_error_detector = DataErrorDetector(self.config.data_error)
        self.bias_detector = BiasDetector(self.config.bias)
        self.drift_detector = DriftDetector(self.config.drift)
        self.cleaner = AutomatedDataCleaner(self.config.remediation)
        self.fairness_remediator = FairnessRemediator(self.config.bias)
        self.retrainer = SelfHealingRetrainer(self.registry, self.config.promotion)
        
        # State
        self.reference_data: Optional[pd.DataFrame] = None
        self.incident_history: List[SelfHealingReport] = []

    def bootstrap_baseline(self, df: pd.DataFrame, target_col: str = "target") -> ModelArtifact:
        """Initialize the baseline champion model on clean reference data."""
        self.reference_data = df.copy()
        
        X = df.drop(columns=[target_col])
        y = df[target_col].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.20, random_state=self.config.random_state, stratify=y
        )

        artifact, metrics = self.retrainer.train_candidate(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            model_type="gradient_boosting"
        )
        artifact.version = "v1_baseline_champion"
        artifact.status = "CHAMPION"
        self.registry.register(artifact)
        self.registry.promote_to_champion(artifact.version, notes="Initial baseline champion model.")
        return artifact

    def diagnose(self, current_df: pd.DataFrame, target_col: str = "target") -> List[DiagnosticFinding]:
        """Perform full multi-dimensional diagnostic scan on production data."""
        findings: List[DiagnosticFinding] = []
        champion = self.registry.get_champion()

        if champion is None:
            raise ValueError("No champion model registered. Call bootstrap_baseline() first.")

        # 1. Detect Data Errors (missingness, schema, outliers)
        feature_cols = [c for c in current_df.columns if c != target_col]
        error_findings = self.data_error_detector.run_all(
            df=current_df[feature_cols],
            numerical_columns=current_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        )
        findings.extend(error_findings)

        # 2. Run model predictions on current data
        clean_temp = current_df[feature_cols].fillna(self.reference_data[feature_cols].median(numeric_only=True))
        y_probs = champion.predict_proba(clean_temp)
        y_preds = champion.predict_with_fair_thresholds(clean_temp, sensitive_df=current_df)

        # Label noise detection if ground truth is present
        if target_col in current_df.columns and current_df[target_col].notnull().sum() > 0:
            y_true = current_df[target_col].dropna().values
            clean_with_target = current_df.dropna(subset=[target_col])
            clean_temp_subset = clean_with_target[feature_cols].fillna(self.reference_data[feature_cols].median(numeric_only=True))
            probs_subset = champion.predict_proba(clean_temp_subset)
            
            f_noise, _ = self.data_error_detector.detect_label_noise(y_true, probs_subset)
            findings.extend(f_noise)

            # 3. Detect Algorithmic Bias
            bias_findings, _ = self.bias_detector.detect_bias(
                df=clean_with_target,
                y_true=y_true,
                y_pred=champion.predict_with_fair_thresholds(clean_temp_subset, sensitive_df=clean_with_target)
            )
            findings.extend(bias_findings)

        # 4. Detect Covariate & Prediction Drift against Reference Baseline
        if self.reference_data is not None:
            drift_findings, _ = self.drift_detector.detect_feature_drift(
                reference_df=self.reference_data,
                current_df=current_df,
                feature_columns=feature_cols
            )
            findings.extend(drift_findings)

            # Prediction drift
            ref_clean = self.reference_data[feature_cols].fillna(self.reference_data[feature_cols].median(numeric_only=True))
            ref_probs = champion.predict_proba(ref_clean)[:, 1]
            pred_drift_findings, _ = self.drift_detector.detect_prediction_drift(
                ref_predictions=ref_probs,
                curr_predictions=y_probs[:, 1]
            )
            findings.extend(pred_drift_findings)

        return findings

    def heal(
        self,
        current_df: pd.DataFrame,
        target_col: str = "target",
        gold_eval_df: Optional[pd.DataFrame] = None
    ) -> SelfHealingReport:
        """Autonomous self-healing execution:

        Diagnosis -> Remediation Plan Formulation -> Cleansing -> Debiasing -> Retraining -> Canary Gate -> Promotion
        """
        cycle_id = f"HEAL_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report = SelfHealingReport(cycle_id=cycle_id)
        champion_before = self.registry.get_champion()
        report.champion_version_before = champion_before.version if champion_before else None

        # 1. DIAGNOSIS
        findings = self.diagnose(current_df, target_col=target_col)
        for f in findings:
            report.add_finding(f)

        # Evaluate degraded performance on current data before healing
        X_curr_features = current_df.drop(columns=[target_col]) if target_col in current_df.columns else current_df
        temp_filled = X_curr_features.fillna(self.reference_data[X_curr_features.columns].median(numeric_only=True))
        
        if target_col in current_df.columns:
            y_curr = current_df[target_col].values
            valid_mask = ~np.isnan(y_curr)
            if valid_mask.sum() > 0:
                report.degraded_metrics = self.retrainer.evaluate_model(
                    champion_before.model,
                    champion_before.preprocessor,
                    temp_filled.loc[valid_mask],
                    y_curr[valid_mask]
                )
                _, deg_fair = self.bias_detector.detect_bias(
                    current_df.loc[valid_mask],
                    y_curr[valid_mask],
                    champion_before.predict_with_fair_thresholds(temp_filled.loc[valid_mask], sensitive_df=current_df.loc[valid_mask])
                )
                for s_col, f_data in deg_fair.items():
                    report.degraded_metrics[f"di_{s_col}"] = f_data["disparate_impact"]

        # 2. PLAN FORMULATION & DATA CLEANSING
        # Check if label noise was flagged
        suspected_noise_mask = None
        has_label_noise = any(f.issue_type == IssueType.LABEL_NOISE for f in findings)
        if has_label_noise and target_col in current_df.columns:
            y_true = current_df[target_col].values
            probs = champion_before.predict_proba(temp_filled)
            _, suspected_noise_mask = self.data_error_detector.detect_label_noise(y_true, probs)

        # Fit automated data cleaner on current data
        cleaned_df, cleaning_steps = self.cleaner.fit_and_clean(
            df=current_df,
            target_col=target_col,
            suspected_noisy_indices=suspected_noise_mask
        )
        for step in cleaning_steps:
            report.add_step(step)

        # 3. FAIRNESS DEBIASING INTERVENTIONS
        sample_weights = None
        group_thresholds: Dict[str, Dict[Any, float]] = {}

        for sens_col in self.config.bias.sensitive_features:
            if sens_col in cleaned_df.columns and target_col in cleaned_df.columns:
                priv_group = self.config.bias.privileged_groups.get(sens_col, "Male")
                
                # Check if this sensitive attribute has bias findings
                has_bias = any(
                    f.affected_target == sens_col and f.issue_type in [IssueType.DISPARATE_IMPACT, IssueType.DEMOGRAPHIC_PARITY, IssueType.EQUALIZED_ODDS]
                    for f in findings
                )
                # Only reweight sensitive attributes that exhibit bias
                if has_bias:
                    weights, rw_step = self.fairness_remediator.compute_fair_sample_weights(
                        df=cleaned_df,
                        y=cleaned_df[target_col].values,
                        sensitive_col=sens_col,
                        privileged_group=priv_group
                    )
                    sample_weights = weights if sample_weights is None else (sample_weights * weights)
                    report.add_step(rw_step)

        # 4. SPLIT & RETRAIN HEALED CANDIDATE
        X_clean = cleaned_df.drop(columns=[target_col])
        y_clean = cleaned_df[target_col].values

        X_train, X_val, y_train, y_val = train_test_split(
            X_clean, y_clean, test_size=0.25, random_state=self.config.random_state, stratify=y_clean
        )

        train_weights = sample_weights[X_train.index.to_numpy()] if sample_weights is not None else None

        # Train healed candidate model
        challenger, candidate_metrics = self.retrainer.train_candidate(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            sample_weights=train_weights,
            model_type="gradient_boosting",
            val_sensitive_df=X_val
        )
        challenger.remediation_notes = "Trained with AutoHeal cleansed dataset and fair sample weights."

        # Optimize post-processing fair thresholds on validation set for biased attributes
        val_probs = challenger.predict_proba(X_val)[:, 1]
        biased_sens_cols = [
            f.affected_target for f in findings
            if f.issue_type in [IssueType.DISPARATE_IMPACT, IssueType.DEMOGRAPHIC_PARITY, IssueType.EQUALIZED_ODDS]
        ]

        for sens_col in self.config.bias.sensitive_features:
            if sens_col in X_val.columns:
                priv_group = self.config.bias.privileged_groups.get(sens_col, "Male")
                raw_fair = self.bias_detector.evaluate_fairness(
                    X_val, y_val, (val_probs >= 0.50).astype(int), sens_col, priv_group
                )
                raw_di = raw_fair.get("disparate_impact", 1.0)
                # Tune threshold only if this attribute was flagged for bias
                if sens_col in biased_sens_cols:
                    t_map, t_step = self.fairness_remediator.optimize_fair_thresholds(
                        df=X_val,
                        y_true=y_val,
                        predicted_probs=val_probs,
                        sensitive_col=sens_col,
                        privileged_group=priv_group,
                        target_disparate_impact=0.85
                    )
                    group_thresholds[sens_col] = t_map
                    report.add_step(t_step)

        challenger.group_thresholds = group_thresholds

        # Register challenger in registry
        self.registry.register(challenger)

        # 5. CANARY VERIFICATION & PROMOTION
        # Use gold evaluation dataset if provided, else use validation split
        eval_df = gold_eval_df if gold_eval_df is not None else cleaned_df.loc[X_val.index]
        eval_X = eval_df.drop(columns=[target_col])
        eval_y = eval_df[target_col].values

        challenger_preds = challenger.predict_with_fair_thresholds(eval_X, sensitive_df=eval_df)
        _, ch_fairness = self.bias_detector.detect_bias(eval_df, eval_y, challenger_preds)

        canary_passed, reason, canary_step = self.retrainer.canary_evaluation(
            challenger=challenger,
            champion=champion_before,
            challenger_fairness=ch_fairness,
            champion_current_metrics=report.degraded_metrics,
            targeted_bias_attributes=biased_sens_cols
        )
        report.add_step(canary_step)
        report.canary_gate_passed = canary_passed

        # Compute healed metrics
        healed_metrics = self.retrainer.evaluate_model(
            challenger.model,
            challenger.preprocessor,
            eval_X,
            eval_y,
            group_thresholds=challenger.group_thresholds,
            sensitive_df=eval_df
        )
        for s_col, f_data in ch_fairness.items():
            healed_metrics[f"di_{s_col}"] = f_data["disparate_impact"]
            healed_metrics[f"dpd_{s_col}"] = f_data["demographic_parity_diff"]
        report.healed_metrics = healed_metrics

        # Compare Before vs After
        for metric_name in ["accuracy", "f1", "roc_auc"]:
            b_val = report.degraded_metrics.get(metric_name, 0.0)
            a_val = report.healed_metrics.get(metric_name, 0.0)
            pct = ((a_val - b_val) / max(1e-6, b_val)) * 100.0 if b_val > 0 else 0.0
            report.comparisons.append(
                MetricComparison(
                    metric_name=metric_name.upper(),
                    before_value=b_val,
                    after_value=a_val,
                    relative_change_pct=pct,
                    improved=(a_val >= b_val - 0.01),
                )
            )

        for sens_col in self.config.bias.sensitive_features:
            key = f"di_{sens_col}"
            if key in report.degraded_metrics and key in report.healed_metrics:
                b_val = report.degraded_metrics[key]
                a_val = report.healed_metrics[key]
                pct = ((a_val - b_val) / max(1e-6, b_val)) * 100.0 if b_val > 0 else 0.0
                report.comparisons.append(
                    MetricComparison(
                        metric_name=f"Disparate Impact ({sens_col})",
                        before_value=b_val,
                        after_value=a_val,
                        relative_change_pct=pct,
                        improved=(a_val >= b_val),
                    )
                )

        # Execute Promotion or Rollback
        if canary_passed:
            self.registry.promote_to_champion(challenger.version, notes="Autonomous self-healing promotion passed.")
            report.status = "HEALED_AND_PROMOTED"
            report.promoted_version_after = challenger.version
            report.executive_summary = (
                f"Successfully diagnosed {len(report.findings)} issues. Executed {len(report.remediation_plan)} "
                f"autonomous corrective actions. Promoted healed model '{challenger.version}' as new Champion "
                f"with F1={healed_metrics.get('f1', 0):.3f} and restored fairness."
            )
        else:
            self.registry.reject_candidate(challenger.version, reason=reason)
            report.status = "REVERTED"
            report.executive_summary = f"Remediation canary checks did not pass. Preserved previous champion. Reason: {reason}"

        report.completed_at = datetime.now().isoformat()
        self.incident_history.append(report)
        return report
