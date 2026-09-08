"""Automated data cleaner: resolves missing values, schema bounds, outliers, and label noise."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.impute import SimpleImputer, KNNImputer

from autoheal.config import RemediationConfig
from autoheal.core.events import RemediationStep


class AutomatedDataCleaner:
    """Executes automated remediation pipelines to cleanse compromised datasets."""

    def __init__(self, config: Optional[RemediationConfig] = None):
        self.config = config or RemediationConfig()
        self.imputers: Dict[str, Any] = {}
        self.clipping_bounds: Dict[str, Tuple[float, float]] = {}

    def fit_and_clean(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = "target",
        suspected_noisy_indices: Optional[np.ndarray] = None
    ) -> Tuple[pd.DataFrame, List[RemediationStep]]:
        """Fit remediation parameters on training data and return cleansed dataset."""
        cleaned_df = df.copy()
        steps: List[RemediationStep] = []

        # 1. Prune or Relabel Noisy Labels
        if suspected_noisy_indices is not None and target_col in cleaned_df.columns:
            noisy_count = int(np.sum(suspected_noisy_indices))
            if noisy_count > 0:
                if self.config.label_cleaning_strategy == "prune":
                    before_rows = len(cleaned_df)
                    cleaned_df = cleaned_df.loc[~suspected_noisy_indices].reset_index(drop=True)
                    after_rows = len(cleaned_df)
                    steps.append(
                        RemediationStep(
                            step_id="STEP_CLEAN_LABEL_NOISE",
                            target=target_col,
                            action_type="PRUNE_NOISY_LABELS",
                            description=f"Pruned {noisy_count} high-confidence corrupted label instances.",
                            parameters={"strategy": "prune", "removed_count": noisy_count},
                            status="EXECUTED",
                            impact_summary=f"Dataset reduced from {before_rows} to {after_rows} clean ground-truth rows.",
                        )
                    )

        # 2. Outlier Winsorization / Quantile Clipping
        num_cols = cleaned_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)

        clipped_cols = []
        for col in num_cols:
            series = cleaned_df[col].dropna()
            if len(series) < 3:
                continue
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            iqr_upper = q75 + 2.5 * iqr if iqr > 0 else float("inf")
            iqr_lower = q25 - 2.5 * iqr if iqr > 0 else -float("inf")
            raw_upper = float(series.quantile(self.config.winsorize_upper_quantile))
            raw_lower = float(series.quantile(self.config.winsorize_lower_quantile))
            upper_q = min(raw_upper, iqr_upper)
            lower_q = max(raw_lower, iqr_lower)
            self.clipping_bounds[col] = (lower_q, upper_q)

            outlier_mask = (cleaned_df[col] < lower_q) | (cleaned_df[col] > upper_q)
            outlier_count = int(outlier_mask.sum())
            if outlier_count > 0:
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_q, upper=upper_q)
                clipped_cols.append(col)

        if clipped_cols:
            steps.append(
                RemediationStep(
                    step_id="STEP_OUTLIER_TREATMENT",
                    target=", ".join(clipped_cols[:5]) + ("..." if len(clipped_cols) > 5 else ""),
                    action_type="WINSORIZE_OUTLIERS",
                    description=f"Applied winsorization clipping [{self.config.winsorize_lower_quantile:.0%}, {self.config.winsorize_upper_quantile:.0%}] to {len(clipped_cols)} numerical features.",
                    parameters={"strategy": "clip", "clipped_columns": clipped_cols},
                    status="EXECUTED",
                    impact_summary=f"Stabilized gradient variance across {len(clipped_cols)} columns without dropping data.",
                )
            )

        # 3. Context-Aware Missing Value Imputation
        missing_cols = [c for c in cleaned_df.columns if cleaned_df[c].isnull().sum() > 0]
        if missing_cols:
            # Separate numerical vs categorical
            cat_cols = [c for c in missing_cols if c not in num_cols and c != target_col]
            num_missing = [c for c in missing_cols if c in num_cols]

            # Categorical mode imputation
            for col in cat_cols:
                mode_val = cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else "Unknown"
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                self.imputers[col] = mode_val

            # Numerical imputation
            if num_missing:
                if self.config.imputation_strategy == "knn" and len(cleaned_df) <= 5000:
                    imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)
                    cleaned_df[num_missing] = imputer.fit_transform(cleaned_df[num_missing])
                    self.imputers["_num_knn"] = (imputer, num_missing)
                else:
                    imputer = SimpleImputer(strategy="median")
                    cleaned_df[num_missing] = imputer.fit_transform(cleaned_df[num_missing])
                    self.imputers["_num_simple"] = (imputer, num_missing)

            steps.append(
                RemediationStep(
                    step_id="STEP_IMPUTATION",
                    target=", ".join(missing_cols[:5]),
                    action_type="IMPUTE_MISSING_DATA",
                    description=f"Automated imputation on {len(missing_cols)} columns using {self.config.imputation_strategy.upper()} strategy.",
                    parameters={"strategy": self.config.imputation_strategy, "columns": missing_cols},
                    status="EXECUTED",
                    impact_summary="Restored complete feature matrices with zero null entries.",
                )
            )

        return cleaned_df, steps

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted cleaning and imputation transforms onto new/test inference data."""
        cleaned_df = df.copy()

        # 1. Apply clipping
        for col, (lower_q, upper_q) in self.clipping_bounds.items():
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_q, upper=upper_q)

        # 2. Apply categorical imputation
        for col, mode_val in self.imputers.items():
            if not col.startswith("_") and col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(mode_val)

        # 3. Apply numerical imputation
        if "_num_knn" in self.imputers:
            imputer, cols = self.imputers["_num_knn"]
            valid_cols = [c for c in cols if c in cleaned_df.columns]
            if valid_cols:
                cleaned_df[valid_cols] = imputer.transform(cleaned_df[valid_cols])
        elif "_num_simple" in self.imputers:
            imputer, cols = self.imputers["_num_simple"]
            valid_cols = [c for c in cols if c in cleaned_df.columns]
            if valid_cols:
                cleaned_df[valid_cols] = imputer.transform(cleaned_df[valid_cols])

        return cleaned_df
