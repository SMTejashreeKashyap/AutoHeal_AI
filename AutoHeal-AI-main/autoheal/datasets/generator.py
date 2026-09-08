"""Synthetic and realistic benchmark generator with controllable data errors, bias, and drift injections."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple


class LoanDatasetGenerator:
    """Generates synthetic Credit & Loan underwriting datasets with controllable anomaly injections."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_clean_dataset(self, n_samples: int = 2500) -> pd.DataFrame:
        """Create clean baseline dataset representing unbiased loan applications."""
        rng = self.rng

        # Sensitive demographic features
        gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48])
        age_group = rng.choice(["Young", "Middle-Aged", "Senior"], size=n_samples, p=[0.30, 0.45, 0.25])

        # Financial continuous features
        income = rng.lognormal(mean=11.0, sigma=0.55, size=n_samples)  # ~$60,000 median
        income = np.clip(income, 18000, 350000)

        credit_score = rng.normal(loc=680, scale=60, size=n_samples)
        credit_score = np.clip(credit_score, 450, 850)

        debt_to_income = rng.beta(a=2.2, b=5.0, size=n_samples) * 0.75  # ~0.25 mean
        debt_to_income = np.clip(debt_to_income, 0.05, 0.65)

        loan_amount = rng.normal(loc=25000, scale=12000, size=n_samples)
        loan_amount = np.clip(loan_amount, 3000, 100000)

        employment_length = rng.exponential(scale=6.0, size=n_samples)
        employment_length = np.clip(employment_length, 0.5, 35.0)

        savings_balance = rng.lognormal(mean=9.5, sigma=1.0, size=n_samples)
        savings_balance = np.clip(savings_balance, 500, 200000)

        # Unbiased ground-truth approval logic (Log-Odds based strictly on financial merit)
        z = (
            -3.0
            + 0.000035 * income
            + 0.012 * (credit_score - 600)
            - 4.2 * debt_to_income
            - 0.000015 * loan_amount
            + 0.08 * employment_length
        )
        prob = 1.0 / (1.0 + np.exp(-z))
        target = rng.binomial(n=1, p=prob, size=n_samples)

        df = pd.DataFrame({
            "income": np.round(income, 2),
            "credit_score": np.round(credit_score, 1),
            "debt_to_income": np.round(debt_to_income, 4),
            "loan_amount": np.round(loan_amount, 2),
            "employment_length": np.round(employment_length, 1),
            "savings_balance": np.round(savings_balance, 2),
            "gender": gender,
            "age_group": age_group,
            "target": target,
        })
        return df

    def inject_data_errors(
        self,
        df: pd.DataFrame,
        missing_rate: float = 0.12,
        outlier_rate: float = 0.04,
        label_noise_rate: float = 0.08
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Introduce missing values, out-of-bounds outliers, and corrupted labels into a dataset."""
        corrupted = df.copy()
        n = len(corrupted)
        rng = np.random.RandomState(self.random_state + 101)

        # 1. Missing values injection in debt_to_income, savings_balance, and employment_length
        missing_cols = ["debt_to_income", "savings_balance", "employment_length"]
        for col in missing_cols:
            mask = rng.rand(n) < missing_rate
            corrupted.loc[mask, col] = np.nan

        # 2. Outliers injection (extreme impossible values)
        outlier_cols = ["debt_to_income", "income", "credit_score"]
        for col in outlier_cols:
            outlier_mask = rng.rand(n) < outlier_rate
            if col == "debt_to_income":
                corrupted.loc[outlier_mask, col] = rng.uniform(8.0, 45.0, size=outlier_mask.sum())
            elif col == "income":
                corrupted.loc[outlier_mask, col] = rng.uniform(5_000_000, 25_000_000, size=outlier_mask.sum())
            elif col == "credit_score":
                corrupted.loc[outlier_mask, col] = rng.uniform(950, 1500, size=outlier_mask.sum())

        # 3. Label noise injection (ground truth flips)
        label_noise_mask = rng.rand(n) < label_noise_rate
        corrupted.loc[label_noise_mask, "target"] = 1 - corrupted.loc[label_noise_mask, "target"]

        metadata = {
            "missing_rate_injected": missing_rate,
            "outlier_rate_injected": outlier_rate,
            "label_noise_rate_injected": label_noise_rate,
            "label_noise_count": int(label_noise_mask.sum()),
        }
        return corrupted, metadata

    def inject_bias(
        self,
        df: pd.DataFrame,
        sensitive_col: str = "gender",
        unprivileged_val: str = "Female",
        disparity_penalty: float = 0.50
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Artificially suppress approvals for the unprivileged group to simulate historical or algorithmic bias."""
        biased = df.copy()
        rng = np.random.RandomState(self.random_state + 202)

        # Identify unprivileged records with positive target
        mask = (biased[sensitive_col] == unprivileged_val) & (biased["target"] == 1)
        candidates = biased[mask].index.values

        # Flip a fraction of positive approvals to rejections
        num_to_flip = int(len(candidates) * disparity_penalty)
        if num_to_flip > 0:
            flipped_indices = rng.choice(candidates, size=num_to_flip, replace=False)
            biased.loc[flipped_indices, "target"] = 0

        metadata = {
            "sensitive_col": sensitive_col,
            "unprivileged_val": unprivileged_val,
            "disparity_penalty": disparity_penalty,
            "suppressed_approvals_count": num_to_flip,
        }
        return biased, metadata

    def inject_drift(
        self,
        df: pd.DataFrame,
        drift_magnitude: float = 0.35,
        drift_type: str = "covariate"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Simulate economic shock / covariate drift or concept drift."""
        drifted = df.copy()
        rng = np.random.RandomState(self.random_state + 303)

        if drift_type == "covariate":
            # Macroeconomic shock: DTI increases significantly, credit scores drop, loan amounts inflate
            drifted["debt_to_income"] = drifted["debt_to_income"] * (1.0 + drift_magnitude)
            drifted["credit_score"] = np.clip(drifted["credit_score"] - (drift_magnitude * 70), 300, 850)
            drifted["loan_amount"] = drifted["loan_amount"] * (1.0 + drift_magnitude * 0.8)
        elif drift_type == "concept":
            # Approval rules shift: credit score requirement becomes much harsher, DTI penalized 3x
            z = (
                -8.0
                + 0.000025 * drifted["income"]
                + 0.018 * (drifted["credit_score"] - 670)
                - 8.5 * drifted["debt_to_income"]
            )
            prob = 1.0 / (1.0 + np.exp(-z))
            drifted["target"] = rng.binomial(n=1, p=prob, size=len(drifted))

        metadata = {
            "drift_type": drift_type,
            "drift_magnitude": drift_magnitude,
        }
        return drifted, metadata
