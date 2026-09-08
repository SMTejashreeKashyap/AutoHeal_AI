"""Unit tests for remediation modules (cleaner, debiasing, retrainer)."""

import unittest
import numpy as np
import pandas as pd

from autoheal.core.registry import ModelRegistry
from autoheal.remediators.cleaner import AutomatedDataCleaner
from autoheal.remediators.debiasing import FairnessRemediator
from autoheal.remediators.retrainer import SelfHealingRetrainer


class TestRemediators(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            "income": [50000.0, 60000.0, np.nan, 75000.0, 10_000_000.0, 80000.0, 55000.0, 62000.0, 71000.0, 58000.0],
            "debt_to_income": [0.25, 0.30, 0.40, np.nan, 0.20, 0.35, 0.28, 0.31, 0.22, 0.29],
            "gender": ["Male", "Female", "Female", "Male", "Female", "Male", "Female", "Male", "Female", "Male"],
            "target": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        })

    def test_cleaner_imputation_and_clipping(self):
        cleaner = AutomatedDataCleaner()
        cleaned_df, steps = cleaner.fit_and_clean(self.df, target_col="target")

        # Verify no NaN remains
        self.assertEqual(cleaned_df["income"].isnull().sum(), 0)
        self.assertEqual(cleaned_df["debt_to_income"].isnull().sum(), 0)
        # Verify extreme outlier (10,000,000) was clipped
        self.assertLess(cleaned_df["income"].max(), 9_000_000.0)
        self.assertGreater(len(steps), 0)

    def test_cleaner_label_noise_pruning(self):
        cleaner = AutomatedDataCleaner()
        # Mark row index 0 as noisy
        noise_mask = np.zeros(len(self.df), dtype=bool)
        noise_mask[0] = True

        cleaned_df, steps = cleaner.fit_and_clean(
            self.df, target_col="target", suspected_noisy_indices=noise_mask
        )
        self.assertEqual(len(cleaned_df), len(self.df) - 1)
        self.assertTrue(any(s.action_type == "PRUNE_NOISY_LABELS" for s in steps))

    def test_fairness_reweighting(self):
        remediator = FairnessRemediator()
        df = pd.DataFrame({
            "gender": ["Male"] * 100 + ["Female"] * 100,
            "target": [1] * 80 + [0] * 20 + [1] * 20 + [0] * 80,
        })
        weights, step = remediator.compute_fair_sample_weights(df, df["target"].values, "gender", "Male")
        self.assertEqual(len(weights), len(df))
        # Unprivileged positive cases should get boosted weights
        female_pos_idx = (df["gender"] == "Female") & (df["target"] == 1)
        male_pos_idx = (df["gender"] == "Male") & (df["target"] == 1)
        self.assertGreater(weights[female_pos_idx][0], weights[male_pos_idx][0])

    def test_retrainer_candidate(self):
        registry = ModelRegistry()
        retrainer = SelfHealingRetrainer(registry)

        X_train = pd.DataFrame({
            "num1": np.random.normal(0, 1, 100),
            "num2": np.random.normal(10, 2, 100),
            "cat": np.random.choice(["A", "B"], 100),
        })
        y_train = np.random.choice([0, 1], 100)

        X_val = pd.DataFrame({
            "num1": np.random.normal(0, 1, 30),
            "num2": np.random.normal(10, 2, 30),
            "cat": np.random.choice(["A", "B"], 30),
        })
        y_val = np.random.choice([0, 1], 30)

        artifact, metrics = retrainer.train_candidate(
            X_train, y_train, X_val, y_val, model_type="logistic_regression"
        )
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertEqual(artifact.status, "CANDIDATE")


if __name__ == "__main__":
    unittest.main()
