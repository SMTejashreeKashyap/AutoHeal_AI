"""Unit tests for Data Error, Bias, and Drift detectors."""

import unittest
import numpy as np
import pandas as pd

from autoheal.config import DataErrorConfig, BiasConfig, DriftConfig
from autoheal.detectors.data_errors import DataErrorDetector
from autoheal.detectors.bias import BiasDetector
from autoheal.detectors.drift import DriftDetector, calculate_psi
from autoheal.core.events import IssueType


class TestDetectors(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Synthetic baseline dataframe
        self.df = pd.DataFrame({
            "income": np.random.normal(60000, 15000, 500),
            "debt_to_income": np.random.uniform(0.1, 0.5, 500),
            "credit_score": np.random.normal(700, 50, 500),
            "gender": np.random.choice(["Male", "Female"], 500),
            "target": np.random.choice([0, 1], 500),
        })

    def test_missing_values_detection(self):
        detector = DataErrorDetector(DataErrorConfig(missing_rate_alert_threshold=0.05))
        corrupted_df = self.df.copy()
        # Inject 15% missing into debt_to_income
        corrupted_df.loc[0:75, "debt_to_income"] = np.nan

        findings, stats = detector.detect_missing_values(corrupted_df)
        self.assertTrue(any(f.affected_target == "debt_to_income" for f in findings))
        self.assertGreaterEqual(stats["debt_to_income"], 0.15)

    def test_outlier_detection(self):
        detector = DataErrorDetector(DataErrorConfig(outlier_method="iqr"))
        corrupted_df = self.df.copy()
        # Inject extreme outliers
        corrupted_df.loc[0:25, "income"] = 5_000_000

        findings, masks = detector.detect_outliers(corrupted_df, numerical_columns=["income"])
        self.assertTrue(any(f.issue_type == IssueType.OUTLIERS for f in findings))
        self.assertGreater(masks["income"].sum(), 20)

    def test_label_noise_detection(self):
        detector = DataErrorDetector()
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        # High confidence in opposite class
        probs = np.array([
            [0.1, 0.9],  # true 0, predicted 1 (noise!)
            [0.9, 0.1],  # correct
            [0.2, 0.8],  # true 0, predicted 1 (noise!)
            [0.1, 0.9],  # correct
            [0.85, 0.15], # true 1, predicted 0 (noise!)
            [0.1, 0.9],  # correct
            [0.8, 0.2],  # correct
            [0.05, 0.95], # correct
        ])
        findings, noise_mask = detector.detect_label_noise(y_true, probs)
        self.assertTrue(noise_mask[0])
        self.assertTrue(noise_mask[2])
        self.assertTrue(noise_mask[4])
        self.assertFalse(noise_mask[1])

    def test_bias_detection(self):
        detector = BiasDetector(BiasConfig())
        biased_df = pd.DataFrame({
            "gender": ["Male"] * 200 + ["Female"] * 200,
        })
        # Favorable outcome rate: Male 80%, Female 20%
        y_true = np.array([1] * 200 + [1] * 200)
        y_pred = np.array([1] * 160 + [0] * 40 + [1] * 40 + [0] * 160)

        findings, metrics = detector.detect_bias(biased_df, y_true, y_pred)
        self.assertTrue(any(f.issue_type == IssueType.DISPARATE_IMPACT for f in findings))
        self.assertLess(metrics["gender"]["disparate_impact"], 0.50)

    def test_drift_detection(self):
        detector = DriftDetector(DriftConfig(ks_p_value_threshold=0.05, psi_warning_threshold=0.10))
        ref_df = pd.DataFrame({
            "credit_score": np.random.normal(720, 40, 500)
        })
        # Shifted distribution
        curr_df = pd.DataFrame({
            "credit_score": np.random.normal(640, 50, 500)
        })

        findings, metrics = detector.detect_feature_drift(ref_df, curr_df)
        self.assertTrue(any(f.issue_type == IssueType.COVARIATE_DRIFT for f in findings))
        self.assertGreater(metrics["credit_score"]["psi"], 0.10)
        self.assertLess(metrics["credit_score"]["ks_p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
