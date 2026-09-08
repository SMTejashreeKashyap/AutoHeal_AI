"""Integration tests for the full Autonomous Self-Healing Pipeline."""

import unittest
import numpy as np

from autoheal.config import AutoHealConfig
from autoheal.datasets.generator import LoanDatasetGenerator
from autoheal.pipeline.orchestrator import SelfHealingOrchestrator
from autoheal.core.events import IssueType


class TestSelfHealingOrchestrator(unittest.TestCase):

    def test_full_autonomous_self_healing_cycle(self):
        generator = LoanDatasetGenerator(random_state=42)
        clean_df = generator.generate_clean_dataset(n_samples=600)

        # 1. Initialize orchestrator and bootstrap baseline champion
        orchestrator = SelfHealingOrchestrator()
        champion = orchestrator.bootstrap_baseline(clean_df, target_col="target")
        self.assertIsNotNone(champion)
        self.assertEqual(champion.status, "CHAMPION")

        # 2. Simulate streaming chaos (inject data errors, bias, and drift)
        corrupted_df, _ = generator.inject_data_errors(clean_df, missing_rate=0.10, outlier_rate=0.03, label_noise_rate=0.06)
        biased_df, _ = generator.inject_bias(corrupted_df, sensitive_col="gender", disparity_penalty=0.40)
        drifted_df, _ = generator.inject_drift(biased_df, drift_magnitude=0.30, drift_type="covariate")

        # 3. Test Diagnostic Engine
        findings = orchestrator.diagnose(drifted_df, target_col="target")
        self.assertGreater(len(findings), 0)
        issue_types = [f.issue_type for f in findings]
        self.assertIn(IssueType.MISSING_VALUES, issue_types)

        # 4. Trigger Autonomous Self-Healing
        report = orchestrator.heal(drifted_df, target_col="target")

        # 5. Assertions on the Self-Healing outcome
        self.assertIn(report.status, ["HEALED_AND_PROMOTED", "REVERTED"])
        self.assertGreater(len(report.remediation_plan), 0)
        self.assertTrue(report.canary_gate_passed)
        self.assertIsNotNone(report.promoted_version_after)
        
        # Verify new champion is promoted in registry
        new_champ = orchestrator.registry.get_champion()
        self.assertEqual(new_champ.version, report.promoted_version_after)


if __name__ == "__main__":
    unittest.main()
