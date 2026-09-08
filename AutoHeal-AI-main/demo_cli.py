"""Interactive CLI Demonstration of AutoHeal-AI: Teaching AI to Fix Itself.

Perfect for quick terminal demos, academic presentations, and verification.
Run:
    python demo_cli.py
"""

import sys
import time
from autoheal.config import AutoHealConfig
from autoheal.datasets.generator import LoanDatasetGenerator
from autoheal.pipeline.orchestrator import SelfHealingOrchestrator
from autoheal.core.reporter import SelfHealingReporter


def print_banner():
    print("=" * 75)
    print("       AUTOHEAL-AI: TEACHING AI TO FIX ITSELF")
    print("  Autonomous Data Debugging, Bias Mitigation & Model Drift Remediation")
    print("=" * 75)


def run_demo():
    print_banner()

    # Step 1: Generate clean dataset & train baseline champion
    print("\n[STEP 1/5] Bootstrapping Pristine Baseline Model...")
    generator = LoanDatasetGenerator(random_state=42)
    clean_df = generator.generate_clean_dataset(n_samples=1000)
    print(f"  > Generated {len(clean_df)} loan records across 8 financial & demographic features.")
    
    orchestrator = SelfHealingOrchestrator()
    champ = orchestrator.bootstrap_baseline(clean_df, target_col="target")
    f1 = champ.metrics.get("f1", 0.0)
    acc = champ.metrics.get("accuracy", 0.0)
    print(f"  > Baseline Champion '{champ.version}' active: Accuracy={acc:.1%}, F1-Score={f1:.3f}")

    # Step 2: Inject Chaos
    print("\n[STEP 2/5] Simulating Real-World Production Failures (Chaos Injection)...")
    corrupted_df, err_meta = generator.inject_data_errors(
        clean_df, missing_rate=0.12, outlier_rate=0.04, label_noise_rate=0.08
    )
    biased_df, bias_meta = generator.inject_bias(
        corrupted_df, sensitive_col="gender", unprivileged_val="Female", disparity_penalty=0.45
    )
    drifted_df, drift_meta = generator.inject_drift(
        biased_df, drift_magnitude=0.35, drift_type="covariate"
    )
    print(f"  > Injected 12% missing fields, 4% extreme outliers, and 8% corrupted ground-truth labels.")
    print(f"  > Injected 45% demographic approval penalty on Female applicants.")
    print(f"  > Injected 35% macroeconomic covariate drift (elevated DTI, dropped credit scores).")

    # Step 3: Run Diagnostic Engine (Spotting)
    print("\n[STEP 3/5] Running Autonomous Diagnostic Telemetry Scan...")
    findings = orchestrator.diagnose(drifted_df, target_col="target")
    print(f"  > Diagnostic Engine Spotted {len(findings)} Total Anomalies:")
    for i, f in enumerate(findings, 1):
        print(f"    [{i:02d}] {f.severity.value:<8} | {f.issue_type.value:<18} | Target: {f.affected_target:<18} | {f.description[:60]}...")

    # Step 4: Execute Autonomous Self-Healing
    print("\n[STEP 4/5] Triggering Autonomous Self-Healing Feedback Loop...")
    print("  > 1. Formulating remediation action plan...")
    print("  > 2. Imputing nulls with KNN & winsorizing outliers...")
    print("  > 3. Pruning suspected noisy/flipped labels via confident learning...")
    print("  > 4. Calculating Kamiran-Calders fair sample reweighting...")
    print("  > 5. Calibrating group-specific fair decision thresholds...")
    print("  > 6. Retraining Gradient Boosting candidate model...")
    print("  > 7. Evaluating canary safety gates against current champion...")
    
    report = orchestrator.heal(drifted_df, target_col="target")
    print(f"\n  [SELF-HEALING STATUS]: {report.status}")
    print(f"  > Canary Gate Passed: {report.canary_gate_passed}")
    print(f"  > Promoted Champion: {report.promoted_version_after}")

    # Step 5: Before vs After Impact Verification
    print("\n[STEP 5/5] Before vs After Verification Metrics:")
    print("-" * 75)
    print(f"  {'METRIC':<28} | {'BEFORE (DEGRADED)':<18} | {'AFTER (HEALED)':<16} | {'DELTA':<8}")
    print("-" * 75)
    for c in report.comparisons:
        sign = "+" if c.relative_change_pct > 0 else ""
        print(f"  {c.metric_name:<28} | {c.before_value:<18.4f} | {c.after_value:<16.4f} | {sign}{c.relative_change_pct:.1f}%")
    print("-" * 75)

    print("\n[EXECUTIVE SUMMARY]:")
    print(f"  {report.executive_summary}\n")
    print("=" * 75)
    print("  To launch the interactive Web Dashboard: python app.py")
    print("  Access dashboard in your browser: http://127.0.0.1:8000")
    print("=" * 75)


if __name__ == "__main__":
    run_demo()
