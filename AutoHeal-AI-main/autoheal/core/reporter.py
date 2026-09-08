"""Automated report generation for incident diagnosis and remediation audits."""

from typing import Dict, Any, List
from autoheal.core.events import SelfHealingReport, FindingSeverity


class SelfHealingReporter:
    """Generates structured incident reports in Markdown, JSON, and HTML."""

    @staticmethod
    def generate_markdown(report: SelfHealingReport) -> str:
        """Render comprehensive markdown report suitable for display or export."""
        lines = []
        lines.append(f"# AutoHeal-AI Incident & Self-Healing Audit Report")
        lines.append(f"**Cycle ID**: `{report.cycle_id}` | **Status**: `{report.status}`")
        lines.append(f"**Triggered**: {report.triggered_at} | **Completed**: {report.completed_at or 'Pending'}")
        lines.append("")

        # Executive Summary
        lines.append("## 1. Executive Summary")
        lines.append(report.executive_summary or "Autonomous self-healing cycle executed successfully.")
        lines.append("")

        # Canary Gate Status
        status_icon = "PASS" if report.canary_gate_passed else "FAIL"
        lines.append(f"> **Canary Safety Gate Status**: `{status_icon}`")
        lines.append(f"> **Champion Before**: `{report.champion_version_before or 'None'}` -> **Champion After**: `{report.promoted_version_after or report.champion_version_before or 'Unchanged'}`")
        lines.append("")

        # Diagnostic Findings
        lines.append("## 2. Diagnostic Findings (Issues Spotted)")
        if not report.findings:
            lines.append("*No anomalies or degradation detected.*")
        else:
            lines.append("| Severity | Issue Type | Target | Metric | Observed | Threshold | Description |")
            lines.append("|---|---|---|---|---|---|---|")
            for f in report.findings:
                lines.append(
                    f"| `{f.severity.value}` | `{f.issue_type.value}` | `{f.affected_target}` | `{f.metric_name}` | {f.metric_value:.4f} | {f.threshold_value:.4f} | {f.description} |"
                )
        lines.append("")

        # Remediation Steps Taken
        lines.append("## 3. Autonomous Remediation Actions Taken")
        if not report.remediation_plan:
            lines.append("*No corrective actions required.*")
        else:
            lines.append("| Step | Action Type | Target | Status | Impact / Result |")
            lines.append("|---|---|---|---|---|")
            for step in report.remediation_plan:
                lines.append(
                    f"| `{step.step_id}` | `{step.action_type}` | `{step.target}` | `{step.status}` | {step.impact_summary or step.description} |"
                )
        lines.append("")

        # Before vs After Impact Analysis
        lines.append("## 4. Before vs After Verification")
        if not report.comparisons:
            lines.append("*No metric comparisons available.*")
        else:
            lines.append("| Metric | Before (Degraded) | After (Healed) | Delta % | Status |")
            lines.append("|---|---|---|---|---|")
            for c in report.comparisons:
                verdict = "IMPROVED" if c.improved else "NEUTRAL/SLIGHT"
                lines.append(
                    f"| **{c.metric_name}** | {c.before_value:.4f} | {c.after_value:.4f} | {c.relative_change_pct:+.2f}% | `{verdict}` |"
                )
        lines.append("")

        lines.append("---")
        lines.append("*Report automatically compiled by AutoHeal-AI Autonomous Telemetry Engine.*")
        return "\n".join(lines)
