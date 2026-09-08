"""Data structures and models for incident detection, audit logging, and reporting."""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy scalars / bools to native Python types for JSON."""
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class FindingSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class IssueType(str, Enum):
    MISSING_VALUES = "MISSING_VALUES"
    OUTLIERS = "OUTLIERS"
    LABEL_NOISE = "LABEL_NOISE"
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    COVARIATE_DRIFT = "COVARIATE_DRIFT"
    CONCEPT_DRIFT = "CONCEPT_DRIFT"
    DISPARATE_IMPACT = "DISPARATE_IMPACT"
    DEMOGRAPHIC_PARITY = "DEMOGRAPHIC_PARITY"
    EQUALIZED_ODDS = "EQUALIZED_ODDS"


@dataclass
class DiagnosticFinding:
    """A specific issue spotted by one of the detectors."""
    issue_type: IssueType
    severity: FindingSeverity
    title: str
    description: str
    affected_target: str  # Column name, demographic slice, or model component
    metric_name: str
    metric_value: float
    threshold_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["issue_type"] = self.issue_type.value
        data["severity"] = self.severity.value
        return _sanitize(data)


@dataclass
class RemediationStep:
    """A corrective action formulated or executed by the auto-healing system."""
    step_id: str
    target: str
    action_type: str  # e.g., 'IMPUTE', 'CLIP_OUTLIERS', 'PRUNE_NOISY_LABELS', 'REWEIGHT_SAMPLES', 'RETRAIN_MODEL', 'TUNE_THRESHOLDS'
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: str = "PLANNED"  # PLANNED, EXECUTED, FAILED, SKIPPED
    impact_summary: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize(asdict(self))


@dataclass
class MetricComparison:
    """Before vs After comparison for any evaluation or fairness metric."""
    metric_name: str
    before_value: float
    after_value: float
    relative_change_pct: float
    improved: bool
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize(asdict(self))


@dataclass
class SelfHealingReport:
    """Comprehensive incident report documenting an entire self-healing cycle."""
    cycle_id: str
    triggered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    status: str = "IN_PROGRESS"  # IN_PROGRESS, HEALED_AND_PROMOTED, REVERTED, FAILED
    
    # Detected issues
    findings: List[DiagnosticFinding] = field(default_factory=list)
    
    # Formulated and executed steps
    remediation_plan: List[RemediationStep] = field(default_factory=list)
    
    # Before and after metrics
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    degraded_metrics: Dict[str, float] = field(default_factory=dict)
    healed_metrics: Dict[str, float] = field(default_factory=dict)
    comparisons: List[MetricComparison] = field(default_factory=list)
    
    # Model versions
    champion_version_before: Optional[str] = None
    promoted_version_after: Optional[str] = None
    
    # Canary evaluation summary
    canary_gate_passed: bool = False
    executive_summary: str = ""

    def add_finding(self, finding: DiagnosticFinding) -> None:
        self.findings.append(finding)

    def add_step(self, step: RemediationStep) -> None:
        self.remediation_plan.append(step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "triggered_at": self.triggered_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "remediation_plan": [s.to_dict() for s in self.remediation_plan],
            "baseline_metrics": self.baseline_metrics,
            "degraded_metrics": self.degraded_metrics,
            "healed_metrics": self.healed_metrics,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "champion_version_before": self.champion_version_before,
            "promoted_version_after": self.promoted_version_after,
            "canary_gate_passed": self.canary_gate_passed,
            "executive_summary": self.executive_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
