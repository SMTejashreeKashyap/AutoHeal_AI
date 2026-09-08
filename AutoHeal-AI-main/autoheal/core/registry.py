"""Model Registry managing Champion and Challenger models, versions, and metadata."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import copy


@dataclass
class ModelArtifact:
    """Represents a trained model version with preprocessing pipeline and metadata."""
    version: str
    model: Any
    model_type: str
    feature_names: List[str]
    target_name: str
    preprocessor: Optional[Any] = None
    group_thresholds: Dict[str, Dict[Any, float]] = field(default_factory=dict)
    default_threshold: float = 0.50
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "CANDIDATE"  # CHAMPION, CANDIDATE, ARCHIVED, REJECTED
    training_data_signature: Dict[str, Any] = field(default_factory=dict)
    remediation_notes: str = ""

    def predict_proba(self, X: Any) -> Any:
        """Predict probability of positive class."""
        X_trans = X
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_trans)
        elif hasattr(self.model, "decision_function"):
            import numpy as np
            scores = self.model.decision_function(X_trans)
            # Sigmoid
            prob = 1.0 / (1.0 + np.exp(-scores))
            return np.vstack([1.0 - prob, prob]).T
        else:
            raise ValueError("Underlying model does not support probability estimation.")

    def predict_with_fair_thresholds(self, X: Any, sensitive_df: Optional[Any] = None) -> Any:
        """Predict binary class using group-specific fairness thresholds if configured."""
        import numpy as np
        probs = self.predict_proba(X)[:, 1]
        
        # If no group thresholds configured or no sensitive metadata provided, use default
        if not self.group_thresholds or sensitive_df is None:
            return (probs >= self.default_threshold).astype(int)
        
        predictions = np.zeros(len(probs), dtype=int)
        # Apply group-specific threshold for each record (taking most favorable if multiple apply)
        for i in range(len(probs)):
            applied_thresholds = []
            for sens_col, thresh_map in self.group_thresholds.items():
                if sens_col in sensitive_df.columns:
                    val = sensitive_df.iloc[i][sens_col]
                    if val in thresh_map:
                        applied_thresholds.append(thresh_map[val])
            record_thresh = min(applied_thresholds) if applied_thresholds else self.default_threshold
            predictions[i] = int(probs[i] >= record_thresh)
            
        return predictions


class ModelRegistry:
    """Manages the lifecycle, versioning, champion promotion, and history of models."""

    def __init__(self):
        self._models: Dict[str, ModelArtifact] = {}
        self._champion_version: Optional[str] = None
        self._history: List[str] = []

    def register(self, artifact: ModelArtifact) -> str:
        """Register a new candidate model artifact."""
        self._models[artifact.version] = artifact
        self._history.append(artifact.version)
        if self._champion_version is None:
            artifact.status = "CHAMPION"
            self._champion_version = artifact.version
        return artifact.version

    def get_champion(self) -> Optional[ModelArtifact]:
        """Retrieve current champion model."""
        if self._champion_version and self._champion_version in self._models:
            return self._models[self._champion_version]
        return None

    def get_model(self, version: str) -> Optional[ModelArtifact]:
        """Retrieve model artifact by version."""
        return self._models.get(version)

    def promote_to_champion(self, version: str, notes: str = "") -> bool:
        """Promote a challenger candidate to champion."""
        if version not in self._models:
            return False
        
        # Archive existing champion
        if self._champion_version and self._champion_version in self._models:
            self._models[self._champion_version].status = "ARCHIVED"
            
        new_champ = self._models[version]
        new_champ.status = "CHAMPION"
        if notes:
            new_champ.remediation_notes = notes
        self._champion_version = version
        return True

    def reject_candidate(self, version: str, reason: str = "") -> None:
        """Reject a candidate that failed canary checks."""
        if version in self._models:
            self._models[version].status = "REJECTED"
            self._models[version].remediation_notes = f"Rejected: {reason}"

    def list_models(self) -> List[Dict[str, Any]]:
        """List summary of all models in the registry."""
        return [
            {
                "version": m.version,
                "model_type": m.model_type,
                "status": m.status,
                "metrics": m.metrics,
                "created_at": m.created_at,
                "remediation_notes": m.remediation_notes,
            }
            for m in self._models.values()
        ]
