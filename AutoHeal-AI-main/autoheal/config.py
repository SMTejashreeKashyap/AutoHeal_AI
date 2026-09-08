"""Configuration settings and threshold parameters for AutoHeal-AI."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class DataErrorConfig:
    """Thresholds for detecting data errors and corruption."""
    # Flag column as having excessive missing data if ratio exceeds this:
    missing_rate_alert_threshold: float = 0.05
    missing_rate_critical_threshold: float = 0.25
    
    # Outlier detection configuration
    outlier_method: str = "iqr"  # 'iqr', 'zscore', or 'isolation_forest'
    outlier_iqr_multiplier: float = 2.5
    outlier_zscore_threshold: float = 3.5
    outlier_contamination: float = 0.05
    outlier_rate_alert_threshold: float = 0.03
    
    # Confident learning / label noise detection threshold
    # Flags sample as noisy if model confidence on counter-class exceeds this margin
    label_noise_confidence_threshold: float = 0.65
    label_noise_rate_alert_threshold: float = 0.04


@dataclass
class BiasConfig:
    """Thresholds and configurations for fairness and algorithmic bias auditing."""
    # List of sensitive protected columns (e.g., 'gender', 'age_group')
    sensitive_features: List[str] = field(default_factory=lambda: ["gender", "age_group"])
    
    # Mapping of sensitive attribute -> privileged group value
    privileged_groups: Dict[str, Any] = field(
        default_factory=lambda: {
            "gender": "Male",
            "age_group": "Middle-Aged"
        }
    )
    
    # Disparate Impact (80% / four-fifths rule threshold: [0.80, 1.25])
    disparate_impact_min: float = 0.80
    disparate_impact_max: float = 1.25
    
    # Max allowed difference in favorable outcome probability |P(Y=1|unprivileged) - P(Y=1|privileged)|
    demographic_parity_max_diff: float = 0.10
    
    # Max allowed difference in True Positive Rate (TPR) or False Positive Rate (FPR)
    equalized_odds_max_diff: float = 0.12


@dataclass
class DriftConfig:
    """Thresholds for covariate and concept drift detection."""
    # Kolmogorov-Smirnov test p-value threshold for continuous features
    # If p-value < threshold, null hypothesis (same distribution) is rejected -> Drift detected!
    ks_p_value_threshold: float = 0.05
    
    # Population Stability Index (PSI)
    # PSI < 0.10: No significant change
    # 0.10 <= PSI < 0.20: Moderate drift (warning)
    # PSI >= 0.20: Significant drift (action required)
    psi_warning_threshold: float = 0.10
    psi_critical_threshold: float = 0.20
    
    # Wasserstein / Earth Mover's Distance normalized threshold
    wasserstein_relative_threshold: float = 0.15
    
    # Categorical Chi-Square test p-value threshold
    chi2_p_value_threshold: float = 0.05
    
    # Page-Hinkley test parameter for concept drift / rolling performance degradation
    page_hinkley_threshold: float = 25.0
    page_hinkley_delta: float = 0.005


@dataclass
class RemediationConfig:
    """Strategies for autonomous self-healing and data remediation."""
    # Imputation: 'knn', 'median_mode', or 'iterative'
    imputation_strategy: str = "knn"
    knn_neighbors: int = 5
    
    # Outlier handling: 'clip' (winsorization) or 'drop'
    outlier_strategy: str = "clip"
    winsorize_lower_quantile: float = 0.01
    winsorize_upper_quantile: float = 0.99
    
    # Label noise correction: 'prune' (drop suspected flipped samples) or 'relabel' (flip to confident class)
    label_cleaning_strategy: str = "prune"
    
    # Debiasing: 'reweight' (pre-processing sample weights), 'threshold_tuning' (post-processing), or 'both'
    debiasing_strategy: str = "both"
    
    # Model candidate types to train during self-healing
    retrain_model_types: List[str] = field(default_factory=lambda: ["gradient_boosting", "random_forest", "logistic_regression"])


@dataclass
class PromotionConfig:
    """Canary gates and promotion thresholds for the champion-challenger pipeline."""
    # Allowable predictive performance degradation tolerance (e.g., F1 must not drop by more than 2% while fixing bias)
    f1_tolerance: float = 0.02
    min_accuracy_threshold: float = 0.70
    
    # Promotion requires meeting all bias fairness gates
    enforce_fairness_in_promotion: bool = True
    
    # Canary evaluation split ratio
    canary_test_size: float = 0.25


@dataclass
class AutoHealConfig:
    """Unified master configuration for the AutoHeal system."""
    data_error: DataErrorConfig = field(default_factory=DataErrorConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    remediation: RemediationConfig = field(default_factory=RemediationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    random_state: int = 42
