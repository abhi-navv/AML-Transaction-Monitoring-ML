"""Data Transfer Objects and Pydantic schemas for Explainable AI (XAI)."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from src.explainability.config import AUDIT_DISCLAIMER


class ExplanationMethod(str, Enum):
    """Supported explanation algorithms."""

    SHAP = "shap"
    LIME = "lime"


class DirectionalImpact(str, Enum):
    """Directional classification of feature contribution toward AML risk."""

    SUSPICIOUS_PUSHER = (
        "suspicious_pusher"  # Increases risk score toward Suspicious class
    )
    LEGITIMACY_ANCHOR = (
        "legitimacy_anchor"  # Decreases risk score toward Legitimate class
    )
    NEUTRAL = "neutral"  # Negligible or zero impact


class RawContribution(BaseModel):
    """Fine-grained feature attribution at the model input level."""

    model_config = ConfigDict(extra="ignore")

    feature_name: str = Field(
        ...,
        description="Exact model feature name (e.g. cat__Payment_type_Cross-border)",
    )
    feature_value: Optional[Union[float, int, str, bool]] = Field(
        default=None, description="Transformed or encoded value consumed by the model"
    )
    attribution: float = Field(
        ..., description="Signed contribution value (log-odds or probability delta)"
    )
    absolute_attribution: float = Field(
        ..., description="Absolute magnitude of attribution"
    )
    direction: DirectionalImpact = Field(
        ..., description="Whether feature increases or decreases risk"
    )
    rank: Optional[int] = Field(
        default=None, description="Importance rank (1 = highest magnitude)"
    )


class DomainContribution(BaseModel):
    """Aggregated feature attribution mapped to human-understandable transaction fields."""

    model_config = ConfigDict(extra="ignore")

    domain_feature: str = Field(
        ..., description="Original domain attribute name (e.g. Payment_type, Amount)"
    )
    domain_value: Optional[Any] = Field(
        default=None,
        description="Raw unencoded value from incoming transaction (e.g. 'Cross-border', 12500.0)",
    )
    net_attribution: float = Field(
        ...,
        description="Net signed attribution sum across all sub-components for this domain field",
    )
    absolute_attribution: float = Field(
        ...,
        description="Sum of absolute attributions across all sub-components for this domain field",
    )
    direction: DirectionalImpact = Field(
        ..., description="Net directional effect on AML risk"
    )
    sub_components: List[RawContribution] = Field(
        default_factory=list,
        description="List of fine-grained model features contributing to this domain field",
    )
    rank: Optional[int] = Field(
        default=None, description="Importance rank based on absolute attribution"
    )


class LocalExplanation(BaseModel):
    """Complete dual-granularity local explanation for a single transaction."""

    model_config = ConfigDict(extra="ignore")

    transaction_id: Optional[str] = Field(
        default=None, description="Transaction or account identifier"
    )
    method: ExplanationMethod = Field(
        ..., description="Explanation algorithm used (shap or lime)"
    )
    model_name: str = Field(..., description="Underlying ML model class name")
    prediction_class: str = Field(
        default="Suspicious", description="Target class explained"
    )
    predicted_risk_score: float = Field(
        ..., description="Predicted risk score / probability for target class"
    )
    base_value: float = Field(
        ..., description="Baseline reference / expected value E[f(x)]"
    )
    raw_contributions: List[RawContribution] = Field(
        default_factory=list, description="Top model-level feature attributions"
    )
    domain_contributions: List[DomainContribution] = Field(
        default_factory=list,
        description="Aggregated domain-level feature attributions for compliance review",
    )
    latency_ms: Optional[float] = Field(
        default=None, description="Execution time in milliseconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context or diagnostics"
    )
    disclaimer: str = Field(
        default=AUDIT_DISCLAIMER, description="Regulatory and causal disclaimer"
    )


class GlobalFeatureImportance(BaseModel):
    """Global feature importance entry."""

    model_config = ConfigDict(extra="ignore")

    feature_name: str = Field(..., description="Feature name (raw or domain)")
    mean_absolute_value: float = Field(
        ..., description="Mean absolute SHAP value or global weight"
    )
    rank: int = Field(..., description="Global ranking")


class GlobalExplanation(BaseModel):
    """Global model transparency explanation over a cohort of transactions."""

    model_config = ConfigDict(extra="ignore")

    method: ExplanationMethod = Field(..., description="Explanation algorithm used")
    model_name: str = Field(..., description="Model class name")
    prediction_class: str = Field(
        default="Suspicious", description="Target class explained"
    )
    sample_size: int = Field(
        ..., description="Number of transactions sampled for global calculation"
    )
    raw_feature_importances: List[GlobalFeatureImportance] = Field(
        default_factory=list,
        description="Global importance across model-level features",
    )
    domain_feature_importances: List[GlobalFeatureImportance] = Field(
        default_factory=list,
        description="Global importance aggregated to domain features",
    )
    disclaimer: str = Field(
        default=AUDIT_DISCLAIMER, description="Regulatory and causal disclaimer"
    )
