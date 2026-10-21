"""Explainable AI (XAI) Module for AML Transaction Monitoring.

Provides model-independent architectures for generating transparent, audit-ready
SHAP and LIME explanations at both model-input and domain-feature levels.
"""

from src.explainability.config import (
    AUDIT_DISCLAIMER,
    DEFAULT_CONFIG,
    ExplainabilityConfig,
)
from src.explainability.schemas import (
    DirectionalImpact,
    DomainContribution,
    ExplanationMethod,
    GlobalExplanation,
    GlobalFeatureImportance,
    LocalExplanation,
    RawContribution,
)
from src.explainability.base import (
    BaseExplainer,
    ContaminatedBackgroundError,
    ExplainabilityError,
    IncompatibleFeatureSchemaError,
    UnsupportedModelExplainerError,
)
from src.explainability.aggregator import FeatureAggregator
from src.explainability.shap_adapter import ShapAdapter
from src.explainability.lime_adapter import LimeAdapter
from src.explainability.background import BackgroundSelector
from src.explainability.explainer_factory import ExplainerFactory
from src.explainability.service import ExplainabilityService

__all__ = [
    # Configuration
    "AUDIT_DISCLAIMER",
    "DEFAULT_CONFIG",
    "ExplainabilityConfig",
    # Schemas & DTOs
    "DirectionalImpact",
    "DomainContribution",
    "ExplanationMethod",
    "GlobalExplanation",
    "GlobalFeatureImportance",
    "LocalExplanation",
    "RawContribution",
    # Base Classes & Exceptions
    "BaseExplainer",
    "ContaminatedBackgroundError",
    "ExplainabilityError",
    "IncompatibleFeatureSchemaError",
    "UnsupportedModelExplainerError",
    # Feature Aggregation Engine
    "FeatureAggregator",
    # Model Adapters
    "ShapAdapter",
    "LimeAdapter",
    # Background, Factory & Service Facade
    "BackgroundSelector",
    "ExplainerFactory",
    "ExplainabilityService",
]
