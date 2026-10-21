"""Abstract Base Classes and Exceptions for Explainable AI (XAI) modules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.explainability.schemas import GlobalExplanation, LocalExplanation


class ExplainabilityError(Exception):
    """Base exception for all explainability-related failures."""

    pass


class UnsupportedModelExplainerError(ExplainabilityError):
    """Raised when an explainer cannot be constructed for a given model architecture."""

    def __init__(self, model: Any, explainer_type: str, reason: str):
        if isinstance(model, str):
            self.model_type = model
        elif model is not None:
            self.model_type = getattr(model, "__name__", type(model).__name__)
        else:
            self.model_type = "None"
        self.explainer_type = explainer_type
        self.reason = reason
        super().__init__(
            f"Cannot configure {explainer_type} for model '{self.model_type}': {reason}"
        )


class IncompatibleFeatureSchemaError(ExplainabilityError):
    """Raised when incoming features do not align with the model or explainer feature space."""

    pass


class ContaminatedBackgroundError(ExplainabilityError):
    """Raised when synthetic (e.g. SMOTE) or invalid samples are detected in reference background data."""

    pass


class BaseExplainer(ABC):
    """Abstract protocol for all model explainers (SHAP, LIME, or custom surrogates)."""

    @abstractmethod
    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.DataFrame, pd.Series],
        raw_record: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> LocalExplanation:
        """Generate a local dual-granularity explanation for an individual transaction.

        Args:
            instance: Transformed feature vector/row matching model input dimension.
            raw_record: Optional dictionary of original unencoded transaction fields.
            top_k: Maximum number of top features to return in raw and domain views.

        Returns:
            LocalExplanation containing raw and aggregated domain-level contributions.
        """
        pass

    @abstractmethod
    def explain_global(
        self,
        dataset: Union[np.ndarray, pd.DataFrame],
        top_k: int = 20,
    ) -> GlobalExplanation:
        """Generate a global feature-importance summary across a dataset sample.

        Args:
            dataset: Matrix of transformed transactions.
            top_k: Maximum number of top features to rank.

        Returns:
            GlobalExplanation containing raw and domain-level feature importances.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return the list of model-level feature names in exact order expected by the model."""
        pass
