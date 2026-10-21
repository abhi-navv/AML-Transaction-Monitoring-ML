"""Explainer Factory: Dynamic instantiation of model-compatible SHAP and LIME explainers."""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import scipy.sparse
import pandas as pd

from src.explainability.aggregator import FeatureAggregator
from src.explainability.base import (
    BaseExplainer,
    ExplainabilityError,
    UnsupportedModelExplainerError,
)
from src.explainability.config import DEFAULT_CONFIG, ExplainabilityConfig
from src.explainability.lime_adapter import LimeAdapter
from src.explainability.schemas import ExplanationMethod
from src.explainability.shap_adapter import ShapAdapter

logger = logging.getLogger("aml_explainability.factory")


class ExplainerFactory:
    """Factory for dynamically creating model-compatible explainers based on trained estimator introspection."""

    @classmethod
    def create_shap_explainer(
        cls,
        model: Any,
        background_data: Optional[
            Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]
        ] = None,
        feature_names: Optional[List[str]] = None,
        config: Optional[ExplainabilityConfig] = None,
        aggregator: Optional[FeatureAggregator] = None,
    ) -> ShapAdapter:
        """Create a model-compatible SHAP adapter (Tree, Linear, or Kernel).

        Args:
            model: Trained classifier.
            background_data: Background reference matrix (required for LinearExplainer).
            feature_names: Registered feature names.
            config: Explainability configuration.
            aggregator: FeatureAggregator for dual-granularity mapping.

        Returns:
            Configured ShapAdapter instance.
        """
        if model is None:
            raise ExplainabilityError(
                "Cannot create SHAP explainer: Model object is None."
            )

        return ShapAdapter(
            model=model,
            background_data=background_data,
            feature_names=feature_names,
            config=config,
            aggregator=aggregator,
        )

    @classmethod
    def create_lime_explainer(
        cls,
        model: Any,
        training_data: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        config: Optional[ExplainabilityConfig] = None,
        aggregator: Optional[FeatureAggregator] = None,
    ) -> LimeAdapter:
        """Create a LIME tabular adapter.

        Args:
            model: Trained classifier.
            training_data: Reference background matrix for local perturbation distributions.
            feature_names: Registered feature names.
            categorical_features: Column indices of categorical features.
            config: Explainability configuration.
            aggregator: FeatureAggregator for dual-granularity mapping.

        Returns:
            Configured LimeAdapter instance.
        """
        if model is None:
            raise ExplainabilityError(
                "Cannot create LIME explainer: Model object is None."
            )

        if training_data is None:
            raise ExplainabilityError(
                "Cannot create LIME explainer: training_data reference is required."
            )

        return LimeAdapter(
            model=model,
            training_data=training_data,
            feature_names=feature_names,
            categorical_features=categorical_features,
            config=config,
            aggregator=aggregator,
        )

    @classmethod
    def create_explainer(
        cls,
        method: Union[str, ExplanationMethod],
        model: Any,
        background_data: Optional[
            Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]
        ] = None,
        feature_names: Optional[List[str]] = None,
        config: Optional[ExplainabilityConfig] = None,
        aggregator: Optional[FeatureAggregator] = None,
        **kwargs: Any,
    ) -> BaseExplainer:
        """Create an explainer based on requested method name ('shap' or 'lime').

        Args:
            method: 'shap' or 'lime' (case-insensitive).
            model: Trained classifier.
            background_data: Background reference matrix.
            feature_names: Feature names list.
            config: Configuration container.
            aggregator: FeatureAggregator.

        Returns:
            Instantiated BaseExplainer.
        """
        method_str = str(method).lower()

        if method_str in ("shap", ExplanationMethod.SHAP.value):
            return cls.create_shap_explainer(
                model=model,
                background_data=background_data,
                feature_names=feature_names,
                config=config,
                aggregator=aggregator,
            )

        elif method_str in ("lime", ExplanationMethod.LIME.value):
            if background_data is None:
                raise ExplainabilityError(
                    "LIME explainer requires reference background_data."
                )
            return cls.create_lime_explainer(
                model=model,
                training_data=background_data,
                feature_names=feature_names,
                config=config,
                aggregator=aggregator,
                **kwargs,
            )

        else:
            raise UnsupportedModelExplainerError(
                model,
                method_str,
                f"Unknown explanation method '{method}'. Supported methods are 'shap' and 'lime'.",
            )
