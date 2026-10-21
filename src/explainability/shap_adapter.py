"""SHAP Adapter: Model-independent SHAP integration with dynamic explainer dispatch."""

import time
import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import scipy.sparse
import shap

from src.explainability.aggregator import FeatureAggregator
from src.explainability.base import (
    BaseExplainer,
    ExplainabilityError,
    IncompatibleFeatureSchemaError,
    UnsupportedModelExplainerError,
)
from src.explainability.config import DEFAULT_CONFIG, ExplainabilityConfig
from src.explainability.schemas import (
    ExplanationMethod,
    GlobalExplanation,
    GlobalFeatureImportance,
    LocalExplanation,
    RawContribution,
)

logger = logging.getLogger("aml_explainability.shap")


def _extract_target_shap_vector(shap_out: Any, target_idx: int = 1) -> np.ndarray:
    """Normalize SHAP output into a 1D vector of Shapley values for the target class."""
    if hasattr(shap_out, "values"):
        shap_out = shap_out.values

    if scipy.sparse.issparse(shap_out):
        shap_out = shap_out.toarray()

    # If list of ndarrays [class_0_vals, class_1_vals]
    if isinstance(shap_out, list):
        idx = target_idx if len(shap_out) > target_idx else -1
        arr = shap_out[idx]
        if scipy.sparse.issparse(arr):
            arr = arr.toarray()
        if arr.ndim == 2:
            return np.asarray(arr[0], dtype=float)
        elif arr.ndim == 1:
            return np.asarray(arr, dtype=float)
        return np.asarray(arr, dtype=float).ravel()

    arr = np.asarray(shap_out, dtype=float)
    # If 3D array: (samples, features, classes) e.g. (1, M, 2)
    if arr.ndim == 3:
        idx = target_idx if arr.shape[2] > target_idx else -1
        return arr[0, :, idx]
    # If 2D array: (samples, features) e.g. (1, M)
    elif arr.ndim == 2:
        return arr[0]
    # If 1D array: (features,)
    elif arr.ndim == 1:
        return arr
    else:
        return arr.flatten()


def _extract_target_base_value(ev: Any, target_idx: int = 1) -> float:
    """Extract expected reference baseline value E[f(x)] for target class."""
    if isinstance(ev, (list, tuple)):
        idx = target_idx if len(ev) > target_idx else -1
        return float(ev[idx])
    elif isinstance(ev, np.ndarray):
        if ev.ndim == 0:
            return float(ev)
        idx = target_idx if ev.size > target_idx else -1
        return float(ev.ravel()[idx])
    else:
        return float(ev)


class ShapAdapter(BaseExplainer):
    """Model-independent SHAP explainer adapter.

    Dynamically selects between:
    - TreeExplainer: for tree ensembles (LightGBM, Random Forest, XGBoost, CatBoost)
    - LinearExplainer: for linear models (LogisticRegression) with reference background
    - KernelExplainer: model-agnostic fallback for small feature sets
    """

    TREE_MODEL_TYPES = (
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "DecisionTreeClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
        "LGBMClassifier",
        "XGBClassifier",
        "CatBoostClassifier",
    )

    LINEAR_MODEL_TYPES = (
        "LogisticRegression",
        "SGDClassifier",
        "LinearSVC",
        "RidgeClassifier",
    )

    def __init__(
        self,
        model: Any,
        background_data: Optional[
            Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]
        ] = None,
        feature_names: Optional[List[str]] = None,
        config: Optional[ExplainabilityConfig] = None,
        aggregator: Optional[FeatureAggregator] = None,
    ):
        """Initialize SHAP adapter for a trained model.

        Args:
            model: Trained estimator.
            background_data: Reference background samples (required for LinearExplainer).
            feature_names: Ordered list of model feature names.
            config: Explainability configuration.
            aggregator: FeatureAggregator for dual-granularity mapping.
        """
        self.model = model
        self.config = config or DEFAULT_CONFIG
        self.aggregator = aggregator or FeatureAggregator(
            self.config.known_domain_features
        )
        self.model_name = getattr(model, "__class__", type(model)).__name__
        self.explainer_type: str = "Unknown"
        self.explainer: Any = None
        self.feature_names: List[str] = feature_names or []

        self._initialize_explainer(background_data)

    def _initialize_explainer(
        self,
        background_data: Optional[
            Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame]
        ],
    ) -> None:
        """Inspect model class hierarchy and instantiate compatible SHAP explainer."""
        model_cls_name = self.model_name

        # 1. Check Tree-based ensembles
        if any(tree_cls in model_cls_name for tree_cls in self.TREE_MODEL_TYPES):
            try:
                # TreeExplainer path-dependent algorithm is exact and natively supports sparse matrices
                if background_data is not None and not scipy.sparse.issparse(
                    background_data
                ):
                    self.explainer = shap.TreeExplainer(
                        self.model, data=background_data
                    )
                else:
                    self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = "TreeExplainer"
                logger.info(f"Initialized SHAP TreeExplainer for {model_cls_name}")
                return
            except Exception as e:
                # Try path-dependent fallback if interventional failed
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    self.explainer_type = "TreeExplainer"
                    logger.info(
                        f"Initialized SHAP TreeExplainer (path-dependent fallback) for {model_cls_name}"
                    )
                    return
                except Exception as inner_e:
                    logger.warning(
                        f"TreeExplainer failed for {model_cls_name} ({e}, {inner_e}). Evaluating fallbacks."
                    )

        # 2. Check Linear models
        if any(lin_cls in model_cls_name for lin_cls in self.LINEAR_MODEL_TYPES):
            if background_data is None:
                raise ExplainabilityError(
                    f"Background data is strictly required to initialize SHAP LinearExplainer for {model_cls_name}."
                )
            try:
                self.explainer = shap.LinearExplainer(self.model, background_data)
                self.explainer_type = "LinearExplainer"
                logger.info(f"Initialized SHAP LinearExplainer for {model_cls_name}")
                return
            except Exception as e:
                raise UnsupportedModelExplainerError(
                    self.model,
                    "LinearExplainer",
                    f"LinearExplainer initialization failed: {e}",
                )

        # 3. Model-Agnostic KernelExplainer Fallback
        if hasattr(self.model, "predict_proba") and background_data is not None:
            # Check feature dimensionality to prevent catastrophic hang
            n_features = (
                background_data.shape[1]
                if hasattr(background_data, "shape")
                else len(self.feature_names)
            )
            if n_features > 500:
                raise UnsupportedModelExplainerError(
                    self.model,
                    "KernelExplainer",
                    f"Feature dimension ({n_features}) exceeds tractable limit for KernelExplainer. "
                    "Use a tree or linear model architecture.",
                )

            try:
                self.explainer = shap.KernelExplainer(
                    self.model.predict_proba, background_data
                )
                self.explainer_type = "KernelExplainer"
                logger.warning(
                    f"Initialized fallback SHAP KernelExplainer for {model_cls_name}. "
                    "Expect higher computation latency."
                )
                return
            except Exception as e:
                raise UnsupportedModelExplainerError(
                    self.model,
                    "KernelExplainer",
                    f"KernelExplainer initialization failed: {e}",
                )

        # 4. Explicit Unsupported Model Error
        raise UnsupportedModelExplainerError(
            self.model,
            "SHAP",
            f"No compatible SHAP explainer available for architecture '{model_cls_name}'. "
            "Model must be a supported tree ensemble, linear model with background data, "
            "or provide predict_proba for low-dimensional KernelExplainer.",
        )

    def get_feature_names(self) -> List[str]:
        """Return registered feature names."""
        return list(self.feature_names)

    def explain_instance(
        self,
        instance: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame, pd.Series],
        raw_record: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> LocalExplanation:
        """Generate dual-granularity SHAP explanation for an individual transaction."""
        top_k = top_k or self.config.default_top_k
        start_time = time.perf_counter()

        # 1. Format input instance to 2D
        if isinstance(instance, pd.Series):
            x_2d = instance.to_numpy().reshape(1, -1)
        elif isinstance(instance, pd.DataFrame):
            x_2d = instance
        elif isinstance(instance, np.ndarray) and instance.ndim == 1:
            x_2d = instance.reshape(1, -1)
        else:
            x_2d = instance

        # 2. Derive feature names if not explicitly registered
        n_features = x_2d.shape[1]
        if not self.feature_names:
            if isinstance(x_2d, pd.DataFrame):
                self.feature_names = list(x_2d.columns)
            else:
                self.feature_names = [f"feature_{i}" for i in range(n_features)]

        if len(self.feature_names) != n_features:
            raise IncompatibleFeatureSchemaError(
                f"Feature names count ({len(self.feature_names)}) does not match input dimensionality ({n_features})."
            )

        # 3. Predict target risk score
        try:
            if hasattr(self.model, "predict_proba"):
                risk_score = float(
                    self.model.predict_proba(x_2d)[0, self.config.target_class_index]
                )
            elif hasattr(self.model, "decision_function"):
                score = float(self.model.decision_function(x_2d)[0])
                risk_score = float(1.0 / (1.0 + np.exp(-score)))
            else:
                risk_score = float(self.model.predict(x_2d)[0])
        except Exception:
            risk_score = 0.5

        # 4. Compute SHAP values
        try:
            shap_out = self.explainer.shap_values(x_2d)
        except Exception:
            if scipy.sparse.issparse(x_2d):
                shap_out = self.explainer.shap_values(x_2d.toarray())
            else:
                raise

        shap_vector = _extract_target_shap_vector(
            shap_out, self.config.target_class_index
        )
        base_value = _extract_target_base_value(
            getattr(self.explainer, "expected_value", 0.0),
            self.config.target_class_index,
        )

        # 5. Extract feature values for inspection
        if scipy.sparse.issparse(x_2d):
            feature_vals = x_2d.toarray().ravel()
        elif isinstance(x_2d, pd.DataFrame):
            feature_vals = x_2d.iloc[0].values
        else:
            feature_vals = np.asarray(x_2d).ravel()

        # 6. Formulate Raw Contributions
        raw_contribs: List[RawContribution] = []
        for feat_name, feat_val, attr_val in zip(
            self.feature_names, feature_vals, shap_vector
        ):
            abs_attr = abs(attr_val)
            direction = self.aggregator.classify_direction(attr_val)
            raw_contribs.append(
                RawContribution(
                    feature_name=feat_name,
                    feature_value=(
                        float(feat_val)
                        if isinstance(feat_val, (int, float, np.number))
                        else str(feat_val)
                    ),
                    attribution=round(float(attr_val), 6),
                    absolute_attribution=round(float(abs_attr), 6),
                    direction=direction,
                )
            )

        # Rank raw contributions
        raw_contribs.sort(key=lambda c: c.absolute_attribution, reverse=True)
        for rank_idx, contrib in enumerate(raw_contribs, start=1):
            contrib.rank = rank_idx

        # 7. Formulate Domain Contributions via Aggregator
        domain_contribs = self.aggregator.aggregate_contributions(
            raw_contributions=raw_contribs,
            raw_record=raw_record,
            top_k=top_k,
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        txn_id = str(raw_record.get("Sender_account", "TXN")) if raw_record else None

        return LocalExplanation(
            transaction_id=txn_id,
            method=ExplanationMethod.SHAP,
            model_name=self.model_name,
            prediction_class=self.config.target_class_name,
            predicted_risk_score=round(risk_score, 4),
            base_value=round(base_value, 6),
            raw_contributions=raw_contribs[:top_k],
            domain_contributions=domain_contribs,
            latency_ms=elapsed_ms,
            metadata={
                "explainer_type": self.explainer_type,
                "total_raw_features": n_features,
            },
        )

    def explain_global(
        self,
        dataset: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
        top_k: Optional[int] = None,
    ) -> GlobalExplanation:
        """Generate global feature-importance summary across a dataset sample."""
        top_k = top_k or 20
        sample_size = dataset.shape[0]

        try:
            shap_out = self.explainer.shap_values(dataset)
        except Exception:
            if scipy.sparse.issparse(dataset):
                shap_out = self.explainer.shap_values(dataset.toarray())
            else:
                raise

        # Handle list vs array vs sparse outputs
        if scipy.sparse.issparse(shap_out):
            shap_matrix = shap_out.toarray()
        elif isinstance(shap_out, list):
            idx = (
                self.config.target_class_index
                if len(shap_out) > self.config.target_class_index
                else -1
            )
            val = shap_out[idx]
            shap_matrix = (
                val.toarray() if scipy.sparse.issparse(val) else np.asarray(val)
            )
        elif hasattr(shap_out, "values"):
            shap_matrix = shap_out.values
            if scipy.sparse.issparse(shap_matrix):
                shap_matrix = shap_matrix.toarray()
            elif isinstance(shap_matrix, np.ndarray) and shap_matrix.ndim == 3:
                shap_matrix = shap_matrix[:, :, self.config.target_class_index]
        else:
            shap_matrix = (
                shap_out.toarray()
                if scipy.sparse.issparse(shap_out)
                else np.asarray(shap_out)
            )
            if shap_matrix.ndim == 3:
                shap_matrix = shap_matrix[:, :, self.config.target_class_index]

        mean_abs_values = np.mean(np.abs(shap_matrix), axis=0)

        # 1. Raw Feature Importance
        raw_importances: List[GlobalFeatureImportance] = []
        for feat_name, mean_abs in zip(self.feature_names, mean_abs_values):
            raw_importances.append(
                GlobalFeatureImportance(
                    feature_name=feat_name,
                    mean_absolute_value=round(float(mean_abs), 6),
                    rank=0,
                )
            )
        raw_importances.sort(key=lambda g: g.mean_absolute_value, reverse=True)
        for rank_idx, imp in enumerate(raw_importances, start=1):
            imp.rank = rank_idx

        # 2. Domain Feature Importance (aggregated)
        domain_totals: Dict[str, float] = {}
        for imp in raw_importances:
            domain_name = self.aggregator.resolve_domain_name(imp.feature_name)
            domain_totals[domain_name] = (
                domain_totals.get(domain_name, 0.0) + imp.mean_absolute_value
            )

        domain_importances: List[GlobalFeatureImportance] = [
            GlobalFeatureImportance(
                feature_name=d_name,
                mean_absolute_value=round(val, 6),
                rank=0,
            )
            for d_name, val in domain_totals.items()
        ]
        domain_importances.sort(key=lambda g: g.mean_absolute_value, reverse=True)
        for rank_idx, imp in enumerate(domain_importances, start=1):
            imp.rank = rank_idx

        return GlobalExplanation(
            method=ExplanationMethod.SHAP,
            model_name=self.model_name,
            prediction_class=self.config.target_class_name,
            sample_size=sample_size,
            raw_feature_importances=raw_importances[:top_k],
            domain_feature_importances=domain_importances[:top_k],
        )
