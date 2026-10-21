"""LIME Adapter: Model-independent LIME tabular integration with stability evaluation."""

import time
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union
import numpy as np
import pandas as pd
import scipy.sparse
from lime.lime_tabular import LimeTabularExplainer

from src.explainability.aggregator import FeatureAggregator
from src.explainability.base import (
    BaseExplainer,
    ExplainabilityError,
    IncompatibleFeatureSchemaError,
)
from src.explainability.config import DEFAULT_CONFIG, ExplainabilityConfig
from src.explainability.schemas import (
    ExplanationMethod,
    GlobalExplanation,
    GlobalFeatureImportance,
    LocalExplanation,
    RawContribution,
)

logger = logging.getLogger("aml_explainability.lime")


class LimeAdapter(BaseExplainer):
    """Model-independent LIME explainer adapter for tabular transaction monitoring."""

    def __init__(
        self,
        model: Any,
        training_data: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None,
        config: Optional[ExplainabilityConfig] = None,
        aggregator: Optional[FeatureAggregator] = None,
    ):
        """Initialize LIME tabular explainer.

        Args:
            model: Trained classifier (with predict_proba or decision_function).
            training_data: Representative background dataset used for perturbation stats.
            feature_names: Ordered list of model feature names.
            categorical_features: List of column indices for categorical features.
            config: Explainability configuration.
            aggregator: FeatureAggregator for dual-granularity mapping.
        """
        self.model = model
        self.config = config or DEFAULT_CONFIG
        self.aggregator = aggregator or FeatureAggregator(
            self.config.known_domain_features
        )
        self.model_name = getattr(model, "__class__", type(model)).__name__

        # Convert sparse reference to dense for LIME background statistics
        if scipy.sparse.issparse(training_data):
            self._is_model_sparse = True
            train_dense = training_data.toarray()
        else:
            self._is_model_sparse = False
            train_dense = np.asarray(training_data)

        self.feature_names: List[str] = feature_names or [
            f"f_{i}" for i in range(train_dense.shape[1])
        ]
        if len(self.feature_names) != train_dense.shape[1]:
            raise IncompatibleFeatureSchemaError(
                f"Feature names count ({len(self.feature_names)}) does not match training data cols ({train_dense.shape[1]})."
            )

        self.explainer = LimeTabularExplainer(
            training_data=train_dense,
            feature_names=self.feature_names,
            class_names=[
                self.config.baseline_class_name,
                self.config.target_class_name,
            ],
            categorical_features=categorical_features,
            mode="classification",
            random_state=self.config.random_seed,
        )

        self._predict_fn = self._build_predict_function()

    def _build_predict_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap model prediction to guarantee 2D class probabilities [P(0), P(1)]."""
        is_sparse = self._is_model_sparse
        model = self.model

        def predict_wrapper(x_dense: np.ndarray) -> np.ndarray:
            x_in = scipy.sparse.csr_matrix(x_dense) if is_sparse else x_dense

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_in)
                if probs.ndim == 1:
                    probs = np.column_stack([1.0 - probs, probs])
                return probs

            elif hasattr(model, "decision_function"):
                scores = model.decision_function(x_in)
                probs_1 = 1.0 / (1.0 + np.exp(-scores))
                return np.column_stack([1.0 - probs_1, probs_1])

            else:
                preds = model.predict(x_in)
                probs_1 = preds.astype(float)
                return np.column_stack([1.0 - probs_1, probs_1])

        return predict_wrapper

    def get_feature_names(self) -> List[str]:
        """Return registered feature names."""
        return list(self.feature_names)

    def explain_instance(
        self,
        instance: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame, pd.Series],
        raw_record: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        num_samples: Optional[int] = None,
        compute_stability: bool = False,
    ) -> LocalExplanation:
        """Generate dual-granularity LIME explanation for an individual transaction."""
        top_k = top_k or self.config.default_top_k
        num_samples = num_samples or self.config.default_lime_num_samples
        start_time = time.perf_counter()

        # 1. Format input instance to 1D dense vector
        if scipy.sparse.issparse(instance):
            x_1d = instance.toarray().ravel()
        elif isinstance(instance, (pd.DataFrame, pd.Series)):
            x_1d = np.asarray(instance).ravel()
        else:
            x_1d = np.asarray(instance).ravel()

        if len(x_1d) != len(self.feature_names):
            raise IncompatibleFeatureSchemaError(
                f"Instance dimension ({len(x_1d)}) does not match explainer features ({len(self.feature_names)})."
            )

        # 2. Predict risk score
        probs = self._predict_fn(x_1d.reshape(1, -1))[0]
        risk_score = float(probs[self.config.target_class_index])

        # 3. Generate LIME local explanation
        exp = self.explainer.explain_instance(
            x_1d,
            self._predict_fn,
            num_features=len(self.feature_names),
            labels=[self.config.target_class_index],
            num_samples=num_samples,
        )

        base_value = float(exp.intercept[self.config.target_class_index])
        feature_weights = exp.as_map()[self.config.target_class_index]

        # 4. Formulate Raw Contributions
        raw_contribs: List[RawContribution] = []
        for feat_idx, weight in feature_weights:
            feat_name = self.feature_names[feat_idx]
            feat_val = float(x_1d[feat_idx])
            abs_attr = abs(weight)
            direction = self.aggregator.classify_direction(weight)

            raw_contribs.append(
                RawContribution(
                    feature_name=feat_name,
                    feature_value=round(feat_val, 4),
                    attribution=round(float(weight), 6),
                    absolute_attribution=round(float(abs_attr), 6),
                    direction=direction,
                )
            )

        # Sort raw contributions by absolute magnitude descending
        raw_contribs.sort(key=lambda c: c.absolute_attribution, reverse=True)
        for rank_idx, contrib in enumerate(raw_contribs, start=1):
            contrib.rank = rank_idx

        # 5. Formulate Domain Contributions via Aggregator
        domain_contribs = self.aggregator.aggregate_contributions(
            raw_contributions=raw_contribs,
            raw_record=raw_record,
            top_k=top_k,
        )

        # 6. Evaluate optional stability score
        metadata: Dict[str, Any] = {
            "explainer_type": "LimeTabularExplainer",
            "num_samples": num_samples,
            "local_score_r2": getattr(exp, "score", None),
        }

        if compute_stability:
            stability_score = self.evaluate_stability(
                x_1d, top_k=5, num_runs=3, num_samples=500
            )
            metadata["stability_jaccard_top5"] = stability_score

        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        txn_id = str(raw_record.get("Sender_account", "TXN")) if raw_record else None

        return LocalExplanation(
            transaction_id=txn_id,
            method=ExplanationMethod.LIME,
            model_name=self.model_name,
            prediction_class=self.config.target_class_name,
            predicted_risk_score=round(risk_score, 4),
            base_value=round(base_value, 6),
            raw_contributions=raw_contribs[:top_k],
            domain_contributions=domain_contribs,
            latency_ms=elapsed_ms,
            metadata=metadata,
        )

    def evaluate_stability(
        self,
        instance: np.ndarray,
        num_runs: int = 3,
        top_k: int = 5,
        num_samples: int = 500,
    ) -> float:
        """Evaluate explanation stability across repeated stochastic perturbations using Jaccard similarity.

        Returns:
            Mean Jaccard index of top-k features across runs (1.0 = completely stable).
        """
        top_feature_sets: List[Set[int]] = []

        for _ in range(num_runs):
            exp = self.explainer.explain_instance(
                instance,
                self._predict_fn,
                num_features=top_k,
                labels=[self.config.target_class_index],
                num_samples=num_samples,
            )
            feature_indices = {
                feat_idx
                for feat_idx, _ in exp.as_map()[self.config.target_class_index][:top_k]
            }
            top_feature_sets.append(feature_indices)

        # Compute pairwise Jaccard similarities
        similarities = []
        for i in range(len(top_feature_sets)):
            for j in range(i + 1, len(top_feature_sets)):
                set_a = top_feature_sets[i]
                set_b = top_feature_sets[j]
                union = set_a | set_b
                intersection = set_a & set_b
                jaccard = len(intersection) / len(union) if union else 1.0
                similarities.append(jaccard)

        return round(float(np.mean(similarities)), 4) if similarities else 1.0

    def explain_global(
        self,
        dataset: Union[np.ndarray, scipy.sparse.spmatrix, pd.DataFrame],
        top_k: Optional[int] = None,
    ) -> GlobalExplanation:
        """Generate global feature-importance summary across a dataset sample using sample LIME averaging."""
        top_k = top_k or 20
        sample_matrix = (
            dataset.toarray() if scipy.sparse.issparse(dataset) else np.asarray(dataset)
        )
        sample_size = min(len(sample_matrix), 50)  # Subsample to keep latency bounded

        accumulated_weights = np.zeros(len(self.feature_names), dtype=float)

        for row_idx in range(sample_size):
            row = sample_matrix[row_idx]
            exp = self.explainer.explain_instance(
                row,
                self._predict_fn,
                num_features=len(self.feature_names),
                labels=[self.config.target_class_index],
                num_samples=200,
            )
            for feat_idx, weight in exp.as_map()[self.config.target_class_index]:
                accumulated_weights[feat_idx] += abs(weight)

        mean_abs_values = accumulated_weights / sample_size

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
            method=ExplanationMethod.LIME,
            model_name=self.model_name,
            prediction_class=self.config.target_class_name,
            sample_size=sample_size,
            raw_feature_importances=raw_importances[:top_k],
            domain_feature_importances=domain_importances[:top_k],
        )
