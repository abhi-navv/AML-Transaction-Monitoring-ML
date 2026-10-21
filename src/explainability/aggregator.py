"""Feature Aggregation Engine: Maps model-level attributions to domain-level explanations."""

import re
from typing import Any, Dict, List, Optional, Set
from src.explainability.config import DEFAULT_CONFIG
from src.explainability.schemas import (
    DirectionalImpact,
    DomainContribution,
    RawContribution,
)


class FeatureAggregator:
    """Aggregates fine-grained model input attributions into human-understandable domain explanations.

    Mathematical Properties:
    - Additivity: For an original domain feature D_k with associated transformed
      model features G(D_k), the net attribution is defined as:
          Phi(D_k) = sum_{j in G(D_k)} phi_j
      Because Shapley values are additive, this guarantees that:
          sum_k Phi(D_k) = sum_j phi_j = f(x) - E[f(x)]
      preserving the efficiency axiom of cooperative game theory.
    - Total Absolute Impact:
          |Phi|(D_k) = sum_{j in G(D_k)} |phi_j|
      capturing the gross magnitude of signal originating from this transaction attribute.
    """

    def __init__(self, known_domain_features: Optional[List[str]] = None):
        """Initialize aggregator with known domain feature names.

        Args:
            known_domain_features: List of canonical domain feature names. Defaults to SAML-D schema.
        """
        features = known_domain_features or DEFAULT_CONFIG.known_domain_features
        # Sort by length descending so longer matching prefixes take precedence (e.g. Sender_bank_location before Sender_account)
        self.known_domain_features: List[str] = sorted(features, key=len, reverse=True)
        self._domain_set: Set[str] = set(self.known_domain_features)

    def resolve_domain_name(self, raw_feature_name: str) -> str:
        """Resolve a model-level feature name to its root domain feature name.

        Handles common transformer naming patterns:
        - scikit-learn ColumnTransformer prefixes: 'num__', 'cat__', 'remainder__'
        - OneHotEncoder dummy expansions: 'cat__Payment_type_Cross-border' -> 'Payment_type'
        - Direct matches or fallback to base name.

        Args:
            raw_feature_name: Model input feature name.

        Returns:
            Resolved domain feature name.
        """
        # 1. Strip standard scikit-learn transformer prefixes
        cleaned = re.sub(
            r"^(num__|cat__|remainder__|pipeline-\d+__)", "", raw_feature_name
        )

        # 2. Check exact match with known domain features
        if cleaned in self._domain_set:
            return cleaned

        # 3. Match against known domain features as prefix (for one-hot encoded dummies)
        for domain_feat in self.known_domain_features:
            # Check if cleaned starts with '{domain_feat}_' or '{domain_feat}='
            if cleaned.startswith(f"{domain_feat}_") or cleaned.startswith(
                f"{domain_feat}="
            ):
                return domain_feat

        # 4. If delimited by double-underscore without recognized prefix
        if "__" in cleaned:
            parts = cleaned.split("__")
            candidate = parts[-1]
            for domain_feat in self.known_domain_features:
                if candidate.startswith(f"{domain_feat}_") or candidate == domain_feat:
                    return domain_feat
            return candidate

        # 5. Fallback: split on first underscore if matches known pattern
        return cleaned

    def classify_direction(
        self, attribution: float, tolerance: float = 1e-6
    ) -> DirectionalImpact:
        """Classify attribution impact direction toward AML risk."""
        if attribution > tolerance:
            return DirectionalImpact.SUSPICIOUS_PUSHER
        elif attribution < -tolerance:
            return DirectionalImpact.LEGITIMACY_ANCHOR
        return DirectionalImpact.NEUTRAL

    def aggregate_contributions(
        self,
        raw_contributions: List[RawContribution],
        raw_record: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[DomainContribution]:
        """Aggregate model-level raw contributions into domain-level contributions.

        Args:
            raw_contributions: List of fine-grained model feature attributions.
            raw_record: Optional dictionary of original unencoded transaction fields.
            top_k: Optional maximum number of top domain contributions to return.

        Returns:
            List of DomainContribution objects ranked by absolute attribution descending.
        """
        raw_record = raw_record or {}
        domain_groups: Dict[str, List[RawContribution]] = {}

        # 1. Group raw contributions by resolved domain feature name
        for raw_contrib in raw_contributions:
            domain_name = self.resolve_domain_name(raw_contrib.feature_name)
            if domain_name not in domain_groups:
                domain_groups[domain_name] = []
            domain_groups[domain_name].append(raw_contrib)

        # 2. Compute mathematically defensible aggregate statistics
        domain_contributions: List[DomainContribution] = []
        for domain_feat, sub_contribs in domain_groups.items():
            net_attr = sum(c.attribution for c in sub_contribs)
            abs_attr = sum(c.absolute_attribution for c in sub_contribs)
            direction = self.classify_direction(net_attr)

            # Retrieve unencoded domain value from incoming raw record if available
            domain_val = raw_record.get(domain_feat, None)

            # Sort sub-components within domain by absolute attribution descending
            sorted_subs = sorted(
                sub_contribs, key=lambda c: c.absolute_attribution, reverse=True
            )

            domain_contributions.append(
                DomainContribution(
                    domain_feature=domain_feat,
                    domain_value=domain_val,
                    net_attribution=round(net_attr, 6),
                    absolute_attribution=round(abs_attr, 6),
                    direction=direction,
                    sub_components=sorted_subs,
                )
            )

        # 3. Rank domain contributions by absolute attribution descending
        domain_contributions.sort(key=lambda d: d.absolute_attribution, reverse=True)

        for rank_idx, contrib in enumerate(domain_contributions, start=1):
            contrib.rank = rank_idx

        # 4. Truncate to top_k if requested
        if top_k is not None and top_k > 0:
            domain_contributions = domain_contributions[:top_k]

        return domain_contributions
