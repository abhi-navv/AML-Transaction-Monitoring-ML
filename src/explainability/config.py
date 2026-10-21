"""Configuration settings and defaults for Explainable AI (XAI) modules."""

from dataclasses import dataclass, field
from typing import List

AUDIT_DISCLAIMER = (
    "A SHAP or LIME attribution explains why the machine learning algorithm assigned a high risk score; "
    "it does not constitute legal or causal proof of criminal money laundering."
)


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Immutable configuration container for XAI operations."""

    # Default explanation method
    default_method: str = "shap"

    # Reference background dataset configuration
    default_background_size: int = 200
    max_background_size: int = 1000

    # LIME perturbation parameters
    default_lime_num_samples: int = 1000
    default_lime_num_features: int = 10

    # Top-K features returned in local / global views
    default_top_k: int = 10

    # Reproducibility
    random_seed: int = 42

    # Target class specification (AML Suspicious class is positive index 1)
    target_class_index: int = 1
    target_class_name: str = "Suspicious"
    baseline_class_name: str = "Legitimate"

    # Known domain feature names from SAML-D schema
    known_domain_features: List[str] = field(
        default_factory=lambda: [
            "Time",
            "Date",
            "Sender_account",
            "Receiver_account",
            "Amount",
            "Payment_currency",
            "Received_currency",
            "Sender_bank_location",
            "Receiver_bank_location",
            "Payment_type",
            "Laundering_type",
            # Engineered features if present
            "Hour",
            "DayOfWeek",
            "Is_weekend",
            "Is_night",
            "Same_currency",
            "Cross_border",
            "Log_amount",
        ]
    )


# Global default instance
DEFAULT_CONFIG = ExplainabilityConfig()
