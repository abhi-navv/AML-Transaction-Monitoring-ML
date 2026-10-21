"""Background Reference Dataset Selector for AML Explainability.

Enforces strict AML domain safeguards:
- Synthetic SMOTE transactions are completely isolated from explainability baselines.
- Reference distributions are sampled exclusively from empirical legitimate (class 0) transactions.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import scipy.sparse

from src.explainability.base import (
    ContaminatedBackgroundError,
    ExplainabilityError,
)
from src.explainability.config import DEFAULT_CONFIG, ExplainabilityConfig

logger = logging.getLogger("aml_explainability.background")

try:
    from src.config import PROCESSED_DATA_DIR
except ImportError:
    PROCESSED_DATA_DIR = (
        Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    )


class BackgroundSelector:
    """Selects and manages reference background samples for SHAP and LIME.

    AML Invariant:
    Synthetic samples created by oversampling techniques (SMOTE, BorderlineSMOTE, ADASYN)
    interpolate artificial feature values that distort expected baseline values (E[f(x)])
    and misguide Shapley attribution. Background references MUST represent verified,
    empirical legitimate transactions.
    """

    def __init__(self, config: Optional[ExplainabilityConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self._cached_background: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = (
            None
        )

    def sample_legitimate_background(
        self,
        data: Optional[Union[np.ndarray, scipy.sparse.spmatrix]] = None,
        labels: Optional[np.ndarray] = None,
        sample_size: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> Union[np.ndarray, scipy.sparse.spmatrix]:
        """Extract uncontaminated background reference transactions.

        Args:
            data: Optional transaction feature matrix. If None, loads from empirical test set.
            labels: Optional transaction labels. If provided, filters to legitimate (class 0).
            sample_size: Number of background records to extract.
            random_seed: Seed for deterministic sampling.

        Returns:
            Subsampled reference matrix of legitimate transactions.
        """
        size = sample_size or self.config.default_background_size
        seed = random_seed if random_seed is not None else self.config.random_seed
        rng = np.random.default_rng(seed)

        # 1. If explicit data provided
        if data is not None:
            if labels is not None:
                labels_arr = np.asarray(labels).ravel()
                if len(labels_arr) != data.shape[0]:
                    raise ExplainabilityError(
                        f"Data rows ({data.shape[0]}) does not match labels length ({len(labels_arr)})."
                    )

                # Check for 100% positive data (which would imply SMOTE or contaminated subset)
                legit_mask = labels_arr == 0
                if not np.any(legit_mask):
                    raise ContaminatedBackgroundError(
                        "Provided background candidate contains no legitimate (class 0) transactions. "
                        "Background references must represent empirical non-suspicious activity."
                    )

                legit_indices = np.where(legit_mask)[0]
                n_sample = min(size, len(legit_indices))
                chosen_idx = rng.choice(legit_indices, size=n_sample, replace=False)
                sampled_bg = data[chosen_idx]
            else:
                n_rows = data.shape[0]
                n_sample = min(size, n_rows)
                chosen_idx = rng.choice(n_rows, size=n_sample, replace=False)
                sampled_bg = data[chosen_idx]

            self._cached_background = sampled_bg
            return sampled_bg

        # 2. Return cached background if already loaded
        if self._cached_background is not None:
            return self._cached_background

        # 3. Default: load from empirical unresampled test data
        x_test_path = PROCESSED_DATA_DIR / "X_test_selected.npz"
        y_test_path = PROCESSED_DATA_DIR / "y_test.npy"

        if not x_test_path.exists() or not y_test_path.exists():
            raise FileNotFoundError(
                f"Empirical test artifacts not found at '{x_test_path}' and '{y_test_path}'. "
                "Explicit background data must be provided."
            )

        try:
            logger.info(
                "Loading uncontaminated background baseline from empirical test split..."
            )
            y_test = np.load(y_test_path, mmap_mode="r")
            legit_indices = np.where(y_test == 0)[0]

            n_sample = min(size, len(legit_indices))
            chosen_idx = rng.choice(legit_indices, size=n_sample, replace=False)

            X_test = scipy.sparse.load_npz(x_test_path)
            sampled_bg = X_test[chosen_idx]

            self._cached_background = sampled_bg
            logger.info(f"Loaded {n_sample} empirical legitimate background samples.")
            return sampled_bg

        except Exception as e:
            raise ExplainabilityError(
                f"Failed loading empirical background baseline: {e}"
            ) from e

    def get_cached_background(
        self,
    ) -> Optional[Union[np.ndarray, scipy.sparse.spmatrix]]:
        """Return the currently cached background matrix if available."""
        return self._cached_background

    def clear_cache(self) -> None:
        """Clear cached background samples."""
        self._cached_background = None
