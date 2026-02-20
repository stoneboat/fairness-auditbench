"""Private-PGM synthesizer implementation using dpmm."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from fairness_auditbench.datasets.base import DatasetSpec
from fairness_auditbench.synthesizers.base import BaseSynthesizer
from fairness_auditbench.synthesizers.registry import register_synthesizer

logger = logging.getLogger(__name__)


@register_synthesizer("private_pgm")
class PrivatePGMSynthesizer(BaseSynthesizer):
    """Synthesizer using Private-PGM (via dpmm library).

    Discretizes numerical features into a configurable number of bins,
    applies DP marginal modeling, and reconstructs synthetic samples.
    """

    def __init__(self, bins: int = 32, degree: int = 2, max_cardinality: int = 2000, **kwargs):
        """
        Args:
            bins: Number of bins to discretize numerical columns into.
            degree: Maximum degree of marginals to measure. Default is 2 (2-way).
            max_cardinality: Warn/collapse if a categorical column exceeds this.
        """
        self.bins = bins
        self.degree = degree
        self.max_cardinality = max_cardinality

        # Internal state learned during fit()
        self._fitted = False
        self._model = None
        self._encoders: Dict[str, Any] = {}
        self._numeric_bins: Dict[str, np.ndarray] = {}
        self._synthetic_data: Optional[pd.DataFrame] = None
        self._spec: Optional[DatasetSpec] = None

    def fit(
        self,
        df: pd.DataFrame,
        spec: DatasetSpec,
        epsilon: float,
        delta: float,
        seed: int,
    ) -> None:
        """Fit the Private-PGM model on the given dataset."""
        try:
            from dpmm.models import MSTGM
        except ImportError:
            raise RuntimeError(
                "The 'dpmm' library is required for PrivatePGMSynthesizer. "
                "Please run `pip install dpmm`."
            )

        self._spec = spec
        df_discrete, domain = self._preprocess(df, spec)

        logger.info(
            f"Fitting Private-PGM (MSTGM) (epsilon={epsilon}, delta={delta}, bins={self.bins})..."
        )

        # Initialize and fit the underlying model
        # MSTGM typically uses degree=2 internally; passing degree directly is not necessary
        # However, we must set the categorical domain schema so it knows the cardinality of variables.
        self._model = MSTGM(epsilon=epsilon, delta=delta, n_jobs=-1)
        self._model.set_random_state(np.random.RandomState(seed))
        
        # dpmm requires domain mapping: e.g. {"col1": 2, "col2": 32}
        # which we already build in `self._preprocess()`
        self._model.set_domain(domain)
        
        # We pass public=False to ensure DP noise is added
        self._model.fit(df_discrete, public=False)
        self._fitted = True
        logger.info("dpmm MSTGM fitting complete.")

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample synthetic records."""
        if not self._fitted:
            raise RuntimeError("Synthesizer must be fitted before calling sample().")

        if self._model is None:
            raise RuntimeError("Model not yet wired up — see fit() RuntimeError for dpmm API instructions.")

        logger.info(f"Generating {n} synthetic records using dpmm MSTGM...")
        # GenerativeModel uses `generate(n_records)`
        df_synth_discrete = self._model.generate(n_records=n)

        return self._postprocess(df_synth_discrete)

    def _preprocess(self, df: pd.DataFrame, spec: DatasetSpec) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Convert df to discrete integers and compute domain capacities."""
        df_discrete = pd.DataFrame()
        domain = {}

        # 1. Handle categorical / group / label columns
        cat_cols = list(spec.categorical_cols)
        if spec.label_col not in cat_cols:
            cat_cols.append(spec.label_col)
        if spec.sensitive_col not in cat_cols:
            cat_cols.append(spec.sensitive_col)
            
        # Deduplicate
        cat_cols = list(dict.fromkeys(cat_cols))

        for col in cat_cols:
            if col not in df.columns:
                continue
            
            # Warn on high cardinality
            n_unique = df[col].nunique()
            if n_unique > self.max_cardinality:
                logger.warning(
                    f"Column '{col}' has high cardinality ({n_unique} > {self.max_cardinality}). "
                    "Consider collapsing rare categories to avoid memory/DP budgeting issues."
                )

            # Map to dense integers
            unique_vals = df[col].dropna().unique()
            # Sort to ensure reproducibility
            unique_vals.sort()
            
            mapping = {val: i for i, val in enumerate(unique_vals)}
            self._encoders[col] = {
                "type": "categorical",
                "mapping": mapping,
                "inverse": {i: val for val, i in mapping.items()}
            }
            # Fill NaNs with a special category if needed, here we just drop or keep them?
            # DPMM typically expects non-null integers. Let's map NaNs to the last bucket or ignore.
            # Assuming data is clean for this prototype.
            df_discrete[col] = df[col].map(mapping).fillna(0).astype(int)
            domain[col] = len(mapping)

        # 2. Handle numerical columns
        for col in spec.numerical_cols:
            if col not in df.columns:
                continue
            
            vals = df[col].dropna()
            if len(vals) == 0:
                domain[col] = 1
                self._encoders[col] = {"type": "numerical", "empty": True}
                df_discrete[col] = 0
                continue
                
            min_val, max_val = vals.min(), vals.max()
            # If min == max, binning is trivial
            if min_val == max_val:
                bins = np.array([-np.inf, np.inf])
            else:
                bins = np.linspace(min_val, max_val, self.bins + 1)
                bins[0] = -np.inf
                bins[-1] = np.inf
                
            self._numeric_bins[col] = bins
            
            # Calculate midpoints for reconstruction
            finite_bins = np.linspace(min_val, max_val, self.bins + 1)
            midpoints = (finite_bins[:-1] + finite_bins[1:]) / 2
            
            self._encoders[col] = {
                "type": "numerical",
                "midpoints": midpoints,
                "empty": False
            }
            
            # Digitize returns indices [1, bins], subtract 1 to get [0, bins-1]
            discrete_vals = np.digitize(df[col], bins) - 1
            # Clip to valid range just in case
            discrete_vals = np.clip(discrete_vals, 0, self.bins - 1)
            
            df_discrete[col] = discrete_vals
            domain[col] = self.bins

        return df_discrete, domain

    def _postprocess(self, df_discrete: pd.DataFrame) -> pd.DataFrame:
        """Invert the discretization to recover original domain values."""
        df_real = pd.DataFrame()
        
        for col, meta in self._encoders.items():
            if col not in df_discrete.columns:
                continue
                
            if meta["type"] == "categorical":
                inverse_map = meta["inverse"]
                df_real[col] = df_discrete[col].map(inverse_map)
            elif meta["type"] == "numerical":
                if meta.get("empty", False):
                    df_real[col] = 0.0
                else:
                    midpoints = meta["midpoints"]
                    # Map bin indices back to bin midpoints
                    # Ensure indices are within bounds
                    indices = np.clip(df_discrete[col].fillna(0).astype(int), 0, len(midpoints)-1)
                    df_real[col] = midpoints[indices]
                    
        return df_real
