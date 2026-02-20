"""Torch-oriented tabular preprocessing.

Converts a pandas DataFrame into integer category codes and normalised
numerical tensors suitable for embedding-based models like FT-Transformer.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TorchTabularPreprocessor:
    """Fit category vocabularies + numeric mean/std, then transform DataFrames
    to numpy arrays ready for ``torch.from_numpy``."""

    def __init__(self):
        self.cat_vocabs: Dict[str, Dict] = {}  # col -> {val: code}
        self.num_mean: Optional[np.ndarray] = None
        self.num_std: Optional[np.ndarray] = None
        self.categorical_cols: List[str] = []
        self.numerical_cols: List[str] = []
        self._fitted = False

    # ----- fit / transform --------------------------------------------------

    def fit(self, df: pd.DataFrame, categorical_cols: List[str], numerical_cols: List[str]):
        self.categorical_cols = list(categorical_cols)
        self.numerical_cols = list(numerical_cols)

        # Category vocabularies (unknown → 0, known values → 1..V)
        for col in self.categorical_cols:
            unique_vals = sorted(df[col].dropna().unique(), key=str)
            self.cat_vocabs[col] = {v: i + 1 for i, v in enumerate(unique_vals)}

        # Numeric statistics
        if self.numerical_cols:
            num_data = df[self.numerical_cols].values.astype(np.float32)
            self.num_mean = np.nanmean(num_data, axis=0)
            self.num_std = np.nanstd(num_data, axis=0)
            self.num_std[self.num_std < 1e-8] = 1.0  # avoid division by zero

        self._fitted = True
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cat_codes [N, C_cat], num_values [N, C_num]) as numpy arrays.

        ``cat_codes`` are ``int64``; ``num_values`` are ``float32``.
        """
        assert self._fitted, "Call fit() first."

        # Categorical → integer codes
        cat_list = []
        for col in self.categorical_cols:
            vocab = self.cat_vocabs[col]
            codes = df[col].map(vocab).fillna(0).astype(np.int64).values
            cat_list.append(codes)
        if cat_list:
            cat_codes = np.stack(cat_list, axis=1)
        else:
            cat_codes = np.empty((len(df), 0), dtype=np.int64)

        # Numerical → z-scored float
        if self.numerical_cols:
            num_values = df[self.numerical_cols].values.astype(np.float32)
            num_values = np.nan_to_num(num_values, nan=0.0)
            num_values = (num_values - self.num_mean) / self.num_std
        else:
            num_values = np.empty((len(df), 0), dtype=np.float32)

        return cat_codes, num_values

    # ----- helpers -----------------------------------------------------------

    def vocab_sizes(self) -> List[int]:
        """Return vocabulary size (including unknown=0) for each categorical column."""
        return [len(v) + 1 for v in self.cat_vocabs.values()]

    def n_numerical(self) -> int:
        return len(self.numerical_cols)
