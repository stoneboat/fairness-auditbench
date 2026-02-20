"""Base class for all DP synthesizers."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from fairness_auditbench.datasets.base import DatasetSpec


class BaseSynthesizer(ABC):
    """Interface for diff-priv synthesizers operating on audit tables."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        spec: DatasetSpec,
        epsilon: float,
        delta: float,
        seed: int,
    ) -> None:
        """Fit the synthesizer on the given dataset using approximate differential privacy.

        Args:
            df: The real dataset to learn from.
            spec: The DatasetSpec defining the schema (categorical/numeric/label/groups).
            epsilon: The privacy budget parameter.
            delta: The privacy bound (relaxation).
            seed: General seed for reproducibility.
        """
        ...

    @abstractmethod
    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample `n` synthetic records from the fitted model.

        Args:
            n: Number of records to generate.
            seed: Seed for the generation process.

        Returns:
            pd.DataFrame: A synthetic DataFrame matching the original schema.
        """
        ...

    def save(self, path: Path) -> None:
        """Save fitted synthesizer metadata.
        
        Optional default implementation just saves the basic info.
        Override if your synthesizer has weights to save.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"name": self.name}, f)

    @classmethod
    def load(cls, path: Path) -> "BaseSynthesizer":
        """Load an existing synthesizer. Optional."""
        raise NotImplementedError("load() is not implemented for this synthesizer.")
