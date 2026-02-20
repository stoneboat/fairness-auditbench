"""Abstract base class for all datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd


@dataclass
class DatasetSpec:
    """Describes the schema of a tabular dataset after loading."""

    label_col: str
    sensitive_col: str
    categorical_cols: List[str] = field(default_factory=list)
    numerical_cols: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label_col": self.label_col,
            "sensitive_col": self.sensitive_col,
            "categorical_cols": self.categorical_cols,
            "numerical_cols": self.numerical_cols,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetSpec":
        return cls(**d)


class BaseDataset(ABC):
    """Interface every dataset plugin must implement."""

    @abstractmethod
    def get_splits(
        self, seed: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DatasetSpec]:
        """Return (train_df, val_df, test_df, spec).

        Each DataFrame contains features + label + sensitive attribute columns.
        """
        ...
