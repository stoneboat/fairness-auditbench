"""Abstract base for trainable models."""

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd

from fairness_auditbench.config import TrainConfig
from fairness_auditbench.datasets.base import DatasetSpec


class BaseModel(ABC):
    """Interface every model must implement."""

    @abstractmethod
    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        spec: DatasetSpec,
        config: TrainConfig,
    ) -> Dict:
        """Train the model and return a metrics dict."""
        ...

    @abstractmethod
    def save(self, output_dir: str) -> None:
        """Persist trained artefacts to *output_dir*."""
        ...
