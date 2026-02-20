"""Training configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    """Configuration for a single training run."""

    # Dataset
    dataset: str = "acs_public_coverage"
    states: List[str] = field(default_factory=lambda: ["CA"])
    year: int = 2018
    horizon: str = "1-Year"
    survey: str = "person"
    sensitive_col: Optional[str] = None  # None → use dataset default

    # Model
    model: str = "logreg"  # "logreg" or "ft_transformer"

    # Reproducibility
    seed: int = 0

    # Dev / debug
    fast_dev_run: bool = False
    fast_dev_n: int = 500

    # Paths
    data_dir: str = "data"
    output_dir: str = "results"

    # Training hyper-params (FT-Transformer)
    max_epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-4
    patience: int = 5

    # Logistic regression
    logreg_max_iter: int = 1000
