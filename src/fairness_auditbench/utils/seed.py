"""Deterministic seeding for reproducibility."""

import random
import numpy as np


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy, and (optionally) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
