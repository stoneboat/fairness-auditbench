"""Device selection helper."""

import logging

logger = logging.getLogger(__name__)


def get_device() -> "torch.device":
    """Return the best available torch device (CUDA → CPU)."""
    import torch

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        dev = torch.device("cpu")
        logger.info("CUDA not available – using CPU")
    return dev
