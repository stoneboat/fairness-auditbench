"""Dataset plugin registry."""

from fairness_auditbench.datasets.folktables_acs import ACSPublicCoverageDataset

DATASET_REGISTRY = {
    "acs_public_coverage": ACSPublicCoverageDataset,
}


def get_dataset(name: str, **kwargs):
    """Instantiate a dataset by name."""
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[name](**kwargs)
