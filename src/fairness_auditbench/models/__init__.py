"""Model registry."""

from fairness_auditbench.models.logreg import LogisticRegressionModel
from fairness_auditbench.models.ft_transformer import FTTransformerModel

MODEL_REGISTRY = {
    "logreg": LogisticRegressionModel,
    "ft_transformer": FTTransformerModel,
}


def get_model(name: str):
    """Return an *uninstantiated* model class by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]
