"""Sklearn-oriented tabular preprocessing."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairness_auditbench.datasets.base import DatasetSpec


def build_sklearn_preprocessor(spec: DatasetSpec) -> ColumnTransformer:
    """Return a ``ColumnTransformer`` that one-hot-encodes categoricals and
    standard-scales numericals."""
    transformers = []
    if spec.categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                spec.categorical_cols,
            )
        )
    if spec.numerical_cols:
        transformers.append(("num", StandardScaler(), spec.numerical_cols))
    return ColumnTransformer(transformers, remainder="drop")
