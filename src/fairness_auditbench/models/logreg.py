"""Logistic Regression model (sklearn pipeline)."""

import logging
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline

from fairness_auditbench.config import TrainConfig
from fairness_auditbench.datasets.base import DatasetSpec
from fairness_auditbench.models.base import BaseModel
from fairness_auditbench.preprocess.tabular_sklearn import build_sklearn_preprocessor
from fairness_auditbench.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


class LogisticRegressionModel(BaseModel):
    """Sklearn Pipeline: ColumnTransformer → LogisticRegression."""

    def __init__(self):
        self.pipeline = None
        self._metrics: Dict = {}

    def train_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        spec: DatasetSpec,
        config: TrainConfig,
    ) -> Dict:
        feature_cols = spec.categorical_cols + spec.numerical_cols
        X_train = train_df[feature_cols]
        y_train = train_df[spec.label_col].values.astype(int)
        X_val = val_df[feature_cols]
        y_val = val_df[spec.label_col].values.astype(int)

        preprocessor = build_sklearn_preprocessor(spec)
        clf = LogisticRegression(
            max_iter=config.logreg_max_iter,
            n_jobs=-1,
            random_state=config.seed,
        )
        self.pipeline = Pipeline([("preprocess", preprocessor), ("clf", clf)])

        logger.info("Training Logistic Regression (max_iter=%d)…", config.logreg_max_iter)
        self.pipeline.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pipeline.predict(X_val)
        y_prob = self.pipeline.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val, y_pred)
        try:
            auroc = roc_auc_score(y_val, y_prob)
        except ValueError:
            auroc = float("nan")
            logger.warning("AUROC undefined (single class in val set)")

        self._metrics = {"accuracy": float(acc), "auroc": float(auroc)}
        logger.info("Logistic Regression → accuracy=%.4f, AUROC=%.4f", acc, auroc)
        return self._metrics

    def save(self, output_dir: str) -> None:
        out = ensure_dir(Path(output_dir))
        joblib.dump(self.pipeline, out / "pipeline.joblib")
        save_json(self._metrics, out / "metrics.json")
        logger.info("Saved LogReg artefacts to %s", out)
