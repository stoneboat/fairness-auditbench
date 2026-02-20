"""Orchestrates dataset loading → model training → artifact saving."""

import logging
from datetime import datetime
from pathlib import Path

from fairness_auditbench.config import TrainConfig
from fairness_auditbench.datasets import get_dataset
from fairness_auditbench.models import get_model
from fairness_auditbench.utils.io import ensure_dir, save_json
from fairness_auditbench.utils.seed import seed_everything

logger = logging.getLogger(__name__)


def run_training(config: TrainConfig) -> dict:
    """End-to-end training run. Returns summary dict."""

    seed_everything(config.seed)

    # ----- dataset -----
    ds = get_dataset(
        config.dataset,
        states=config.states,
        year=config.year,
        horizon=config.horizon,
        survey=config.survey,
        sensitive_col=config.sensitive_col,
        data_dir=config.data_dir,
        fast_dev_run=config.fast_dev_run,
        fast_dev_n=config.fast_dev_n,
    )
    train_df, val_df, test_df, spec = ds.get_splits(seed=config.seed)

    logger.info(
        "Dataset '%s' loaded: train=%d, val=%d, test=%d",
        config.dataset,
        len(train_df),
        len(val_df),
        len(test_df),
    )
    logger.info(
        "Spec: label=%s, sensitive=%s, #cat=%d, #num=%d",
        spec.label_col,
        spec.sensitive_col,
        len(spec.categorical_cols),
        len(spec.numerical_cols),
    )

    # ----- model -----
    ModelCls = get_model(config.model)
    model = ModelCls()
    metrics = model.train_model(train_df, val_df, spec, config)

    # ----- save artefacts -----
    model_dir = (
        Path(config.output_dir)
        / "models"
        / config.dataset
        / config.model
        / f"seed={config.seed}"
    )
    model.save(str(model_dir))

    # ----- run summary -----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_summary = {
        "timestamp": ts,
        "dataset": config.dataset,
        "model": config.model,
        "seed": config.seed,
        "states": config.states,
        "year": config.year,
        "fast_dev_run": config.fast_dev_run,
        "metrics": metrics,
        "model_dir": str(model_dir),
    }
    runs_dir = ensure_dir(Path(config.output_dir) / "runs")
    summary_path = runs_dir / f"{ts}_{config.model}_{config.dataset}.json"
    save_json(run_summary, summary_path)
    logger.info("Run summary saved to %s", summary_path)

    return run_summary
