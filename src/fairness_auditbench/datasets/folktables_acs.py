"""ACS Public Coverage dataset via folktables."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fairness_auditbench.datasets.base import BaseDataset, DatasetSpec
from fairness_auditbench.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


def _infer_column_types(
    df: pd.DataFrame, label_col: str, sensitive_col: str
) -> Tuple[List[str], List[str]]:
    """Heuristic to separate categorical vs numerical columns.

    A column is treated as **categorical** when any of these hold:
    - It has ``object`` or ``CategoricalDtype``.
    - It is integer-typed with ``nunique <= 20``.
    - It is float-typed but all non-NaN values are whole numbers **and**
      ``nunique <= 20``.  This catches float-encoded categoricals that
      folktables (and many census datasets) produce.

    Everything else is numerical.
    """
    categorical, numerical = [], []
    for col in df.columns:
        if col in (label_col, sensitive_col):
            continue
        if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 20:
            categorical.append(col)
        elif pd.api.types.is_float_dtype(df[col]) and df[col].nunique() <= 20:
            # Check that all non-NaN values are whole numbers (e.g. 1.0, 2.0)
            vals = df[col].dropna()
            if len(vals) == 0 or (vals == vals.astype(int).astype(float)).all():
                categorical.append(col)
            else:
                numerical.append(col)
        else:
            numerical.append(col)
    return categorical, numerical


class ACSPublicCoverageDataset(BaseDataset):
    """Folktables ACS Public Coverage task.

    Parameters
    ----------
    states : list of state abbreviations (default ``["CA"]``)
    year : survey year (default ``2018``)
    horizon : ``"1-Year"`` or ``"5-Year"``
    survey : ``"person"`` (default)
    sensitive_col : override for the sensitive attribute column name.
        If *None* the folktables default group label (``RAC1P``) is used.
    data_dir : root data directory (default ``"data"``)
    fast_dev_run : if *True*, subsample the data to ``fast_dev_n`` rows
    fast_dev_n : number of rows when ``fast_dev_run`` is active
    """

    def __init__(
        self,
        states: Optional[List[str]] = None,
        year: int = 2018,
        horizon: str = "1-Year",
        survey: str = "person",
        sensitive_col: Optional[str] = None,
        data_dir: str = "data",
        fast_dev_run: bool = False,
        fast_dev_n: int = 500,
    ):
        self.states = states or ["CA"]
        self.year = year
        self.horizon = horizon
        self.survey = survey
        self._sensitive_col_override = sensitive_col
        self.data_dir = Path(data_dir)
        self.fast_dev_run = fast_dev_run
        self.fast_dev_n = fast_dev_n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_splits(
        self, seed: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DatasetSpec]:
        cache_dir = self._cache_path(seed)
        if (cache_dir / "train.parquet").exists():
            logger.info("Loading cached splits from %s", cache_dir)
            train_df = pd.read_parquet(cache_dir / "train.parquet")
            val_df = pd.read_parquet(cache_dir / "val.parquet")
            test_df = pd.read_parquet(cache_dir / "test.parquet")
            from fairness_auditbench.utils.io import load_json
            spec = DatasetSpec.from_dict(load_json(cache_dir / "spec.json"))
            return train_df, val_df, test_df, spec

        # ----- download / extract -----
        df, label_col, sensitive_col = self._load_raw()

        if self.fast_dev_run:
            n = min(self.fast_dev_n, len(df))
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
            logger.info("fast_dev_run: subsampled to %d rows", n)

        # ----- split -----
        train_df, temp_df = train_test_split(
            df, test_size=0.3, random_state=seed, stratify=df[label_col]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=2 / 3, random_state=seed, stratify=temp_df[label_col]
        )

        # ----- infer types -----
        categorical_cols, numerical_cols = _infer_column_types(
            train_df, label_col, sensitive_col
        )

        spec = DatasetSpec(
            label_col=label_col,
            sensitive_col=sensitive_col,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
        )

        # ----- persist -----
        ensure_dir(cache_dir)
        train_df.to_parquet(cache_dir / "train.parquet", index=False)
        val_df.to_parquet(cache_dir / "val.parquet", index=False)
        test_df.to_parquet(cache_dir / "test.parquet", index=False)
        save_json(spec.to_dict(), cache_dir / "spec.json")
        logger.info(
            "Saved splits to %s  (train=%d, val=%d, test=%d)",
            cache_dir,
            len(train_df),
            len(val_df),
            len(test_df),
        )

        return train_df, val_df, test_df, spec

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, seed: int) -> Path:
        states_tag = "-".join(sorted(self.states))
        subdir = "acs_public_coverage"
        if self.fast_dev_run:
            subdir += f"_fast{self.fast_dev_n}"
        return (
            self.data_dir
            / "processed"
            / subdir
            / f"states={states_tag}"
            / f"year={self.year}"
            / f"seed={seed}"
        )

    def _load_raw(self) -> Tuple[pd.DataFrame, str, str]:
        """Download / cache ACS data via folktables and return a flat DataFrame."""
        from folktables import ACSDataSource, ACSPublicCoverage

        raw_dir = self.data_dir / "raw"
        ensure_dir(raw_dir)

        data_source = ACSDataSource(
            survey_year=str(self.year),
            horizon=self.horizon,
            survey=self.survey,
            root_dir=str(raw_dir),
        )

        acs_data = data_source.get_data(states=self.states, download=True)

        features, label, group = ACSPublicCoverage.df_to_pandas(acs_data)

        label_col = ACSPublicCoverage.target
        feature_cols = list(ACSPublicCoverage.features)

        # Determine sensitive column
        if self._sensitive_col_override and self._sensitive_col_override in feature_cols:
            sensitive_col = self._sensitive_col_override
        else:
            sensitive_col = ACSPublicCoverage.group
            if self._sensitive_col_override:
                logger.warning(
                    "Requested sensitive_col='%s' not in features; "
                    "falling back to default '%s'",
                    self._sensitive_col_override,
                    sensitive_col,
                )

        # Build a single DataFrame
        df = features.copy()
        df[label_col] = label.values
        # Ensure the sensitive column from the group output is in the df
        if sensitive_col not in df.columns:
            df[sensitive_col] = group.values

        logger.info(
            "Loaded ACS Public Coverage: %d rows, %d features, "
            "label='%s', sensitive='%s'",
            len(df),
            len(feature_cols),
            label_col,
            sensitive_col,
        )
        return df, label_col, sensitive_col
