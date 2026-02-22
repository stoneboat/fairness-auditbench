"""DPCTGAN synthesizer implementation using smartnoise-synth."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import torch

from fairness_auditbench.datasets.base import DatasetSpec
from fairness_auditbench.synthesizers.base import BaseSynthesizer
from fairness_auditbench.synthesizers.registry import register_synthesizer

logger = logging.getLogger(__name__)


def _set_deterministic_seeds(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # smartnoise-synth handles varying internal seeding, but we can set global states
    # to maintain as much determinism as possible within the CTGAN framework


@register_synthesizer("dpctgan")
class DPCTGANSynthesizer(BaseSynthesizer):
    """Synthesizer using DPCTGAN from smartnoise-synth."""

    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: Tuple[int, int] = (256, 256),
        discriminator_dim: Tuple[int, int] = (256, 256),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        epochs: int = 300,
        pac: int = 1,
        discriminator_steps: int = 1,
        sigma: float = 5,
        max_per_sample_grad_norm: float = 1.0,
        loss: str = "cross_entropy",
        preprocessor_eps: Optional[float] = None,
        cuda: bool = True,
        verbose: bool = True,
        disabled_dp: bool = False,
        nullable: bool = False,
        **kwargs
    ):
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.generator_decay = generator_decay
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.pac = pac
        self.discriminator_steps = discriminator_steps
        self.sigma = sigma
        self.max_per_sample_grad_norm = max_per_sample_grad_norm
        self.loss = loss
        self.preprocessor_eps = preprocessor_eps
        self.cuda = cuda
        self.verbose = verbose
        self.disabled_dp = disabled_dp
        self.nullable = nullable

        self._fitted = False
        self._synth = None
        self._columns: List[str] = []
        self._spec_snapshot: Dict[str, Any] = {}
        self._fit_meta: Dict[str, Any] = {}

    def fit(
        self,
        df: pd.DataFrame,
        spec: DatasetSpec,
        epsilon: float,
        delta: float,
        seed: int,
    ) -> None:
        """Fit the DPCTGAN model on the given dataset."""
        try:
            from snsynth import Synthesizer
        except ImportError:
            raise RuntimeError(
                "The 'smartnoise-synth' library is required for DPCTGANSynthesizer. "
                "Please run `pip install smartnoise-synth`."
            )

        _set_deterministic_seeds(seed)

        categorical_cols = list(spec.categorical_cols)
        numerical_cols = list(spec.numerical_cols)
        label_col = spec.label_col
        sensitive_col = spec.sensitive_col

        if label_col not in categorical_cols and label_col not in numerical_cols:
            categorical_cols.append(label_col)
        if sensitive_col not in categorical_cols:
            categorical_cols.append(sensitive_col)

        # Deduplicate
        dpctgan_categorical = list(dict.fromkeys(categorical_cols))
        dpctgan_continuous = list(dict.fromkeys(numerical_cols))
        
        # Ensure all columns exist in the DataFrame
        dpctgan_categorical = [c for c in dpctgan_categorical if c in df.columns]
        dpctgan_continuous = [c for c in dpctgan_continuous if c in df.columns]

        # Allocate preprocess budget
        if self.preprocessor_eps is None:
            # Safe starting point if not provided, allocate a fraction of privacy budget
            prep_eps = min(0.1 * epsilon, 0.9)
        else:
            prep_eps = self.preprocessor_eps
            
        # Ensure preprocessing budget is strictly less than total epsilon
        if prep_eps >= epsilon:
             prep_eps = epsilon * 0.1
             logger.warning(f"preprocessor_eps was >= epsilon. Adjusted to {prep_eps}")

        logger.info(
            f"Fitting DPCTGAN (epsilon={epsilon}, delta={delta}, preprocessor_eps={prep_eps}, epochs={self.epochs})..."
        )

        synth = Synthesizer.create(
            "dpctgan",
            epsilon=float(epsilon),
            delta=float(delta),
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            generator_decay=self.generator_decay,
            discriminator_decay=self.discriminator_decay,
            batch_size=self.batch_size,
            epochs=self.epochs,
            pac=self.pac,
            discriminator_steps=self.discriminator_steps,
            sigma=self.sigma,
            max_per_sample_grad_norm=self.max_per_sample_grad_norm,
            loss=self.loss,
            cuda=self.cuda,
            verbose=self.verbose,
            disabled_dp=self.disabled_dp,
        )

        synth.fit(
            df,
            categorical_columns=dpctgan_categorical,
            continuous_columns=dpctgan_continuous,
            preprocessor_eps=float(prep_eps),
            nullable=self.nullable,
        )

        self._synth = synth
        self._columns = list(df.columns)
        self._spec_snapshot = {
            "categorical_cols": list(spec.categorical_cols),
            "numerical_cols": list(spec.numerical_cols),
            "label_col": spec.label_col,
            "sensitive_col": spec.sensitive_col,
            "dpctgan_categorical": dpctgan_categorical,
            "dpctgan_continuous": dpctgan_continuous
        }
        self._fit_meta = {
            "epsilon": epsilon,
            "delta": delta,
            "preprocessor_eps": prep_eps,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        
        # Best effort attempt to record actual spent epsilon string representation from smartnoise if it exists
        if hasattr(synth, "odometer"):
             # Optional: depending on snsynth version it may show how much budget was used
             pass
             
        self._fitted = True
        logger.info("DPCTGAN fitting complete.")

    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample synthetic records."""
        if not self._fitted or self._synth is None:
            raise RuntimeError("Synthesizer must be fitted before calling sample().")

        _set_deterministic_seeds(seed)
        logger.info(f"Generating {n} synthetic records using DPCTGAN...")

        df_syn = self._synth.sample(int(n))
        
        # Ensure column matches
        missing_cols = set(self._columns) - set(df_syn.columns)
        if missing_cols:
            logger.warning(f"Synthetic data is missing columns: {missing_cols}")
            for col in missing_cols:
                df_syn[col] = np.nan # Or some default logic

        extra_cols = set(df_syn.columns) - set(self._columns)
        if extra_cols:
             logger.warning(f"Synthetic data has unexpected columns: {extra_cols}. Dropping them.")
             df_syn = df_syn.drop(columns=list(extra_cols))
             
        # Enforce column order
        df_syn = df_syn[self._columns]

        return df_syn

    def save(self, path: Path) -> None:
        """Save fitted synthesizer metadata and model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "name": self.name,
            "hyperparameters": {
                "embedding_dim": self.embedding_dim,
                "generator_dim": self.generator_dim,
                "discriminator_dim": self.discriminator_dim,
                "generator_lr": self.generator_lr,
                "discriminator_lr": self.discriminator_lr,
                "generator_decay": self.generator_decay,
                "discriminator_decay": self.discriminator_decay,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "pac": self.pac,
                "discriminator_steps": self.discriminator_steps,
                "sigma": self.sigma,
                "max_per_sample_grad_norm": self.max_per_sample_grad_norm,
                "loss": self.loss,
                "preprocessor_eps": self.preprocessor_eps,
                "cuda": self.cuda,
            },
            "columns": self._columns,
            "spec_snapshot": self._spec_snapshot,
            "fit_meta": self._fit_meta
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        if self._synth is not None:
             try:
                 joblib.dump(self._synth, path / "snsynth.joblib", compress=3)
             except Exception as e:
                 logger.warning(f"Failed to serialize smartnoise-synth object to {path / 'snsynth.joblib'}: {e}")

    @classmethod
    def load(cls, path: Path) -> "DPCTGANSynthesizer":
        """Load an existing synthesizer."""
        meta_path = path / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
            
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            
        # Reconstruct
        instance = cls(**metadata.get("hyperparameters", {}))
        instance._columns = metadata.get("columns", [])
        instance._spec_snapshot = metadata.get("spec_snapshot", {})
        instance._fit_meta = metadata.get("fit_meta", {})
        
        model_path = path / "snsynth.joblib"
        if model_path.exists():
             try:
                 instance._synth = joblib.load(model_path)
                 instance._fitted = True
             except Exception as e:
                 logger.warning(f"Failed to load smartnoise-synth object from {model_path}: {e}")
                 instance._fitted = False
        else:
             logger.warning(f"No saved model found at {model_path}. Resampling requires refitting.")
             
        return instance

@register_synthesizer("patectgan")
class PATECTGANSynthesizer(DPCTGANSynthesizer):
    """Synthesizer using PATECTGAN from smartnoise-synth. 
    Shares the exact same interface as DPCTGAN but overrides the synth creation."""
    
    def fit(self, df: pd.DataFrame, spec: DatasetSpec, epsilon: float, delta: float, seed: int) -> None:
        try:
            from snsynth import Synthesizer
        except ImportError:
            raise RuntimeError(
                "The 'smartnoise-synth' library is required for PATECTGANSynthesizer. "
                "Please run `pip install smartnoise-synth`."
            )

        _set_deterministic_seeds(seed)

        categorical_cols = list(spec.categorical_cols)
        numerical_cols = list(spec.numerical_cols)
        label_col = spec.label_col
        sensitive_col = spec.sensitive_col

        if label_col not in categorical_cols and label_col not in numerical_cols:
            categorical_cols.append(label_col)
        if sensitive_col not in categorical_cols:
            categorical_cols.append(sensitive_col)

        dpctgan_categorical = list(dict.fromkeys(categorical_cols))
        dpctgan_continuous = list(dict.fromkeys(numerical_cols))
        dpctgan_categorical = [c for c in dpctgan_categorical if c in df.columns]
        dpctgan_continuous = [c for c in dpctgan_continuous if c in df.columns]

        if self.preprocessor_eps is None:
            prep_eps = min(0.1 * epsilon, 0.9)
        else:
            prep_eps = self.preprocessor_eps
            
        if prep_eps >= epsilon:
             prep_eps = epsilon * 0.1
             logger.warning(f"preprocessor_eps was >= epsilon. Adjusted to {prep_eps}")

        logger.info(
            f"Fitting PATECTGAN (epsilon={epsilon}, delta={delta}, preprocessor_eps={prep_eps}, epochs={self.epochs})..."
        )

        synth = Synthesizer.create(
            "patectgan",
            epsilon=float(epsilon),
            delta=float(delta),
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            discriminator_lr=self.discriminator_lr,
            generator_decay=self.generator_decay,
            discriminator_decay=self.discriminator_decay,
            batch_size=self.batch_size,
            epochs=self.epochs,
            pac=self.pac,
            discriminator_steps=self.discriminator_steps,
            sigma=self.sigma,
            max_per_sample_grad_norm=self.max_per_sample_grad_norm,
            loss=self.loss,
            cuda=self.cuda,
            verbose=self.verbose,
            disabled_dp=self.disabled_dp,
        )

        synth.fit(
            df,
            categorical_columns=dpctgan_categorical,
            continuous_columns=dpctgan_continuous,
            preprocessor_eps=float(prep_eps),
            nullable=self.nullable,
        )

        self._synth = synth
        self._columns = list(df.columns)
        self._spec_snapshot = {
            "categorical_cols": list(spec.categorical_cols),
            "numerical_cols": list(spec.numerical_cols),
            "label_col": spec.label_col,
            "sensitive_col": spec.sensitive_col,
            "dpctgan_categorical": dpctgan_categorical,
            "dpctgan_continuous": dpctgan_continuous
        }
        self._fit_meta = {
            "epsilon": epsilon,
            "delta": delta,
            "preprocessor_eps": prep_eps,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        self._fitted = True
        logger.info("PATECTGAN fitting complete.")
