"""Non-DP CTGAN synthesizer."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

import pandas as pd
import numpy as np

try:
    from ctgan import CTGAN
except ImportError:
    CTGAN = None

from fairness_auditbench.datasets.folktables_acs import DatasetSpec
from fairness_auditbench.synthesizers.base import BaseSynthesizer
from fairness_auditbench.synthesizers.registry import register_synthesizer
from fairness_auditbench.utils.io import save_json, load_json

logger = logging.getLogger(__name__)

def _set_deterministic_seeds(seed: int) -> None:
    import torch
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@register_synthesizer("ctgan")
class CTGANSynthesizer(BaseSynthesizer):
    """
    Wrapper for non-DP CTGAN from the SDV ecosystem.
    NOTE: This synthesizer provides NO differential privacy guarantees.
    Epsilon and delta parameters are ignored during fit.
    """
    
    name: str = "ctgan"
    
    def __init__(
        self,
        embedding_dim: int = 128,
        generator_dim: tuple = (256, 256),
        discriminator_dim: tuple = (256, 256),
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        generator_decay: float = 1e-6,
        discriminator_decay: float = 1e-6,
        batch_size: int = 500,
        epochs: int = 300,
        pac: int = 10,
        discriminator_steps: int = 1,
        log_frequency: bool = True,
        enable_gpu: bool = True,
        verbose: bool = False,
        seed: int = 0,
        **kwargs
    ):
        if CTGAN is None:
            raise ImportError(
                "CTGAN is not installed. Please install it with 'pip install ctgan'."
            )
            
        self.embedding_dim = embedding_dim
        
        if isinstance(generator_dim, str):
            self.generator_dim = tuple(int(x.strip()) for x in generator_dim.split(","))
        else:
            self.generator_dim = generator_dim
            
        if isinstance(discriminator_dim, str):
            self.discriminator_dim = tuple(int(x.strip()) for x in discriminator_dim.split(","))
        else:
            self.discriminator_dim = discriminator_dim

        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.generator_decay = generator_decay
        self.discriminator_decay = discriminator_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.pac = pac
        self.discriminator_steps = discriminator_steps
        self.log_frequency = log_frequency
        self.verbose = verbose
        self.cuda = enable_gpu
        self.seed = seed
        
        self.spec: Optional[DatasetSpec] = None
        
        self._ctgan = None
        self._numeric_medians = {}
        self._columns = []
        self._dtypes = {}
        
    def fit(self, df: pd.DataFrame, spec: DatasetSpec, epsilon: float, delta: float, seed: int) -> None:
        """Fit the non-DP CTGAN model to the data."""
        self.spec = spec
        _set_deterministic_seeds(seed)
        
        df_prepped = df.copy()
        self._columns = list(df.columns)
        self._dtypes = df.dtypes.to_dict()
        
        # Build discrete column list ensuring they actually exist in the dataframe
        discrete_set = set(spec.categorical_cols)
        if spec.label_col:
            discrete_set.add(spec.label_col)
        if spec.sensitive_col:
            discrete_set.add(spec.sensitive_col)
            
        discrete_columns = list(discrete_set & set(df_prepped.columns))
        
        # 1. Handle Categorical columns (fill missing with __MISSING__)
        for col in discrete_columns:
            df_prepped[col] = df_prepped[col].fillna("__MISSING__").astype(str)

        # 2. Handle Numeric columns (median imputation)
        for col in spec.numerical_cols:
            if col in df_prepped.columns:
                median_val = df_prepped[col].median()
                if pd.isna(median_val):
                    median_val = 0.0 # Fallback if entirely NaNs
                self._numeric_medians[col] = float(median_val)
                df_prepped[col] = df_prepped[col].fillna(median_val).astype(np.float32)

        # NOTE: ignores epsilon and delta!
        self._ctgan = CTGAN(
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=self.generator_lr,
            generator_decay=self.generator_decay,
            discriminator_lr=self.discriminator_lr,
            discriminator_decay=self.discriminator_decay,
            batch_size=self.batch_size,
            discriminator_steps=self.discriminator_steps,
            log_frequency=self.log_frequency,
            verbose=self.verbose,
            epochs=self.epochs,
            pac=self.pac,
            cuda=self.cuda
        )
        
        if hasattr(self._ctgan, 'set_random_state'):
            self._ctgan.set_random_state(seed)
            
        logger.info(f"Fitting non-DP CTGAN with {self.epochs} epochs & batch_size {self.batch_size}...")
        self._ctgan.fit(df_prepped, discrete_columns=discrete_columns)
        
    def sample(self, n: int, seed: int) -> pd.DataFrame:
        """Sample from the trained CTGAN model."""
        if self._ctgan is None:
            raise RuntimeError("Synthesizer has not been fitted yet.")
            
        _set_deterministic_seeds(seed)
        if hasattr(self._ctgan, 'set_random_state'):
            self._ctgan.set_random_state(seed)
            
        df_syn = self._ctgan.sample(n)
        
        # Postprocess: reorder columns and restore types
        for col in self._columns:
            if col not in df_syn.columns:
                df_syn[col] = np.nan
                
        df_syn = df_syn[self._columns].copy()
        
        for col, dtype in self._dtypes.items():
            if col not in df_syn.columns:
                continue
            try:
                if dtype == bool:
                    df_syn[col] = df_syn[col].map(lambda x: True if str(x).lower() in ['true', '1', '1.0'] else False)
                else:
                    df_syn[col] = df_syn[col].astype(dtype)
            except Exception as e:
                logger.debug(f"Could not cast {col} to {dtype}: {e}")
                
        return df_syn

    def save(self, path: Path) -> None:
        if self._ctgan is None:
            raise RuntimeError("Model not fitted.")
            
        import joblib
        
        ctgan_path = path / "ctgan_model.pkl"
        try:
            self._ctgan.save(str(ctgan_path))
        except Exception as e:
            logger.warning(f"CTGAN save failed, falling back to joblib: {e}")
            joblib.dump(self._ctgan, ctgan_path)
            
        metadata = {
            "name": self.name,
            "numeric_medians": self._numeric_medians,
            "columns": self._columns,
            "dtypes": {k: str(v) for k, v in self._dtypes.items()},
            "privacy": {
                "is_dp": False,
                "epsilon": None,
                "delta": None,
                "note": "non-DP CTGAN (ctgan package). Provided epsilon/delta ignored."
            }
        }
        save_json(metadata, path / "ctgan_metadata.json")

    @classmethod
    def load(cls, path: Path) -> "CTGANSynthesizer":
        instance = cls()
        metadata = load_json(path / "ctgan_metadata.json")
        instance._numeric_medians = metadata.get("numeric_medians", {})
        instance._columns = metadata.get("columns", [])
        
        # Load model logic
        ctgan_path = path / "ctgan_model.pkl"
        try:
            instance._ctgan = CTGAN.load(str(ctgan_path))
        except Exception as e:
            logger.warning(f"CTGAN.load failed, attempting joblib: {e}")
            try:
                import joblib
                instance._ctgan = joblib.load(ctgan_path)
            except Exception as e2:
                logger.error(f"Failed to load CTGAN model natively or via joblib: {e2}")
                raise
            
        return instance
