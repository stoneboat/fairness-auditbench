"""Runner for synthesizing DP datasets."""

import logging
import time
from pathlib import Path
from typing import Optional

from fairness_auditbench.datasets.folktables_acs import ACSPublicCoverageDataset
from fairness_auditbench.synthesizers.registry import get_synthesizer
import fairness_auditbench.synthesizers.private_pgm  # imported for side-effect of registration
import fairness_auditbench.synthesizers.dp_1way      # imported for side-effect of registration
from fairness_auditbench.utils.io import ensure_dir, save_json

logger = logging.getLogger(__name__)


def run_synthesis(
    dataset: str,
    states: list[str],
    year: int,
    audit_split: str,
    synth_name: str,
    epsilon: Optional[float],
    delta: float,
    seed: int,
    bins: int,
    degree: int,
    out_dir: str,
    fast_dev_run: bool,
    sensitive_col: Optional[str] = None,
    **kwargs,
):
    """Orchestrate dataset loading, DP synthesis, and saving artifacts."""
    logger.info("Starting synthesis runner: %s on %s...", synth_name, dataset)
    
    # 1. Load dataset splits and spec
    if dataset == "acs_public_coverage":
        ds = ACSPublicCoverageDataset(
            states=states,
            year=year,
            sensitive_col=sensitive_col,
            fast_dev_run=fast_dev_run,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    train_df, val_df, test_df, spec = ds.get_splits(seed=seed)
    
    # 2. Choose audit split
    if audit_split == "train":
        audit_df = train_df
    elif audit_split == "val":
        audit_df = val_df
    elif audit_split == "test":
        audit_df = test_df
    else:
        raise ValueError(f"Unknown split '{audit_split}'")
        
    logger.info("Using audit split '%s' with %d rows.", audit_split, len(audit_df))
    
    # 3. Fit synthesizer
    start_time = time.time()
    logger.info("Instantiating synthesizer '%s'...", synth_name)
    
    synth = get_synthesizer(synth_name, bins=bins, degree=degree, max_cardinality=2000, **kwargs)
    synth.fit(audit_df, spec=spec, epsilon=epsilon, delta=delta, seed=seed)
    
    fit_time = time.time() - start_time
    logger.info("Fitting complete in %.2fs.", fit_time)
    
    # 4. Sample
    start_time = time.time()
    n_samples = len(audit_df)
    logger.info("Sampling %d rows...", n_samples)
    synth_df = synth.sample(n=n_samples, seed=seed)
    sample_time = time.time() - start_time
    logger.info("Sampling complete in %.2fs.", sample_time)

    # 5. Save artifacts
    states_tag = "-".join(sorted(states))
    dataset_tag = f"{dataset}_fast" if fast_dev_run else dataset
    
    eps_str = f"eps={epsilon}" if epsilon is not None else "eps=none"
    
    out_path = Path(out_dir) / "synth" / dataset_tag / synth_name / eps_str / f"seed={seed}" / f"audit_split={audit_split}"
    ensure_dir(out_path)
    
    parquet_path = out_path / "synthetic.parquet"
    synth_df.to_parquet(parquet_path, index=False)
    
    param_meta = {
        "dataset": dataset,
        "states": states,
        "year": year,
        "audit_split": audit_split,
        "n_samples": n_samples,
        "synth_name": synth_name,
        "epsilon": epsilon,
        "delta": delta,
        "seed": seed,
        "bins": bins,
        "degree": degree,
        "fast_dev_run": fast_dev_run,
        "fit_time_s": fit_time,
        "sample_time_s": sample_time,
        "spec_summary": spec.to_dict()
    }
    # Include any kwargs in metadata
    param_meta.update(kwargs)
    
    save_json(param_meta, out_path / "metadata.json")
    
    # Optional: Save synthesizer model if supported
    try:
        synth.save(out_path)
    except Exception as e:
        logger.warning(f"Failed to save synthesizer model: {e}")
    
    logger.info("Synthetic dataset and metadata saved to %s", out_path)
    return out_path
