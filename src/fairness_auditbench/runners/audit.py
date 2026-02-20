"""Runner for auditing fairness of a model on real vs synthetic data."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import torch

from fairness_auditbench.datasets.folktables_acs import ACSPublicCoverageDataset
from fairness_auditbench.metrics.fairness import compute_fairness_metrics
from fairness_auditbench.utils.io import ensure_dir

logger = logging.getLogger(__name__)

def evaluate_model(df: pd.DataFrame, model_type: str, model_dir: Path, spec) -> pd.DataFrame:
    """Load model artifacts and generated predictions."""
    df = df.copy()
    
    if model_type == "logreg":
        # Load sklearn pipeline
        model_path = model_dir / "pipeline.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Logistic Regression model not found at {model_path}")
            
        pipeline = joblib.load(model_path)
        
        # We need to drop label and sensitive cols if they were not in the training features.
        # Assuming the pipeline expects just X categorical/numerical.
        X = df.drop(columns=[spec.label_col, spec.sensitive_col], errors="ignore")
        
        # predict_proba returns [P(y=0), P(y=1)]
        preds = pipeline.predict_proba(X)[:, 1]
        df["_y_pred"] = preds
        
    elif model_type == "ft_transformer":
        # Load model and preprocessor
        from fairness_auditbench.models.ft_transformer import _FTTransformerNet
        import json
        
        preprocessor_path = model_dir / "preprocessor.joblib"
        hparams_path = model_dir / "hparams.json"
        model_path = model_dir / "model.pt"
        
        if not all(p.exists() for p in [preprocessor_path, hparams_path, model_path]):
            raise FileNotFoundError(f"FT-Transformer artifacts not found at {model_dir}")
            
        preprocessor = joblib.load(preprocessor_path)
        with open(hparams_path, "r") as f:
            hparams = json.load(f)
            
        net = _FTTransformerNet(**hparams)
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
        net.eval()
        
        # Transform data using preprocessor
        cat_codes, num_values = preprocessor.transform(df)
        
        # Convert to tensors
        cat_t = torch.from_numpy(cat_codes).long()
        num_t = torch.from_numpy(num_values).float()
        
        # Predict
        with torch.no_grad():
            logits = net(cat_t, num_t)
            probs = torch.sigmoid(logits).numpy()
            
        df["_y_pred"] = probs
        
    else:
        raise ValueError(f"Unknown model type '{model_type}'")
        
    return df

def run_audit(
    dataset: str,
    states: list[str],
    year: int,
    model_type: str,
    model_seed: int,
    audit_split: str,
    synth_name: str,
    epsilon: float,
    synth_seed: int,
    out_dir: str,
    fast_dev_run: bool,
    sensitive_col: Optional[str] = None,
):
    """Run fairness audit on real vs synthetic data and append results to metrics.jsonl."""
    logger.info("Starting fairness audit using model '%s' on %s data...", model_type, "synthetic " + synth_name)
    
    # 1. Load real metadata / data
    if dataset == "acs_public_coverage":
        ds = ACSPublicCoverageDataset(
            states=states,
            year=year,
            sensitive_col=sensitive_col,
            fast_dev_run=fast_dev_run,
        )
    else:
        raise ValueError(f"Unknown dataset '{dataset}'")

    train_df, val_df, test_df, spec = ds.get_splits(seed=model_seed)
    
    if audit_split == "train":
        real_df = train_df
    elif audit_split == "val":
        real_df = val_df
    elif audit_split == "test":
        real_df = test_df
    else:
        raise ValueError(f"Unknown split '{audit_split}'")
        
    # 2. Find and load model direction (Path must match train_model.py where it's saved as: models/{dataset}/{model_type}/seed={seed})
    # train_model saves without fast tags and without state/year
    dataset_tag = f"{dataset}_fast" if fast_dev_run else dataset
    model_dir = Path(out_dir) / "models" / dataset / model_type / f"seed={model_seed}"
    
    # 3. Predict on Real Data
    logger.info("Evaluating on REAL audit data...")
    real_preds_df = evaluate_model(real_df, model_type, model_dir, spec)
    metrics_real = compute_fairness_metrics(
        real_preds_df, 
        y_true_col=spec.label_col, 
        y_pred_col="_y_pred", 
        sensitive_col=spec.sensitive_col
    )
    
    # 4. Load Synthetic Data
    synth_dir = Path(out_dir) / "synth" / dataset_tag / synth_name / f"eps={epsilon}" / f"seed={synth_seed}" / f"audit_split={audit_split}"
    synth_path = synth_dir / "synthetic.parquet"
    
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic data not found at {synth_path}. Run synthesis first.")
        
    synth_df = pd.read_parquet(synth_path)
    
    metadata_path = synth_dir / "metadata.json"
    synth_meta = {}
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            synth_meta = json.load(f)
    
    # 5. Predict on Synthetic Data
    logger.info("Evaluating on SYNTHETIC audit data...")
    synth_preds_df = evaluate_model(synth_df, model_type, model_dir, spec)
    metrics_synth = compute_fairness_metrics(
        synth_preds_df,
        y_true_col=spec.label_col,
        y_pred_col="_y_pred",
        sensitive_col=spec.sensitive_col
    )
    
    # 6. Compute Audit Error
    audit_error = {
        k: abs(metrics_real.get(k, 0.0) - metrics_synth.get(k, 0.0))
        for k in ["demographic_parity_score", "equal_opportunity_score", "equalized_odds_score"]
    }
    
    # 7. Package structured metrics for logging
    log_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": dataset,
        "states": states,
        "year": year,
        "fast_dev_run": fast_dev_run,
        "model_name": model_type,
        "model_seed": model_seed,
        "synth_name": synth_name,
        "epsilon": epsilon,
        "synth_seed": synth_seed,
        "bins": synth_meta.get("bins"),
        "degree": synth_meta.get("degree"),
        "audit_split": audit_split,
        "audit_error": audit_error,
        "metrics_real": metrics_real,
        "metrics_synth": metrics_synth,
        "model_path": str(model_dir),
        "synth_path": str(synth_path)
    }
    
    # 8. Append to JSONL
    out_dir_path = Path(out_dir)
    ensure_dir(out_dir_path)
    metrics_log_path = out_dir_path / "metrics.jsonl"
    
    with open(metrics_log_path, "a") as f:
        f.write(json.dumps(log_record) + "\n")
        
    logger.info("Audit complete. Error in Equalized Odds Score: %.4f", audit_error['equalized_odds_score'])
    logger.info("Appended results to %s", metrics_log_path)
