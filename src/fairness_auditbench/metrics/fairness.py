"""Fairness metrics computation."""

import numpy as np
import pandas as pd
from typing import Dict, Any

def compute_fairness_metrics(
    df: pd.DataFrame, 
    y_true_col: str, 
    y_pred_col: str, 
    sensitive_col: str
) -> Dict[str, Any]:
    """Compute fairness metrics for multi-group sensitive attributes.
    
    Args:
        df: DataFrame containing the true labels, predictions, and sensitive groups.
        y_true_col: Column name of true labels (binary).
        y_pred_col: Column name of predicted probabilities or binary predictions.
        sensitive_col: Column name of the sensitive attribute.
        
    Returns:
        Dictionary of fairness metrics and group statistics.
    """
    groups = df[sensitive_col].unique()
    
    # We'll binarize predictions if they are probabilities (threshold=0.5)
    if pd.api.types.is_float_dtype(df[y_pred_col]):
        df["_y_pred_bin"] = (df[y_pred_col] >= 0.5).astype(int)
    else:
        df["_y_pred_bin"] = df[y_pred_col].astype(int)
        
    y_true = df[y_true_col].astype(int)
    y_pred = df["_y_pred_bin"]
    
    group_stats = {}
    pos_rates = []
    tprs = []
    fprs = []
    
    for g in sorted(groups):
        mask = df[sensitive_col] == g
        n_group = int(mask.sum())
        
        if n_group == 0:
            continue
            
        g_y_true = y_true[mask]
        g_y_pred = y_pred[mask]
        
        n_pos = int(g_y_true.sum())
        n_neg = int(n_group - n_pos)
        
        # Positive prediction rate P(Y_hat=1 | A=g)
        pr = g_y_pred.mean()
        pos_rates.append(pr)
        
        # True Positive Rate P(Y_hat=1 | Y=1, A=g)
        if n_pos > 0:
            tpr = g_y_pred[g_y_true == 1].mean()
            tprs.append(tpr)
        else:
            tpr = np.nan
            
        # False Positive Rate P(Y_hat=1 | Y=0, A=g)
        if n_neg > 0:
            fpr = g_y_pred[g_y_true == 0].mean()
            fprs.append(fpr)
        else:
            fpr = np.nan
            
        group_stats[str(g)] = {
            "n": n_group,
            "positives": n_pos,
            "negatives": n_neg,
            "pr": float(pr),
            "tpr": float(tpr),
            "fpr": float(fpr)
        }
        
    # Drop NaNs for diff computation
    valid_tprs = [x for x in tprs if not np.isnan(x)]
    valid_fprs = [x for x in fprs if not np.isnan(x)]
    
    dp_diff = np.max(pos_rates) - np.min(pos_rates) if pos_rates else 0.0
    
    eo_diff = np.max(valid_tprs) - np.min(valid_tprs) if valid_tprs else 0.0
    
    if valid_tprs and valid_fprs:
        tpr_gap = np.max(valid_tprs) - np.min(valid_tprs)
        fpr_gap = np.max(valid_fprs) - np.min(valid_fprs)
        eq_odds_diff = max(tpr_gap, fpr_gap)
    else:
        eq_odds_diff = 0.0
        
    return {
        "demographic_parity_score": float(dp_diff),
        "equal_opportunity_score": float(eo_diff),
        "equalized_odds_score": float(eq_odds_diff),
        "groups": group_stats
    }
