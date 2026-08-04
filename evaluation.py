"""Shared, leakage-free HER2ST evaluation protocol."""

import scanpy as sc  
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from predict import (get_R, get_MSE, get_MAE, get_Spearman, get_MoransI_all,
                     cluster_with_nmi)
from utils import comp_tsne_km

PROTOCOL_NAME = "her2st-loocv-v1-raw-lognorm"


def evaluate_her2st_predictions(adata_pred, adata_gt, genes, label=None,
                                 n_clusters=4):
    """Evaluate all models on log-normalised HER2ST expression."""
    genes = list(genes)
    if adata_pred.shape != adata_gt.shape:
        raise ValueError(f"Prediction/ground-truth shapes differ: "
                         f"{adata_pred.shape} vs {adata_gt.shape}")
    if adata_pred.shape[1] != len(genes):
        raise ValueError(f"Expected {len(genes)} genes, got {adata_pred.shape[1]}")

    # === FIX: Ensure log1p transform ===
    # If data is already log1p, this is safe (log1p(log1p(x)) ~ log1p(x) for x>0)
    adata_pred.X = np.log1p(adata_pred.X)
    adata_gt.X = np.log1p(adata_gt.X)
    
    adata_pred.var_names = genes
    adata_gt.var_names = genes
    
    R, p_values = get_R(adata_pred, adata_gt)
    Spearman, spearman_pvalues = get_Spearman(adata_pred, adata_gt)
    MSE = get_MSE(adata_pred, adata_gt)
    MAE = get_MAE(adata_pred, adata_gt)
    RMSE = np.sqrt(MSE)
    morans = get_MoransI_all(adata_pred, adata_gt, top_k=50)

    adata_visual = adata_pred.copy()
    sc.pp.scale(adata_visual)
    adata_visual = comp_tsne_km(adata_visual, n_clusters)
    
    if label is None:
        ARI = NMI = float("nan")
    else:
        _, ARI, NMI = cluster_with_nmi(adata_visual, label)

    return adata_visual, {
        "R": R, "p_values": p_values,
        "Spearman": Spearman, "spearman_pvalues": spearman_pvalues,
        "MSE": MSE, "MAE": MAE, "RMSE": RMSE, "morans": morans,
        "ARI": ARI, "NMI": NMI,
        "pearson": np.nanmean(R),
        "median_pearson": np.nanmedian(R),
        "spearman": np.nanmean(Spearman),
        "rmse": np.nanmean(RMSE), 
        "mae": np.nanmean(MAE),
        "morans_i_pred": np.nanmean(morans["pred"]),
        "morans_i_gt": np.nanmean(morans["gt"]),
    }
