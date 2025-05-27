"""
metrics.py

Purpose:
    Compute ranking and classification metrics for recommendation evaluation.

Key Functions:
    - precision_at_k(y_true, y_score, k)
    - recall_at_k(y_true, y_score, k)
    - ndcg_at_k(y_true, y_score, k)
    - auc_roc(y_true, y_score)
    - mae(y_true, y_score)
    - rmse(y_true, y_score)

Inputs:
    - y_true: Ground truth binary labels (1 for relevant, 0 for not).
    - y_score: Model scores for each candidate.
    - k: Top-K cutoff for ranking metrics.

Outputs:
    - Metric values (float).

High-Level Logic:
    1. Sort candidates by score.
    2. Compute metrics at cutoff K.
    3. For AUC, use sklearn's roc_auc_score.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

def precision_at_k(y_true, y_score, k):
    idx = np.argsort(y_score)[::-1][:k]
    return np.sum(np.array(y_true)[idx]) / k

def recall_at_k(y_true, y_score, k):
    idx = np.argsort(y_score)[::-1][:k]
    return np.sum(np.array(y_true)[idx]) / np.sum(y_true)

def ndcg_at_k(y_true, y_score, k):
    idx = np.argsort(y_score)[::-1][:k]
    dcg = np.sum((2**np.array(y_true)[idx] - 1) / np.log2(np.arange(2, k + 2)))
    ideal = np.sum((2**np.sort(y_true)[::-1][:k] - 1) / np.log2(np.arange(2, k + 2)))
    return dcg / ideal if ideal > 0 else 0.0

def auc_roc(y_true, y_score):
    if len(set(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)

def mae(y_true, y_score):
    """Calculates Mean Absolute Error."""
    return mean_absolute_error(y_true, y_score)

def rmse(y_true, y_score):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_score))
