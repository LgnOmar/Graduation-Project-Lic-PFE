"""
Evaluation metrics for recommendation systems.
This module provides functions to evaluate the quality of recommendations.
"""

import numpy as np
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def precision_at_k(predictions: List[List[Any]], ground_truth: List[List[Any]], k: int) -> float:
    """
    Calculate precision@k for recommendations.
    
    Args:
        predictions: List of lists where each inner list contains recommended items for a user.
        ground_truth: List of lists where each inner list contains relevant items for a user.
        k: The number of top recommendations to consider.
        
    Returns:
        float: Precision@k score.
    """
    precision_scores = []
    
    for i, pred in enumerate(predictions):
        if i >= len(ground_truth):
            continue
            
        # Consider only the top k predictions
        pred_k = pred[:k]
        
        # Count relevant items in the top k predictions
        relevant_count = sum(1 for item in pred_k if item in ground_truth[i])
        
        # Calculate precision for this user
        if len(pred_k) > 0:
            precision_scores.append(relevant_count / len(pred_k))
        else:
            precision_scores.append(0.0)
    
    # Average precision across users
    return np.mean(precision_scores) if precision_scores else 0.0


def recall_at_k(predictions: List[List[Any]], ground_truth: List[List[Any]], k: int) -> float:
    """
    Calculate recall@k for recommendations.
    
    Args:
        predictions: List of lists where each inner list contains recommended items for a user.
        ground_truth: List of lists where each inner list contains relevant items for a user.
        k: The number of top recommendations to consider.
        
    Returns:
        float: Recall@k score.
    """
    recall_scores = []
    
    for i, pred in enumerate(predictions):
        if i >= len(ground_truth) or not ground_truth[i]:
            continue
            
        # Consider only the top k predictions
        pred_k = pred[:k]
        
        # Count relevant items in the top k predictions
        relevant_count = sum(1 for item in pred_k if item in ground_truth[i])
        
        # Calculate recall for this user
        recall_scores.append(relevant_count / len(ground_truth[i]))
    
    # Average recall across users
    return np.mean(recall_scores) if recall_scores else 0.0


def ndcg_at_k(predictions: List[List[Any]], ground_truth: List[List[Any]], k: int) -> float:
    """
    Calculate normalized discounted cumulative gain (NDCG) at k.
    
    Args:
        predictions: List of lists where each inner list contains recommended items for a user.
        ground_truth: List of lists where each inner list contains relevant items for a user.
        k: The number of top recommendations to consider.
        
    Returns:
        float: NDCG@k score.
    """
    ndcg_scores = []
    
    for i, pred in enumerate(predictions):
        if i >= len(ground_truth) or not ground_truth[i]:
            continue
        
        # Consider only the top k predictions
        pred_k = pred[:k]
        
        # Create relevance array (1 if item is relevant, 0 otherwise)
        relevance = np.array([1 if item in ground_truth[i] else 0 for item in pred_k])
        
        # Calculate DCG
        discounts = np.log2(np.arange(2, len(relevance) + 2))  # [log2(2), log2(3), ...]
        dcg = np.sum(relevance / discounts)
        
        # Calculate ideal DCG (IDCG)
        # The ideal is to have all relevant items at the top
        n_rel = min(len(ground_truth[i]), k)
        ideal_relevance = np.ones(n_rel)
        ideal_discounts = np.log2(np.arange(2, n_rel + 2))
        idcg = np.sum(ideal_relevance / ideal_discounts)
        
        # Calculate NDCG
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    
    # Average NDCG across users
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE) between actual and predicted ratings.
    
    Args:
        actual: Array of actual ratings.
        predicted: Array of predicted ratings.
        
    Returns:
        float: MAE score.
    """
    if len(actual) == 0:
        return 0.0
    return np.mean(np.abs(actual - predicted))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) between actual and predicted ratings.
    
    Args:
        actual: Array of actual ratings.
        predicted: Array of predicted ratings.
        
    Returns:
        float: RMSE score.
    """
    if len(actual) == 0:
        return 0.0
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_recommendation_metrics(
    recommendations: List[List[Any]],
    ground_truth: List[List[Any]],
    actual_ratings: np.ndarray = None,
    predicted_ratings: np.ndarray = None,
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Calculate only precision, recall, ndcg, MAE, RMSE at different values of k.
    
    Args:
        recommendations: List of lists where each inner list contains recommended items for a user.
        ground_truth: List of lists where each inner list contains relevant items for a user.
        actual_ratings: Array of actual ratings (for MAE, RMSE calculation).
        predicted_ratings: Array of predicted ratings (for MAE, RMSE calculation).
        k_values: List of k values to calculate metrics for.
        
    Returns:
        Dict[str, float]: Dictionary with metrics.
    """
    metrics = {}
    for k in k_values:
        metrics[f'precision@{k}'] = precision_at_k(recommendations, ground_truth, k)
        metrics[f'recall@{k}'] = recall_at_k(recommendations, ground_truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(recommendations, ground_truth, k)
    if actual_ratings is not None and predicted_ratings is not None:
        metrics['mae'] = mean_absolute_error(actual_ratings, predicted_ratings)
        metrics['rmse'] = root_mean_squared_error(actual_ratings, predicted_ratings)
    return metrics
