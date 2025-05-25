"""
Evaluation metrics module for JibJob recommendation system.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, ndcg_score

logger = logging.getLogger(__name__)

class RecommendationEvaluator:
    """
    Class responsible for evaluating recommendation performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RecommendationEvaluator with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.metrics_config = config.get('evaluation', {})
        
        # Set default k values for top-k metrics
        self.k_values = self.metrics_config.get('k_values', [1, 5, 10, 20])
        
    def calculate_hit_rate(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        k: int = 10
    ) -> float:
        """
        Calculate hit rate at k (HR@k).
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            ground_truth: Dictionary mapping user ID to list of relevant job IDs.
            k: Number of recommendations to consider.
            
        Returns:
            float: HR@k value.
        """
        hits = 0
        total = 0
        
        for user_id, gt_jobs in ground_truth.items():
            if user_id in recommendations:
                rec_jobs = recommendations[user_id][:k]
                # Check if at least one relevant item is in the top-k recommendations
                if any(job_id in gt_jobs for job_id in rec_jobs):
                    hits += 1
                total += 1
                
        if total == 0:
            return 0.0
            
        return hits / total
        
    def calculate_precision(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        k: int = 10
    ) -> float:
        """
        Calculate precision at k (P@k).
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            ground_truth: Dictionary mapping user ID to list of relevant job IDs.
            k: Number of recommendations to consider.
            
        Returns:
            float: P@k value.
        """
        precisions = []
        
        for user_id, gt_jobs in ground_truth.items():
            if user_id in recommendations:
                rec_jobs = recommendations[user_id][:k]
                # Count relevant items in the top-k recommendations
                relevant_count = sum(1 for job_id in rec_jobs if job_id in gt_jobs)
                precisions.append(relevant_count / min(k, len(rec_jobs)) if rec_jobs else 0)
                
        if not precisions:
            return 0.0
            
        return sum(precisions) / len(precisions)
        
    def calculate_recall(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        k: int = 10
    ) -> float:
        """
        Calculate recall at k (R@k).
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            ground_truth: Dictionary mapping user ID to list of relevant job IDs.
            k: Number of recommendations to consider.
            
        Returns:
            float: R@k value.
        """
        recalls = []
        
        for user_id, gt_jobs in ground_truth.items():
            if user_id in recommendations and gt_jobs:  # Only consider users with ground truth
                rec_jobs = recommendations[user_id][:k]
                # Count relevant items in the top-k recommendations
                relevant_count = sum(1 for job_id in rec_jobs if job_id in gt_jobs)
                recalls.append(relevant_count / len(gt_jobs))
                
        if not recalls:
            return 0.0
            
        return sum(recalls) / len(recalls)
        
    def calculate_ndcg(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        k: int = 10
    ) -> float:
        """
        Calculate normalized discounted cumulative gain at k (NDCG@k).
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            ground_truth: Dictionary mapping user ID to list of relevant job IDs.
            k: Number of recommendations to consider.
            
        Returns:
            float: NDCG@k value.
        """
        ndcg_values = []
        
        for user_id, gt_jobs in ground_truth.items():
            if user_id in recommendations and gt_jobs:  # Only consider users with ground truth
                rec_jobs = recommendations[user_id][:k]
                
                # Create relevance scores (1 if item is relevant, 0 otherwise)
                relevance = np.zeros(min(k, len(rec_jobs)))
                for i, job_id in enumerate(rec_jobs):
                    if job_id in gt_jobs:
                        relevance[i] = 1
                
                # Create ideal relevance scores (all relevant items at the top)
                ideal_relevance = np.zeros(min(k, len(gt_jobs)))
                ideal_relevance[:len(gt_jobs)] = 1
                
                # Calculate NDCG using scikit-learn
                if np.sum(relevance) > 0:  # Only calculate if there's at least one relevant item
                    ndcg = ndcg_score([ideal_relevance], [relevance])
                    ndcg_values.append(ndcg)
                else:
                    ndcg_values.append(0.0)
                
        if not ndcg_values:
            return 0.0
            
        return sum(ndcg_values) / len(ndcg_values)
        
    def calculate_map(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        k: int = 10
    ) -> float:
        """
        Calculate Mean Average Precision at k (MAP@k).
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            ground_truth: Dictionary mapping user ID to list of relevant job IDs.
            k: Number of recommendations to consider.
            
        Returns:
            float: MAP@k value.
        """
        ap_values = []
        
        for user_id, gt_jobs in ground_truth.items():
            if user_id in recommendations and gt_jobs:  # Only consider users with ground truth
                rec_jobs = recommendations[user_id][:k]
                
                # Calculate average precision
                hits = 0
                sum_precisions = 0
                
                for i, job_id in enumerate(rec_jobs):
                    if job_id in gt_jobs:
                        hits += 1
                        # Precision at position i+1
                        precision_at_i = hits / (i + 1)
                        sum_precisions += precision_at_i
                
                if hits > 0:
                    ap = sum_precisions / min(len(gt_jobs), k)
                    ap_values.append(ap)
                else:
                    ap_values.append(0.0)
                
        if not ap_values:
            return 0.0
            
        return sum(ap_values) / len(ap_values)
        
    def calculate_diversity(
        self,
        recommendations: Dict[str, List[str]],
        job_categories: Dict[str, str],
        k: int = 10
    ) -> float:
        """
        Calculate recommendation diversity based on categories.
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            job_categories: Dictionary mapping job ID to category ID.
            k: Number of recommendations to consider.
            
        Returns:
            float: Average diversity score.
        """
        diversity_scores = []
        
        for user_id, rec_jobs in recommendations.items():
            rec_jobs = rec_jobs[:k]
            
            # Get categories for recommended jobs
            categories = [job_categories.get(job_id) for job_id in rec_jobs if job_id in job_categories]
            
            # Calculate diversity (unique categories / total categories)
            if categories:
                unique_categories = len(set(categories))
                diversity = unique_categories / len(categories)
                diversity_scores.append(diversity)
                
        if not diversity_scores:
            return 0.0
            
        return sum(diversity_scores) / len(diversity_scores)
        
    def calculate_coverage(
        self,
        recommendations: Dict[str, List[str]],
        all_items: List[str],
        k: int = 10
    ) -> float:
        """
        Calculate catalog coverage.
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            all_items: List of all available job IDs.
            k: Number of recommendations to consider.
            
        Returns:
            float: Coverage score (percentage of catalog recommended).
        """
        # Collect all recommended items
        recommended_items = set()
        for user_id, rec_jobs in recommendations.items():
            rec_jobs = rec_jobs[:k]
            recommended_items.update(rec_jobs)
            
        # Calculate coverage
        if not all_items:
            return 0.0
            
        return len(recommended_items) / len(all_items)
        
    def calculate_popularity_bias(
        self,
        recommendations: Dict[str, List[str]],
        item_popularity: Dict[str, int],
        k: int = 10
    ) -> float:
        """
        Calculate popularity bias in recommendations.
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            item_popularity: Dictionary mapping job ID to its popularity (e.g., interaction count).
            k: Number of recommendations to consider.
            
        Returns:
            float: Average popularity of recommended items.
        """
        popularity_scores = []
        
        for user_id, rec_jobs in recommendations.items():
            rec_jobs = rec_jobs[:k]
            
            # Get popularity scores for recommended jobs
            popularities = [item_popularity.get(job_id, 0) for job_id in rec_jobs]
            
            if popularities:
                avg_popularity = sum(popularities) / len(popularities)
                popularity_scores.append(avg_popularity)
                
        if not popularity_scores:
            return 0.0
            
        return sum(popularity_scores) / len(popularity_scores)
        
    def calculate_serendipity(
        self,
        recommendations: Dict[str, List[str]],
        user_profiles: Dict[str, List[str]],
        job_categories: Dict[str, str],
        k: int = 10
    ) -> float:
        """
        Calculate serendipity (unexpected but valuable recommendations).
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            user_profiles: Dictionary mapping user ID to list of category IDs they are interested in.
            job_categories: Dictionary mapping job ID to category ID.
            k: Number of recommendations to consider.
            
        Returns:
            float: Average serendipity score.
        """
        serendipity_scores = []
        
        for user_id, rec_jobs in recommendations.items():
            if user_id in user_profiles:
                rec_jobs = rec_jobs[:k]
                user_categories = set(user_profiles[user_id])
                
                # Calculate serendipity for each recommendation
                job_scores = []
                for job_id in rec_jobs:
                    if job_id in job_categories:
                        job_category = job_categories[job_id]
                        # If job category is not in user profile categories, it's serendipitous
                        if job_category not in user_categories:
                            job_scores.append(1.0)
                        else:
                            job_scores.append(0.0)
                
                if job_scores:
                    avg_score = sum(job_scores) / len(job_scores)
                    serendipity_scores.append(avg_score)
                
        if not serendipity_scores:
            return 0.0
            
        return sum(serendipity_scores) / len(serendipity_scores)
        
    def evaluate_all_metrics(
        self,
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        job_categories: Dict[str, str] = None,
        user_profiles: Dict[str, List[str]] = None,
        item_popularity: Dict[str, int] = None,
        all_items: List[str] = None
    ) -> Dict[str, Dict[int, float]]:
        """
        Evaluate recommendations using multiple metrics.
        
        Args:
            recommendations: Dictionary mapping user ID to list of recommended job IDs.
            ground_truth: Dictionary mapping user ID to list of relevant job IDs.
            job_categories: Dictionary mapping job ID to category ID.
            user_profiles: Dictionary mapping user ID to list of category IDs they are interested in.
            item_popularity: Dictionary mapping job ID to its popularity.
            all_items: List of all available job IDs.
            
        Returns:
            Dict[str, Dict[int, float]]: Dictionary of metric names to {k: value} dictionaries.
        """
        results = {}
        
        logger.info(f"Evaluating recommendations for {len(recommendations)} users")
        
        # Basic metrics that require only recommendations and ground truth
        for k in self.k_values:
            # Create metrics dictionary if it doesn't exist
            for metric_name in ['hit_rate', 'precision', 'recall', 'ndcg', 'map']:
                if metric_name not in results:
                    results[metric_name] = {}
                    
            # Calculate basic metrics
            results['hit_rate'][k] = self.calculate_hit_rate(recommendations, ground_truth, k)
            results['precision'][k] = self.calculate_precision(recommendations, ground_truth, k)
            results['recall'][k] = self.calculate_recall(recommendations, ground_truth, k)
            results['ndcg'][k] = self.calculate_ndcg(recommendations, ground_truth, k)
            results['map'][k] = self.calculate_map(recommendations, ground_truth, k)
            
            logger.info(f"Metrics at k={k}: "
                      f"HR={results['hit_rate'][k]:.4f}, "
                      f"P={results['precision'][k]:.4f}, "
                      f"R={results['recall'][k]:.4f}, "
                      f"NDCG={results['ndcg'][k]:.4f}, "
                      f"MAP={results['map'][k]:.4f}")
                      
        # Additional metrics that require extra information
        if job_categories:
            results['diversity'] = {}
            for k in self.k_values:
                results['diversity'][k] = self.calculate_diversity(recommendations, job_categories, k)
                logger.info(f"Diversity at k={k}: {results['diversity'][k]:.4f}")
                
        if all_items:
            results['coverage'] = {}
            for k in self.k_values:
                results['coverage'][k] = self.calculate_coverage(recommendations, all_items, k)
                logger.info(f"Coverage at k={k}: {results['coverage'][k]:.4f}")
                
        if item_popularity:
            results['popularity_bias'] = {}
            for k in self.k_values:
                results['popularity_bias'][k] = self.calculate_popularity_bias(recommendations, item_popularity, k)
                logger.info(f"Popularity bias at k={k}: {results['popularity_bias'][k]:.4f}")
                
        if job_categories and user_profiles:
            results['serendipity'] = {}
            for k in self.k_values:
                results['serendipity'][k] = self.calculate_serendipity(
                    recommendations, user_profiles, job_categories, k)
                logger.info(f"Serendipity at k={k}: {results['serendipity'][k]:.4f}")
                
        logger.info("Evaluation complete")
        return results
