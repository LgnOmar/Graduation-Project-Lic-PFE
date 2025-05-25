"""
A/B testing framework for the JibJob recommendation system.

This module provides tools for conducting A/B tests on recommendation algorithms 
and evaluating their performance against control groups.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import torch
import pickle
import logging
import tempfile
import shutil
import json
import random
from typing import List, Dict, Tuple, Any, Callable
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jibjob_recommender_system.data_handling.data_loader import DataLoader
from jibjob_recommender_system.feature_engineering.graph_features import GraphFeatureProcessor
from jibjob_recommender_system.graph_construction.graph_builder import GraphBuilder
from jibjob_recommender_system.models.gcn_model import HeteroGCNLinkPredictor
from jibjob_recommender_system.training.trainer import ModelTrainer
from jibjob_recommender_system.inference.recommender_service import JobRecommender
from jibjob_recommender_system.evaluation.evaluation import calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ABTestingFramework:
    """Framework for conducting A/B tests on recommendation models."""
    
    def __init__(self, data_dir: str, model_dir: str):
        """
        Initialize the A/B testing framework.
        
        Args:
            data_dir: Directory containing data files
            model_dir: Directory for storing model files
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize variant trackers
        self.variants = {}
        self.variant_assignments = {}
        self.variant_results = {}
    
    def add_variant(self, name: str, model_factory: Callable, params: Dict[str, Any] = None):
        """
        Add a variant to the A/B test.
        
        Args:
            name: Name of the variant
            model_factory: Function that creates and returns a model
            params: Parameters to pass to the model factory
        """
        self.variants[name] = {
            'model_factory': model_factory,
            'params': params or {}
        }
        logger.info(f"Added variant '{name}' to A/B test")
    
    def assign_users_to_variants(self, users: List[str], weights: List[float] = None):
        """
        Assign users to variants using weighted random assignment.
        
        Args:
            users: List of user IDs
            weights: Optional weights for each variant (default: equal weights)
        """
        variant_names = list(self.variants.keys())
        n_variants = len(variant_names)
        
        if not weights:
            weights = [1/n_variants] * n_variants
        
        # Normalize weights
        total = sum(weights)
        weights = [w/total for w in weights]
        
        # Assign users to variants
        self.variant_assignments = {}
        for user_id in users:
            variant = random.choices(variant_names, weights=weights, k=1)[0]
            self.variant_assignments[user_id] = variant
        
        # Count assignments
        counts = {name: 0 for name in variant_names}
        for variant in self.variant_assignments.values():
            counts[variant] += 1
        
        logger.info(f"Assigned {len(users)} users to {n_variants} variants: {counts}")
    
    def train_variants(self, data_loader: DataLoader, graph_builder: GraphBuilder, trainer_params: Dict[str, Any] = None):
        """
        Train all variants in the A/B test.
        
        Args:
            data_loader: DataLoader instance
            graph_builder: GraphBuilder instance
            trainer_params: Parameters for the trainer
        
        Returns:
            Dict of variant names to trained model paths
        """
        trainer_params = trainer_params or {}
        model_paths = {}
        
        # Load data
        users_df, jobs_df, interactions_df = data_loader.load_data()
        
        # Load job embeddings
        with open(os.path.join(self.data_dir, 'job_embeddings.pkl'), 'rb') as f:
            job_embeddings = pickle.load(f)
        
        # Build graph
        graph, job_id_mapping, user_id_mapping = graph_builder.build_graph(
            users_df, jobs_df, interactions_df, job_embeddings
        )
        
        # Train each variant
        for name, variant in self.variants.items():
            logger.info(f"Training variant '{name}'")
            
            # Create model
            model = variant['model_factory'](**variant['params'], 
                                          node_types=list(graph.x_dict.keys()),
                                          edge_types=list(graph.edge_index_dict.keys()))
            
            # Create trainer
            variant_model_dir = os.path.join(self.model_dir, name)
            os.makedirs(variant_model_dir, exist_ok=True)
            
            trainer = ModelTrainer(
                model=model,
                graph=graph,
                output_dir=variant_model_dir
            )
            
            # Train model
            results = trainer.train(**trainer_params)
            model_paths[name] = results['best_model_path']
            
            # Save results
            self.variant_results[name] = results
            
            logger.info(f"Variant '{name}' trained with metrics: AUC={results['test_auc']:.4f}")
        
        return model_paths
    
    def evaluate_recommendations(self, recommenders: Dict[str, JobRecommender], test_users: List[str], top_n: int = 10):
        """
        Evaluate recommendations from each variant.
        
        Args:
            recommenders: Dict of variant names to recommender instances
            test_users: List of user IDs for testing
            top_n: Number of recommendations to generate
            
        Returns:
            Dict of variant evaluation metrics
        """
        variant_metrics = {}
        
        for variant_name, recommender in recommenders.items():
            logger.info(f"Evaluating recommendations for variant '{variant_name}'")
            
            # Generate recommendations for each user
            all_user_recs = []
            for user_id in test_users:
                try:
                    recs = recommender.get_recommendations(user_id, top_n=top_n)
                    all_user_recs.append({
                        'user_id': user_id,
                        'recommendations': recs
                    })
                except Exception as e:
                    logger.warning(f"Error generating recommendations for user {user_id} with variant {variant_name}: {e}")
            
            # Calculate metrics
            avg_score = 0
            category_diversity = {}
            location_diversity = {}
            score_distribution = []
            
            if all_user_recs:
                # Load job data for diversity calculation
                jobs_df = pd.read_csv(os.path.join(self.data_dir, 'jobs_df.csv'))
                
                for user_rec in all_user_recs:
                    # Collect scores
                    scores = [score for _, score in user_rec['recommendations']]
                    avg_score += sum(scores) / len(scores) if scores else 0
                    score_distribution.extend(scores)
                    
                    # Collect categories
                    job_ids = [job_id for job_id, _ in user_rec['recommendations']]
                    job_info = jobs_df[jobs_df['job_id'].isin(job_ids)]
                    
                    # Category diversity
                    for category in job_info['categorie'].unique():
                        if category in category_diversity:
                            category_diversity[category] += 1
                        else:
                            category_diversity[category] = 1
                    
                    # Location diversity
                    for location in job_info['ville'].unique():
                        if location in location_diversity:
                            location_diversity[location] += 1
                        else:
                            location_diversity[location] = 1
                
                # Calculate average score
                avg_score /= len(all_user_recs)
                
                # Normalize diversity counts
                total_categories = sum(category_diversity.values())
                category_diversity = {k: v/total_categories for k, v in category_diversity.items()}
                
                total_locations = sum(location_diversity.values())
                location_diversity = {k: v/total_locations for k, v in location_diversity.items()}
                
                # Calculate score distribution stats
                score_mean = np.mean(score_distribution)
                score_std = np.std(score_distribution)
                score_min = np.min(score_distribution)
                score_max = np.max(score_distribution)
                
                # Save metrics
                variant_metrics[variant_name] = {
                    'avg_score': avg_score,
                    'category_diversity': category_diversity,
                    'location_diversity': location_diversity,
                    'score_distribution': {
                        'mean': score_mean,
                        'std': score_std,
                        'min': score_min,
                        'max': score_max
                    },
                    'num_successful_users': len(all_user_recs),
                    'coverage': len(all_user_recs) / len(test_users) if test_users else 0
                }
                
                logger.info(f"Variant '{variant_name}' metrics: Avg score={avg_score:.4f}, " + 
                           f"Coverage={variant_metrics[variant_name]['coverage']:.2f}")
            else:
                logger.warning(f"No successful recommendations for variant '{variant_name}'")
                variant_metrics[variant_name] = {
                    'avg_score': 0,
                    'category_diversity': {},
                    'location_diversity': {},
                    'score_distribution': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
                    'num_successful_users': 0,
                    'coverage': 0
                }
        
        return variant_metrics
    
    def simulate_user_feedback(self, variant_recommenders: Dict[str, JobRecommender], users_df: pd.DataFrame):
        """
        Simulate user feedback based on the variant assignments and recommendations.
        
        Args:
            variant_recommenders: Dict of variant names to recommender instances
            users_df: DataFrame of user information
            
        Returns:
            DataFrame of simulated interaction data
        """
        new_interactions = []
        
        # Get users with variant assignments
        assigned_users = self.variant_assignments.keys()
        
        # Load jobs data for category matching
        jobs_df = pd.read_csv(os.path.join(self.data_dir, 'jobs_df.csv'))
        
        # For each assigned user
        for user_id in assigned_users:
            if user_id not in users_df['user_id'].values:
                continue
                
            variant = self.variant_assignments[user_id]
            recommender = variant_recommenders.get(variant)
            
            if not recommender:
                continue
                
            try:
                # Get recommendations
                recommendations = recommender.get_recommendations(user_id, top_n=5)
                
                # Get user preferences (using category as a proxy)
                user_category = users_df.loc[users_df['user_id'] == user_id, 'categorie'].iloc[0]
                
                for job_id, score in recommendations:
                    # Check if job matches user's category (proxy for relevance)
                    job_category = jobs_df.loc[jobs_df['job_id'] == job_id, 'categorie'].iloc[0] \
                        if job_id in jobs_df['job_id'].values else None
                    
                    # Simulate rating based on category match and score
                    if job_category == user_category:
                        # Higher rating for matching categories
                        rating = min(5, max(1, int(4 + score)))
                        interaction_type = 'apply'
                    else:
                        # Lower rating for non-matching categories
                        rating = min(4, max(1, int(2 + score)))
                        interaction_type = 'view'
                    
                    # Add some randomness
                    if random.random() < 0.7:  # 70% chance of interacting
                        new_interactions.append({
                            'user_id': user_id,
                            'job_id': job_id,
                            'rating': rating,
                            'interaction_type': interaction_type,
                            'variant': variant,
                            'commentaire_texte_anglais': f"Simulated feedback from variant {variant}"
                        })
            except Exception as e:
                logger.warning(f"Error simulating feedback for user {user_id}: {e}")
        
        return pd.DataFrame(new_interactions) if new_interactions else None
    
    def compare_variants(self, metrics: Dict[str, Dict]):
        """
        Compare variants and determine the winner.
        
        Args:
            metrics: Dict of variant metrics
            
        Returns:
            Dict with comparison results and winner
        """
        if not metrics:
            return {'winner': None, 'reason': 'No metrics available'}
        
        # Define scoring weights for different metrics
        weights = {
            'avg_score': 0.4,
            'coverage': 0.3,
            'category_diversity_entropy': 0.15,
            'location_diversity_entropy': 0.15
        }
        
        scores = {}
        
        for variant, variant_metrics in metrics.items():
            # Calculate diversity entropy (higher is better)
            category_entropy = self._calculate_entropy(variant_metrics['category_diversity'])
            location_entropy = self._calculate_entropy(variant_metrics['location_diversity'])
            
            # Calculate weighted score
            score = (
                weights['avg_score'] * variant_metrics['avg_score'] +
                weights['coverage'] * variant_metrics['coverage'] +
                weights['category_diversity_entropy'] * category_entropy +
                weights['location_diversity_entropy'] * location_entropy
            )
            
            scores[variant] = {
                'total_score': score,
                'avg_score': variant_metrics['avg_score'],
                'coverage': variant_metrics['coverage'],
                'category_entropy': category_entropy,
                'location_entropy': location_entropy
            }
        
        # Find winner
        winner = max(scores, key=lambda k: scores[k]['total_score'])
        
        comparison = {
            'scores': scores,
            'winner': winner,
            'reason': f"Highest total score: {scores[winner]['total_score']:.4f}"
        }
        
        logger.info(f"A/B test winner: {winner} ({comparison['reason']})")
        
        return comparison
    
    @staticmethod
    def _calculate_entropy(distribution: Dict):
        """Calculate entropy of a distribution (measure of diversity)."""
        values = list(distribution.values())
        total = sum(values)
        
        if total == 0:
            return 0
            
        probabilities = [v/total for v in values]
        entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probabilities)
        return entropy


class TestABTesting(unittest.TestCase):
    """Test class for A/B testing framework."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_path = os.path.join(cls.temp_dir, 'data')
        cls.model_path = os.path.join(cls.temp_dir, 'models')
        
        # Create necessary directories
        os.makedirs(cls.data_path, exist_ok=True)
        os.makedirs(cls.model_path, exist_ok=True)
        
        # Create sample data
        cls._create_sample_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_sample_data(cls):
        """Create and save sample data for testing."""
        # Create sample users
        users_df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(30)],
            'age': np.random.randint(18, 60, 30),
            'ville': np.random.choice(['Alger', 'Oran', 'Constantine', 'Annaba', 'Setif'], 30),
            'categorie': np.random.choice(['Informatique', 'Plomberie', 'Electricité', 'Beauté', 'Transport'], 30),
            'experience': np.random.choice(['1-3 ans', '3-5 ans', '5+ ans', 'Débutant'], 30)
        })
        
        # Create sample jobs
        jobs_df = pd.DataFrame({
            'job_id': [f'job_{i}' for i in range(50)],
            'titre_emploi': [f'Job title {i}' for i in range(50)],
            'description_mission_anglais': [
                f'This is a sample job description for job {i}. It requires skills in various domains.' 
                for i in range(50)
            ],
            'ville': np.random.choice(['Alger', 'Oran', 'Constantine', 'Annaba', 'Setif'], 50),
            'categorie': np.random.choice(['Informatique', 'Plomberie', 'Electricité', 'Beauté', 'Transport'], 50)
        })
        
        # Create sample interactions
        interactions = []
        for i in range(100):
            user_idx = np.random.randint(0, 30)
            job_idx = np.random.randint(0, 50)
            rating = np.random.choice([1, 2, 3, 4, 5])
            interactions.append({
                'user_id': f'user_{user_idx}',
                'job_id': f'job_{job_idx}',
                'rating': rating,
                'commentaire_texte_anglais': f'Sample comment for job {job_idx} by user {user_idx}.'
            })
        
        interactions_df = pd.DataFrame(interactions)
        
        # Save data to files
        users_df.to_csv(os.path.join(cls.data_path, 'users_df.csv'), index=False)
        jobs_df.to_csv(os.path.join(cls.data_path, 'jobs_df.csv'), index=False)
        interactions_df.to_csv(os.path.join(cls.data_path, 'interactions_df.csv'), index=False)
        
        # Create mock job embeddings (for faster testing)
        job_embeddings = {job_id: np.random.randn(768) for job_id in jobs_df['job_id']}
        with open(os.path.join(cls.data_path, 'job_embeddings.pkl'), 'wb') as f:
            pickle.dump(job_embeddings, f)
        
        # Create processed interactions file
        processed_interactions = interactions_df.copy()
        processed_interactions['enhanced_rating'] = processed_interactions['rating'] * (1 + np.random.randn(len(processed_interactions)) * 0.1)
        processed_interactions.to_csv(os.path.join(cls.data_path, 'processed_interactions.csv'), index=False)
    
    def _create_model_factory(self, hidden_channels, dropout):
        """Create a model factory function with specific parameters."""
        def factory(node_types, edge_types, **kwargs):
            return HeteroGCNLinkPredictor(
                node_types=node_types,
                edge_types=edge_types,
                hidden_channels=hidden_channels,
                num_layers=2,
                dropout=dropout
            )
        return factory
    
    def test_abtest_framework(self):
        """Test the A/B testing framework."""
        # Initialize framework
        ab_framework = ABTestingFramework(data_dir=self.data_path, model_dir=self.model_path)
        
        # Add variants
        ab_framework.add_variant(
            name='control',
            model_factory=self._create_model_factory(hidden_channels=32, dropout=0.1)
        )
        ab_framework.add_variant(
            name='variant_a',
            model_factory=self._create_model_factory(hidden_channels=64, dropout=0.2)
        )
        
        # Load data
        data_loader = DataLoader(self.data_path)
        users_df, jobs_df, interactions_df = data_loader.load_data()
        
        # Process features
        feature_processor = GraphFeatureProcessor(self.data_path)
        interactions_df = feature_processor.process_interactions(interactions_df)
        
        # Load job embeddings
        with open(os.path.join(self.data_path, 'job_embeddings.pkl'), 'rb') as f:
            job_embeddings = pickle.load(f)
        
        # Assign users to variants
        users = users_df['user_id'].tolist()
        ab_framework.assign_users_to_variants(users, weights=[0.5, 0.5])
        
        # Create graph builder
        graph_builder = GraphBuilder()
        
        # Train variants (minimal training for testing)
        trainer_params = {
            'num_epochs': 2,
            'batch_size': 32,
            'learning_rate': 0.01,
            'val_ratio': 0.2,
            'test_ratio': 0.2,
            'early_stopping_patience': 2
        }
        
        model_paths = ab_framework.train_variants(data_loader, graph_builder, trainer_params)
        
        # Make sure both variants were trained
        self.assertEqual(len(model_paths), 2)
        self.assertIn('control', model_paths)
        self.assertIn('variant_a', model_paths)
        
        # Verify model files exist
        for path in model_paths.values():
            self.assertTrue(os.path.exists(path))
        
        # Create recommenders for each variant
        recommenders = {}
        graph, job_id_mapping, user_id_mapping = graph_builder.build_graph(
            users_df, jobs_df, interactions_df, job_embeddings
        )
        
        for variant, model_path in model_paths.items():
            # Load model
            checkpoint = torch.load(model_path)
            model = HeteroGCNLinkPredictor(
                node_types=list(graph.x_dict.keys()),
                edge_types=list(graph.edge_index_dict.keys()),
                hidden_channels=64 if variant == 'variant_a' else 32,
                num_layers=2,
                dropout=0.2 if variant == 'variant_a' else 0.1
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Create recommender
            recommenders[variant] = JobRecommender(
                model=model,
                graph=graph,
                job_id_mapping=job_id_mapping,
                user_id_mapping=user_id_mapping
            )
        
        # Evaluate recommendations
        test_users = users_df['user_id'].iloc[:10].tolist()
        metrics = ab_framework.evaluate_recommendations(recommenders, test_users, top_n=5)
        
        # Check metrics
        self.assertEqual(len(metrics), 2)
        self.assertIn('control', metrics)
        self.assertIn('variant_a', metrics)
        
        # Compare variants
        comparison = ab_framework.compare_variants(metrics)
        
        # Verify comparison results
        self.assertIn('winner', comparison)
        self.assertIn('scores', comparison)
        self.assertIn('reason', comparison)
        
        # Winner should be one of the variants
        self.assertIn(comparison['winner'], ['control', 'variant_a'])
        
        # Test simulated feedback
        feedback_df = ab_framework.simulate_user_feedback(recommenders, users_df)
        
        if feedback_df is not None:
            self.assertGreater(len(feedback_df), 0)
            self.assertIn('variant', feedback_df.columns)
            self.assertIn('rating', feedback_df.columns)
            self.assertIn('interaction_type', feedback_df.columns)

if __name__ == '__main__':
    unittest.main()
