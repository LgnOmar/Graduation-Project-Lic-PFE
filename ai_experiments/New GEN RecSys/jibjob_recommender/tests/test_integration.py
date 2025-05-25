"""
Integration tests for the JibJob recommendation system.

This module tests the interaction between different components of the system
to ensure they work together correctly in various scenarios.
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
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jibjob_recommender_system.data_handling.data_loader import DataLoader
from jibjob_recommender_system.feature_engineering.graph_features import GraphFeatureProcessor
from jibjob_recommender_system.feature_engineering.text_embeddings import BERTEmbedder
from jibjob_recommender_system.graph_construction.graph_builder import GraphBuilder
from jibjob_recommender_system.models.gcn_model import HeteroGCNLinkPredictor
from jibjob_recommender_system.training.trainer import ModelTrainer
from jibjob_recommender_system.inference.recommender_service import JobRecommender
from jibjob_recommender_system.evaluation.evaluation import calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestIntegration(unittest.TestCase):
    """Test class for integration tests of the JibJob recommendation system components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment before running tests."""
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
        """Clean up test environment after running tests."""
        shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_sample_data(cls):
        """Create and save sample data for testing."""
        # Create sample users
        users_df = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(20)],
            'age': np.random.randint(18, 60, 20),
            'ville': np.random.choice(['Alger', 'Oran', 'Constantine', 'Annaba', 'Setif'], 20),
            'categorie': np.random.choice(['Informatique', 'Plomberie', 'Electricité', 'Beauté', 'Transport'], 20),
            'experience': np.random.choice(['1-3 ans', '3-5 ans', '5+ ans', 'Débutant'], 20)
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
            user_idx = np.random.randint(0, 20)
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
    
    def test_full_pipeline(self):
        """Test the complete recommendation pipeline from data loading to recommendation."""
        # Step 1: Load data
        logger.info("Step 1: Loading data")
        data_loader = DataLoader(self.data_path)
        users_df, jobs_df, interactions_df = data_loader.load_data()
        
        self.assertIsNotNone(users_df)
        self.assertIsNotNone(jobs_df)
        self.assertIsNotNone(interactions_df)
        self.assertGreater(len(users_df), 0)
        self.assertGreater(len(jobs_df), 0)
        self.assertGreater(len(interactions_df), 0)
        
        # Step 2: Process features
        logger.info("Step 2: Processing features")
        feature_processor = GraphFeatureProcessor(self.data_path)
        interactions_df = feature_processor.process_interactions(interactions_df)
        
        self.assertIn('enhanced_rating', interactions_df.columns)
        
        # Load job embeddings (in a real scenario these would be computed, but for testing we use mock data)
        with open(os.path.join(self.data_path, 'job_embeddings.pkl'), 'rb') as f:
            job_embeddings = pickle.load(f)
        
        # Step 3: Build graph
        logger.info("Step 3: Building graph")
        graph_builder = GraphBuilder()
        graph, job_id_mapping, user_id_mapping = graph_builder.build_graph(
            users_df, jobs_df, interactions_df, job_embeddings
        )
        
        self.assertIsNotNone(graph)
        self.assertIsNotNone(job_id_mapping)
        self.assertIsNotNone(user_id_mapping)
        self.assertTrue(hasattr(graph, 'x_dict'))
        self.assertTrue(hasattr(graph, 'edge_index_dict'))
        
        # Step 4: Create and train model
        logger.info("Step 4: Creating and training model")
        model = HeteroGCNLinkPredictor(
            node_types=list(graph.x_dict.keys()),
            edge_types=list(graph.edge_index_dict.keys()),
            hidden_channels=32,
            num_layers=2
        )
        
        trainer = ModelTrainer(
            model=model,
            graph=graph,
            output_dir=self.model_path
        )
        
        # Since we're just testing integration, we'll do minimal training
        results = trainer.train(
            num_epochs=2,
            batch_size=64,
            learning_rate=0.01,
            val_ratio=0.2,
            test_ratio=0.2,
            early_stopping_patience=3
        )
        
        self.assertIsNotNone(results)
        self.assertIn('test_auc', results)
        self.assertIn('best_model_path', results)
        
        # Step 5: Load the model for inference
        logger.info("Step 5: Loading model for inference")
        saved_model = torch.load(results['best_model_path'])
        model.load_state_dict(saved_model['model_state_dict'])
        model.eval()
        
        # Step 6: Create recommender service
        logger.info("Step 6: Creating recommender service")
        recommender = JobRecommender(
            model=model,
            graph=graph,
            job_id_mapping=job_id_mapping,
            user_id_mapping=user_id_mapping
        )
        
        # Step 7: Get recommendations for a user
        logger.info("Step 7: Getting recommendations")
        user_id = users_df['user_id'].iloc[0]
        recommendations = recommender.get_recommendations(
            user_id=user_id,
            top_n=5
        )
        
        self.assertGreater(len(recommendations), 0)
        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(isinstance(job_id, str) for job_id, _ in recommendations))
        self.assertTrue(all(isinstance(score, float) for _, score in recommendations))
        
        # Step 8: Verify different user gets different recommendations
        logger.info("Step 8: Verifying recommendations are different for different users")
        user_id2 = users_df['user_id'].iloc[1]
        recommendations2 = recommender.get_recommendations(
            user_id=user_id2,
            top_n=5
        )
        
        # It's statistically unlikely that two users would get the exact same recommendations
        # but we can check that they are not identical
        rec1_job_ids = [job_id for job_id, _ in recommendations]
        rec2_job_ids = [job_id for job_id, _ in recommendations2]
        self.assertNotEqual(set(rec1_job_ids), set(rec2_job_ids))
        
        # Step 9: Test error handling
        logger.info("Step 9: Testing error handling")
        with self.assertRaises(Exception):
            recommender.get_recommendations(
                user_id="non_existent_user",
                top_n=5
            )

    def test_data_to_feature_pipeline(self):
        """Test the pipeline from data loading to feature processing."""
        # Step 1: Load data
        data_loader = DataLoader(self.data_path)
        users_df, jobs_df, interactions_df = data_loader.load_data()
        
        # Step 2: Process interactions for sentiment
        feature_processor = GraphFeatureProcessor(self.data_path)
        processed_interactions = feature_processor.process_interactions(interactions_df)
        
        # Validate the output has the expected columns and transformations
        self.assertIn('enhanced_rating', processed_interactions.columns)
        self.assertTrue(all(isinstance(val, float) for val in processed_interactions['enhanced_rating']))
        
        # Step 3: Test saving and loading processed data
        processed_file = os.path.join(self.data_path, 'processed_interactions.csv')
        processed_interactions.to_csv(processed_file, index=False)
        
        loaded_df = pd.read_csv(processed_file)
        self.assertEqual(len(loaded_df), len(processed_interactions))
        self.assertListEqual(list(loaded_df.columns), list(processed_interactions.columns))

    def test_feature_to_model_pipeline(self):
        """Test the pipeline from feature processing to model training."""
        # Step 1: Load data
        data_loader = DataLoader(self.data_path)
        users_df, jobs_df, interactions_df = data_loader.load_data()
        
        # Step 2: Process features
        feature_processor = GraphFeatureProcessor(self.data_path)
        interactions_df = feature_processor.process_interactions(interactions_df)
        
        # Load job embeddings
        with open(os.path.join(self.data_path, 'job_embeddings.pkl'), 'rb') as f:
            job_embeddings = pickle.load(f)
        
        # Step 3: Build graph
        graph_builder = GraphBuilder()
        graph, job_id_mapping, user_id_mapping = graph_builder.build_graph(
            users_df, jobs_df, interactions_df, job_embeddings
        )
        
        # Step 4: Create and train model (minimal training)
        model = HeteroGCNLinkPredictor(
            node_types=list(graph.x_dict.keys()),
            edge_types=list(graph.edge_index_dict.keys()),
            hidden_channels=32,
            num_layers=2
        )
        
        trainer = ModelTrainer(
            model=model,
            graph=graph,
            output_dir=self.model_path
        )
        
        results = trainer.train(
            num_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            val_ratio=0.2,
            test_ratio=0.2,
            early_stopping_patience=3
        )
        
        # Validate training produces expected outputs
        self.assertIn('test_auc', results)
        self.assertIn('train_loss', results)
        self.assertIn('val_auc', results)
        self.assertIn('best_model_path', results)
        self.assertTrue(os.path.exists(results['best_model_path']))
        
        # Verify model checkpoint contains expected components
        checkpoint = torch.load(results['best_model_path'])
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('epoch', checkpoint)
        self.assertIn('best_val_auc', checkpoint)

    def test_model_to_inference_pipeline(self):
        """Test the pipeline from model training to inference."""
        # Load data
        data_loader = DataLoader(self.data_path)
        users_df, jobs_df, interactions_df = data_loader.load_data()
        
        # Process features
        feature_processor = GraphFeatureProcessor(self.data_path)
        interactions_df = feature_processor.process_interactions(interactions_df)
        
        # Load job embeddings
        with open(os.path.join(self.data_path, 'job_embeddings.pkl'), 'rb') as f:
            job_embeddings = pickle.load(f)
        
        # Build graph
        graph_builder = GraphBuilder()
        graph, job_id_mapping, user_id_mapping = graph_builder.build_graph(
            users_df, jobs_df, interactions_df, job_embeddings
        )
        
        # Create model
        model = HeteroGCNLinkPredictor(
            node_types=list(graph.x_dict.keys()),
            edge_types=list(graph.edge_index_dict.keys()),
            hidden_channels=32,
            num_layers=2
        )
        
        # Train minimally
        trainer = ModelTrainer(
            model=model,
            graph=graph,
            output_dir=self.model_path
        )
        
        results = trainer.train(
            num_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            val_ratio=0.2,
            test_ratio=0.2,
            early_stopping_patience=3
        )
        
        # Load trained model
        model_path = results['best_model_path']
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create recommender
        recommender = JobRecommender(
            model=model,
            graph=graph,
            job_id_mapping=job_id_mapping,
            user_id_mapping=user_id_mapping
        )
        
        # Get recommendations for multiple users
        for i in range(5):
            user_id = users_df['user_id'].iloc[i]
            recommendations = recommender.get_recommendations(
                user_id=user_id,
                top_n=5
            )
            
            # Validate recommendations
            self.assertEqual(len(recommendations), 5)
            
            # Check recommendations format
            for job_id, score in recommendations:
                self.assertTrue(job_id in jobs_df['job_id'].values)
                self.assertTrue(0 <= score <= 1)
            
            # Check that scores are in descending order
            scores = [score for _, score in recommendations]
            self.assertEqual(scores, sorted(scores, reverse=True))

if __name__ == '__main__':
    unittest.main()
