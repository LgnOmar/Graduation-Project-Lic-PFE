"""
Continuous integration tests for the JibJob recommendation system.

This module tests the entire pipeline from data loading to API service in various end-to-end scenarios.
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
from pathlib import Path
import multiprocessing
import time
import requests
import subprocess

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jibjob_recommender_system.data_handling.data_loader import DataLoader
from jibjob_recommender_system.feature_engineering.graph_features import GraphFeatureProcessor
from jibjob_recommender_system.graph_construction.graph_builder import GraphBuilder
from jibjob_recommender_system.models.gcn_model import HeteroGCNLinkPredictor
from jibjob_recommender_system.training.trainer import ModelTrainer
from jibjob_recommender_system.inference.recommender_service import JobRecommender
from jibjob_recommender_system.main import run_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestContinuousIntegration(unittest.TestCase):
    """Test class for end-to-end and continuous integration tests."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment for CI tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.data_path = os.path.join(cls.temp_dir, 'data')
        cls.model_path = os.path.join(cls.temp_dir, 'models')
        
        # Create necessary directories
        os.makedirs(cls.data_path, exist_ok=True)
        os.makedirs(cls.model_path, exist_ok=True)
        
        # Create sample data
        cls._create_sample_data()
        
        # Setup environment variables for tests
        os.environ['JIBJOB_DATA_DIR'] = cls.data_path
        os.environ['JIBJOB_MODEL_DIR'] = cls.model_path
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after tests."""
        shutil.rmtree(cls.temp_dir)
        # Clear environment variables
        if 'JIBJOB_DATA_DIR' in os.environ:
            del os.environ['JIBJOB_DATA_DIR']
        if 'JIBJOB_MODEL_DIR' in os.environ:
            del os.environ['JIBJOB_MODEL_DIR']
    
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
        for i in range(200):
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
    
    def _start_api_server(self, port=8000):
        """Start the API server for testing."""
        # Import here to avoid circular imports
        from jibjob_recommender_system.inference.api import app
        import uvicorn
        
        # Start server in a separate process
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=port)
        
        self.server_process = multiprocessing.Process(target=run_server)
        self.server_process.start()
        
        # Wait for the server to start
        time.sleep(2)
    
    def _stop_api_server(self):
        """Stop the API server."""
        if hasattr(self, 'server_process') and self.server_process.is_alive():
            self.server_process.terminate()
            self.server_process.join(timeout=3)
            
            # Force kill if didn't join properly
            if self.server_process.is_alive():
                self.server_process.kill()
    
    def test_end_to_end_main_pipeline(self):
        """Test the end-to-end pipeline using the main module."""
        logger.info("Testing end-to-end main pipeline")
        
        # Run the pipeline
        success, output_paths = run_pipeline(
            data_dir=self.data_path,
            model_dir=self.model_path,
            epochs=2,  # Low number for testing
            batch_size=32,
            learning_rate=0.01
        )
        
        # Check pipeline completion
        self.assertTrue(success)
        self.assertIn('model_path', output_paths)
        self.assertTrue(os.path.exists(output_paths['model_path']))
        
        # Check that all expected files were created
        self.assertTrue(os.path.exists(os.path.join(self.data_path, 'processed_interactions.csv')))
        self.assertTrue(os.path.exists(os.path.join(self.model_path, 'best_model.pt')))
        self.assertTrue(os.path.exists(os.path.join(self.model_path, 'test_results.json')))
        
        # Load and check test results
        with open(os.path.join(self.model_path, 'test_results.json'), 'r') as f:
            results = json.load(f)
        
        self.assertIn('test_auc', results)
        self.assertIn('test_precision', results)
        self.assertIn('test_recall', results)

    def test_pipeline_with_data_changes(self):
        """Test the pipeline's ability to handle data changes."""
        logger.info("Testing pipeline with data changes")
        
        # Run the initial pipeline
        success, output_paths = run_pipeline(
            data_dir=self.data_path,
            model_dir=self.model_path,
            epochs=2,  # Low number for testing
            batch_size=32,
            learning_rate=0.01
        )
        
        self.assertTrue(success)
        
        # Record initial model
        initial_model_path = output_paths['model_path']
        initial_model_stats = os.stat(initial_model_path)
        
        # Add new data (users and interactions)
        users_df = pd.read_csv(os.path.join(self.data_path, 'users_df.csv'))
        new_users = pd.DataFrame({
            'user_id': [f'user_{i}' for i in range(len(users_df), len(users_df) + 10)],
            'age': np.random.randint(18, 60, 10),
            'ville': np.random.choice(['Alger', 'Oran', 'Constantine', 'Annaba', 'Setif'], 10),
            'categorie': np.random.choice(['Informatique', 'Plomberie', 'Electricité', 'Beauté', 'Transport'], 10),
            'experience': np.random.choice(['1-3 ans', '3-5 ans', '5+ ans', 'Débutant'], 10)
        })
        
        users_df = pd.concat([users_df, new_users], ignore_index=True)
        users_df.to_csv(os.path.join(self.data_path, 'users_df.csv'), index=False)
        
        # Add new interactions
        interactions_df = pd.read_csv(os.path.join(self.data_path, 'interactions_df.csv'))
        new_interactions = []
        for i in range(50):
            user_idx = np.random.randint(len(users_df) - 10, len(users_df))  # Use new users
            job_idx = np.random.randint(0, 50)
            rating = np.random.choice([1, 2, 3, 4, 5])
            new_interactions.append({
                'user_id': f'user_{user_idx}',
                'job_id': f'job_{job_idx}',
                'rating': rating,
                'commentaire_texte_anglais': f'New comment for job {job_idx} by user {user_idx}.'
            })
        
        new_interactions_df = pd.DataFrame(new_interactions)
        interactions_df = pd.concat([interactions_df, new_interactions_df], ignore_index=True)
        interactions_df.to_csv(os.path.join(self.data_path, 'interactions_df.csv'), index=False)
        
        # Delete processed data to force regeneration
        if os.path.exists(os.path.join(self.data_path, 'processed_interactions.csv')):
            os.remove(os.path.join(self.data_path, 'processed_interactions.csv'))
        
        # Run pipeline again with updated data
        success, output_paths = run_pipeline(
            data_dir=self.data_path,
            model_dir=self.model_path,
            epochs=2,
            batch_size=32,
            learning_rate=0.01
        )
        
        self.assertTrue(success)
        
        # Verify that a new model was created
        new_model_path = output_paths['model_path']
        new_model_stats = os.stat(new_model_path)
        
        # Either file size or modification time should be different
        self.assertTrue(
            initial_model_stats.st_size != new_model_stats.st_size or
            initial_model_stats.st_mtime != new_model_stats.st_mtime
        )
    
    def test_api_service(self):
        """Test the API service."""
        logger.info("Testing API service")
        
        try:
            # Setup the model
            run_pipeline(
                data_dir=self.data_path,
                model_dir=self.model_path,
                epochs=2,
                batch_size=32,
                learning_rate=0.01
            )
            
            # Start API server
            port = 8765  # Use a different port to avoid conflicts
            api_process = subprocess.Popen(
                [sys.executable, "-m", "jibjob_recommender_system.inference.api", 
                 "--port", str(port), 
                 "--model", os.path.join(self.model_path, "best_model.pt"),
                 "--data", self.data_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Try to get recommendations for users
            users_df = pd.read_csv(os.path.join(self.data_path, 'users_df.csv'))
            for i in range(3):  # Test with first 3 users
                user_id = users_df['user_id'].iloc[i]
                response = requests.get(f"http://localhost:{port}/recommendations/{user_id}?top_n=5")
                
                # Check response
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn('job_ids', data)
                self.assertIn('scores', data)
                self.assertEqual(len(data['job_ids']), 5)
                self.assertEqual(len(data['scores']), 5)
                
                # Check that all scores are between 0 and 1
                for score in data['scores']:
                    self.assertTrue(0 <= score <= 1)
                
                # Check that job IDs are strings
                for job_id in data['job_ids']:
                    self.assertTrue(isinstance(job_id, str))
            
            # Test error handling with non-existent user
            response = requests.get(f"http://localhost:{port}/recommendations/nonexistent_user?top_n=5")
            self.assertEqual(response.status_code, 404)
            
        finally:
            # Clean up
            if 'api_process' in locals():
                api_process.terminate()
                try:
                    api_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    api_process.kill()

    def test_incremental_training(self):
        """Test the system's ability to retrain incrementally."""
        logger.info("Testing incremental training")
        
        # Run the initial pipeline
        success, output_paths = run_pipeline(
            data_dir=self.data_path,
            model_dir=self.model_path,
            epochs=2,
            batch_size=32,
            learning_rate=0.01
        )
        
        self.assertTrue(success)
        
        # Get initial recommendations for a user
        users_df = pd.read_csv(os.path.join(self.data_path, 'users_df.csv'))
        test_user_id = users_df['user_id'].iloc[0]
        
        # Load the recommender
        model_path = output_paths['model_path']
        data_loader = DataLoader(self.data_path)
        users_df, jobs_df, interactions_df = data_loader.load_data()
        
        # Load job embeddings
        with open(os.path.join(self.data_path, 'job_embeddings.pkl'), 'rb') as f:
            job_embeddings = pickle.load(f)
        
        # Build graph
        graph_builder = GraphBuilder()
        graph, job_id_mapping, user_id_mapping = graph_builder.build_graph(
            users_df, jobs_df, interactions_df, job_embeddings
        )
        
        # Create model
        checkpoint = torch.load(model_path)
        model = HeteroGCNLinkPredictor(
            node_types=list(graph.x_dict.keys()),
            edge_types=list(graph.edge_index_dict.keys()),
            hidden_channels=32,
            num_layers=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create recommender
        recommender = JobRecommender(
            model=model,
            graph=graph,
            job_id_mapping=job_id_mapping,
            user_id_mapping=user_id_mapping
        )
        
        # Get initial recommendations
        initial_recs = recommender.get_recommendations(
            user_id=test_user_id,
            top_n=5
        )
        
        # Add new interactions for the user that favor a specific category
        jobs_df = pd.read_csv(os.path.join(self.data_path, 'jobs_df.csv'))
        target_category = 'Informatique'
        informatique_jobs = jobs_df[jobs_df['categorie'] == target_category]['job_id'].tolist()
        
        if not informatique_jobs:
            # If no jobs in target category, create some
            for i in range(5):
                idx = i + len(jobs_df)
                new_job = {
                    'job_id': f'job_{idx}',
                    'titre_emploi': f'IT Job {i}',
                    'description_mission_anglais': f'Advanced IT job requiring programming skills.',
                    'ville': 'Alger',
                    'categorie': target_category
                }
                jobs_df = pd.concat([jobs_df, pd.DataFrame([new_job])], ignore_index=True)
            
            # Update job embeddings
            for job_id in [f'job_{i + len(jobs_df) - 5}' for i in range(5)]:
                job_embeddings[job_id] = np.random.randn(768)
            
            # Save updated jobs
            jobs_df.to_csv(os.path.join(self.data_path, 'jobs_df.csv'), index=False)
            with open(os.path.join(self.data_path, 'job_embeddings.pkl'), 'wb') as f:
                pickle.dump(job_embeddings, f)
            
            # Update informatique_jobs list
            informatique_jobs = jobs_df[jobs_df['categorie'] == target_category]['job_id'].tolist()
        
        # Add interactions with the target category jobs
        interactions_df = pd.read_csv(os.path.join(self.data_path, 'interactions_df.csv'))
        new_interactions = []
        for job_id in informatique_jobs[:3]:  # Use first 3 jobs from the category
            new_interactions.append({
                'user_id': test_user_id,
                'job_id': job_id,
                'rating': 5,  # High rating to indicate strong preference
                'commentaire_texte_anglais': f'Great job! Highly recommended.'
            })
        
        new_interactions_df = pd.DataFrame(new_interactions)
        interactions_df = pd.concat([interactions_df, new_interactions_df], ignore_index=True)
        interactions_df.to_csv(os.path.join(self.data_path, 'interactions_df.csv'), index=False)
        
        # Delete processed data to force regeneration
        if os.path.exists(os.path.join(self.data_path, 'processed_interactions.csv')):
            os.remove(os.path.join(self.data_path, 'processed_interactions.csv'))
        
        # Run pipeline again with updated data
        success, output_paths = run_pipeline(
            data_dir=self.data_path,
            model_dir=self.model_path,
            epochs=2,
            batch_size=32,
            learning_rate=0.01
        )
        
        self.assertTrue(success)
        
        # Load the new recommender
        model_path = output_paths['model_path']
        data_loader = DataLoader(self.data_path)
        users_df, jobs_df, interactions_df = data_loader.load_data()
        
        # Load job embeddings
        with open(os.path.join(self.data_path, 'job_embeddings.pkl'), 'rb') as f:
            job_embeddings = pickle.load(f)
        
        # Build graph
        graph_builder = GraphBuilder()
        graph, job_id_mapping, user_id_mapping = graph_builder.build_graph(
            users_df, jobs_df, interactions_df, job_embeddings
        )
        
        # Create model
        checkpoint = torch.load(model_path)
        model = HeteroGCNLinkPredictor(
            node_types=list(graph.x_dict.keys()),
            edge_types=list(graph.edge_index_dict.keys()),
            hidden_channels=32,
            num_layers=2
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create recommender
        recommender = JobRecommender(
            model=model,
            graph=graph,
            job_id_mapping=job_id_mapping,
            user_id_mapping=user_id_mapping
        )
        
        # Get new recommendations
        new_recs = recommender.get_recommendations(
            user_id=test_user_id,
            top_n=5
        )
        
        # Extract job IDs for comparison
        initial_job_ids = [job_id for job_id, _ in initial_recs]
        new_job_ids = [job_id for job_id, _ in new_recs]
        
        # Check if recommendations changed
        self.assertNotEqual(set(initial_job_ids), set(new_job_ids))
        
        # Check if the new recommendations include more jobs from the target category
        initial_category_count = sum(
            1 for job_id in initial_job_ids
            if jobs_df.loc[jobs_df['job_id'] == job_id, 'categorie'].iloc[0] == target_category
        )
        
        new_category_count = sum(
            1 for job_id in new_job_ids
            if jobs_df.loc[jobs_df['job_id'] == job_id, 'categorie'].iloc[0] == target_category
        )
        
        # The count of recommended jobs from the target category should increase
        self.assertGreaterEqual(new_category_count, initial_category_count)

if __name__ == '__main__':
    unittest.main()
