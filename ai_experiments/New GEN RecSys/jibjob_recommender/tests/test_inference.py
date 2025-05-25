"""
Tests for the inference modules of JibJob recommendation system.
"""

import unittest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from jibjob_recommender_system.inference.recommender_service import RecommenderService
from jibjob_recommender_system.inference.predict import RecommendationCLI

class TestRecommenderService(unittest.TestCase):
    """Test cases for the RecommenderService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model
        self.mock_model = MagicMock()
        self.mock_model.forward = MagicMock(return_value=torch.tensor([
            [0.9, 0.8, 0.5],  # node 0 embedding
            [0.7, 0.6, 0.3],  # node 1 embedding
            [0.5, 0.4, 0.1],  # node 2 embedding
            [0.3, 0.2, 0.0],  # node 3 embedding
            [0.1, 0.0, -0.1]  # node 4 embedding
        ]))
        
        # Configuration
        self.config = {
            'recommendation': {
                'top_n': 2,
                'location_radius': 50.0,  # km
                'min_category_match': 1,
                'fallback_strategy': 'popularity',
                'embedding_dim': 3,
                'text_similarity_weight': 0.3,
                'category_weight': 0.4,
                'distance_weight': 0.3
            }
        }
        
        # Sample data
        self.user_data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'user_type': ['professional', 'professional', 'employer'],
            'categories': [['cat1', 'cat2'], ['cat2', 'cat3'], ['cat1']],
            'latitude': [34.05, 34.10, 34.15],
            'longitude': [-118.25, -118.30, -118.20],
            'embedding': [[0.9, 0.8, 0.5], [0.7, 0.6, 0.3], [0.5, 0.4, 0.1]]
        }).set_index('user_id')
        
        self.job_data = pd.DataFrame({
            'job_id': ['j1', 'j2', 'j3', 'j4'],
            'employer_id': ['u3', 'u3', 'e2', 'e3'],
            'categories': [['cat1'], ['cat2'], ['cat1', 'cat3'], ['cat4']],
            'latitude': [34.07, 34.12, 34.08, 34.20],
            'longitude': [-118.27, -118.32, -118.22, -118.15],
            'embedding': [[0.3, 0.2, 0.0], [0.1, 0.0, -0.1], [0.5, 0.4, 0.1], [0.7, 0.6, 0.3]],
            'popularity': [10, 5, 8, 3]
        }).set_index('job_id')
        
        # User and job mappings
        self.user_mapping = {'u1': 0, 'u2': 1, 'u3': 2}
        self.job_mapping = {'j1': 3, 'j2': 4, 'j3': 5, 'j4': 6}
        
        # Initialize the service with all mocks
        with patch('jibjob_recommender_system.inference.recommender_service.GCNRecommender', return_value=self.mock_model):
            with patch('torch.load', return_value={
                'model_state_dict': {},
                'user_mapping': self.user_mapping,
                'job_mapping': self.job_mapping
            }):
                self.recommender_service = RecommenderService(
                    config=self.config,
                    model_path='dummy_path.pt',
                    user_data=self.user_data,
                    job_data=self.job_data
                )
    
    def test_filter_jobs_by_location(self):
        """Test filtering jobs by location."""
        # User location
        user_lat, user_lon = 34.05, -118.25
        
        # Test with default radius
        filtered_jobs = self.recommender_service._filter_jobs_by_location(
            self.job_data, user_lat, user_lon, radius_km=50
        )
        
        # All jobs should be within this radius for our test data
        self.assertEqual(len(filtered_jobs), 4)
        
        # Test with smaller radius
        filtered_jobs = self.recommender_service._filter_jobs_by_location(
            self.job_data, user_lat, user_lon, radius_km=5
        )
        
        # Should return fewer jobs
        self.assertTrue(len(filtered_jobs) < 4)
    
    def test_filter_jobs_by_category(self):
        """Test filtering jobs by category."""
        # User categories
        user_categories = ['cat1', 'cat2']
        
        # Test with min_matches=1
        filtered_jobs = self.recommender_service._filter_jobs_by_category(
            self.job_data, user_categories, min_matches=1
        )
        
        # Jobs with cat1 or cat2: j1, j2, j3
        self.assertEqual(len(filtered_jobs), 3)
        self.assertTrue('j1' in filtered_jobs.index)
        self.assertTrue('j2' in filtered_jobs.index)
        self.assertTrue('j3' in filtered_jobs.index)
        
        # Test with min_matches=2
        filtered_jobs = self.recommender_service._filter_jobs_by_category(
            self.job_data, user_categories, min_matches=2
        )
        
        # No jobs have both cat1 and cat2
        self.assertEqual(len(filtered_jobs), 0)
    
    def test_get_text_embedding(self):
        """Test text embedding generation."""
        # Mock the embedder
        with patch('sentence_transformers.SentenceTransformer') as mock_transformer:
            # Set up mock
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
            mock_transformer.return_value = mock_model
            
            # Get embedding
            embedding = self.recommender_service._get_text_embedding("Sample text")
            
            # Check result
            self.assertEqual(len(embedding), 3)
            self.assertEqual(embedding[0], 0.1)
            self.assertEqual(embedding[1], 0.2)
            self.assertEqual(embedding[2], 0.3)
            
            # Transformer should have been called
            mock_transformer.assert_called_once()
            mock_model.encode.assert_called_once_with("Sample text")
    
    def test_preprocess_user_data(self):
        """Test user data preprocessing."""
        # New user data
        new_user = {
            'user_id': 'u4',
            'user_type': 'professional',
            'categories': ['cat1', 'cat4'],
            'latitude': 34.06,
            'longitude': -118.26,
            'profile_text': 'Software engineer'
        }
        
        # Mock embedding function
        with patch.object(self.recommender_service, '_get_text_embedding', return_value=[0.1, 0.2, 0.3]):
            processed = self.recommender_service._preprocess_user_data(new_user)
            
            # Check result
            self.assertEqual(processed['user_id'], 'u4')
            self.assertEqual(processed['user_type'], 'professional')
            self.assertEqual(processed['categories'], ['cat1', 'cat4'])
            self.assertEqual(processed['latitude'], 34.06)
            self.assertEqual(processed['longitude'], -118.26)
            self.assertEqual(processed['embedding'], [0.1, 0.2, 0.3])
    
    def test_get_recommendations_from_model(self):
        """Test model-based recommendation generation."""
        # Mock user and node embeddings
        user_embedding = torch.tensor([0.9, 0.8, 0.5])
        job_embeddings = {
            'j1': torch.tensor([0.3, 0.2, 0.0]),
            'j2': torch.tensor([0.1, 0.0, -0.1]),
            'j3': torch.tensor([0.5, 0.4, 0.1]),
            'j4': torch.tensor([0.7, 0.6, 0.3])
        }
        
        # Get recommendations
        with patch.object(self.recommender_service, '_get_node_embeddings', return_value=(user_embedding, job_embeddings)):
            recommendations = self.recommender_service._get_recommendations_from_model('u1', self.job_data)
            
            # Check result
            self.assertEqual(len(recommendations), 4)
            self.assertIn('score', recommendations.columns)
            
            # Check order (should be sorted by score)
            job_ids = recommendations.index.tolist()
            scores = recommendations['score'].tolist()
            self.assertEqual(job_ids[0], 'j4')  # Highest similarity with user embedding
            self.assertTrue(scores[0] > scores[1])  # Scores should be in descending order
    
    def test_recommend_for_user(self):
        """Test recommendation generation for existing user."""
        # Mock get_recommendations_from_model
        mock_recommendations = pd.DataFrame({
            'score': [0.9, 0.7, 0.5, 0.3]
        }, index=['j4', 'j3', 'j1', 'j2'])
        
        with patch.object(self.recommender_service, '_get_recommendations_from_model', return_value=mock_recommendations):
            # Get recommendations
            results = self.recommender_service.recommend_for_user('u1', top_n=2)
            
            # Check result
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]['job_id'], 'j4')
            self.assertEqual(results[1]['job_id'], 'j3')
            
            # Each result should have score and job_id
            self.assertIn('score', results[0])
            self.assertIn('job_id', results[0])
    
    def test_recommend_for_new_user(self):
        """Test recommendation for new user."""
        # New user data
        new_user = {
            'user_id': 'new_user',
            'user_type': 'professional',
            'categories': ['cat1', 'cat4'],
            'latitude': 34.06,
            'longitude': -118.26,
            'profile_text': 'Software engineer'
        }
        
        # Mock necessary methods
        with patch.object(self.recommender_service, '_preprocess_user_data', return_value={
            'user_id': 'new_user',
            'user_type': 'professional',
            'categories': ['cat1', 'cat4'],
            'latitude': 34.06,
            'longitude': -118.26,
            'embedding': [0.1, 0.2, 0.3]
        }):
            with patch.object(self.recommender_service, '_get_recommendations_for_new_user', return_value=pd.DataFrame({
                'score': [0.8, 0.6, 0.4, 0.2]
            }, index=['j3', 'j1', 'j4', 'j2'])):
                # Get recommendations
                results = self.recommender_service.recommend_for_new_user(new_user, top_n=2)
                
                # Check result
                self.assertEqual(len(results), 2)
                self.assertEqual(results[0]['job_id'], 'j3')
                self.assertEqual(results[1]['job_id'], 'j1')
                
                # Each result should have score and job_id
                self.assertIn('score', results[0])
                self.assertIn('job_id', results[0])
    
    def test_get_fallback_recommendations(self):
        """Test fallback recommendation strategy."""
        # User categories
        user_categories = ['cat2']
        
        # Get recommendations
        recommendations = self.recommender_service._get_fallback_recommendations(
            self.job_data, user_categories, top_n=2
        )
        
        # Check result
        self.assertEqual(len(recommendations), 2)
        
        # Should prefer jobs with matching categories and higher popularity
        # j2 has cat2 with popularity 5
        self.assertEqual(recommendations.index[0], 'j2')
        
        # For the second recommendation, it should use popularity
        # among remaining jobs (j1, j3, j4), j1 has highest popularity (10)
        self.assertEqual(recommendations.index[1], 'j1')


class TestRecommendationCLI(unittest.TestCase):
    """Test cases for the RecommendationCLI class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock recommender service
        self.mock_service = MagicMock()
        self.mock_service.recommend_for_user.return_value = [
            {'job_id': 'j1', 'score': 0.9},
            {'job_id': 'j2', 'score': 0.7}
        ]
        
        self.mock_service.recommend_for_new_user.return_value = [
            {'job_id': 'j3', 'score': 0.8},
            {'job_id': 'j4', 'score': 0.6}
        ]
        
        # Mock args
        self.args = MagicMock()
        self.args.user_id = 'u1'
        self.args.top_n = 2
        self.args.output_format = 'json'
        
        # Initialize CLI
        with patch('jibjob_recommender_system.inference.predict.RecommenderService', return_value=self.mock_service):
            self.cli = RecommendationCLI(self.args)
    
    def test_generate_recommendations_for_user(self):
        """Test recommendation generation for existing user."""
        # Generate recommendations
        recommendations = self.cli.generate_recommendations_for_user('u1')
        
        # Check result
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]['job_id'], 'j1')
        self.assertEqual(recommendations[1]['job_id'], 'j2')
        
        # Recommender service should have been called
        self.mock_service.recommend_for_user.assert_called_once_with('u1', top_n=2)
    
    def test_generate_recommendations_for_new_user(self):
        """Test recommendation generation for new user."""
        # New user data
        new_user = {
            'user_id': 'new_user',
            'categories': ['cat1'],
            'latitude': 34.05,
            'longitude': -118.25,
            'profile_text': 'Sample profile'
        }
        
        # Generate recommendations
        recommendations = self.cli.generate_recommendations_for_new_user(new_user)
        
        # Check result
        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]['job_id'], 'j3')
        self.assertEqual(recommendations[1]['job_id'], 'j4')
        
        # Recommender service should have been called
        self.mock_service.recommend_for_new_user.assert_called_once()
    
    def test_format_output_json(self):
        """Test JSON output formatting."""
        # Sample recommendations
        recommendations = [
            {'job_id': 'j1', 'score': 0.9},
            {'job_id': 'j2', 'score': 0.7}
        ]
        
        # Format output
        output = self.cli.format_output(recommendations, 'json')
        
        # Check result
        self.assertIn('"job_id": "j1"', output)
        self.assertIn('"score": 0.9', output)
        self.assertIn('"job_id": "j2"', output)
        self.assertIn('"score": 0.7', output)
    
    def test_format_output_csv(self):
        """Test CSV output formatting."""
        # Sample recommendations
        recommendations = [
            {'job_id': 'j1', 'score': 0.9},
            {'job_id': 'j2', 'score': 0.7}
        ]
        
        # Format output
        output = self.cli.format_output(recommendations, 'csv')
        
        # Check result
        self.assertIn('job_id,score', output)
        self.assertIn('j1,0.9', output)
        self.assertIn('j2,0.7', output)
    
    @patch.object(RecommendationCLI, 'generate_recommendations_for_user')
    def test_run_existing_user(self, mock_generate):
        """Test CLI run for existing user."""
        # Set up mock
        mock_generate.return_value = [
            {'job_id': 'j1', 'score': 0.9},
            {'job_id': 'j2', 'score': 0.7}
        ]
        
        # Run CLI
        with patch('sys.stdout') as mock_stdout:
            self.cli.run()
            
            # generate_recommendations_for_user should have been called
            mock_generate.assert_called_once_with('u1')
            
            # Output should have been printed
            mock_stdout.write.assert_called()


if __name__ == '__main__':
    unittest.main()
