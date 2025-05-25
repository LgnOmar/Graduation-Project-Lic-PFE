"""
Tests for the recommender service module of JibJob recommendation system.
"""

import unittest
import numpy as np
import pandas as pd
import torch
from jibjob_recommender_system.inference.recommender_service import RecommenderService
from unittest.mock import MagicMock, patch

class TestRecommenderService(unittest.TestCase):
    """Test cases for the RecommenderService class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock config
        self.config = {
            'recommendation': {
                'top_n': 10,
                'location_radius': 50.0,  # km
                'min_category_match': 1,
                'fallback_strategy': 'popularity',
                'embedding_dim': 64,
                'distance_weight': 0.3,
                'category_weight': 0.4,
                'text_similarity_weight': 0.3
            }
        }
        
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = torch.tensor([
            [0.9, 0.8, 0.7, 0.6, 0.5],  # Scores for user 1
            [0.5, 0.9, 0.4, 0.8, 0.3]   # Scores for user 2
        ])
        
        # Sample data
        self.user_data = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'user_type': ['professional', 'professional', 'employer'],
            'categories': [['cat1', 'cat2'], ['cat2', 'cat3'], ['cat1']],
            'latitude': [34.05, 34.10, 34.15],
            'longitude': [-118.25, -118.30, -118.20],
            'embedding': [np.random.rand(64).tolist() for _ in range(3)]
        })
        
        self.job_data = pd.DataFrame({
            'job_id': ['j1', 'j2', 'j3', 'j4', 'j5'],
            'employer_id': ['u3', 'u3', 'e2', 'e3', 'e1'],
            'categories': [['cat1'], ['cat2'], ['cat1', 'cat3'], ['cat4'], ['cat2', 'cat3']],
            'latitude': [34.07, 34.12, 34.08, 34.20, 34.11],
            'longitude': [-118.27, -118.32, -118.22, -118.15, -118.29],
            'embedding': [np.random.rand(64).tolist() for _ in range(5)],
            'popularity': [10, 5, 8, 3, 7]
        })
        
        # Initialize the service with mocks
        with patch('jibjob_recommender_system.inference.recommender_service.GCNRecommender', return_value=self.mock_model):
            self.recommender_service = RecommenderService(
                config=self.config,
                model_path='dummy_path.pt',
                user_data=self.user_data,
                job_data=self.job_data
            )
        
    def test_preprocess_user_data(self):
        """Test user data preprocessing."""
        # Test with a new user
        new_user = {
            'user_id': 'new_user',
            'user_type': 'professional',
            'categories': ['cat1', 'cat4'],
            'latitude': 34.06,
            'longitude': -118.26,
            'profile_text': 'Software engineer with Python experience'
        }
        
        # Mock the text embedding function
        with patch.object(self.recommender_service, '_get_text_embedding', return_value=np.random.rand(64).tolist()):
            processed_user = self.recommender_service._preprocess_user_data(new_user)
            
        # Check user has required fields
        self.assertEqual(processed_user['user_id'], 'new_user')
        self.assertEqual(processed_user['user_type'], 'professional')
        self.assertEqual(len(processed_user['embedding']), 64)  # Should have embedding of correct dimension
    
    def test_filter_jobs_by_location(self):
        """Test location-based job filtering."""
        # User location
        user_lat, user_lon = 34.05, -118.25
        
        # Get jobs within 50 km
        filtered_jobs = self.recommender_service._filter_jobs_by_location(
            self.job_data, user_lat, user_lon, radius_km=50
        )
        
        # All jobs in our sample should be within the radius
        self.assertEqual(len(filtered_jobs), 5)
        
        # Test with smaller radius (10 km)
        filtered_jobs = self.recommender_service._filter_jobs_by_location(
            self.job_data, user_lat, user_lon, radius_km=10
        )
        
        # Should return fewer jobs
        self.assertTrue(len(filtered_jobs) < 5)
    
    def test_filter_jobs_by_category(self):
        """Test category-based job filtering."""
        # User categories
        user_categories = ['cat1', 'cat2']
        
        # Get jobs with at least 1 matching category
        filtered_jobs = self.recommender_service._filter_jobs_by_category(
            self.job_data, user_categories, min_matches=1
        )
        
        # Jobs j1, j2, j3, j5 have at least one matching category
        self.assertEqual(len(filtered_jobs), 4)
        
        # Test with higher minimum matches (2)
        filtered_jobs = self.recommender_service._filter_jobs_by_category(
            self.job_data, user_categories, min_matches=2
        )
        
        # No jobs have 2 matching categories with user
        self.assertEqual(len(filtered_jobs), 0)
    
    def test_recommend_for_user(self):
        """Test recommendation generation for an existing user."""
        # Get recommendations for user u1
        with patch.object(self.recommender_service, '_get_recommendations_from_model',
                         return_value=pd.DataFrame({
                             'job_id': ['j1', 'j2', 'j3', 'j4', 'j5'],
                             'score': [0.9, 0.8, 0.7, 0.6, 0.5]
                         })):
            recommendations = self.recommender_service.recommend_for_user('u1', top_n=3)
        
        # Check number of recommendations
        self.assertEqual(len(recommendations), 3)
        
        # Check recommendation structure
        self.assertTrue(all(field in recommendations[0] for field in ['job_id', 'score']))
        
        # Check order (should be sorted by score)
        self.assertEqual(recommendations[0]['job_id'], 'j1')
        self.assertEqual(recommendations[1]['job_id'], 'j2')
        self.assertEqual(recommendations[2]['job_id'], 'j3')
    
    def test_recommend_for_new_user(self):
        """Test recommendation generation for a new user."""
        # New user data
        new_user = {
            'user_id': 'new_user',
            'user_type': 'professional',
            'categories': ['cat1', 'cat4'],
            'latitude': 34.06,
            'longitude': -118.26,
            'profile_text': 'Software engineer with Python experience'
        }
        
        # Mock required methods
        with patch.object(self.recommender_service, '_get_text_embedding', return_value=np.random.rand(64).tolist()):
            with patch.object(self.recommender_service, '_get_recommendations_for_new_user',
                            return_value=pd.DataFrame({
                                'job_id': ['j1', 'j3', 'j5', 'j2', 'j4'],
                                'score': [0.85, 0.75, 0.65, 0.55, 0.45]
                            })):
                recommendations = self.recommender_service.recommend_for_new_user(new_user, top_n=3)
        
        # Check number of recommendations
        self.assertEqual(len(recommendations), 3)
        
        # Check recommendation structure
        self.assertTrue(all(field in recommendations[0] for field in ['job_id', 'score']))
        
        # Check order (should be sorted by score)
        self.assertEqual(recommendations[0]['job_id'], 'j1')
        self.assertEqual(recommendations[1]['job_id'], 'j3')
        self.assertEqual(recommendations[2]['job_id'], 'j5')
    
    def test_get_fallback_recommendations(self):
        """Test fallback recommendation strategy."""
        # Get fallback recommendations
        recommendations = self.recommender_service._get_fallback_recommendations(
            self.job_data, user_categories=['cat2'], top_n=3
        )
        
        # Check number of recommendations
        self.assertEqual(len(recommendations), 3)
        
        # Check that we get jobs with matching categories and sorted by popularity
        # j2 has cat2 with popularity 5
        # j5 has cat2 (and cat3) with popularity 7
        # Others don't have cat2 but should be sorted by popularity
        self.assertEqual(recommendations.iloc[0]['job_id'], 'j5')  # Most popular with matching category
        self.assertEqual(recommendations.iloc[1]['job_id'], 'j2')  # Second most popular with matching category

if __name__ == '__main__':
    unittest.main()
