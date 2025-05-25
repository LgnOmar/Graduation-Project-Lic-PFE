"""
Tests for the graph features module of JibJob recommendation system.
"""

import unittest
import numpy as np
import pandas as pd
from jibjob_recommender_system.feature_engineering.graph_features import GraphFeatureProcessor

class TestGraphFeatureProcessor(unittest.TestCase):
    """Test cases for the GraphFeatureProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'feature_engineering': {
                'embedding_dim': 64,
                'distance_weight_factor': 0.5,
                'category_weight_factor': 1.0,
                'text_similarity_threshold': 0.3
            }
        }
        self.graph_processor = GraphFeatureProcessor(config=self.config)
        
        # Sample user data
        self.users_df = pd.DataFrame({
            'user_id': ['u1', 'u2', 'u3'],
            'user_type': ['professional', 'professional', 'employer'],
            'categories': [['cat1', 'cat2'], ['cat2', 'cat3'], ['cat1']],
            'latitude': [34.05, 34.10, 34.15],
            'longitude': [-118.25, -118.30, -118.20],
            'embedding': [np.random.rand(64) for _ in range(3)]
        })
        
        # Sample job data
        self.jobs_df = pd.DataFrame({
            'job_id': ['j1', 'j2', 'j3', 'j4'],
            'employer_id': ['u3', 'u3', 'e2', 'e3'],
            'categories': [['cat1'], ['cat2'], ['cat1', 'cat3'], ['cat4']],
            'latitude': [34.07, 34.12, 34.08, 34.20],
            'longitude': [-118.27, -118.32, -118.22, -118.15],
            'embedding': [np.random.rand(64) for _ in range(4)]
        })
        
    def test_compute_cosine_similarity(self):
        """Test cosine similarity computation."""
        # Create two sample embeddings
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        vec3 = np.array([1.0, 1.0, 0.0])
        
        # Calculate similarities
        sim12 = self.graph_processor.compute_cosine_similarity(vec1, vec2)
        sim13 = self.graph_processor.compute_cosine_similarity(vec1, vec3)
        
        # Check results
        self.assertAlmostEqual(sim12, 0.0)  # Orthogonal vectors
        self.assertAlmostEqual(sim13, 1/np.sqrt(2))  # 45-degree angle
        
    def test_compute_haversine_distance(self):
        """Test haversine distance computation."""
        # New York (approximate coordinates)
        lat1, lon1 = 40.7128, -74.0060
        
        # Los Angeles (approximate coordinates)
        lat2, lon2 = 34.0522, -118.2437
        
        # Distance should be around 3,944 km
        distance = self.graph_processor.compute_haversine_distance(lat1, lon1, lat2, lon2)
        
        # Check if distance is in the expected range (allowing some tolerance)
        self.assertTrue(3900 <= distance <= 4000)
        
    def test_compute_distance_weighted_edges(self):
        """Test distance-based edge weight computation."""
        # Test with simple data
        user_locations = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'latitude': [34.05, 34.10],
            'longitude': [-118.25, -118.30]
        })
        
        job_locations = pd.DataFrame({
            'job_id': ['j1', 'j2'],
            'latitude': [34.07, 34.12],
            'longitude': [-118.27, -118.32]
        })
        
        # Calculate edge weights
        edges = self.graph_processor.compute_distance_weighted_edges(
            user_locations, job_locations, max_distance=10.0
        )
        
        # Check that all user-job pairs are present
        self.assertEqual(len(edges), 4)
        
        # Check edge weight properties
        for _, row in edges.iterrows():
            # Check weight is between 0 and 1
            self.assertTrue(0 <= row['weight'] <= 1)
            
    def test_compute_category_edge_weights(self):
        """Test category-based edge weight computation."""
        # Test with simple data
        user_categories = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'categories': [['cat1', 'cat2'], ['cat2', 'cat3']]
        })
        
        job_categories = pd.DataFrame({
            'job_id': ['j1', 'j2'],
            'categories': [['cat1'], ['cat2', 'cat3']]
        })
        
        # Calculate edge weights
        edges = self.graph_processor.compute_category_edge_weights(
            user_categories, job_categories
        )
        
        # Check that all user-job pairs are present
        self.assertEqual(len(edges), 4)
        
        # Check specific category matches
        # u1-j1: ['cat1', 'cat2'] vs ['cat1'] -> 1 match out of max(2,1) = 2 -> 0.5
        # u1-j2: ['cat1', 'cat2'] vs ['cat2', 'cat3'] -> 1 match out of max(2,2) = 2 -> 0.5
        # u2-j1: ['cat2', 'cat3'] vs ['cat1'] -> 0 matches out of max(2,1) = 2 -> 0.0
        # u2-j2: ['cat2', 'cat3'] vs ['cat2', 'cat3'] -> 2 matches out of max(2,2) = 2 -> 1.0
        expected_weights = {
            ('u1', 'j1'): 0.5,
            ('u1', 'j2'): 0.5,
            ('u2', 'j1'): 0.0,
            ('u2', 'j2'): 1.0
        }
        
        for _, row in edges.iterrows():
            key = (row['user_id'], row['job_id'])
            self.assertAlmostEqual(row['weight'], expected_weights[key])
            
    def test_normalize_features(self):
        """Test feature normalization."""
        # Create sample feature array
        features = np.array([
            [1.0, 5.0, 10.0],
            [2.0, 4.0, 8.0],
            [3.0, 3.0, 6.0],
            [4.0, 2.0, 4.0],
            [5.0, 1.0, 2.0]
        ])
        
        # Normalize features
        normalized = self.graph_processor.normalize_features(features)
        
        # Check shape is preserved
        self.assertEqual(features.shape, normalized.shape)
        
        # Check min and max values (should be close to 0 and 1)
        self.assertAlmostEqual(np.min(normalized), 0.0, places=6)
        self.assertAlmostEqual(np.max(normalized), 1.0, places=6)
        
    def test_combine_similarity_and_distance(self):
        """Test combination of similarity and distance weights."""
        # Create sample edges
        similarity_edges = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2'],
            'job_id': ['j1', 'j2', 'j1', 'j2'],
            'weight': [0.8, 0.6, 0.4, 0.9]
        })
        
        distance_edges = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2'],
            'job_id': ['j1', 'j2', 'j1', 'j2'],
            'weight': [0.7, 0.5, 0.3, 0.8]
        })
        
        # Combine edges with equal weight (alpha=0.5)
        combined = self.graph_processor.combine_similarity_and_distance(
            similarity_edges, distance_edges, alpha=0.5
        )
        
        # Check that all pairs are present
        self.assertEqual(len(combined), 4)
        
        # Check specific combined weights
        # (0.8 * 0.5) + (0.7 * 0.5) = 0.4 + 0.35 = 0.75
        # (0.6 * 0.5) + (0.5 * 0.5) = 0.3 + 0.25 = 0.55
        # (0.4 * 0.5) + (0.3 * 0.5) = 0.2 + 0.15 = 0.35
        # (0.9 * 0.5) + (0.8 * 0.5) = 0.45 + 0.4 = 0.85
        expected_weights = {
            ('u1', 'j1'): 0.75,
            ('u1', 'j2'): 0.55,
            ('u2', 'j1'): 0.35,
            ('u2', 'j2'): 0.85
        }
        
        for _, row in combined.iterrows():
            key = (row['user_id'], row['job_id'])
            self.assertAlmostEqual(row['weight'], expected_weights[key])
            
if __name__ == '__main__':
    unittest.main()
