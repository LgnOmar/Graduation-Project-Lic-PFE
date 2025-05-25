"""
Tests for the dataset classes.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from src.data.dataset import (
    JobRecommendationDataset,
    GraphRecommendationDataset,
    NegativeSamplingDataset
)


class TestDatasets:
    """Tests for the dataset classes."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample user-job interactions
        interactions_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3],
            'job_id': [101, 102, 101, 103, 102],
            'rating': [4.5, 3.0, 5.0, 4.0, 2.5]
        })
        
        # Create mappings
        user_mapping = {1: 0, 2: 1, 3: 2}
        job_mapping = {101: 0, 102: 1, 103: 2}
        
        # Create features
        user_features = torch.randn(3, 5)  # 3 users, 5 features
        job_features = torch.randn(3, 8)   # 3 jobs, 8 features
        
        return {
            'interactions_df': interactions_df,
            'user_mapping': user_mapping,
            'job_mapping': job_mapping,
            'user_features': user_features,
            'job_features': job_features
        }
    
    def test_job_recommendation_dataset(self, sample_data):
        """Test the basic recommendation dataset."""
        # Create dataset
        dataset = JobRecommendationDataset(
            interactions_df=sample_data['interactions_df'],
            user_mapping=sample_data['user_mapping'],
            job_mapping=sample_data['job_mapping'],
            user_features=sample_data['user_features'],
            job_features=sample_data['job_features'],
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating'
        )
        
        # Check length
        assert len(dataset) == len(sample_data['interactions_df'])
        
        # Check item format
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 3
        
        user_features, job_features, rating = item
        assert isinstance(user_features, torch.Tensor)
        assert isinstance(job_features, torch.Tensor)
        assert isinstance(rating, torch.Tensor)
        
        assert user_features.shape == (5,)  # 5 user features
        assert job_features.shape == (8,)   # 8 job features
        assert rating.shape == ()           # Single rating value
    
    def test_graph_recommendation_dataset(self, sample_data):
        """Test the graph-based recommendation dataset."""
        # Create dataset
        dataset = GraphRecommendationDataset(
            interactions_df=sample_data['interactions_df'],
            user_mapping=sample_data['user_mapping'],
            job_mapping=sample_data['job_mapping'],
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating'
        )
        
        # Get the graph
        graph = dataset.get_graph()
        
        # Check graph structure
        assert isinstance(graph, torch.utils.data.Data)
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'edge_weight')
        assert hasattr(graph, 'num_users')
        assert hasattr(graph, 'num_jobs')
        
        # Check dimensions
        assert graph.edge_index.shape[0] == 2
        assert graph.num_users == len(sample_data['user_mapping'])
        assert graph.num_jobs == len(sample_data['job_mapping'])
        
        # Check with node features
        dataset_with_features = GraphRecommendationDataset(
            interactions_df=sample_data['interactions_df'],
            user_mapping=sample_data['user_mapping'],
            job_mapping=sample_data['job_mapping'],
            user_features=sample_data['user_features'],
            job_features=sample_data['job_features'],
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating'
        )
        
        graph_with_features = dataset_with_features.get_graph()
        assert hasattr(graph_with_features, 'user_features')
        assert hasattr(graph_with_features, 'job_features')
        assert graph_with_features.user_features.shape == sample_data['user_features'].shape
        assert graph_with_features.job_features.shape == sample_data['job_features'].shape
    
    def test_negative_sampling_dataset(self, sample_data):
        """Test the negative sampling dataset."""
        # Create dataset
        dataset = NegativeSamplingDataset(
            interactions_df=sample_data['interactions_df'],
            user_mapping=sample_data['user_mapping'],
            job_mapping=sample_data['job_mapping'],
            user_features=sample_data['user_features'],
            job_features=sample_data['job_features'],
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating',
            num_negatives=2
        )
        
        # Check length (should be interactions + negative samples)
        expected_length = len(sample_data['interactions_df']) * (1 + 2)  # original + 2 negatives per interaction
        assert len(dataset) == expected_length
        
        # Check item format
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 3
        
        user_features, job_features, label = item
        assert isinstance(user_features, torch.Tensor)
        assert isinstance(job_features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        
        # Check labels (first N items should be positive, rest negative)
        num_positives = len(sample_data['interactions_df'])
        
        # Positive examples
        for i in range(num_positives):
            _, _, label = dataset[i]
            assert label.item() == 1
        
        # Negative examples
        for i in range(num_positives, len(dataset)):
            _, _, label = dataset[i]
            assert label.item() == 0
            
    def test_negative_sampling_uniqueness(self, sample_data):
        """Test that negative samples are unique per user."""
        # Create dataset with more negative samples
        dataset = NegativeSamplingDataset(
            interactions_df=sample_data['interactions_df'],
            user_mapping=sample_data['user_mapping'],
            job_mapping=sample_data['job_mapping'],
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating',
            num_negatives=5
        )
        
        # Check that each user's negative samples are different from their positive interactions
        num_positives = len(sample_data['interactions_df'])
        
        # Get user-job pairs for positive interactions
        positive_pairs = []
        for i, row in sample_data['interactions_df'].iterrows():
            user_id = sample_data['user_mapping'][row['user_id']]
            job_id = sample_data['job_mapping'][row['job_id']]
            positive_pairs.append((user_id, job_id))
        
        # Check negative samples
        for i in range(num_positives, len(dataset)):
            user_idx = dataset.user_indices[i]
            job_idx = dataset.job_indices[i]
            
            # This negative sample should not be in positive pairs for this user
            assert (user_idx.item(), job_idx.item()) not in positive_pairs
