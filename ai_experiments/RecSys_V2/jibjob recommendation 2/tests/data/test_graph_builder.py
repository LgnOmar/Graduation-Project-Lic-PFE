"""
Tests for the graph building utilities.
"""

import pytest
import torch
import numpy as np
from src.data.graph_builder import build_interaction_graph, build_heterogeneous_graph


class TestGraphBuilder:
    """Tests for the graph building functions."""
    
    def test_build_interaction_graph(self):
        """Test building a simple bipartite interaction graph."""
        # Create sample data
        user_indices = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
        job_indices = torch.tensor([0, 1, 1, 2, 0], dtype=torch.long)
        ratings = torch.tensor([4.0, 3.5, 5.0, 4.5, 3.0], dtype=torch.float)
        
        num_users = 3
        num_jobs = 3
        
        # Build graph
        graph = build_interaction_graph(
            user_indices=user_indices,
            job_indices=job_indices,
            ratings=ratings,
            num_users=num_users,
            num_jobs=num_jobs
        )
        
        # Check graph structure
        assert hasattr(graph, 'edge_index')
        assert hasattr(graph, 'edge_weight')
        assert graph.edge_index.shape[0] == 2  # Source and target nodes
        assert graph.edge_index.shape[1] == len(user_indices) * 2  # Bidirectional edges
        
        # Check that graph attributes were set correctly
        assert graph.num_users == num_users
        assert graph.num_jobs == num_jobs
    
    def test_build_interaction_graph_with_threshold(self):
        """Test building a graph with a rating threshold."""
        # Create sample data
        user_indices = torch.tensor([0, 0, 1, 1, 2], dtype=torch.long)
        job_indices = torch.tensor([0, 1, 1, 2, 0], dtype=torch.long)
        ratings = torch.tensor([4.0, 3.5, 5.0, 4.5, 3.0], dtype=torch.float)
        
        num_users = 3
        num_jobs = 3
        
        # Build graph with threshold
        graph = build_interaction_graph(
            user_indices=user_indices,
            job_indices=job_indices,
            ratings=ratings,
            num_users=num_users,
            num_jobs=num_jobs,
            threshold=4.0  # Only include ratings >= 4.0
        )
        
        # Check that only edges with ratings >= 4.0 are included
        # We expect 3 ratings above threshold, times 2 for bidirectional edges
        assert graph.edge_index.shape[1] == 6
    
    def test_build_interaction_graph_with_normalization(self):
        """Test building a graph with normalized ratings."""
        # Create sample data
        user_indices = torch.tensor([0, 0, 1], dtype=torch.long)
        job_indices = torch.tensor([0, 1, 1], dtype=torch.long)
        ratings = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float)
        
        num_users = 2
        num_jobs = 2
        
        # Build graph with normalization
        graph = build_interaction_graph(
            user_indices=user_indices,
            job_indices=job_indices,
            ratings=ratings,
            num_users=num_users,
            num_jobs=num_jobs,
            normalize_ratings=True
        )
        
        # Check that ratings are normalized to [0, 1] range
        assert torch.all(graph.edge_weight <= 1.0)
        assert torch.all(graph.edge_weight >= 0.0)
        
        # The edge with rating 1.0 should be mapped to 0.0, rating 5.0 to 1.0
        min_weight = graph.edge_weight.min().item()
        max_weight = graph.edge_weight.max().item()
        assert min_weight == 0.0
        assert max_weight == 1.0
    
    def test_build_heterogeneous_graph(self):
        """Test building a heterogeneous graph with multiple node and edge types."""
        # Create sample user-job interactions
        user_job_indices = torch.tensor([
            [0, 0, 1, 2],  # User nodes
            [0, 1, 1, 2]   # Job nodes
        ], dtype=torch.long)
        
        user_job_ratings = torch.tensor([4.0, 3.5, 5.0, 4.5], dtype=torch.float)
        
        # Create sample job-category assignments
        job_category_indices = torch.tensor([
            [0, 1, 2],     # Job nodes
            [0, 1, 0]      # Category nodes
        ], dtype=torch.long)
        
        # Node features
        user_features = torch.randn(3, 5)  # 3 users, 5 features each
        job_features = torch.randn(3, 8)   # 3 jobs, 8 features each
        category_features = torch.randn(2, 3)  # 2 categories, 3 features each
        
        # Build heterogeneous graph
        hetero_graph = build_heterogeneous_graph(
            user_job_indices=user_job_indices,
            user_job_ratings=user_job_ratings,
            job_category_indices=job_category_indices,
            user_features=user_features,
            job_features=job_features,
            category_features=category_features,
            normalize_ratings=True
        )
        
        # Check graph structure
        assert isinstance(hetero_graph, torch.utils.data.Data)
        
        # Check node types
        assert 'user' in hetero_graph.node_types
        assert 'job' in hetero_graph.node_types
        assert 'category' in hetero_graph.node_types
        
        # Check edge types
        assert ('user', 'rates', 'job') in hetero_graph.edge_types
        assert ('job', 'belongs_to', 'category') in hetero_graph.edge_types
        assert ('category', 'has_job', 'job') in hetero_graph.edge_types
        
        # Check node features
        assert hetero_graph['user'].x.shape == user_features.shape
        assert hetero_graph['job'].x.shape == job_features.shape
        assert hetero_graph['category'].x.shape == category_features.shape
        
        # Check edge indices
        assert hetero_graph['user', 'rates', 'job'].edge_index.shape[1] == len(user_job_ratings)
        assert hetero_graph['job', 'belongs_to', 'category'].edge_index.shape[1] == job_category_indices.shape[1]
        
        # Check edge weights
        assert hasattr(hetero_graph['user', 'rates', 'job'], 'edge_weight')
        assert hetero_graph['user', 'rates', 'job'].edge_weight.shape[0] == len(user_job_ratings)
