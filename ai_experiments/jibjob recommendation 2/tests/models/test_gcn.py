"""
Tests for the Graph Convolutional Network (GCN) model.
"""

import pytest
import torch
import numpy as np
from src.models.gcn import GCNRecommender, HeterogeneousGCN
from torch_geometric.data import Data, HeteroData


class TestGCNRecommender:
    """Tests for the GCNRecommender class."""
    
    @pytest.fixture
    def gcn_model(self):
        """Create a test GCN model."""
        return GCNRecommender(
            num_users=100,
            num_jobs=500,
            embedding_dim=32,
            hidden_dim=32,
            num_layers=2
        )
    
    def test_init(self, gcn_model):
        """Test initialization of the model."""
        assert gcn_model is not None
        assert isinstance(gcn_model, torch.nn.Module)
    
        def test_forward(self, gcn_model):
            """Test forward pass with simulated data."""
        # Create a simple bipartite graph
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],  # User nodes
            [0, 1, 1, 2, 2, 3]   # Job nodes
        ], dtype=torch.long)
        
        edge_weight = torch.ones(edge_index.size(1))
        
        # Create graph data object
        data = Data(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_users=3,
            num_jobs=4
        )
        
        # Run forward pass
        output = gcn_model(data)
        
        # Check output shapes
        assert isinstance(output, tuple)
        user_emb, job_emb = output
        assert user_emb.shape == (3, 32)  # 3 users, 32 embedding dim
        assert job_emb.shape == (4, 32)  # 4 jobs, 32 embedding dim
        
    def test_recommend(self, gcn_model):
        """Test recommendation function."""
        # Create a simple bipartite graph
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],  # User nodes
            [0, 1, 1, 2, 2, 3]   # Job nodes
        ], dtype=torch.long)
        
        edge_weight = torch.ones(edge_index.size(1))
        
        # Create graph data object
        data = Data(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_users=3,
            num_jobs=4
        )
        
        # Get recommendations for a single user
        user_id = 0
        top_k = 2
        job_ids, scores = gcn_model.recommend(
            graph=data,
            user_id=user_id,
            top_k=top_k
        )
        
        # Check output format
        assert isinstance(job_ids, list)
        assert isinstance(scores, list)
        assert len(job_ids) <= top_k
        assert len(scores) <= top_k
        assert all(0 <= job_id < 4 for job_id in job_ids)  # Job IDs should be valid
        assert all(isinstance(score, float) for score in scores)
        
        # Test batch recommendation
        user_ids = [0, 1]
        recommendations = gcn_model.recommend_batch(
            graph=data,
            user_ids=user_ids,
            top_k=2
        )
        
        # Check output format for batch recommendations
        assert isinstance(recommendations, dict)
        assert len(recommendations) == 2  # Two users
        assert 0 in recommendations
        assert 1 in recommendations
    
    def test_save_and_load_model(self, gcn_model):
        """Test saving and loading the model."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            gcn_model.save(temp_dir)
            
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "model.pt"))
            assert os.path.exists(os.path.join(temp_dir, "config.json"))
            
            # Create a new model with same architecture and load saved parameters
            loaded_model, _ = GCNRecommender.load(temp_dir)
            
            # Check that parameters are the same
            for p1, p2 in zip(gcn_model.parameters(), loaded_model.parameters()):
                assert torch.all(torch.eq(p1, p2))


class TestHeterogeneousGCN:
    """Tests for the HeterogeneousGCN class."""
    
    @pytest.fixture
    def hetero_gcn_model(self):
        """Create a test heterogeneous GCN model."""
        node_types = ['user', 'job', 'category']
        edge_types = [
            ('user', 'rates', 'job'),
            ('job', 'belongs_to', 'category'),
            ('category', 'has_job', 'job')
        ]
        node_feature_dims = {
            'user': 32,
            'job': 32,
            'category': 32
        }
        
        return HeterogeneousGCN(
            node_types=node_types,
            edge_types=edge_types,
            node_feature_dims=node_feature_dims,
            embedding_dim=32,
            hidden_dim=32
        )
    
    def test_init(self, hetero_gcn_model):
        """Test initialization of the model."""
        assert hetero_gcn_model is not None
        assert isinstance(hetero_gcn_model, torch.nn.Module)
    def test_forward(self, hetero_gcn_model):
        """Test forward pass with simulated heterogeneous graph."""
        # Create a simple heterogeneous graph
        data = HeteroData()
        
        # Add node types
        data['user'].x = torch.randn(3, 32)  # 3 users
        data['job'].x = torch.randn(4, 32)   # 4 jobs
        data['category'].x = torch.randn(2, 32)  # 2 categories
        
        # Add edge types
        data['user', 'rates', 'job'].edge_index = torch.tensor([
            [0, 0, 1, 2],  # User nodes
            [0, 1, 2, 3]   # Job nodes
        ], dtype=torch.long)
        
        data['job', 'belongs_to', 'category'].edge_index = torch.tensor([
            [0, 1, 2, 3],  # Job nodes
            [0, 1, 0, 1]   # Category nodes
        ], dtype=torch.long)
        
        data['category', 'has_job', 'job'].edge_index = torch.tensor([
            [0, 0, 1, 1],  # Category nodes
            [0, 2, 1, 3]   # Job nodes
        ], dtype=torch.long)
        
        # Create user and job indices for testing
        user_indices = torch.tensor([0, 1])
        job_indices = torch.tensor([0, 2])
        
        # Run forward pass
        output = hetero_gcn_model(data, user_indices, job_indices)
        
        # Check output is a tensor of predictions
        assert isinstance(output, torch.Tensor)
        assert output.shape == (2,)  # 2 user-job pairs

    def test_recommend(self, hetero_gcn_model):
        """Test recommendation function."""
        # Create a simple heterogeneous graph
        data = HeteroData()
        
        # Add node types
        data['user'].x = torch.randn(3, 32)  # 3 users
        data['job'].x = torch.randn(4, 32)   # 4 jobs
        data['category'].x = torch.randn(2, 32)  # 2 categories
        
        # Add edge types
        data['user', 'rates', 'job'].edge_index = torch.tensor([
            [0, 0, 1, 2],  # User nodes
            [0, 1, 2, 3]   # Job nodes
        ], dtype=torch.long)
        
        data['job', 'belongs_to', 'category'].edge_index = torch.tensor([
            [0, 1, 2, 3],  # Job nodes
            [0, 1, 0, 1]   # Category nodes
        ], dtype=torch.long)
        
        data['category', 'has_job', 'job'].edge_index = torch.tensor([
            [0, 0, 1, 1],  # Category nodes
            [0, 2, 1, 3]   # Job nodes
        ], dtype=torch.long)
        
        # Get recommendations
        user_id = 0
        top_k = 2
        job_ids, scores = hetero_gcn_model.recommend(
            graph=data,
            user_id=user_id,
            top_k=top_k
        )
        
        # Check output format
        assert isinstance(job_ids, list)
        assert isinstance(scores, list)
        assert len(job_ids) <= top_k
        assert len(scores) <= top_k
        assert all(0 <= job_id < 4 for job_id in job_ids)  # Job IDs should be valid
        assert all(isinstance(score, float) for score in scores)
