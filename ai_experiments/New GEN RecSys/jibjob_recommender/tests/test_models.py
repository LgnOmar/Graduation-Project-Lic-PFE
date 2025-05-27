"""
Tests for the models module of JibJob recommendation system.
"""

import unittest
import torch
import numpy as np
from jibjob_recommender_system.models.gcn_recommender import GCNRecommender
from jibjob_recommender_system.models.base_recommender import BaseRecommender
from unittest.mock import patch, MagicMock

class TestBaseRecommender(unittest.TestCase):
    """Test cases for the BaseRecommender class."""
    
    def test_abstract_methods(self):
        """Test that BaseRecommender is an abstract class with required methods."""
        # Check that BaseRecommender can't be instantiated directly
        with self.assertRaises(TypeError):
            base = BaseRecommender()
        
        # Create a minimal concrete implementation
        class ConcreteRecommender(BaseRecommender):
            def forward(self, x):
                return x
            
            def predict(self, x):
                return x
        
        # Should be able to instantiate concrete implementation
        recommender = ConcreteRecommender()
        self.assertIsNotNone(recommender)


class TestGCNRecommender(unittest.TestCase):
    """Test cases for the GCNRecommender class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock Graph neural network modules to avoid needing the actual libraries
        self.patcher1 = patch('jibjob_recommender_system.models.gcn_recommender.GCNConv')
        self.patcher2 = patch('jibjob_recommender_system.models.gcn_recommender.HeteroConv')
        self.patcher3 = patch('jibjob_recommender_system.models.gcn_recommender.SAGEConv')
        self.patcher4 = patch('jibjob_recommender_system.models.gcn_recommender.GATConv')
        
        # Start the patchers
        self.mockGCNConv = self.patcher1.start()
        self.mockHeteroConv = self.patcher2.start()
        self.mockSAGEConv = self.patcher3.start()
        self.mockGATConv = self.patcher4.start()
        
        # Set up mock behavior
        self.mockGCNConv.return_value = MagicMock()
        self.mockGCNConv.return_value.forward = lambda x, edge_index: x  # Identity function
        self.mockHeteroConv.return_value = MagicMock()
        self.mockSAGEConv.return_value = MagicMock()
        self.mockGATConv.return_value = MagicMock()
        
        # Model parameters
        self.model_params = {
            'input_dim': 64,
            'hidden_dims': [32, 16],
            'output_dim': 8,
            'dropout': 0.2,
            'use_hetero_gnn': False
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
    
    def test_init(self):
        """Test model initialization."""
        # Initialize model
        model = GCNRecommender(self.model_params)
        
        # Check attributes
        self.assertEqual(model.input_dim, 64)
        self.assertEqual(model.output_dim, 8)
        self.assertEqual(model.hidden_dims, [32, 16])
        self.assertAlmostEqual(model.dropout_rate, 0.2)
        self.assertFalse(model.use_hetero_gnn)
        
        # Check layers
        self.assertIsNotNone(model.gcn_layers)
        self.assertEqual(len(model.gcn_layers), 2)  # Should have 2 GCN layers based on hidden_dims
        
    def test_init_hetero(self):
        """Test heterogeneous GNN initialization."""
        # Set parameters for HeteroGNN
        hetero_params = self.model_params.copy()
        hetero_params['use_hetero_gnn'] = True
        hetero_params['node_types'] = ['user', 'job', 'category']
        hetero_params['edge_types'] = [('user', 'applied_to', 'job'), ('user', 'has', 'category')]
        
        # Initialize model
        model = GCNRecommender(hetero_params)
        
        # Check heterogeneous attributes
        self.assertTrue(model.use_hetero_gnn)
        self.assertEqual(model.node_types, ['user', 'job', 'category'])
        self.assertEqual(model.edge_types, [('user', 'applied_to', 'job'), ('user', 'has', 'category')])
        
        # HeteroConv should have been called
        self.mockHeteroConv.assert_called()
    
    @patch('torch.nn.functional.relu')
    @patch('torch.nn.functional.dropout')
    def test_forward(self, mock_dropout, mock_relu):
        """Test forward pass."""
        # Set up mocks
        mock_relu.side_effect = lambda x: x  # Identity function
        mock_dropout.side_effect = lambda x, p, training: x  # Identity function
        
        # Initialize model
        model = GCNRecommender(self.model_params)
        
        # Create dummy input
        x = torch.tensor(np.random.rand(10, 64), dtype=torch.float32)  # 10 nodes, 64 features
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.long)  # 5 edges
        
        # Call forward
        output = model(x, edge_index)
        
        # Check output
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (10, 8))  # 10 nodes, 8 output features
    
    @patch.object(GCNRecommender, 'forward')
    def test_predict(self, mock_forward):
        """Test prediction method."""
        # Set up mock
        mock_output = torch.tensor(np.random.rand(10, 8), dtype=torch.float32)
        mock_forward.return_value = mock_output
        
        # Initialize model
        model = GCNRecommender(self.model_params)
        
        # Create dummy input
        x = torch.tensor(np.random.rand(10, 64), dtype=torch.float32)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=torch.long)
        
        # Call predict
        predictions = model.predict(x, edge_index)
        
        # Check output
        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape, (10, 8))
        
        # forward should have been called
        mock_forward.assert_called_once_with(x, edge_index)
    
    def test_save_and_load(self):
        """Test model saving and loading functionality."""
        # Initialize model
        model = GCNRecommender(self.model_params)
        
        # Mock torch.save and torch.load
        with patch('torch.save') as mock_save:
            with patch('torch.load', return_value={'state_dict': {}, 'params': self.model_params}) as mock_load:
                # Test save
                model.save('dummy_path.pt')
                mock_save.assert_called_once()
                
                # Test load
                loaded_model = GCNRecommender.load('dummy_path.pt')
                mock_load.assert_called_once()
                self.assertIsInstance(loaded_model, GCNRecommender)
                self.assertEqual(loaded_model.input_dim, model.input_dim)
                self.assertEqual(loaded_model.hidden_dims, model.hidden_dims)
                self.assertEqual(loaded_model.output_dim, model.output_dim)


if __name__ == '__main__':
    unittest.main()
