"""
Tests for the training module of JibJob recommendation system.
"""

import unittest
import torch
import numpy as np
import pandas as pd
import networkx as nx
from jibjob_recommender_system.training.trainer import Trainer
from jibjob_recommender_system.training.train_gcn import GCNTrainer
from unittest.mock import patch, MagicMock

class TestTrainer(unittest.TestCase):
    """Test cases for the Trainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model
        self.model = MagicMock()
        self.model.parameters = lambda: [torch.nn.Parameter(torch.randn(1)) for _ in range(3)]  # Mock parameters
        
        # Configuration
        self.params = {
            'num_epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 5e-4,
            'batch_size': 64,
            'early_stopping_patience': 3
        }
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            params=self.params
        )
    
    def test_init(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.num_epochs, self.params['num_epochs'])
        self.assertEqual(self.trainer.batch_size, self.params['batch_size'])
        self.assertIsNotNone(self.trainer.optimizer)
        
    def test_setup_optimizer(self):
        """Test optimizer setup."""
        # Test with default Adam optimizer
        optimizer = self.trainer._setup_optimizer(self.model, lr=0.001, weight_decay=5e-4)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        
        # Test with SGD optimizer
        with patch.dict(self.params, {'optimizer': 'sgd'}):
            trainer = Trainer(self.model, self.params)
            optimizer = trainer._setup_optimizer(self.model, lr=0.001, weight_decay=5e-4)
            self.assertIsInstance(optimizer, torch.optim.SGD)
    
    def test_setup_loss_function(self):
        """Test loss function setup."""
        # Test with default BCE loss
        loss_fn = self.trainer._setup_loss_function()
        self.assertIsInstance(loss_fn, torch.nn.BCEWithLogitsLoss)
        
        # Test with MSE loss
        with patch.dict(self.params, {'loss_function': 'mse'}):
            trainer = Trainer(self.model, self.params)
            loss_fn = trainer._setup_loss_function()
            self.assertIsInstance(loss_fn, torch.nn.MSELoss)
    
    @patch('torch.nn.BCEWithLogitsLoss')
    def test_compute_loss(self, mock_loss_fn):
        """Test loss computation."""
        # Set up mock loss
        mock_loss = MagicMock()
        mock_loss.return_value = torch.tensor(0.5)
        mock_loss_fn.return_value = mock_loss
        
        # Create predictions and targets
        predictions = torch.tensor([[0.7], [0.3]], dtype=torch.float32)
        targets = torch.tensor([[1.0], [0.0]], dtype=torch.float32)
        
        # Compute loss
        trainer = Trainer(self.model, self.params)
        loss = trainer._compute_loss(predictions, targets)
        
        # Check loss value
        self.assertEqual(loss.item(), 0.5)
        
        # loss function should have been called with predictions and targets
        mock_loss.assert_called_once()
    
    def test_early_stopping_check(self):
        """Test early stopping logic."""
        # Initialize early stopping counter and best loss
        self.trainer.early_stopping_counter = 0
        self.trainer.best_val_loss = float('inf')
        
        # Case 1: New loss is better (lower) than best
        self.assertFalse(self.trainer._early_stopping_check(0.5))
        self.assertEqual(self.trainer.early_stopping_counter, 0)
        self.assertEqual(self.trainer.best_val_loss, 0.5)
        
        # Case 2: New loss is worse, increment counter
        self.assertFalse(self.trainer._early_stopping_check(0.6))
        self.assertEqual(self.trainer.early_stopping_counter, 1)
        self.assertEqual(self.trainer.best_val_loss, 0.5)
        
        # Case 3: Multiple worse losses, but not enough for stopping
        self.assertFalse(self.trainer._early_stopping_check(0.7))
        self.assertEqual(self.trainer.early_stopping_counter, 2)
        self.assertFalse(self.trainer._early_stopping_check(0.8))
        self.assertEqual(self.trainer.early_stopping_counter, 3)
        
        # Case 4: Enough worse losses for stopping (patience = 3)
        self.assertTrue(self.trainer._early_stopping_check(0.9))
        
        # Case 5: New best resets counter
        self.trainer.early_stopping_counter = 2
        self.assertFalse(self.trainer._early_stopping_check(0.4))
        self.assertEqual(self.trainer.early_stopping_counter, 0)
        self.assertEqual(self.trainer.best_val_loss, 0.4)


class TestGCNTrainer(unittest.TestCase):
    """Test cases for the GCNTrainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model
        self.model = MagicMock()
        self.model.forward = lambda x, edge_index: x  # Identity function for simplicity
        
        # Mock graph and features
        self.graph = nx.Graph()
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle
        
        self.node_features = np.random.rand(3, 64)  # 3 nodes, 64 features each
        
        # Maps user/job IDs to node indices
        self.user_mapping = {'u1': 0, 'u2': 1}
        self.job_mapping = {'j1': 2}
        
        # Training interactions
        self.train_interactions = pd.DataFrame({
            'user_id': ['u1', 'u2'],
            'job_id': ['j1', 'j1']
        })
        
        # Validation interactions
        self.val_interactions = pd.DataFrame({
            'user_id': ['u1'],
            'job_id': ['j1']
        })
        
        # Configuration
        self.params = {
            'num_epochs': 5,
            'learning_rate': 0.001,
            'weight_decay': 5e-4,
            'batch_size': 2,
            'early_stopping_patience': 3,
            'negative_samples': 2
        }
        
        # Create trainer
        self.trainer = GCNTrainer(
            model=self.model,
            graph=self.graph,
            node_features=torch.tensor(self.node_features, dtype=torch.float32),
            user_mapping=self.user_mapping,
            job_mapping=self.job_mapping,
            train_interactions=self.train_interactions,
            val_interactions=self.val_interactions,
            params=self.params
        )
    
    def test_init(self):
        """Test GCNTrainer initialization."""
        # Check initialization
        self.assertEqual(self.trainer.model, self.model)
        self.assertEqual(self.trainer.graph, self.graph)
        self.assertEqual(self.trainer.user_mapping, self.user_mapping)
        self.assertEqual(self.trainer.job_mapping, self.job_mapping)
        self.assertEqual(len(self.trainer.train_edges), len(self.train_interactions))
        self.assertEqual(len(self.trainer.val_edges), len(self.val_interactions))
        
    def test_generate_negative_samples(self):
        """Test negative sampling logic."""
        # Generate negative samples
        pos_edges = [(0, 2), (1, 2)]  # User 0 -> Job 2, User 1 -> Job 2
        neg_edges = self.trainer._generate_negative_samples(pos_edges, n_samples=3)
        
        # Check number of samples
        self.assertEqual(len(neg_edges), 3)
        
        # Check that negative samples are not in positive edges
        for u, v in neg_edges:
            self.assertNotIn((u, v), pos_edges)
            
            # Check that first index is a user node
            self.assertIn(u, self.user_mapping.values())
            
            # Check that second index is a job node
            self.assertIn(v, self.job_mapping.values())
    
    def test_prepare_batch(self):
        """Test batch preparation."""
        # Sample data
        pos_edges = [(0, 2), (1, 2)]
        neg_edges = [(0, 3), (1, 4)]
        
        # Prepare batch
        pos_tensor, neg_tensor = self.trainer._prepare_batch(pos_edges, neg_edges)
        
        # Check tensor shapes
        self.assertEqual(pos_tensor.shape, (len(pos_edges), 2))
        self.assertEqual(neg_tensor.shape, (len(neg_edges), 2))
        
        # Check tensor values
        for i, (u, v) in enumerate(pos_edges):
            self.assertEqual(pos_tensor[i, 0].item(), u)
            self.assertEqual(pos_tensor[i, 1].item(), v)
            
        for i, (u, v) in enumerate(neg_edges):
            self.assertEqual(neg_tensor[i, 0].item(), u)
            self.assertEqual(neg_tensor[i, 1].item(), v)
    
    @patch('torch.nn.functional.binary_cross_entropy_with_logits')
    def test_compute_loss(self, mock_bce):
        """Test GCN loss computation."""
        # Set up mock
        mock_bce.return_value = torch.tensor(0.5)
        
        # Get node embeddings
        node_embeddings = torch.randn(5, 64)  # 5 nodes, 64 features
        
        # Sample edges
        pos_edges = torch.tensor([[0, 2], [1, 2]])
        neg_edges = torch.tensor([[0, 3], [1, 4]])
        
        # Compute loss
        loss = self.trainer._compute_loss(node_embeddings, pos_edges, neg_edges)
        
        # Check loss value
        self.assertEqual(loss.item(), 0.5)
        
        # BCE should have been called
        mock_bce.assert_called_once()
    
    @patch.object(GCNTrainer, '_compute_metrics')
    @patch.object(GCNTrainer, '_compute_loss')
    def test_evaluate(self, mock_loss, mock_metrics):
        """Test model evaluation."""
        # Set up mocks
        mock_loss.return_value = torch.tensor(0.3)
        mock_metrics.return_value = {'hit_rate': 0.8, 'ndcg': 0.7}
        
        # Evaluate model
        loss, metrics = self.trainer._evaluate()
        
        # Check results
        self.assertEqual(loss, 0.3)
        self.assertEqual(metrics['hit_rate'], 0.8)
        self.assertEqual(metrics['ndcg'], 0.7)
        
        # Both methods should have been called
        mock_loss.assert_called_once()
        mock_metrics.assert_called_once()
    
    @patch.object(GCNTrainer, '_evaluate')
    @patch.object(GCNTrainer, '_train_epoch')
    @patch.object(Trainer, '_early_stopping_check')
    def test_train(self, mock_early_stopping, mock_train_epoch, mock_evaluate):
        """Test training process."""
        # Set up mocks
        mock_train_epoch.return_value = 0.4
        mock_evaluate.return_value = (0.3, {'hit_rate': 0.8, 'ndcg': 0.7})
        mock_early_stopping.side_effect = [False, False, True]  # Stop after 3 epochs
        
        # Train model
        history = self.trainer.train()
        
        # Check history structure
        self.assertIn('train_loss', history)
        self.assertIn('val_loss', history)
        self.assertIn('hit_rate', history)
        self.assertIn('ndcg', history)
        
        # Check number of epochs trained
        self.assertEqual(len(history['train_loss']), 3)
        self.assertEqual(len(history['val_loss']), 3)
        
        # Check call counts
        self.assertEqual(mock_train_epoch.call_count, 3)
        self.assertEqual(mock_evaluate.call_count, 3)
        self.assertEqual(mock_early_stopping.call_count, 3)
    

if __name__ == '__main__':
    unittest.main()
