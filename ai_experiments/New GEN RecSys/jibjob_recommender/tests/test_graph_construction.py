"""
Tests for the graph construction module of JibJob recommendation system.
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
import torch
from jibjob_recommender_system.graph_construction.graph_builder import GraphBuilder
from unittest.mock import MagicMock, patch

class TestGraphBuilder(unittest.TestCase):
    """Test cases for the GraphBuilder class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'graph_construction': {
                'similarity_threshold': 0.7,
                'max_distance_km': 50.0,
                'use_heterogeneous_graph': True,
                'edge_types': ['category', 'location', 'similarity', 'application']
            }
        }
        self.graph_builder = GraphBuilder(config=self.config)
        
        # Sample data for testing
        self.feature_data = {
            'users': pd.DataFrame({
                'user_id': ['u1', 'u2', 'u3'],
                'user_type': ['professional', 'professional', 'employer'],
                'categories': [['cat1', 'cat2'], ['cat2', 'cat3'], ['cat1']],
                'latitude': [34.05, 34.10, 34.15],
                'longitude': [-118.25, -118.30, -118.20],
                'embedding': [np.random.rand(64).tolist() for _ in range(3)]
            }).set_index('user_id'),
            
            'jobs': pd.DataFrame({
                'job_id': ['j1', 'j2', 'j3', 'j4'],
                'employer_id': ['u3', 'u3', 'e2', 'e3'],
                'categories': [['cat1'], ['cat2'], ['cat1', 'cat3'], ['cat4']],
                'latitude': [34.07, 34.12, 34.08, 34.20],
                'longitude': [-118.27, -118.32, -118.22, -118.15],
                'embedding': [np.random.rand(64).tolist() for _ in range(4)]
            }).set_index('job_id'),
            
            'categories': pd.DataFrame({
                'category_id': ['cat1', 'cat2', 'cat3', 'cat4'],
                'category_name': ['Electrician', 'Plumber', 'Tutor', 'Gardener']
            }).set_index('category_id'),
            
            'job_applications': pd.DataFrame({
                'application_id': ['app1', 'app2', 'app3'],
                'job_id': ['j1', 'j2', 'j1'],
                'user_id': ['u1', 'u1', 'u2'],
                'application_status': ['applied', 'applied', 'applied']
            })
        }
    
    def test_initialize_graph(self):
        """Test graph initialization."""
        # Test homogeneous graph initialization
        homogeneous_graph = self.graph_builder._initialize_graph(heterogeneous=False)
        self.assertIsInstance(homogeneous_graph, nx.Graph)
        
        # Test heterogeneous graph initialization
        heterogeneous_graph = self.graph_builder._initialize_graph(heterogeneous=True)
        self.assertIsInstance(heterogeneous_graph, nx.Graph)
        # Check if the heterogeneous flag is stored
        self.assertTrue(hasattr(heterogeneous_graph, 'is_heterogeneous'))
        self.assertTrue(getattr(heterogeneous_graph, 'is_heterogeneous', False))
    
    def test_add_nodes_to_graph(self):
        """Test adding nodes to graph."""
        graph = self.graph_builder._initialize_graph(heterogeneous=True)
        
        # Extract professionals from users
        professionals = self.feature_data['users'][self.feature_data['users']['user_type'] == 'professional']
        
        # Create node mappings
        user_mapping = {}
        node_features = []
        node_types = []
        
        # Add professional nodes
        self.graph_builder._add_nodes_to_graph(
            graph=graph,
            df=professionals,
            node_type='professional',
            id_column=professionals.index.name,
            feature_column='embedding',
            node_mapping=user_mapping,
            all_node_features=node_features,
            all_node_types=node_types,
            node_start_idx=0
        )
        
        # Check graph properties
        self.assertEqual(len(graph.nodes()), len(professionals))
        self.assertEqual(len(user_mapping), len(professionals))
        self.assertEqual(len(node_features), len(professionals))
        self.assertEqual(len(node_types), len(professionals))
        
        # Verify node attributes
        for i, user_id in enumerate(professionals.index):
            self.assertIn(user_id, user_mapping)
            node_idx = user_mapping[user_id]
            self.assertEqual(graph.nodes[node_idx].get('type'), 'professional')
            self.assertEqual(graph.nodes[node_idx].get('original_id'), user_id)
            self.assertEqual(node_types[node_idx], 'professional')
    
    def test_add_category_edges(self):
        """Test adding category-based edges."""
        graph = self.graph_builder._initialize_graph(heterogeneous=True)
        
        # Create node mappings for professionals and categories
        user_mapping = {'u1': 0, 'u2': 1}
        category_mapping = {'cat1': 2, 'cat2': 3, 'cat3': 4}
        
        # Add nodes to graph
        for i, user_id in enumerate(['u1', 'u2']):
            graph.add_node(i, type='professional', original_id=user_id)
            
        for i, cat_id in enumerate(['cat1', 'cat2', 'cat3']):
            graph.add_node(i+2, type='category', original_id=cat_id)
        
        # Create user categories data
        user_categories = {
            'u1': ['cat1', 'cat2'],
            'u2': ['cat2', 'cat3']
        }
        
        # Add edges based on categories
        self.graph_builder._add_category_edges(
            graph=graph,
            user_mapping=user_mapping,
            category_mapping=category_mapping,
            user_categories=user_categories
        )
        
        # Check edge count
        expected_edges = 4  # u1-cat1, u1-cat2, u2-cat2, u2-cat3
        self.assertEqual(len(graph.edges()), expected_edges)
        
        # Verify specific edges
        self.assertTrue(graph.has_edge(0, 2))  # u1-cat1
        self.assertTrue(graph.has_edge(0, 3))  # u1-cat2
        self.assertTrue(graph.has_edge(1, 3))  # u2-cat2
        self.assertTrue(graph.has_edge(1, 4))  # u2-cat3
        
        # Check edge attributes
        for u, v in graph.edges():
            self.assertEqual(graph.edges[u, v].get('type'), 'has_category')
            self.assertEqual(graph.edges[u, v].get('weight'), 1.0)
    
    def test_build_graph(self):
        """Test the complete graph building process."""
        # Mock the internal methods to simplify testing
        with patch.object(self.graph_builder, '_initialize_graph') as mock_init:
            with patch.object(self.graph_builder, '_add_nodes_to_graph') as mock_add_nodes:
                with patch.object(self.graph_builder, '_add_category_edges') as mock_add_categories:
                    with patch.object(self.graph_builder, '_add_application_edges') as mock_add_applications:
                        with patch.object(self.graph_builder, '_add_location_edges') as mock_add_location:
                            with patch.object(self.graph_builder, '_add_similarity_edges') as mock_add_similarity:
                                # Create a mock graph and return it when _initialize_graph is called
                                mock_graph = nx.Graph()
                                mock_init.return_value = mock_graph
                                
                                # Call the build_graph method
                                result = self.graph_builder.build_graph(self.feature_data)
                                
                                # Check if all methods were called with correct parameters
                                mock_init.assert_called_once()
                                self.assertEqual(mock_add_nodes.call_count, 3)  # professionals, jobs, categories
                                mock_add_categories.assert_called_once()
                                mock_add_applications.assert_called_once()
                                mock_add_location.assert_called_once()
                                mock_add_similarity.assert_called_once()
                                
                                # Check result structure
                                self.assertIn('graph', result)
                                self.assertIn('user_mapping', result)
                                self.assertIn('job_mapping', result)
                                self.assertIn('node_features', result)
                                self.assertIn('node_types', result)
                                
    def test_convert_to_pytorch_geometric(self):
        """Test conversion to PyTorch Geometric format."""
        # Create a simple graph
        graph = nx.Graph()
        graph.add_node(0, type='professional')
        graph.add_node(1, type='job')
        graph.add_edge(0, 1, weight=0.8, type='similar')
        
        # Create node features
        node_features = np.array([
            [1.0, 0.0],  # Professional features
            [0.0, 1.0]   # Job features
        ])
        
        # Convert to PyTorch Geometric format
        with patch('torch.tensor', side_effect=lambda x: x):  # Mock torch.tensor to return input
            try:
                # This test will work if PyTorch Geometric is available
                pyg_data = self.graph_builder._convert_to_pytorch_geometric(
                    graph=graph,
                    node_features=node_features,
                    node_types=['professional', 'job']
                )
                
                # Check PyG data structure
                self.assertIn('x', pyg_data)
                self.assertIn('edge_index', pyg_data)
                self.assertIn('edge_attr', pyg_data)
                self.assertIn('node_type', pyg_data)
                
            except ImportError:
                # PyTorch Geometric might not be available in test environment
                self.skipTest("PyTorch Geometric not available, skipping conversion test")


if __name__ == '__main__':
    unittest.main()
