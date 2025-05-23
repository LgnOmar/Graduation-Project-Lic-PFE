"""
Graph Convolutional Network (GCN) model for JibJob recommendation system.
This module handles the graph-based recommendation model that learns from user-job interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv
from torch_geometric.data import Data, HeteroData
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging
import os
import json

# Setup logging
logger = logging.getLogger(__name__) 

class GCNRecommender(nn.Module):
    """
    Graph Convolutional Network model for job recommendations.
    
    This model uses a GCN to learn representations of users and jobs from
    their interaction graph, and produces personalized job recommendations.
    """
    
    def __init__(
        self,
        num_users: int,
        num_jobs: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = 'gcn'
    ):
        """
        Initialize the GCN recommendation model.
        
        Args:
            num_users: Number of users in the system.
            num_jobs: Number of jobs in the system.
            embedding_dim: Dimension of the initial embeddings.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of graph convolution layers.
            dropout: Dropout rate for regularization.
            conv_type: Type of graph convolution ('gcn', 'sage', or 'gat').
        """
        super(GCNRecommender, self).__init__()
        
        self.num_users = num_users
        self.num_jobs = num_jobs
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # User and job embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.job_embedding = nn.Embedding(num_jobs, embedding_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        
        # First layer (input dimension is embedding_dim)
        if conv_type == 'gcn':
            self.convs.append(GCNConv(embedding_dim, hidden_dim))
        elif conv_type == 'sage':
            self.convs.append(SAGEConv(embedding_dim, hidden_dim))
        elif conv_type == 'gat':
            self.convs.append(GATConv(embedding_dim, hidden_dim))
        else:
            raise ValueError(f"Unknown convolution type: {conv_type}")
        
        # Additional layers (input and output dimension are both hidden_dim)
        for _ in range(num_layers - 1):
            if conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
        
        # Final prediction layer
        self.prediction = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize model parameters."""
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.job_embedding.weight, std=0.1)
        
        # Initialize prediction layer
        nn.init.xavier_uniform_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)
        
    def forward(
        self,
        graph: Data,
        user_indices: torch.Tensor,
        job_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the GCN model.
        
        Args:
            graph: PyTorch Geometric Data object containing the graph.
            user_indices: Tensor of user indices.
            job_indices: Tensor of job indices.
            
        Returns:
            torch.Tensor: Predicted ratings/scores for the user-job pairs.
        """
        # Get node features by combining user and job embeddings
        x = torch.cat([
            self.user_embedding.weight,  # [num_users, embedding_dim]
            self.job_embedding.weight    # [num_jobs, embedding_dim]
        ], dim=0)  # [num_users + num_jobs, embedding_dim]
        
        # Apply graph convolutions
        edge_index = graph.edge_index
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # Not the last layer
                x = F.relu(x)
                x = self.dropout(x)
        
        # Get user and job embeddings from the result
        user_emb = x[user_indices]
        job_emb = x[self.num_users + job_indices]
        
        # Concatenate user and job embeddings
        combined_emb = torch.cat([user_emb, job_emb], dim=1)
        
        # Predict ratings
        predictions = self.prediction(combined_emb).squeeze(-1)
        
        return predictions
    
    def recommend(
        self,
        graph: Data,
        user_id: int,
        top_k: int = 10,
        exclude_rated: bool = True,
        rated_indices: Optional[List[int]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        Get top-k job recommendations for a user.
        
        Args:
            graph: PyTorch Geometric Data object containing the graph.
            user_id: ID of the user to recommend jobs for.
            top_k: Number of recommendations to return.
            exclude_rated: Whether to exclude jobs the user has already rated.
            rated_indices: List of job indices the user has already rated.
                           Required if exclude_rated is True.
            
        Returns:
            Tuple[List[int], List[float]]: Job IDs and their scores.
        """
        self.eval()
        with torch.no_grad():
            # Get all jobs
            all_job_indices = torch.arange(self.num_jobs, device=graph.edge_index.device)
            
            # Create user-job pairs for all jobs
            user_indices = torch.full_like(all_job_indices, user_id)
            
            # Get predictions for all pairs
            predictions = self.forward(graph, user_indices, all_job_indices)
            
            # Exclude rated jobs if needed
            if exclude_rated and rated_indices:
                mask = torch.ones(self.num_jobs, dtype=torch.bool, device=predictions.device)
                mask[rated_indices] = False
                predictions = predictions[mask]
                job_indices = all_job_indices[mask]
            else:
                job_indices = all_job_indices
            
            # Get top-k recommendations
            if len(predictions) <= top_k:
                top_indices = torch.argsort(predictions, descending=True)
                top_jobs = job_indices[top_indices].cpu().numpy().tolist()
                top_scores = predictions[top_indices].cpu().numpy().tolist()
            else:
                top_values, top_indices = torch.topk(predictions, top_k)
                top_jobs = job_indices[top_indices].cpu().numpy().tolist()
                top_scores = top_values.cpu().numpy().tolist()
        
        return top_jobs, top_scores
    
    def recommend_batch(
        self,
        graph: Data,
        user_ids: List[int],
        top_k: int = 10,
        exclude_rated: bool = True,
        user_rated_indices: Optional[Dict[int, List[int]]] = None
    ) -> Dict[int, Tuple[List[int], List[float]]]:
        """
        Get top-k job recommendations for multiple users.
        
        Args:
            graph: PyTorch Geometric Data object containing the graph.
            user_ids: List of user IDs to recommend jobs for.
            top_k: Number of recommendations per user.
            exclude_rated: Whether to exclude jobs each user has already rated.
            user_rated_indices: Dictionary mapping user IDs to lists of rated job indices.
                                Required if exclude_rated is True.
            
        Returns:
            Dict[int, Tuple[List[int], List[float]]]: 
                Dictionary mapping user IDs to their recommendations and scores.
        """
        recommendations = {}
        for user_id in user_ids:
            rated_indices = user_rated_indices.get(user_id, []) if user_rated_indices else None
            recs, scores = self.recommend(
                graph, user_id, top_k, exclude_rated, rated_indices
            )
            recommendations[user_id] = (recs, scores)
        
        return recommendations
    
    def save(self, path: str, metadata: Optional[Dict] = None):
        """
        Save the model and metadata.
        
        Args:
            path: Directory path to save the model.
            metadata: Additional metadata to save with the model.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save model configuration
        config = {
            'num_users': self.num_users,
            'num_jobs': self.num_jobs,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout.p
        }
        
        # Add metadata if provided
        if metadata:
            config['metadata'] = metadata
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """
        Load a saved model.
        
        Args:
            path: Directory path where the model is saved.
            device: Device to load the model to ('cpu' or 'cuda').
            
        Returns:
            GCNRecommender: Loaded model.
        """
        # Autoriser les classes torch_geometric lors du chargement
        import torch.serialization
        torch.serialization.add_safe_globals(['torch_geometric.data.data.DataEdgeAttr'])
        
        # Load configuration
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Extract metadata if present
        metadata = config.pop('metadata', None)
        
        # Create model
        model = cls(**config)
        
        # Load model state
        try:
            # Essayer d'abord avec les paramètres par défaut
            state_dict = torch.load(os.path.join(path, 'model.pt'), map_location=device)
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Erreur lors du chargement du modèle: {e}")
            logger.warning("Tentative avec weights_only=False...")
            # Essayer avec weights_only=False pour PyTorch 2.6+
            state_dict = torch.load(
                os.path.join(path, 'model.pt'), 
                map_location=device,
                weights_only=False
            )
            model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        return model, metadata


class HeterogeneousGCN(nn.Module):
    """
    Heterogeneous Graph Convolutional Network for job recommendations.
    
    This model handles heterogeneous graphs with different node and edge types,
    appropriate for complex recommendation scenarios.
    """
    
    def __init__(
        self,
        node_types: List[str],
        edge_types: List[Tuple[str, str, str]],
        node_feature_dims: Dict[str, int],
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_layer_type: str = 'sage'
    ):
        """
        Initialize the heterogeneous GCN recommendation model.
        
        Args:
            node_types: List of node types (e.g., ['user', 'job', 'skill']).
            edge_types: List of edge types as (src_type, relation, dst_type).
            node_feature_dims: Dict mapping node types to their feature dimensions.
            embedding_dim: Dimension of the node embeddings.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of graph convolution layers.
            dropout: Dropout rate for regularization.
        """
        super(HeterogeneousGCN, self).__init__()

        #store the original dimensions passed
        self.input_node_feature_dims = node_feature_dims
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Node type-specific embeddings or feature transformations
        self.node_embeddings = nn.ModuleDict({
            node_type: nn.Linear(feature_dim, embedding_dim)
            for node_type, feature_dim in self.input_node_feature_dims.items()
        })

        # Import here to avoid circular imports
        from torch_geometric.nn import HeteroConv, GCNConv
        
        # Heterogeneous graph convolution layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            conv_dict = {}
            for src_node_type, rel_type, dst_node_type in edge_types:
                # Determine input dimension for this specific GCNConv
                # For the first HeteroConv layer (i=0), input is self.embedding_dim (after projection)
                # For subsequent layers (i>0), input is self.hidden_dim (output of previous layer)
                current_in_channels = self.embedding_dim if i == 0 else self.hidden_dim
                
                # *** SWITCH TO SAGEConv ***
                conv_dict[(src_node_type, rel_type, dst_node_type)] = SAGEConv(
                    in_channels=(current_in_channels, current_in_channels), # SAGEConv for bipartite takes (in_channels_src, in_channels_dst)
                                                                           # After our projection, both user and job features going into
                                                                           # the first SAGEConv layer are of size `embedding_dim`.
                                                                           # For subsequent layers, both are `hidden_dim`.
                    out_channels=self.hidden_dim,
                    # SAGEConv does not have 'add_self_loops' or 'normalize' in the same way GCNConv does.
                    # It has its own aggregation mechanism.
                    # Default aggregator for SAGEConv is 'mean'.
                )
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
        
        # Final prediction layer
        self.prediction = nn.Linear(hidden_dim * 2, 1)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        graph: HeteroData,
        user_indices: torch.Tensor,
        job_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the heterogeneous GCN model.
        
        Args:
            graph: PyTorch Geometric HeteroData object containing the graph.
            user_indices: Tensor of user indices.
            job_indices: Tensor of job indices.
            
        Returns:
            torch.Tensor: Predicted ratings/scores for the user-job pairs.
        """
        # Initial node features
        x_dict = {}
        for node_type in self.node_types:
            # Apply type-specific embedding or transformation
            x_dict[node_type] = self.node_embeddings[node_type](graph[node_type].x)
        
        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            # Apply current convolution layer
            x_dict = conv(x_dict, graph.edge_index_dict)
            
            # Apply non-linearity and dropout
            if i < len(self.convs) - 1:  # Not the last layer
                for node_type in x_dict:
                    x_dict[node_type] = F.relu(x_dict[node_type])
                    x_dict[node_type] = self.dropout(x_dict[node_type])
        
        # Get user and job embeddings
        user_emb = x_dict['user'][user_indices]
        job_emb = x_dict['job'][job_indices]
        
        # Concatenate user and job embeddings
        combined_emb = torch.cat([user_emb, job_emb], dim=1)
        
        # Predict ratings
        predictions = self.prediction(combined_emb).squeeze(-1)
        
        return predictions
    
    def save(self, path: str, metadata: Optional[Dict] = None):
        """
        Save the heterogeneous GCN model and metadata.
        
        Args:
            path: Directory path to save the model.
            metadata: Additional metadata to save with the model.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(path, 'model.pt'))
        
        # Save model configuration
        config = {
            'node_types': self.node_types,
            'edge_types': self.edge_types,
            # 'node_feature_dims' was the constructor argument. We now save what was passed.
            'input_node_feature_dims': self.input_node_feature_dims,  # <-- THIS IS THE ADDED LINE
            'embedding_dim': self.embedding_dim, # Target embedding dim for GCN layers
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout.p
        }
        
        # Add metadata if provided
        if metadata:
            config['metadata'] = metadata
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4) # indent for readability
        logger.info(f"HeterogeneousGCN model configuration saved to {os.path.join(path, 'config.json')}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """
        Load a saved heterogeneous GCN model.
        
        Args:
            path: Directory path where the model is saved.
            node_feature_dims: Dict mapping node types to their feature dimensions.
            device: Device to load the model to ('cpu' or 'cuda').
            
        Returns:
            HeterogeneousGCN: Loaded model.
            Dict: Metadata that was saved with the model.
        """
        # Load configuration
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
            node_feature_dims_loaded = config.pop('input_node_feature_dims')
            model = cls(node_feature_dims=node_feature_dims_loaded, **config)
        
        # Extract metadata if present
        metadata = config.pop('metadata', None)
        
        # Create model
        model = cls(node_feature_dims=node_feature_dims, **config)
        
        # Load model state
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location=device)
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        return model, metadata
