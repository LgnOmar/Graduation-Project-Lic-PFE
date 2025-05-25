"""
GCN Recommender model architecture.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import HeteroConv, Linear, to_hetero
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

class GCNRecommender(torch.nn.Module):
    """
    GCN-based recommender model for homogeneous graphs.
    """
    
    def __init__(self, config: Dict[str, Any], in_channels: int):
        """
        Initialize the GCN recommender model.
        
        Args:
            config: Configuration dictionary.
            in_channels: Number of input features.
        """
        super(GCNRecommender, self).__init__()
        
        self.config = config
        self.model_config = config.get('model', {})
        
        # Model hyperparameters
        self.hidden_dim = self.model_config.get('hidden_dim', 64)
        self.embedding_dim = self.model_config.get('embedding_dim', 128)
        self.num_layers = self.model_config.get('num_layers', 2)
        self.dropout = self.model_config.get('dropout', 0.2)
        
        # GNN layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, self.hidden_dim))
        
        for i in range(1, self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))
            
        self.convs.append(GCNConv(self.hidden_dim, self.embedding_dim))
        
        # Prediction head for link prediction (optional)
        self.link_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        # Initial message passing layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x
        
    def predict_link(self, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict links between source and destination nodes.
        
        Args:
            src_embeddings: Source node embeddings.
            dst_embeddings: Destination node embeddings.
            
        Returns:
            torch.Tensor: Link prediction scores.
        """
        # Concatenate embeddings
        x = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Apply prediction head
        scores = self.link_predictor(x)
        return torch.sigmoid(scores)
        
    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        return self.forward(x, edge_index)


class SAGERecommender(torch.nn.Module):
    """
    GraphSAGE-based recommender model for homogeneous graphs.
    """
    
    def __init__(self, config: Dict[str, Any], in_channels: int):
        """
        Initialize the GraphSAGE recommender model.
        
        Args:
            config: Configuration dictionary.
            in_channels: Number of input features.
        """
        super(SAGERecommender, self).__init__()
        
        self.config = config
        self.model_config = config.get('model', {})
        
        # Model hyperparameters
        self.hidden_dim = self.model_config.get('hidden_dim', 64)
        self.embedding_dim = self.model_config.get('embedding_dim', 128)
        self.num_layers = self.model_config.get('num_layers', 2)
        self.dropout = self.model_config.get('dropout', 0.2)
        
        # SAGE layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, self.hidden_dim))
        
        for i in range(1, self.num_layers - 1):
            self.convs.append(SAGEConv(self.hidden_dim, self.hidden_dim))
            
        self.convs.append(SAGEConv(self.hidden_dim, self.embedding_dim))
        
        # Prediction head for link prediction (optional)
        self.link_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        # Initial message passing layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        return x
        
    def predict_link(self, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict links between source and destination nodes.
        
        Args:
            src_embeddings: Source node embeddings.
            dst_embeddings: Destination node embeddings.
            
        Returns:
            torch.Tensor: Link prediction scores.
        """
        # Concatenate embeddings
        x = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Apply prediction head
        scores = self.link_predictor(x)
        return torch.sigmoid(scores)
        
    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get node embeddings.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            
        Returns:
            torch.Tensor: Node embeddings.
        """
        return self.forward(x, edge_index)


class HeteroGCNRecommender(torch.nn.Module):
    """
    Heterogeneous GCN-based recommender model.
    """
    
    def __init__(self, config: Dict[str, Any], metadata, in_channels_dict: Dict[str, int]):
        """
        Initialize the HeteroGCN recommender model.
        
        Args:
            config: Configuration dictionary.
            metadata: Graph metadata (node types and edge types).
            in_channels_dict: Dictionary mapping node types to input feature dimensions.
        """
        super(HeteroGCNRecommender, self).__init__()
        
        self.config = config
        self.model_config = config.get('model', {})
        self.metadata = metadata
        
        # Model hyperparameters
        self.hidden_dim = self.model_config.get('hidden_dim', 64)
        self.embedding_dim = self.model_config.get('embedding_dim', 128)
        self.num_layers = self.model_config.get('num_layers', 2)
        self.dropout = self.model_config.get('dropout', 0.2)
        
        # Create HeteroConv layers
        self.convs = torch.nn.ModuleList()
        
        # First layer: Map different input dimensions to common hidden dimension
        conv1_dict = {}
        for edge_type in metadata[1]:
            src_type, _, dst_type = edge_type
            conv1_dict[edge_type] = SAGEConv((in_channels_dict[src_type], in_channels_dict[dst_type]), self.hidden_dim)
            
        self.convs.append(HeteroConv(conv1_dict, aggr='mean'))
        
        # Additional layers
        for i in range(1, self.num_layers):
            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv((self.hidden_dim, self.hidden_dim), 
                                             self.embedding_dim if i == self.num_layers - 1 else self.hidden_dim)
                
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))
            
        # Linear transformations for each node type
        self.lins = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.lins[node_type] = Linear(self.embedding_dim, self.embedding_dim)
            
        # Prediction head for link prediction
        self.link_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )
        
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the model.
        
        Args:
            x_dict: Dictionary mapping node types to features.
            edge_index_dict: Dictionary mapping edge types to connectivity.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping node types to embeddings.
        """
        # Initial message passing layers
        for i in range(self.num_layers - 1):
            x_dict = self.convs[i](x_dict, edge_index_dict)
            
            # Apply activation and dropout
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
            
        # Final layer
        x_dict = self.convs[-1](x_dict, edge_index_dict)
        
        # Apply linear transformations
        x_dict = {key: self.lins[key](x) for key, x in x_dict.items()}
        
        return x_dict
        
    def predict_link(self, src_embeddings: torch.Tensor, dst_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict links between source and destination nodes.
        
        Args:
            src_embeddings: Source node embeddings.
            dst_embeddings: Destination node embeddings.
            
        Returns:
            torch.Tensor: Link prediction scores.
        """
        # Concatenate embeddings
        x = torch.cat([src_embeddings, dst_embeddings], dim=1)
        
        # Apply prediction head
        scores = self.link_predictor(x)
        return torch.sigmoid(scores)
        
    def get_embedding(self, x_dict, edge_index_dict):
        """
        Get node embeddings.
        
        Args:
            x_dict: Dictionary mapping node types to features.
            edge_index_dict: Dictionary mapping edge types to connectivity.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping node types to embeddings.
        """
        return self.forward(x_dict, edge_index_dict)
