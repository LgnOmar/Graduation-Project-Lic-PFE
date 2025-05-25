"""
Module containing the GCN model architecture.
"""
import torch
from torch_geometric.nn import HeteroConv, SAGEConv
from typing import Dict, List, Tuple

class HeteroGCNLinkPredictor(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_layers: int,
        data: torch.nn.Module,
        dropout: float = 0.2
    ):
        """
        Initialize the GCN model.
        
        Args:
            hidden_channels: Number of hidden channels in each layer
            num_layers: Number of GCN layers
            data: HeteroData object containing the graph structure
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(p=dropout)
        self.convs = torch.nn.ModuleList()
        
        # Initialize GCN layers
        for i in range(num_layers):
            conv = HeteroConv({
                ('user', 'interacts_with', 'job'): SAGEConv(
                    (-1, -1), hidden_channels),
                ('job', 'rev_interacts_with', 'user'): SAGEConv(
                    (-1, -1), hidden_channels)
            })
            self.convs.append(conv)
        
        # Batch normalization layers
        self.batch_norms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_channels) 
            for _ in range(num_layers)
        ])
        
        # Final MLP for link prediction with regularization
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.ReLU(),
            self.dropout,
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.BatchNorm1d(hidden_channels // 2),
            torch.nn.ReLU(),
            self.dropout,
            torch.nn.Linear(hidden_channels // 2, 1),
            torch.nn.Sigmoid()
        )

    def encode(
        self, 
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode the input features through GCN layers.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Dictionary of node embeddings
        """
        for i, conv in enumerate(self.convs):
            # Apply convolution
            x_dict_conv = conv(x_dict, edge_index_dict)
            
            # Apply batch normalization and nonlinearity to each node type
            x_dict = {}
            for node_type, x in x_dict_conv.items():
                x = self.batch_norms[i](x)
                x = torch.nn.functional.relu(x)
                x = self.dropout(x)
                x_dict[node_type] = x
                
        return x_dict

    def decode(
        self,
        z_dict: Dict[str, torch.Tensor],
        edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode node embeddings to predict link scores.
        
        Args:
            z_dict: Dictionary of node embeddings
            edge_label_index: Indices of edges to predict
            
        Returns:
            Tensor of predicted scores
        """
        user_embeddings = z_dict['user'][edge_label_index[0]]
        job_embeddings = z_dict['job'][edge_label_index[1]]
        
        # Concatenate user and job embeddings
        z = torch.cat([user_embeddings, job_embeddings], dim=-1)
        
        # Pass through MLP to get predictions
        return self.link_predictor(z).squeeze(-1)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            edge_label_index: Indices of edges to predict
            
        Returns:
            Tensor of predicted scores for the input edges
        """
        # Encode node features
        z_dict = self.encode(x_dict, edge_index_dict)
        
        # Decode edge predictions
        pred = self.decode(z_dict, edge_label_index)
        
        return pred

    def get_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Get node embeddings for all nodes.
        
        Args:
            x_dict: Dictionary of node features
            edge_index_dict: Dictionary of edge indices
            
        Returns:
            Dictionary of node embeddings
        """
        self.eval()
        with torch.no_grad():
            return self.encode(x_dict, edge_index_dict)
