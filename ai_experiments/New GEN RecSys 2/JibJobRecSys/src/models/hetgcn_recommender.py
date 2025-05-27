"""
hetgcn_recommender.py

Purpose:
    Define the Heterogeneous GCN recommender model for the JibJob platform using PyTorch Geometric.

Key Classes:
    - HetGCNRecommender(nn.Module)
        - __init__(self, metadata, hidden_channels, out_channels, num_layers, ...)
        - forward(self, data: HeteroData, edge_label_index_dict=None)
        - Uses HGTConv or to_hetero with SAGEConv/GATConv.
        - Prediction head: dot product or MLP.

Inputs:
    - metadata: Tuple of node and edge types for HeteroData.
    - hidden_channels: Hidden dimension size.
    - out_channels: Output dimension size.
    - num_layers: Number of GNN layers.
    - data: HeteroData graph.
    - edge_label_index_dict: Edges for which to compute scores (optional).

Outputs:
    - Forward pass returns prediction scores for (Professional, Job) pairs.

High-Level Logic:
    1. Initialize HGTConv layers for each edge type.
    2. Pass node features through GNN layers.
    3. For link prediction, compute dot product or MLP score for (Professional, Job) pairs.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv

class HetGCNRecommender(nn.Module):
    def __init__(self, metadata, input_feature_dims, hidden_channels=128, out_channels=64, num_layers=2, num_heads=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(HGTConv(
                    in_channels=input_feature_dims,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads
                ))
            else:
                self.layers.append(HGTConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads
                ))
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
    def forward(self, data, edge_label_index_dict=None):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)
        # For link prediction: get embeddings for (Professional, Job) pairs
        if edge_label_index_dict is not None:
            src, dst = edge_label_index_dict[('Professional', 'interacted_with', 'Job')]
            prof_emb = x_dict['Professional'][src]
            job_emb = x_dict['Job'][dst]
            pair_emb = torch.cat([prof_emb, job_emb], dim=1)
            return torch.sigmoid(self.predictor(pair_emb)).squeeze(-1)
        return x_dict
