"""
graph_builder.py

Purpose:
    Build a torch_geometric.data.HeteroData graph from generated CSVs and BERT embeddings.

Key Functions:
    - build_hetero_graph(data_dfs: Dict[str, pd.DataFrame], professional_features, job_features, category_features) -> HeteroData
        - Maps entity IDs to node indices.
        - Populates node features and edge indices for all edge types.
        - Optionally adds symmetric edges.

Inputs:
    - data_dfs: Dict of DataFrames for all entities and relations.
    - professional_features: np.ndarray or torch.Tensor for professional node features.
    - job_features: np.ndarray or torch.Tensor for job node features.
    - category_features: np.ndarray or torch.Tensor for category node features.

Outputs:
    - HeteroData object for PyG.

High-Level Logic:
    1. Map original IDs to 0-based indices for each node type.
    2. Assign node features.
    3. Build edge_index tensors for each edge type.
    4. Return HeteroData object.
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

def build_hetero_graph(data_dfs, professional_features, job_features, category_features):
    # Map IDs to indices
    prof_ids = data_dfs['professionals']['professional_id'].tolist()
    job_ids = data_dfs['jobs']['job_id'].tolist()
    cat_ids = data_dfs['categories']['category_id'].tolist()
    prof_id_map = {pid: i for i, pid in enumerate(prof_ids)}
    job_id_map = {jid: i for i, jid in enumerate(job_ids)}
    cat_id_map = {cid: i for i, cid in enumerate(cat_ids)}
    data = HeteroData()
    data['Professional'].x = torch.tensor(professional_features, dtype=torch.float)
    data['Job'].x = torch.tensor(job_features, dtype=torch.float)
    data['Category'].x = torch.tensor(category_features, dtype=torch.float)
    # Edges: (Professional, selected_category, Category)
    pc = data_dfs['professional_selected_categories']
    edge_index = [
        [prof_id_map[pid] for pid in pc['professional_id']],
        [cat_id_map[cid] for cid in pc['category_id']]
    ]
    data['Professional', 'selected_category', 'Category'].edge_index = torch.tensor(edge_index, dtype=torch.long)
    # Edges: (Job, requires_category, Category)
    jc = data_dfs['job_required_categories']
    edge_index = [
        [job_id_map[jid] for jid in jc['job_id']],
        [cat_id_map[cid] for cid in jc['category_id']]
    ]
    data['Job', 'requires_category', 'Category'].edge_index = torch.tensor(edge_index, dtype=torch.long)
    # Edges: (Professional, interacted_with, Job)
    inter = data_dfs['interactions']
    edge_index = [
        [prof_id_map[pid] for pid in inter['professional_id']],
        [job_id_map[jid] for jid in inter['job_id']]
    ]
    data['Professional', 'interacted_with', 'Job'].edge_index = torch.tensor(edge_index, dtype=torch.long)
    # Make graph undirected
    data = T.ToUndirected()(data)
    return data
