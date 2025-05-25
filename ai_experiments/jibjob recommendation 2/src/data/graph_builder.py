"""
Graph building utilities for JibJob recommendation system.
This module contains functions to build interaction graphs for the recommendation system.
"""

import torch
import numpy as np
from torch_geometric.data import Data, HeteroData
import scipy.sparse as sp
from typing import Tuple, List, Dict, Union, Optional

def build_interaction_graph(
    user_indices: torch.Tensor,
    job_indices: torch.Tensor,
    ratings: torch.Tensor,
    num_users: int,
    num_jobs: int,
    threshold: Optional[float] = None,
    normalize_ratings: bool = False
) -> Data:
    """
    Build a bipartite graph from user-job interactions.
    
    Args:
        user_indices: Tensor with user indices.
        job_indices: Tensor with job indices.
        ratings: Tensor with rating values.
        num_users: Total number of users.
        num_jobs: Total number of jobs.
        threshold: Optional threshold for including interactions.
                  If provided, only interactions with ratings above the threshold will be included.
        normalize_ratings: Whether to normalize ratings to [0, 1] range.
        
    Returns:
        torch_geometric.data.Data: PyTorch Geometric data object representing the graph.
    """
    # Check if tensors are on the same device
    device = user_indices.device
    
    # Filter interactions based on threshold if provided
    if threshold is not None:
        mask = ratings > threshold
        user_indices = user_indices[mask]
        job_indices = job_indices[mask]
        ratings = ratings[mask]
    
    # Normalize ratings if required
    if normalize_ratings and ratings.numel() > 0:
        min_rating = ratings.min()
        max_rating = ratings.max()
        if max_rating > min_rating:
            ratings = (ratings - min_rating) / (max_rating - min_rating)
    
    # Create edge indices for a bipartite graph
    # For each user-job interaction, we create two edges:
    # 1. user -> job
    # 2. job -> user (adding offset to job indices)
    
    # First edge: user -> job
    source_nodes = user_indices
    target_nodes = job_indices + num_users  # Offset job indices
    
    # Second edge: job -> user
    source_nodes_reverse = job_indices + num_users  # Offset job indices
    target_nodes_reverse = user_indices
    
    # Combine edges
    edge_index = torch.stack([
        torch.cat([source_nodes, source_nodes_reverse]),
        torch.cat([target_nodes, target_nodes_reverse])
    ], dim=0)
    
    # Duplicate ratings for the reverse edges
    edge_weights = torch.cat([ratings, ratings])
    
    # Create node features (can be extended with more features)
    # Here we initialize with ones, but in practice you'd want to use
    # more meaningful features
    x = torch.ones((num_users + num_jobs, 1), device=device)
    
    # Create graph data object
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_weights.view(-1, 1)
    )
    
    # Store the number of users and jobs as graph attributes
    graph.num_users = num_users
    graph.num_jobs = num_jobs
    
    return graph

def build_heterogeneous_graph(
    user_indices: torch.Tensor,
    job_indices: torch.Tensor,
    ratings: torch.Tensor,
    user_features: Optional[torch.Tensor] = None,
    job_features: Optional[torch.Tensor] = None,
    num_users: int = None,
    num_jobs: int = None,
    extra_relations: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
) -> HeteroData:
    """
    Build a heterogeneous graph for the recommendation system with multiple node and edge types.
    
    Args:
        user_indices: Tensor with user indices.
        job_indices: Tensor with job indices.
        ratings: Tensor with rating values.
        user_features: Optional features for user nodes.
        job_features: Optional features for job nodes.
        num_users: Total number of users.
        num_jobs: Total number of jobs.
        extra_relations: Optional dictionary containing additional relations.
                         Each key is a relation name, and the value is a dictionary with
                         'edge_index' and 'edge_attr' tensors.
        
    Returns:
        torch_geometric.data.HeteroData: Heterogeneous graph.
    """
    # Infer numbers if not provided
    if num_users is None:
        num_users = user_indices.max().item() + 1
    if num_jobs is None:
        num_jobs = job_indices.max().item() + 1
    
    # Create heterogeneous graph
    hetero_graph = HeteroData()
    
    # Add user nodes
    if user_features is not None:
        hetero_graph['user'].x = user_features
    else:
        # Default: create one-hot encodings
        hetero_graph['user'].x = torch.eye(num_users)
    
    # Add job nodes
    if job_features is not None:
        hetero_graph['job'].x = job_features
    else:
        # Default: create one-hot encodings
        hetero_graph['job'].x = torch.eye(num_jobs)
    
    # Add user-job rating edges
    hetero_graph['user', 'rates', 'job'].edge_index = torch.stack([user_indices, job_indices])
    hetero_graph['user', 'rates', 'job'].edge_attr = ratings.view(-1, 1)
    
    # Add job-user reverse edges
    hetero_graph['job', 'rated_by', 'user'].edge_index = torch.stack([job_indices, user_indices])
    hetero_graph['job', 'rated_by', 'user'].edge_attr = ratings.view(-1, 1)
    
    # Add extra relations if provided
    if extra_relations:
        for relation_name, relation_data in extra_relations.items():
            src_type, rel_type, dst_type = relation_name.split('__')
            hetero_graph[src_type, rel_type, dst_type].edge_index = relation_data['edge_index']
            
            if 'edge_attr' in relation_data:
                hetero_graph[src_type, rel_type, dst_type].edge_attr = relation_data['edge_attr']
    
    return hetero_graph

def adjacency_matrix_to_edge_index(adj_matrix: sp.spmatrix) -> torch.Tensor:
    """
    Convert a sparse adjacency matrix to edge index format.
    
    Args:
        adj_matrix: Sparse adjacency matrix.
        
    Returns:
        torch.Tensor: Edge index tensor with shape [2, num_edges].
    """
    # Convert to COO format if not already
    coo = adj_matrix.tocoo()
    
    # Extract row and column indices
    row = torch.from_numpy(coo.row).long()
    col = torch.from_numpy(coo.col).long()
    
    # Stack to create edge index
    edge_index = torch.stack([row, col], dim=0)
    
    return edge_index

def create_job_similarity_graph(
    job_features: torch.Tensor,
    threshold: float = 0.7,
    metric: str = 'cosine'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a job similarity graph based on feature similarity.
    
    Args:
        job_features: Tensor of job features [num_jobs, feature_dim].
        threshold: Similarity threshold for creating an edge.
        metric: Similarity metric ('cosine', 'dot', or 'euclidean').
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Edge index and edge weights.
    """
    # Normalize features for cosine similarity
    if metric == 'cosine':
        norms = torch.norm(job_features, dim=1, keepdim=True)
        # Avoid division by zero
        norms = torch.where(norms > 0, norms, torch.ones_like(norms))
        normalized_features = job_features / norms
        
        # Compute pairwise cosine similarities
        similarity = torch.mm(normalized_features, normalized_features.t())
    
    elif metric == 'dot':
        # Dot product similarity
        similarity = torch.mm(job_features, job_features.t())
    
    elif metric == 'euclidean':
        # Euclidean distance (converted to similarity)
        n = job_features.size(0)
        dist = torch.cdist(job_features, job_features)
        max_dist = dist.max()
        # Convert distance to similarity (1 - normalized_distance)
        similarity = 1.0 - (dist / max_dist)
    
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")
    
    # Apply threshold
    mask = (similarity > threshold) & (torch.eye(job_features.size(0), device=job_features.device) == 0)
    
    # Extract edge indices and weights
    indices = mask.nonzero(as_tuple=True)
    edge_index = torch.stack(indices)
    edge_weights = similarity[indices]
    
    return edge_index, edge_weights

def create_skill_job_graph(
    job_skill_matrix: Union[np.ndarray, sp.spmatrix, torch.Tensor],
    num_jobs: int,
    num_skills: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a bipartite graph linking jobs and skills.
    
    Args:
        job_skill_matrix: Matrix where rows are jobs and columns are skills.
                          A non-zero value indicates that the job requires the skill.
        num_jobs: Number of jobs.
        num_skills: Number of skills.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Edge index and edge weights.
    """
    # Convert to torch tensor if needed
    if isinstance(job_skill_matrix, np.ndarray):
        job_skill_matrix = torch.from_numpy(job_skill_matrix)
    elif sp.issparse(job_skill_matrix):
        coo = job_skill_matrix.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data)
        job_skill_matrix = torch.sparse_coo_tensor(indices, values, coo.shape)
    
    # Extract non-zero entries
    if job_skill_matrix.is_sparse:
        indices = job_skill_matrix._indices()
        values = job_skill_matrix._values()
    else:
        indices = job_skill_matrix.nonzero(as_tuple=True)
        values = job_skill_matrix[indices]
        indices = torch.stack(indices)
    
    # Create bipartite edge index
    job_indices = indices[0]
    skill_indices = indices[1] + num_jobs  # Offset skill indices
    
    # Forward edges: job -> skill
    edge_index_forward = torch.stack([job_indices, skill_indices])
    
    # Backward edges: skill -> job
    edge_index_backward = torch.stack([skill_indices, job_indices])
    
    # Combine
    edge_index = torch.cat([edge_index_forward, edge_index_backward], dim=1)
    edge_weights = torch.cat([values, values])
    
    return edge_index, edge_weights

def build_full_recommendation_graph(
    user_job_interactions: torch.Tensor,
    user_features: Optional[torch.Tensor] = None,
    job_features: Optional[torch.Tensor] = None,
    job_skill_matrix: Optional[torch.Tensor] = None,
    job_similarity_threshold: float = 0.7,
    num_users: Optional[int] = None,
    num_jobs: Optional[int] = None,
    num_skills: Optional[int] = None
) -> HeteroData:
    """
    Build a comprehensive heterogeneous graph for the recommendation system.
    
    Args:
        user_job_interactions: Tensor of shape [num_interactions, 3] with
                               [user_id, job_id, rating] for each interaction.
        user_features: Optional tensor of user features.
        job_features: Optional tensor of job features.
        job_skill_matrix: Optional tensor mapping jobs to skills.
        job_similarity_threshold: Threshold for job-job similarity edges.
        num_users: Total number of users.
        num_jobs: Total number of jobs.
        num_skills: Total number of skills.
        
    Returns:
        torch_geometric.data.HeteroData: Complete heterogeneous graph.
    """
    # Extract user-job interactions
    user_indices = user_job_interactions[:, 0].long()
    job_indices = user_job_interactions[:, 1].long()
    ratings = user_job_interactions[:, 2]
    
    # Infer dimensions if not provided
    if num_users is None:
        num_users = user_indices.max().item() + 1
    if num_jobs is None:
        num_jobs = job_indices.max().item() + 1
    if num_skills is None and job_skill_matrix is not None:
        num_skills = job_skill_matrix.size(1)
    
    # Create heterogeneous graph
    hetero_graph = HeteroData()
    
    # Add user nodes
    if user_features is not None:
        hetero_graph['user'].x = user_features
    else:
        hetero_graph['user'].x = torch.eye(num_users)
    
    # Add job nodes
    if job_features is not None:
        hetero_graph['job'].x = job_features
    else:
        hetero_graph['job'].x = torch.eye(num_jobs)
    
    # Add user-job edges
    hetero_graph['user', 'rates', 'job'].edge_index = torch.stack([user_indices, job_indices])
    hetero_graph['user', 'rates', 'job'].edge_attr = ratings.view(-1, 1)
    
    # Add job-user edges (reverse)
    hetero_graph['job', 'rated_by', 'user'].edge_index = torch.stack([job_indices, user_indices])
    hetero_graph['job', 'rated_by', 'user'].edge_attr = ratings.view(-1, 1)
    
    # Add job-job similarity edges if job features are available
    if job_features is not None:
        job_job_edge_index, job_job_edge_weights = create_job_similarity_graph(
            job_features, threshold=job_similarity_threshold
        )
        hetero_graph['job', 'similar', 'job'].edge_index = job_job_edge_index
        hetero_graph['job', 'similar', 'job'].edge_attr = job_job_edge_weights.view(-1, 1)
    
    # Add job-skill edges if skill matrix is available
    if job_skill_matrix is not None:
        # Add skill nodes
        hetero_graph['skill'].x = torch.eye(num_skills)
        
        # Create job-skill connections
        for job_idx in range(num_jobs):
            skill_mask = job_skill_matrix[job_idx].nonzero(as_tuple=True)[0]
            if len(skill_mask) > 0:
                job_indices = torch.full((len(skill_mask),), job_idx, dtype=torch.long)
                
                # Job -> Skill edges
                job_skill_edge_index = torch.stack([job_indices, skill_mask])
                if ('job', 'has_skill', 'skill') not in hetero_graph.edge_types:
                    hetero_graph['job', 'has_skill', 'skill'].edge_index = job_skill_edge_index
                else:
                    prev_edges = hetero_graph['job', 'has_skill', 'skill'].edge_index
                    hetero_graph['job', 'has_skill', 'skill'].edge_index = torch.cat([prev_edges, job_skill_edge_index], dim=1)
                
                # Skill -> Job edges
                skill_job_edge_index = torch.stack([skill_mask, job_indices])
                if ('skill', 'belongs_to', 'job') not in hetero_graph.edge_types:
                    hetero_graph['skill', 'belongs_to', 'job'].edge_index = skill_job_edge_index
                else:
                    prev_edges = hetero_graph['skill', 'belongs_to', 'job'].edge_index
                    hetero_graph['skill', 'belongs_to', 'job'].edge_index = torch.cat([prev_edges, skill_job_edge_index], dim=1)
    
    return hetero_graph
