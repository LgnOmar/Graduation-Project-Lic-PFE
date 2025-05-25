"""
Module for constructing heterogeneous graph for the recommendation system.
"""
import torch
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_graph(
    interactions_df: pd.DataFrame,
    job_embeddings: Dict[str, np.ndarray],
    user_embeddings: Dict[str, np.ndarray] = None
) -> HeteroData:
    """
    Build a heterogeneous graph from interactions and embeddings.
    
    Args:
        interactions_df: Processed DataFrame containing user-job interactions
        job_embeddings: Dictionary mapping job_id to BERT embeddings
        user_embeddings: Optional dictionary mapping user_id to embeddings
        
    Returns:
        PyTorch Geometric HeteroData object
    """
    try:
        logger.info("Building heterogeneous graph...")
        
        # Create mappings from IDs to indices
        unique_users = interactions_df['user_id'].unique()
        unique_jobs = interactions_df['job_id'].unique()
        
        logger.info(f"Graph will have {len(unique_users)} users and {len(unique_jobs)} jobs")
        
        user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        job_to_idx = {jid: idx for idx, jid in enumerate(unique_jobs)}
        
        # Create edge indices and attributes
        user_indices = torch.tensor([user_to_idx[uid] for uid in interactions_df['user_id']])
        job_indices = torch.tensor([job_to_idx[jid] for jid in interactions_df['job_id']])
        edge_index = torch.stack([user_indices, job_indices])
        
        # Get edge attributes (enhanced ratings)
        edge_attr = torch.tensor(interactions_df['enhanced_rating'].values, dtype=torch.float)
        
        # Create the heterogeneous graph
        data = HeteroData()
        
        # Add node features
        if user_embeddings:
            user_features = torch.stack([
                torch.tensor(user_embeddings[uid], dtype=torch.float)
                for uid in unique_users
            ])
        else:
            # If no user embeddings, use one-hot encoding
            user_features = torch.eye(len(unique_users))
        
        # Check that all job IDs have embeddings
        missing_jobs = [jid for jid in unique_jobs if jid not in job_embeddings]
        if missing_jobs:
            logger.warning(f"Missing embeddings for {len(missing_jobs)} jobs. Using zeros instead.")
            for jid in missing_jobs:
                # Use zeros with same dimensionality as other embeddings
                sample_dim = next(iter(job_embeddings.values())).shape[0]
                job_embeddings[jid] = np.zeros(sample_dim)
        
        job_features = torch.stack([
            torch.tensor(job_embeddings[jid], dtype=torch.float)
            for jid in unique_jobs
        ])
        
        # Store ID mappings for later use
        data['user'].id_mapping = user_to_idx
        data['job'].id_mapping = job_to_idx
        
        data['user'].x = user_features
        data['job'].x = job_features
        
        # Add edges
        data['user', 'interacts_with', 'job'].edge_index = edge_index
        data['user', 'interacts_with', 'job'].edge_attr = edge_attr
        
        # Add reverse edges for message passing
        data['job', 'rev_interacts_with', 'user'].edge_index = edge_index.flip(0)
        data['job', 'rev_interacts_with', 'user'].edge_attr = edge_attr
        
        logger.info(f"Graph built successfully with {edge_index.size(1)} interactions")
        
        return data
    
    except Exception as e:
        logger.error(f"Error building graph: {str(e)}")
        raise

def save_graph_data(
    graph: HeteroData,
    output_dir: str = 'data',
    filename: str = 'graph.pt'
) -> None:
    """
    Save the constructed graph.
    
    Args:
        graph: HeteroData object to save
        output_dir: Directory to save the graph
        filename: Name of the file to save the graph
    """
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        torch.save(graph, f'{output_dir}/{filename}')
        logger.info(f"Graph saved to {output_dir}/{filename}")
    except Exception as e:
        logger.error(f"Error saving graph: {str(e)}")
        raise
