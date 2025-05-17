"""
Recommender system implementation using trained GCN model.
"""
import torch
import pickle
import os
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional
import numpy as np
from gcn_model import HeteroGCNLinkPredictor
from graph_construction import build_graph

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobRecommender:
    def __init__(
        self,
        model: HeteroGCNLinkPredictor,
        graph: torch.nn.Module,
        job_id_mapping: Dict[str, int],
        user_id_mapping: Dict[str, int]
    ):
        """
        Initialize the recommender system.
        
        Args:
            model: Trained GCN model
            graph: HeteroData graph object
            job_id_mapping: Mapping from job IDs to indices
            user_id_mapping: Mapping from user IDs to indices
        """
        self.model = model
        self.graph = graph
        self.job_id_mapping = job_id_mapping
        self.user_id_mapping = user_id_mapping
        self.reverse_job_mapping = {v: k for k, v in job_id_mapping.items()}
        
        # Extract and cache all node embeddings
        with torch.no_grad():
            self.node_embeddings = model.get_embeddings(
                graph.x_dict,
                graph.edge_index_dict
            )
    
    def get_recommendations(
        self,
        user_id: str,
        top_n: int = 10,
        exclude_interacted: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Get job recommendations for a user.
        
        Args:
            user_id: ID of the user to get recommendations for
            top_n: Number of recommendations to return
            exclude_interacted: Whether to exclude jobs the user has already interacted with
            
        Returns:
            List of (job_id, score) tuples for recommended jobs
        """
        try:
            # Get user index
            if user_id not in self.user_id_mapping:
                raise KeyError(f"User {user_id} not found")
            user_idx = self.user_id_mapping[user_id]
            
            # Get user embedding
            user_embedding = self.node_embeddings['user'][user_idx]
            
            # Get all job embeddings
            job_embeddings = self.node_embeddings['job']
            
            # Create edge indices for all possible user-job pairs
            num_jobs = len(job_embeddings)
            edge_index = torch.tensor([
                [user_idx] * num_jobs,  # User index repeated
                list(range(num_jobs))    # All job indices
            ])
            
            # Get predictions for all jobs
            self.model.eval()
            with torch.no_grad():
                edge_attr = self.model.decode(
                    self.node_embeddings,
                    edge_index
                )
            scores = edge_attr.cpu().numpy()
            
            # Get indices of top recommendations
            if exclude_interacted:
                # Get indices of jobs the user has already interacted with
                interacted_jobs = set()
                edge_index = self.graph['user', 'interacts_with', 'job'].edge_index
                for i in range(edge_index.size(1)):
                    if edge_index[0, i].item() == user_idx:
                        interacted_jobs.add(edge_index[1, i].item())
                
                # Mask out interacted jobs
                scores = np.ma.array(scores, mask=False)
                scores.mask[[i for i in interacted_jobs]] = True
            
            top_indices = np.argsort(scores)[-top_n:][::-1]
            
            # Convert to job IDs and scores
            recommendations = [
                (self.reverse_job_mapping[idx], float(scores[idx]))
                for idx in top_indices
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

def load_recommender(
    model_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_model.pt'),
    data_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
) -> JobRecommender:
    """
    Load the trained model and create a recommender instance.
    
    Args:
        model_path: Path to the saved model checkpoint
        data_dir: Directory containing the data files
        
    Returns:
        Initialized JobRecommender instance
    """
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path)
        
        # Load processed data
        with open(os.path.join(data_dir, 'job_embeddings.pkl'), 'rb') as f:
            job_data = pickle.load(f)
            
        interactions_df = pd.read_csv(os.path.join(data_dir, 'processed_interactions.csv'))
        
        # Build the graph
        graph = build_graph(interactions_df, job_data)
        
        # Initialize model
        model = HeteroGCNLinkPredictor(
            hidden_channels=64,  # Should match training configuration
            num_layers=2,        # Should match training configuration
            data=graph
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Get ID mappings from the graph
        job_id_mapping = graph['job'].id_mapping
        user_id_mapping = graph['user'].id_mapping
        
        # Create recommender instance
        recommender = JobRecommender(
            model=model,
            graph=graph,
            job_id_mapping=job_id_mapping,
            user_id_mapping=user_id_mapping
        )
        
        return recommender
        
    except Exception as e:
        logger.error(f"Error loading recommender: {str(e)}")
        raise
