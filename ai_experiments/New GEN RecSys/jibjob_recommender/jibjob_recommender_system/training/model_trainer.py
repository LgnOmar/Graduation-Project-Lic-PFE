"""
ModelTrainer class for training GCN recommender models.
This is a wrapper around the Trainer class.
"""

import os
import logging
import torch
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import torch_geometric
from .trainer import Trainer
from ..models.gcn_recommender import GCNRecommender, HeteroGCNRecommender

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Class for training GCN recommender models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelTrainer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.model_class = None  # Will be set by caller
        
    def train(self, graph) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Train a GCN recommender model on the provided graph.
        
        Args:
            graph: The graph data to train on.
            
        Returns:
            Tuple[torch.nn.Module, Dict[str, Any]]: (Trained model, Training statistics)
        """
        logger.info("Initializing model...")
        
        # Create model instance
        if self.model_class is None:
            raise ValueError("Model class not set. Set model_class to GCNRecommender or HeteroGCNRecommender.")
            
        model = self.model_class(self.config)
        
        # Create trainer
        trainer = Trainer(self.config, model)
        
        # Check if we're using a heterogeneous model
        is_heterogeneous = self.model_class == HeteroGCNRecommender
        
        # Extract training pairs and labels from graph
        if is_heterogeneous:
            # Get professional-job pairs and labels
            # This is simplified and should be adapted to your actual graph structure
            professional_job_pairs = graph.edge_index_dict.get(('professional', 'applies_to', 'job'), 
                                                             torch.zeros((2, 0), dtype=torch.long))
            positive_labels = torch.ones(professional_job_pairs.size(1))
            
            # Generate negative samples
            total_professionals = graph.x_dict['professional'].size(0)
            total_jobs = graph.x_dict['job'].size(0)
            
            all_pairs, all_labels = trainer.generate_negative_samples(
                professional_job_pairs,
                total_professionals,
                total_jobs,
                n_neg_per_pos=5
            )
            
            # Train heterogeneous model
            stats = trainer.train_heterogeneous_model(graph, all_pairs, all_labels)
            
        else:
            # Extract edges between professionals and jobs
            # This assumes a homogeneous graph with professional and job nodes
            # You'll need to adapt this to your specific graph structure
            total_nodes = graph.x.size(0)
            
            # Get professional-job pairs from edge_index
            # Assuming edge_index contains all edges, including professional-job edges
            professional_job_pairs = graph.edge_index[:, graph.edge_type == 'professional_job']
            positive_labels = torch.ones(professional_job_pairs.size(1))
            
            # Generate negative samples
            # For this, you need to know how many professional and job nodes you have
            # This is a simplified approach
            n_professionals = total_nodes // 2  # Simplified assumption
            n_jobs = total_nodes - n_professionals
            
            all_pairs, all_labels = trainer.generate_negative_samples(
                professional_job_pairs,
                n_professionals,
                n_jobs,
                n_neg_per_pos=5
            )
            
            # Train homogeneous model
            stats = trainer.train_homogeneous_model(graph, all_pairs, all_labels)
            
        logger.info(f"Model training completed with final validation AUC: {stats.get('val_aucs', [0])[-1]:.4f}")
        
        return model, stats
        
    def save_model(self, model: torch.nn.Module, model_path: str) -> None:
        """
        Save model to file.
        
        Args:
            model: Model to save.
            model_path: Path to save the model to.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model state dict
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
        
    def load_model(self, model_path: str) -> Optional[torch.nn.Module]:
        """
        Load model from file.
        
        Args:
            model_path: Path to load the model from.
            
        Returns:
            Optional[torch.nn.Module]: Loaded model or None if loading failed.
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Create model
            if self.model_class == HeteroGCNRecommender:
                model = HeteroGCNRecommender(self.config)
            else:
                model = GCNRecommender(self.config)
                
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info(f"Model loaded from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
