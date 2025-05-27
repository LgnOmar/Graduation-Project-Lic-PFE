"""
Base Recommender class defining the interface for recommendation models.
"""

import logging
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseRecommender(ABC):
    """
    Abstract base class for recommender models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the BaseRecommender.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        
    @abstractmethod
    def fit(self, data: Any) -> None:
        """
        Train the recommender model.
        
        Args:
            data: Training data.
        """
        pass
        
    @abstractmethod
    def recommend(self, professional_id: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a professional.
        
        Args:
            professional_id: ID of the professional user.
            top_n: Number of recommendations to generate.
            
        Returns:
            List[Dict[str, Any]]: List of recommended jobs with metadata.
        """
        pass
        
    @abstractmethod
    def evaluate(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the recommender model.
        
        Args:
            test_data: Test data.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        pass
        
    def save_model(self, path: str) -> bool:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model to.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {path}: {str(e)}")
            return False
            
    def load_model(self, path: str) -> bool:
        """
        Load the model from disk.
        
        Args:
            path: Path to load the model from.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            return False
