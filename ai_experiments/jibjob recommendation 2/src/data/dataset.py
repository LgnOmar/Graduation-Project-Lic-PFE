"""
Dataset classes for JibJob recommendation system.
This module provides PyTorch dataset implementations for recommendation data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserJobInteractionDataset(Dataset):
    """
    Dataset for user-job interactions.
    
    This dataset handles user-job interactions with ratings and optional features.
    """
    
    def __init__(
        self,
        user_indices: torch.Tensor,
        job_indices: torch.Tensor,
        ratings: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        job_features: Optional[torch.Tensor] = None
    ):
        """
        Initialize the user-job interaction dataset.
        
        Args:
            user_indices: Tensor with user indices.
            job_indices: Tensor with job indices.
            ratings: Tensor with rating values.
            user_features: Optional tensor with user features.
            job_features: Optional tensor with job features.
        """
        assert len(user_indices) == len(job_indices) == len(ratings), "Input tensors must have the same length"
        
        self.user_indices = user_indices
        self.job_indices = job_indices
        self.ratings = ratings
        self.user_features = user_features
        self.job_features = job_features
        
        # Get number of users and jobs
        self.num_users = int(user_indices.max().item()) + 1
        self.num_jobs = int(job_indices.max().item()) + 1
    
    def __len__(self) -> int:
        """Get number of interactions in the dataset."""
        return len(self.user_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an interaction by index.
        
        Args:
            idx: Index of the interaction.
            
        Returns:
            Dict[str, torch.Tensor]: Interaction data.
        """
        result = {
            'user_idx': self.user_indices[idx],
            'job_idx': self.job_indices[idx],
            'rating': self.ratings[idx]
        }
        
        # Add features if available
        if self.user_features is not None:
            user_idx = self.user_indices[idx].item()
            if user_idx < len(self.user_features):
                result['user_features'] = self.user_features[user_idx]
        
        if self.job_features is not None:
            job_idx = self.job_indices[idx].item()
            if job_idx < len(self.job_features):
                result['job_features'] = self.job_features[job_idx]
        
        return result
    
    def get_user_features(self, user_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get features for specified users.
        
        Args:
            user_indices: Tensor with user indices.
            
        Returns:
            Optional[torch.Tensor]: User features or None if not available.
        """
        if self.user_features is None:
            return None
        return self.user_features[user_indices]
    
    def get_job_features(self, job_indices: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Get features for specified jobs.
        
        Args:
            job_indices: Tensor with job indices.
            
        Returns:
            Optional[torch.Tensor]: Job features or None if not available.
        """
        if self.job_features is None:
            return None
        return self.job_features[job_indices]
    
    def get_all_users(self) -> torch.Tensor:
        """
        Get indices of all users.
        
        Returns:
            torch.Tensor: Tensor with all user indices.
        """
        return torch.arange(self.num_users)
    
    def get_all_jobs(self) -> torch.Tensor:
        """
        Get indices of all jobs.
        
        Returns:
            torch.Tensor: Tensor with all job indices.
        """
        return torch.arange(self.num_jobs)
    
    def get_user_interactions(self, user_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get all interactions for a specific user.
        
        Args:
            user_idx: User index.
            
        Returns:
            Dict[str, torch.Tensor]: User's interactions.
        """
        mask = self.user_indices == user_idx
        
        return {
            'job_indices': self.job_indices[mask],
            'ratings': self.ratings[mask]
        }
    
    def get_job_interactions(self, job_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get all interactions for a specific job.
        
        Args:
            job_idx: Job index.
            
        Returns:
            Dict[str, torch.Tensor]: Job's interactions.
        """
        mask = self.job_indices == job_idx
        
        return {
            'user_indices': self.user_indices[mask],
            'ratings': self.ratings[mask]
        }
    
    @classmethod
    def from_dataframe(
        cls,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        job_col: str = 'job_id',
        rating_col: str = 'rating',
        user_features_df: Optional[pd.DataFrame] = None,
        job_features_df: Optional[pd.DataFrame] = None
    ) -> 'UserJobInteractionDataset':
        """
        Create a dataset from DataFrames.
        
        Args:
            interactions_df: DataFrame with user-job interactions.
            user_col: Name of the user ID column.
            job_col: Name of the job ID column.
            rating_col: Name of the rating column.
            user_features_df: Optional DataFrame with user features.
            job_features_df: Optional DataFrame with job features.
            
        Returns:
            UserJobInteractionDataset: Dataset created from the DataFrames.
        """
        # Create user and job mappings
        unique_users = interactions_df[user_col].unique()
        unique_jobs = interactions_df[job_col].unique()
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        job_to_idx = {job_id: idx for idx, job_id in enumerate(unique_jobs)}
        
        # Convert IDs to indices
        user_indices = torch.tensor([
            user_to_idx[user_id] for user_id in interactions_df[user_col]
        ], dtype=torch.long)
        
        job_indices = torch.tensor([
            job_to_idx[job_id] for job_id in interactions_df[job_col]
        ], dtype=torch.long)
        
        # Get ratings
        if rating_col in interactions_df.columns:
            ratings = torch.tensor(interactions_df[rating_col].values, dtype=torch.float)
        else:
            # Default to ones if no ratings
            ratings = torch.ones(len(interactions_df), dtype=torch.float)
        
        # Process user features if available
        user_features = None
        if user_features_df is not None:
            # Match with mapping
            user_features_df = user_features_df[user_features_df[user_col].isin(unique_users)]
            
            # Sort by mapped index
            user_features_df['idx'] = user_features_df[user_col].map(user_to_idx)
            user_features_df = user_features_df.sort_values('idx')
            
            # Get feature columns (exclude ID and idx)
            feature_cols = [col for col in user_features_df.columns 
                          if col != user_col and col != 'idx']
            
            if feature_cols:
                user_features = torch.tensor(
                    user_features_df[feature_cols].values,
                    dtype=torch.float
                )
        
        # Process job features if available
        job_features = None
        if job_features_df is not None:
            # Match with mapping
            job_features_df = job_features_df[job_features_df[job_col].isin(unique_jobs)]
            
            # Sort by mapped index
            job_features_df['idx'] = job_features_df[job_col].map(job_to_idx)
            job_features_df = job_features_df.sort_values('idx')
            
            # Get feature columns (exclude ID and idx)
            feature_cols = [col for col in job_features_df.columns 
                          if col != job_col and col != 'idx']
            
            if feature_cols:
                job_features = torch.tensor(
                    job_features_df[feature_cols].values,
                    dtype=torch.float
                )
        
        return cls(
            user_indices=user_indices,
            job_indices=job_indices,
            ratings=ratings,
            user_features=user_features,
            job_features=job_features
        )
    
    def to_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size for the DataLoader.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes.
            
        Returns:
            DataLoader: DataLoader for this dataset.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
    
    def split(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_state: int = 42
    ) -> Tuple['UserJobInteractionDataset', 'UserJobInteractionDataset', 'UserJobInteractionDataset']:
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            val_ratio: Fraction of data to use for validation.
            test_ratio: Fraction of data to use for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple[UserJobInteractionDataset, UserJobInteractionDataset, UserJobInteractionDataset]:
                Training, validation, and test datasets.
        """
        # Set random seed
        torch.manual_seed(random_state)
        
        # Get number of samples
        n_samples = len(self)
        
        # Calculate split sizes
        test_size = int(n_samples * test_ratio)
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - test_size - val_size
        
        # Generate random permutation
        indices = torch.randperm(n_samples)
        
        # Split into train, val, test
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = UserJobInteractionDataset(
            user_indices=self.user_indices[train_indices],
            job_indices=self.job_indices[train_indices],
            ratings=self.ratings[train_indices],
            user_features=self.user_features,
            job_features=self.job_features
        )
        
        val_dataset = UserJobInteractionDataset(
            user_indices=self.user_indices[val_indices],
            job_indices=self.job_indices[val_indices],
            ratings=self.ratings[val_indices],
            user_features=self.user_features,
            job_features=self.job_features
        )
        
        test_dataset = UserJobInteractionDataset(
            user_indices=self.user_indices[test_indices],
            job_indices=self.job_indices[test_indices],
            ratings=self.ratings[test_indices],
            user_features=self.user_features,
            job_features=self.job_features
        )
        
        return train_dataset, val_dataset, test_dataset


class UserJobGraphDataset(Dataset):
    """
    Dataset for graph-based learning with user-job interactions.
    
    This dataset is designed for use with graph-based recommendation models.
    """
    
    def __init__(
        self,
        graph_data: Any,  # PyTorch Geometric Data object
        user_indices: torch.Tensor,
        job_indices: torch.Tensor,
        ratings: torch.Tensor
    ):
        """
        Initialize the graph dataset.
        
        Args:
            graph_data: PyTorch Geometric Data object with the graph.
            user_indices: Tensor with user indices for interactions.
            job_indices: Tensor with job indices for interactions.
            ratings: Tensor with rating values for interactions.
        """
        self.graph_data = graph_data
        self.user_indices = user_indices
        self.job_indices = job_indices
        self.ratings = ratings
    
    def __len__(self) -> int:
        """Get number of interactions in the dataset."""
        return len(self.user_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an interaction by index.
        
        Args:
            idx: Index of the interaction.
            
        Returns:
            Dict[str, torch.Tensor]: Interaction data.
        """
        return {
            'graph_data': self.graph_data,
            'user_idx': self.user_indices[idx],
            'job_idx': self.job_indices[idx],
            'rating': self.ratings[idx]
        }
    
    def get_full_graph(self) -> Any:
        """
        Get the complete graph.
        
        Returns:
            Any: PyTorch Geometric Data object with the graph.
        """
        return self.graph_data
    
    def to_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> DataLoader:
        """
        Create a DataLoader for this dataset.
        
        Args:
            batch_size: Batch size for the DataLoader.
            shuffle: Whether to shuffle the data.
            num_workers: Number of worker processes.
            
        Returns:
            DataLoader: DataLoader for this dataset.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching.
        
        This ensures that the entire graph is shared across batch items
        instead of being duplicated.
        
        Args:
            batch: List of interaction dictionaries.
            
        Returns:
            Dict[str, torch.Tensor]: Batched data.
        """
        # Get the graph only from the first item (all have the same graph)
        graph_data = batch[0]['graph_data']
        
        # Stack other tensors
        user_indices = torch.stack([item['user_idx'] for item in batch])
        job_indices = torch.stack([item['job_idx'] for item in batch])
        ratings = torch.stack([item['rating'] for item in batch])
        
        return {
            'graph_data': graph_data,
            'user_indices': user_indices,
            'job_indices': job_indices,
            'ratings': ratings
        }
    
    @classmethod
    def from_interaction_dataset(
        cls,
        dataset: UserJobInteractionDataset,
        graph_data: Any  # PyTorch Geometric Data object
    ) -> 'UserJobGraphDataset':
        """
        Create a graph dataset from an interaction dataset.
        
        Args:
            dataset: User-job interaction dataset.
            graph_data: PyTorch Geometric Data object with the graph.
            
        Returns:
            UserJobGraphDataset: Graph dataset.
        """
        return cls(
            graph_data=graph_data,
            user_indices=dataset.user_indices,
            job_indices=dataset.job_indices,
            ratings=dataset.ratings
        )
    
    def split(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_state: int = 42
    ) -> Tuple['UserJobGraphDataset', 'UserJobGraphDataset', 'UserJobGraphDataset']:
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            val_ratio: Fraction of data to use for validation.
            test_ratio: Fraction of data to use for testing.
            random_state: Random seed for reproducibility.
            
        Returns:
            Tuple[UserJobGraphDataset, UserJobGraphDataset, UserJobGraphDataset]:
                Training, validation, and test datasets.
        """
        # Set random seed
        torch.manual_seed(random_state)
        
        # Get number of samples
        n_samples = len(self)
        
        # Calculate split sizes
        test_size = int(n_samples * test_ratio)
        val_size = int(n_samples * val_ratio)
        train_size = n_samples - test_size - val_size
        
        # Generate random permutation
        indices = torch.randperm(n_samples)
        
        # Split into train, val, test
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_dataset = UserJobGraphDataset(
            graph_data=self.graph_data,
            user_indices=self.user_indices[train_indices],
            job_indices=self.job_indices[train_indices],
            ratings=self.ratings[train_indices]
        )
        
        val_dataset = UserJobGraphDataset(
            graph_data=self.graph_data,
            user_indices=self.user_indices[val_indices],
            job_indices=self.job_indices[val_indices],
            ratings=self.ratings[val_indices]
        )
        
        test_dataset = UserJobGraphDataset(
            graph_data=self.graph_data,
            user_indices=self.user_indices[test_indices],
            job_indices=self.job_indices[test_indices],
            ratings=self.ratings[test_indices]
        )
        
        return train_dataset, val_dataset, test_dataset
