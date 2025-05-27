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

from src.data.user_types import UserType, JobCategory, ProfessionalProfile

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


class ProfessionalJobMatchDataset(Dataset):
    """
    Dataset for matching professionals with suitable jobs.
    
    This dataset is specifically designed for the professional-job matching use case,
    where the goal is to recommend jobs to professionals based on their categories
    and location preferences.
    """
    
    def __init__(
        self,
        professionals_df: pd.DataFrame,
        jobs_df: pd.DataFrame,
        interactions_df: Optional[pd.DataFrame] = None,
        user_id_col: str = 'user_id',
        job_id_col: str = 'job_id',
        user_type_col: str = 'user_type',
        user_categories_col: str = 'categories',
        user_location_col: str = 'location',
        job_category_col: str = 'category',
        job_location_col: str = 'location',
        rating_col: Optional[str] = 'rating'
    ):
        """
        Initialize the professional-job match dataset.
        
        Args:
            professionals_df: DataFrame containing professional data
            jobs_df: DataFrame containing job data
            interactions_df: Optional DataFrame containing past interactions
            user_id_col: Column name for user IDs in professionals_df
            job_id_col: Column name for job IDs in jobs_df
            user_type_col: Column name for user type in professionals_df
            user_categories_col: Column name for professional categories in professionals_df
            user_location_col: Column name for professional location in professionals_df
            job_category_col: Column name for job category in jobs_df
            job_location_col: Column name for job location in jobs_df
            rating_col: Column name for ratings in interactions_df
        """
        # Filter to only include professionals
        self.professionals_df = professionals_df[
            professionals_df[user_type_col] == UserType.PROFESSIONAL
        ].copy()
        
        self.jobs_df = jobs_df.copy()
        self.interactions_df = interactions_df.copy() if interactions_df is not None else None
        
        # Store column names
        self.user_id_col = user_id_col
        self.job_id_col = job_id_col
        self.user_categories_col = user_categories_col
        self.user_location_col = user_location_col
        self.job_category_col = job_category_col
        self.job_location_col = job_location_col
        self.rating_col = rating_col
        
        # Create professional profiles
        self.professional_profiles = {}
        for _, row in self.professionals_df.iterrows():
            user_id = row[user_id_col]
            
            # Handle categories (could be string or list)
            categories = row[user_categories_col]
            if isinstance(categories, str):
                # Parse string representation of list if needed
                if categories.startswith('[') and categories.endswith(']'):
                    categories = eval(categories)
                else:
                    categories = [categories]
            
            # Create profile
            self.professional_profiles[user_id] = ProfessionalProfile(
                user_id=user_id,
                categories=categories,
                location=row[user_location_col] if user_location_col in row else None
            )
        
        # Create job information dictionary
        self.job_info = {}
        for _, row in self.jobs_df.iterrows():
            job_id = row[job_id_col]
            self.job_info[job_id] = {
                'category': row[job_category_col],
                'location': row[job_location_col] if job_location_col in row else None
            }
        
        # Create list of professional-job pairs
        self.pairs = []
        for p_id, profile in self.professional_profiles.items():
            for j_id, job in self.job_info.items():
                # Only include pairs where the job category matches a professional category
                if job['category'] in profile.categories:
                    self.pairs.append((p_id, j_id))
    
    def __len__(self) -> int:
        """Get the number of potential professional-job matches."""
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a professional-job pair by index.
        
        Args:
            idx: Index of the pair.
            
        Returns:
            Dict[str, Any]: Professional-job pair data.
        """
        p_id, j_id = self.pairs[idx]
        
        result = {
            'professional_id': p_id,
            'job_id': j_id,
            'category_match': 1.0,  # Since we filtered for matching categories
            'location_match': 1.0 if self.professional_profiles[p_id].location == self.job_info[j_id]['location'] else 0.0
        }
        
        # Add rating if available
        if self.interactions_df is not None:
            interaction = self.interactions_df[
                (self.interactions_df[self.user_id_col] == p_id) & 
                (self.interactions_df[self.job_id_col] == j_id)
            ]
            if not interaction.empty and self.rating_col in interaction.columns:
                result['rating'] = float(interaction.iloc[0][self.rating_col])
        
        return result
    
    def get_matching_jobs_for_professional(
        self, 
        professional_id: Any,
        top_k: Optional[int] = None,
        min_category_match: float = 1.0,
        min_location_match: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get jobs that match a professional's profile.
        
        Args:
            professional_id: ID of the professional
            top_k: Maximum number of jobs to return
            min_category_match: Minimum category match score (0-1)
            min_location_match: Minimum location match score (0-1)
            
        Returns:
            List[Dict[str, Any]]: List of matching jobs with match scores
        """
        if professional_id not in self.professional_profiles:
            return []
        
        profile = self.professional_profiles[professional_id]
        
        matches = []
        for job_id, job in self.job_info.items():
            # Check category match
            category_match = 1.0 if job['category'] in profile.categories else 0.0
            
            # Skip if category doesn't match threshold
            if category_match < min_category_match:
                continue
            
            # Check location match
            location_match = 1.0 if profile.location == job['location'] else 0.0
            
            # Skip if location doesn't match threshold
            if location_match < min_location_match:
                continue
            
            # Calculate overall match score
            match_score = (category_match + location_match) / 2.0
            
            matches.append({
                'job_id': job_id,
                'category': job['category'],
                'location': job['location'],
                'category_match': category_match,
                'location_match': location_match,
                'match_score': match_score
            })
        
        # Sort by match score descending
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None:
            matches = matches[:top_k]
        
        return matches
