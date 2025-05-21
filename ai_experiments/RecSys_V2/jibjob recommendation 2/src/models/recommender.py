"""
Main recommender system for JibJob that integrates all components.
This module serves as the main entry point for the recommendation functionality.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from pathlib import Path
import logging
import os
import json
import time
from tqdm import tqdm
import torch.nn.functional as F

from src.models.bert_embeddings import BERTEmbeddings
from src.models.sentiment_analysis import SentimentAnalysis
from src.models.gcn import GCNRecommender, HeterogeneousGCN
from src.data.graph_builder import build_interaction_graph
from src.utils.metrics import calculate_recommendation_metrics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobRecommender:
    """
    Main recommendation system for JibJob that integrates text understanding, 
    sentiment analysis, and graph-based recommendation.
    
    This class serves as a high-level interface for the entire recommendation system.
    """
    
    def __init__(
        self,
        embedding_model: Optional[BERTEmbeddings] = None,
        sentiment_model: Optional[SentimentAnalysis] = None,
        graph_model: Optional[Union[GCNRecommender, HeterogeneousGCN]] = None,
        device: Optional[str] = None,
        sentiment_weight: float = 0.5,
        rating_weight: float = 0.5,
        embedding_model_name: str = "bert-base-multilingual-cased",
        sentiment_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        embedding_cache_dir: Optional[str] = None,
        sentiment_cache_dir: Optional[str] = None
    ):
        """
        Initialize the JibJob recommendation system.
        
        Args:
            embedding_model: Pre-initialized BERT embedding model.
            sentiment_model: Pre-initialized sentiment analysis model.
            graph_model: Pre-initialized graph recommendation model.
            device: Device to run models on ('cpu' or 'cuda').
            sentiment_weight: Weight for sentiment scores in the enhanced rating.
            rating_weight: Weight for explicit ratings in the enhanced rating.
            embedding_model_name: Name of the BERT model to use for embeddings if not provided.
            sentiment_model_name: Name of the sentiment model to use if not provided.
            embedding_cache_dir: Directory to cache the embedding model.
            sentiment_cache_dir: Directory to cache the sentiment model.
        """
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set up models
        self.embedding_model = embedding_model or BERTEmbeddings(
            model_name=embedding_model_name,
            device=self.device,
            cache_dir=embedding_cache_dir
        )
        
        self.sentiment_model = sentiment_model or SentimentAnalysis(
            model_name=sentiment_model_name,
            device=self.device,
            cache_dir=sentiment_cache_dir
        )
        
        self.graph_model = graph_model
        
        # Configuration
        self.sentiment_weight = sentiment_weight
        self.rating_weight = rating_weight
        
        # User and job mappings
        self.user_to_idx: Dict[Any, int] = {}  # Maps user IDs to indices
        self.idx_to_user: Dict[int, Any] = {}  # Maps indices to user IDs
        self.job_to_idx: Dict[Any, int] = {}   # Maps job IDs to indices
        self.idx_to_job: Dict[int, Any] = {}   # Maps indices to job IDs
        
        # Store the interaction graph
        self.graph = None
        
        # Metadata
        self.metadata = {
            "last_trained": None,
            "num_users": 0,
            "num_jobs": 0,
            "num_interactions": 0,
            "version": "1.0.0"
        }
    
    def calculate_enhanced_rating(
        self,
        explicit_rating: Optional[float],
        comment_text: Optional[str]
    ) -> float:
        """
        Calculate enhanced rating by combining explicit rating and sentiment score.
        
        Args:
            explicit_rating: Explicit rating (e.g., 1-5 stars), normalized to 0-1.
                            Can be None if no explicit rating was provided.
            comment_text: Comment text to analyze sentiment from.
                          Can be None if no comment was provided.
            
        Returns:
            float: Enhanced rating between 0 and 1.
        """
        sentiment_score = None
        final_rating = 0.5  # Default to neutral if no information
        
        # Get sentiment score if comment provided
        if comment_text and self.sentiment_model:
            sentiment_score = self.sentiment_model.analyze_sentiment(comment_text)
        
        # Calculate enhanced rating
        if explicit_rating is not None and sentiment_score is not None:
            # Both explicit rating and sentiment available
            final_rating = (
                explicit_rating * self.rating_weight +
                sentiment_score * self.sentiment_weight
            )
        elif explicit_rating is not None:
            # Only explicit rating available
            final_rating = explicit_rating
        elif sentiment_score is not None:
            # Only sentiment score available
            final_rating = sentiment_score
        
        return final_rating
    
    def calculate_batch_enhanced_ratings(
        self,
        data: pd.DataFrame,
        rating_col: str = 'rating',
        comment_col: str = 'comment',
        normalize_ratings: bool = True,
        rating_max: float = 5.0,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Calculate enhanced ratings for a batch of data.
        
        Args:
            data: DataFrame containing ratings and comments.
            rating_col: Name of the column containing explicit ratings.
            comment_col: Name of the column containing comments.
            normalize_ratings: Whether to normalize ratings to 0-1.
            rating_max: Maximum value for ratings (used for normalization).
            batch_size: Batch size for sentiment analysis.
            show_progress: Whether to show progress bar.
            
        Returns:
            np.ndarray: Array of enhanced ratings.
        """
        n_samples = len(data)
        enhanced_ratings = np.zeros(n_samples)
        
        # Process explicit ratings
        explicit_ratings = None
        if rating_col in data.columns:
            explicit_ratings = data[rating_col].values
            
            # Handle missing ratings
            mask_missing = pd.isna(explicit_ratings)
            explicit_ratings = np.where(mask_missing, 0.0, explicit_ratings)
            
            # Normalize ratings if needed
            if normalize_ratings:
                explicit_ratings = explicit_ratings / rating_max
        
        # Process comments for sentiment
        sentiment_scores = None
        if comment_col in data.columns and self.sentiment_model:
            comments = data[comment_col].fillna('').values
            
            # Process non-empty comments
            non_empty_indices = [i for i, c in enumerate(comments) if c.strip()]
            non_empty_comments = [comments[i] for i in non_empty_indices]
            
            if non_empty_comments:
                # Analyze sentiment for non-empty comments
                non_empty_scores = self.sentiment_model.batch_analyze_sentiment(
                    non_empty_comments,
                    batch_size=batch_size,
                    show_progress=show_progress
                )
                
                # Initialize sentiment scores
                sentiment_scores = np.full(n_samples, np.nan)
                
                # Assign scores to non-empty comments
                for i, score in zip(non_empty_indices, non_empty_scores):
                    sentiment_scores[i] = score
        
        # Calculate enhanced ratings
        for i in range(n_samples):
            rating = explicit_ratings[i] if explicit_ratings is not None else None
            sentiment = sentiment_scores[i] if sentiment_scores is not None and i < len(sentiment_scores) and not np.isnan(sentiment_scores[i]) else None
            
            if rating is not None and sentiment is not None:
                # Both rating and sentiment available
                enhanced_ratings[i] = rating * self.rating_weight + sentiment * self.sentiment_weight
            elif rating is not None:
                # Only rating available
                enhanced_ratings[i] = rating
            elif sentiment is not None:
                # Only sentiment available
                enhanced_ratings[i] = sentiment
            else:
                # No information, default to neutral
                enhanced_ratings[i] = 0.5
        
        return enhanced_ratings
    
    def prepare_interaction_data(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        job_col: str = 'job_id',
        rating_col: Optional[str] = 'rating',
        comment_col: Optional[str] = 'comment',
        rating_max: float = 5.0,
        rebuild_mappings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare interaction data for the recommendation model.
        
        Args:
            interactions_df: DataFrame with user-job interactions.
            user_col: Name of the user ID column.
            job_col: Name of the job ID column.
            rating_col: Name of the explicit rating column (optional).
            comment_col: Name of the comment text column (optional).
            rating_max: Maximum value for ratings (used for normalization).
            rebuild_mappings: Whether to rebuild user and job ID mappings.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                user_indices, job_indices, ratings tensors.
        """
        # Create or update user/job mappings
        if rebuild_mappings or not self.user_to_idx:
            unique_users = interactions_df[user_col].unique()
            self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
            self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        if rebuild_mappings or not self.job_to_idx:
            unique_jobs = interactions_df[job_col].unique()
            self.job_to_idx = {job: idx for idx, job in enumerate(unique_jobs)}
            self.idx_to_job = {idx: job for job, idx in self.job_to_idx.items()}
        
        # Convert user and job IDs to indices
        user_indices = torch.tensor([
            self.user_to_idx[user] for user in interactions_df[user_col]
            if user in self.user_to_idx  # Filter out users not in mapping
        ], dtype=torch.long)
        
        job_indices = torch.tensor([
            self.job_to_idx[job] for job in interactions_df[job_col]
            if job in self.job_to_idx  # Filter out jobs not in mapping
        ], dtype=torch.long)
        
        # Calculate enhanced ratings
        enhanced_ratings = None
        if rating_col in interactions_df.columns or comment_col in interactions_df.columns:
            enhanced_ratings = self.calculate_batch_enhanced_ratings(
                interactions_df,
                rating_col=rating_col,
                comment_col=comment_col,
                normalize_ratings=True,
                rating_max=rating_max
            )
            enhanced_ratings = torch.tensor(enhanced_ratings, dtype=torch.float)
        else:
            # Default to neutral ratings if no rating information
            enhanced_ratings = torch.ones(len(interactions_df), dtype=torch.float) * 0.5
        
        # Filter to include only users and jobs in the mappings
        valid_indices = []
        for i, (user, job) in enumerate(zip(interactions_df[user_col], interactions_df[job_col])):
            if user in self.user_to_idx and job in self.job_to_idx:
                valid_indices.append(i)
        
        user_indices = torch.tensor([self.user_to_idx[interactions_df[user_col].iloc[i]] for i in valid_indices], dtype=torch.long)
        job_indices = torch.tensor([self.job_to_idx[interactions_df[job_col].iloc[i]] for i in valid_indices], dtype=torch.long)
        enhanced_ratings = enhanced_ratings[valid_indices]
        
        # Update metadata
        self.metadata["num_users"] = len(self.user_to_idx)
        self.metadata["num_jobs"] = len(self.job_to_idx)
        self.metadata["num_interactions"] = len(valid_indices)
        
        return user_indices, job_indices, enhanced_ratings
    
    def build_graph(
        self, 
        user_indices: torch.Tensor, 
        job_indices: torch.Tensor,
        ratings: torch.Tensor
    ):
        """
        Build the interaction graph for the recommendation model.
        
        Args:
            user_indices: Tensor of user indices.
            job_indices: Tensor of job indices.
            ratings: Tensor of enhanced ratings.
            
        Returns:
            torch_geometric.data.Data: The interaction graph.
        """
        # Build the interaction graph (user-job bipartite graph)
        graph = build_interaction_graph(
            user_indices=user_indices,
            job_indices=job_indices,
            ratings=ratings,
            num_users=len(self.user_to_idx),
            num_jobs=len(self.job_to_idx)
        )
        
        self.graph = graph
        return graph
    
    def create_gcn_model(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = 'gcn'
    ):
        """
        Create a new GCN recommendation model.
        
        Args:
            embedding_dim: Dimension of the initial embeddings.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of graph convolution layers.
            dropout: Dropout rate for regularization.
            conv_type: Type of graph convolution ('gcn', 'sage', or 'gat').
            
        Returns:
            GCNRecommender: The created GCN model.
        """
        self.graph_model = GCNRecommender(
            num_users=len(self.user_to_idx),
            num_jobs=len(self.job_to_idx),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type
        ).to(self.device)
        
        return self.graph_model
    
    def train(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = 'user_id',
        job_col: str = 'job_id',
        rating_col: Optional[str] = 'rating',
        comment_col: Optional[str] = 'comment',
        val_ratio: float = 0.2,
        epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        early_stop_patience: int = 10,
        rebuild_mappings: bool = False,
        create_new_model: bool = True,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = 'gcn'
    ) -> Dict:
        """
        Train the recommendation model on interaction data.
        
        Args:
            interactions_df: DataFrame with user-job interactions.
            user_col: Name of the user ID column.
            job_col: Name of the job ID column.
            rating_col: Name of the explicit rating column (optional).
            comment_col: Name of the comment text column (optional).
            val_ratio: Fraction of data to use for validation.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate for optimizer.
            weight_decay: L2 regularization strength.
            early_stop_patience: Number of epochs to wait before early stopping.
            rebuild_mappings: Whether to rebuild user and job ID mappings.
            create_new_model: Whether to create a new model or use an existing one.
            embedding_dim: Dimension of node embeddings.
            hidden_dim: Dimension of hidden layers.
            num_layers: Number of graph convolution layers.
            dropout: Dropout rate.
            conv_type: Type of graph convolution.
            
        Returns:
            Dict: Training history.
        """
        # Prepare interaction data
        user_indices, job_indices, ratings = self.prepare_interaction_data(
            interactions_df=interactions_df,
            user_col=user_col,
            job_col=job_col,
            rating_col=rating_col,
            comment_col=comment_col,
            rebuild_mappings=rebuild_mappings
        )
        
        # Build the interaction graph
        self.build_graph(user_indices, job_indices, ratings)
        
        # Create a new model or use existing
        if create_new_model or self.graph_model is None:
            self.create_gcn_model(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                conv_type=conv_type
            )
        
        # Split data into train and validation
        num_samples = len(user_indices)
        indices = torch.randperm(num_samples)
        split_idx = int(num_samples * (1 - val_ratio))
        
        train_mask = indices[:split_idx]
        val_mask = indices[split_idx:]
        
        train_user_indices = user_indices[train_mask]
        train_job_indices = job_indices[train_mask]
        train_ratings = ratings[train_mask]
        
        val_user_indices = user_indices[val_mask]
        val_job_indices = job_indices[val_mask]
        val_ratings = ratings[val_mask]
        
        # Move the graph to device
        graph = self.graph.to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.graph_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup loss function
        loss_fn = torch.nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Training
            self.graph_model.train()
            start_time = time.time()
            train_loss = 0.0
            
            # Process in batches
            for i in range(0, len(train_user_indices), batch_size):
                batch_user_indices = train_user_indices[i:i+batch_size].to(self.device)
                batch_job_indices = train_job_indices[i:i+batch_size].to(self.device)
                batch_ratings = train_ratings[i:i+batch_size].to(self.device)
                
                # Forward pass
                predictions = self.graph_model(
                    graph,
                    batch_user_indices,
                    batch_job_indices
                )
                
                # Calculate loss
                loss = loss_fn(predictions, batch_ratings)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_ratings)
            
            # Calculate average training loss
            train_loss /= len(train_user_indices)
            history['train_loss'].append(train_loss)
            
            # Validation
            self.graph_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for i in range(0, len(val_user_indices), batch_size):
                    batch_user_indices = val_user_indices[i:i+batch_size].to(self.device)
                    batch_job_indices = val_job_indices[i:i+batch_size].to(self.device)
                    batch_ratings = val_ratings[i:i+batch_size].to(self.device)
                    
                    # Forward pass
                    predictions = self.graph_model(
                        graph,
                        batch_user_indices,
                        batch_job_indices
                    )
                    
                    # Calculate loss
                    loss = loss_fn(predictions, batch_ratings)
                    val_loss += loss.item() * len(batch_ratings)
            
            # Calculate average validation loss
            val_loss /= len(val_user_indices)
            history['val_loss'].append(val_loss)
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, "
                       f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Update metadata
        self.metadata["last_trained"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        return history
    
    def recommend(
        self, 
        user_id: Any, 
        top_k: int = 10,
        exclude_rated: bool = True,
        rated_job_ids: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate job recommendations for a user.
        
        Args:
            user_id: ID of the user to recommend jobs for.
            top_k: Number of recommendations to return.
            exclude_rated: Whether to exclude jobs the user has already rated.
            rated_job_ids: List of job IDs the user has already rated.
                           Required if exclude_rated is True.
            
        Returns:
            List[Dict[str, Any]]: List of recommended jobs with scores.
        """
        if self.graph is None or self.graph_model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Convert user ID to index
        if user_id not in self.user_to_idx:
            raise ValueError(f"Unknown user ID: {user_id}")
            
        user_idx = self.user_to_idx[user_id]
        
        # Convert rated job IDs to indices if needed
        rated_indices = None
        if exclude_rated and rated_job_ids:
            rated_indices = []
            for job_id in rated_job_ids:
                if job_id in self.job_to_idx:
                    rated_indices.append(self.job_to_idx[job_id])
        
        # Get recommendations
        top_job_indices, top_scores = self.graph_model.recommend(
            graph=self.graph.to(self.device),
            user_id=user_idx,
            top_k=top_k,
            exclude_rated=exclude_rated,
            rated_indices=rated_indices
        )
        
        # Convert indices back to IDs and format results
        recommendations = []
        for job_idx, score in zip(top_job_indices, top_scores):
            job_id = self.idx_to_job[job_idx]
            recommendations.append({
                'job_id': job_id,
                'score': float(score)
            })
        
        return recommendations
    
    def recommend_batch(
        self, 
        user_ids: List[Any], 
        top_k: int = 10,
        exclude_rated: bool = True,
        user_rated_jobs: Optional[Dict[Any, List[Any]]] = None
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Generate job recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs to recommend jobs for.
            top_k: Number of recommendations per user.
            exclude_rated: Whether to exclude jobs each user has already rated.
            user_rated_jobs: Dictionary mapping user IDs to lists of rated job IDs.
                             Required if exclude_rated is True.
            
        Returns:
            Dict[Any, List[Dict[str, Any]]]: 
                Dictionary mapping user IDs to their recommendations.
        """
        if self.graph is None or self.graph_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert user IDs to indices
        valid_user_ids = [user_id for user_id in user_ids if user_id in self.user_to_idx]
        user_indices = [self.user_to_idx[user_id] for user_id in valid_user_ids]
        
        # Convert rated job IDs to indices if needed
        user_rated_indices = None
        if exclude_rated and user_rated_jobs:
            user_rated_indices = {}
            for user_id in valid_user_ids:
                if user_id in user_rated_jobs:
                    rated_indices = []
                    for job_id in user_rated_jobs[user_id]:
                        if job_id in self.job_to_idx:
                            rated_indices.append(self.job_to_idx[job_id])
                    user_rated_indices[self.user_to_idx[user_id]] = rated_indices
        
        # Get recommendations for all users
        recommendations_dict = {}
        
        # Move graph to device
        graph = self.graph.to(self.device)
        
        for user_id in tqdm(valid_user_ids, desc="Generating recommendations"):
            user_idx = self.user_to_idx[user_id]
            
            rated_indices = user_rated_indices.get(user_idx, []) if user_rated_indices else None
            
            top_job_indices, top_scores = self.graph_model.recommend(
                graph=graph,
                user_id=user_idx,
                top_k=top_k,
                exclude_rated=exclude_rated,
                rated_indices=rated_indices
            )
            
            # Convert indices back to IDs and format results
            user_recommendations = []
            for job_idx, score in zip(top_job_indices, top_scores):
                job_id = self.idx_to_job[job_idx]
                user_recommendations.append({
                    'job_id': job_id,
                    'score': float(score)
                })
                
            recommendations_dict[user_id] = user_recommendations
        
        return recommendations_dict
    
    def evaluate(
        self,
        test_interactions: pd.DataFrame,
        user_col: str = 'user_id',
        job_col: str = 'job_id',
        rating_col: Optional[str] = 'rating',
        comment_col: Optional[str] = 'comment',
        top_k: int = 10,
        exclude_rated: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the recommendation model on test data.
        
        Args:
            test_interactions: DataFrame with test interactions.
            user_col: Name of the user ID column.
            job_col: Name of the job ID column.
            rating_col: Name of the rating column.
            comment_col: Name of the comment column.
            top_k: Number of recommendations to generate.
            exclude_rated: Whether to exclude rated items from recommendations.
            
        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        if self.graph is None or self.graph_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get unique users in test set
        test_users = test_interactions[user_col].unique()
        
        # Filter to only include users in our mapping
        test_users = [user for user in test_users if user in self.user_to_idx]
        
        if len(test_users) == 0:
            raise ValueError("No valid test users found.")
        
        # Group test interactions by user
        user_rated_jobs = {}
        user_test_jobs = {}
        
        for user in test_users:
            user_interactions = test_interactions[test_interactions[user_col] == user]
            user_rated_jobs[user] = list(user_interactions[job_col])
            
            # For enhanced ratings, combine explicit ratings and sentiment
            if rating_col in user_interactions.columns or comment_col in user_interactions.columns:
                enhanced_ratings = self.calculate_batch_enhanced_ratings(
                    user_interactions,
                    rating_col=rating_col,
                    comment_col=comment_col
                )
                # Get jobs with positive ratings (above threshold)
                threshold = 0.6  # Consider ratings above 0.6 as positive
                positive_indices = np.where(enhanced_ratings > threshold)[0]
                user_test_jobs[user] = [
                    user_interactions.iloc[i][job_col] for i in positive_indices
                ]
            else:
                # If no ratings, use all jobs as positive
                user_test_jobs[user] = list(user_interactions[job_col])
        
        # Generate recommendations for all test users
        recommendations = self.recommend_batch(
            user_ids=test_users,
            top_k=top_k,
            exclude_rated=exclude_rated,
            user_rated_jobs=user_rated_jobs if exclude_rated else None
        )
        
        # Prepare data for metric calculation
        all_recommendations = []
        all_ground_truth = []
        
        for user in test_users:
            if user in recommendations and user in user_test_jobs:
                user_recs = [rec['job_id'] for rec in recommendations[user]]
                all_recommendations.append(user_recs)
                all_ground_truth.append(user_test_jobs[user])
        
        # Préparation pour MAE/RMSE (corrigé)
        actual_ratings = []
        predicted_ratings = []
        for user in test_users:
            user_interactions = test_interactions[test_interactions[user_col] == user]
            if rating_col in user_interactions.columns:
                for _, row in user_interactions.iterrows():
                    job_id = row[job_col]
                    if job_id in self.job_to_idx:
                        user_idx = self.user_to_idx[user]
                        job_idx = self.job_to_idx[job_id]
                        pred = self.graph_model(
                            self.graph.to(self.device),
                            torch.tensor([user_idx], device=self.device),
                            torch.tensor([job_idx], device=self.device)
                        )
                        actual_ratings.append(row[rating_col])
                        predicted_ratings.append(float(pred.cpu().detach().numpy()[0]))
        if len(actual_ratings) == 0 or len(predicted_ratings) == 0:
            actual_ratings = None
            predicted_ratings = None
        else:
            actual_ratings = np.array(actual_ratings)
            predicted_ratings = np.array(predicted_ratings)
        # Calcul des métriques
        metrics = calculate_recommendation_metrics(
            recommendations=all_recommendations,
            ground_truth=all_ground_truth,
            actual_ratings=actual_ratings,
            predicted_ratings=predicted_ratings,
            k_values=[5, 10]
        )
        # Add actual and predicted ratings to metrics for downstream use (e.g., plotting)
        metrics['actual_ratings'] = actual_ratings
        metrics['predicted_ratings'] = predicted_ratings
        return metrics
    
    def get_job_embeddings(self) -> Dict[Any, np.ndarray]:
        """
        Get learned embeddings for all jobs.
        
        Returns:
            Dict[Any, np.ndarray]: Dictionary mapping job IDs to their embeddings.
        """
        if self.graph is None or self.graph_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Extract job embeddings from the model
        self.graph_model.eval()
        with torch.no_grad():
            # Apply graph convolutions to get final node embeddings
            x = torch.cat([
                self.graph_model.user_embedding.weight,
                self.graph_model.job_embedding.weight
            ], dim=0)
            
            edge_index = self.graph.edge_index.to(self.device)
            
            for i, conv in enumerate(self.graph_model.convs):
                x = conv(x, edge_index)
                if i < len(self.graph_model.convs) - 1:
                    x = F.relu(x)
            
            # Extract job embeddings
            job_embs = x[self.graph_model.num_users:].cpu().numpy()
        
        # Map job indices to IDs
        job_embeddings = {
            self.idx_to_job[idx]: job_embs[idx]
            for idx in range(len(self.idx_to_job))
        }
        
        return job_embeddings
    
    def save_model(self, path: str):
        """
        Save the entire recommendation system.
        
        Args:
            path: Directory path to save the model.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save BERT embedding model
        embedding_path = os.path.join(path, 'bert_embeddings')
        os.makedirs(embedding_path, exist_ok=True)
        self.embedding_model.save_cache(embedding_path)
        
        # Save sentiment analysis model
        sentiment_path = os.path.join(path, 'sentiment')
        os.makedirs(sentiment_path, exist_ok=True)
        self.sentiment_model.save_model(sentiment_path)
        
        # Save graph model if available
        if self.graph_model is not None:
            graph_path = os.path.join(path, 'graph_model')
            os.makedirs(graph_path, exist_ok=True)
            self.graph_model.save(graph_path)
            
            # Save the graph
            if self.graph is not None:
                torch.save(self.graph, os.path.join(path, 'interaction_graph.pt'))
        
        # Save mappings and metadata
        mappings = {
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'job_to_idx': self.job_to_idx,
            'idx_to_job': self.idx_to_job
        }
        
        with open(os.path.join(path, 'mappings.json'), 'w') as f:
            # Convert keys to strings for JSON serialization
            serializable_mappings = {
                'user_to_idx': {str(k): v for k, v in self.user_to_idx.items()},
                'idx_to_user': {str(k): str(v) for k, v in self.idx_to_user.items()},
                'job_to_idx': {str(k): v for k, v in self.job_to_idx.items()},
                'idx_to_job': {str(k): str(v) for k, v in self.idx_to_job.items()}
            }
            json.dump(serializable_mappings, f)
        
        # Save configuration and metadata
        config = {
            'sentiment_weight': self.sentiment_weight,
            'rating_weight': self.rating_weight,        'metadata': self.metadata
        }
        
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None):
        """
        Load a saved recommendation system.
        
        Args:
            path: Directory path where the model is saved.
            device: Device to load the models to ('cpu' or 'cuda').
            
        Returns:
            JobRecommender: Loaded recommendation system.
        """
        # Safety check for CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
            
        # Determine device
        device = device if device else ('cpu')
        
        # Autoriser les classes torch_geometric lors du chargement
        import torch.serialization
        torch.serialization.add_safe_globals(['torch_geometric.data.data.DataEdgeAttr'])
        
        # Load configuration
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Extract weights and metadata
        sentiment_weight = config.get('sentiment_weight', 0.5)
        rating_weight = config.get('rating_weight', 0.5)
        metadata = config.get('metadata', {})
        
        # Load mappings
        with open(os.path.join(path, 'mappings.json'), 'r') as f:
            serialized_mappings = json.load(f)
            
        # Convert string keys back to original types
        mappings = {
            'user_to_idx': {k: int(v) for k, v in serialized_mappings['user_to_idx'].items()},
            'idx_to_user': {int(k): v for k, v in serialized_mappings['idx_to_user'].items()},
            'job_to_idx': {k: int(v) for k, v in serialized_mappings['job_to_idx'].items()},
            'idx_to_job': {int(k): v for k, v in serialized_mappings['idx_to_job'].items()}
        }
        
        # Load models
        embedding_model = BERTEmbeddings.load_from_cache(
            os.path.join(path, 'bert_embeddings'),
            device=device
        )
        
        sentiment_model = SentimentAnalysis.load_model(
            os.path.join(path, 'sentiment'),
            device=device
        )
        
        # Create recommender instance
        recommender = cls(
            embedding_model=embedding_model,
            sentiment_model=sentiment_model,
            device=device,
            sentiment_weight=sentiment_weight,
            rating_weight=rating_weight
        )
        
        # Set mappings
        recommender.user_to_idx = mappings['user_to_idx']
        recommender.idx_to_user = mappings['idx_to_user']
        recommender.job_to_idx = mappings['job_to_idx']
        recommender.idx_to_job = mappings['idx_to_job']
          # Load graph model if available
        graph_model_path = os.path.join(path, 'graph_model')
        if os.path.exists(graph_model_path):
            from src.models.gcn import GCNRecommender
            try:
                recommender.graph_model, _ = GCNRecommender.load(graph_model_path, device=device)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du modèle GCN: {e}")
                logger.warning("Tentative de chargement manuel avec weights_only=False...")
                
                # Chargement manuel avec weights_only=False
                gcn_path = os.path.join(graph_model_path, 'model.pt')
                if os.path.exists(gcn_path):
                    # Obtenir les paramètres du modèle depuis config.json
                    with open(os.path.join(graph_model_path, 'config.json'), 'r') as f:
                        model_config = json.load(f)
                    
                    # Créer une nouvelle instance
                    recommender.graph_model = GCNRecommender(
                        embedding_dim=model_config.get('embedding_dim', 64),
                        hidden_dim=model_config.get('hidden_dim', 32),
                        num_users=model_config.get('num_users', len(recommender.user_to_idx)),
                        num_jobs=model_config.get('num_jobs', len(recommender.job_to_idx)),
                        num_layers=model_config.get('num_layers', 2),
                        dropout=model_config.get('dropout', 0.1),
                        conv_type=model_config.get('conv_type', 'GCNConv')
                    )
                    
                    # Charger les poids
                    recommender.graph_model.load_state_dict(
                        torch.load(gcn_path, map_location=device, weights_only=False)
                    )
                    recommender.graph_model.to(device)
                    recommender.graph_model.eval()
          # Load the graph if available
        graph_path = os.path.join(path, 'interaction_graph.pt')
        if os.path.exists(graph_path):
            try:
                # Essayer d'abord avec weights_only=True (par défaut depuis PyTorch 2.6)
                recommender.graph = torch.load(graph_path, map_location=device)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du graphe: {e}")
                logger.warning("Tentative avec weights_only=False...")
                recommender.graph = torch.load(graph_path, map_location=device, weights_only=False)
        
        # Set metadata
        recommender.metadata = metadata
        
        return recommender
    
    def find_similar_jobs(self, job_id: Any, top_k: int = 5) -> list:
        """
        Find the most similar jobs to a given job using cosine similarity of embeddings.
        Args:
            job_id: The job ID to find similarities for.
            top_k: Number of similar jobs to return.
        Returns:
            List of tuples (job_id, similarity) sorted by similarity descending.
        """
        job_embeddings = self.get_job_embeddings()
        if job_id not in job_embeddings:
            raise ValueError(f"Job ID {job_id} not found in job embeddings.")
        target_emb = job_embeddings[job_id]
        similarities = []
        for other_id, emb in job_embeddings.items():
            if other_id == job_id:
                continue
            sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb))
            similarities.append((other_id, float(sim)))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
