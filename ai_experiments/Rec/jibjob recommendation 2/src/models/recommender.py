"""
Main recommender system for JibJob that integrates all components.
This module serves as the main entry point for the recommendation functionality.
"""

import torch
from torch_geometric.data import HeteroData 
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
        # graph_model: Optional[Union[GCNRecommender, HeterogeneousGCN]] = None,
        graph_model: Optional[HeterogeneousGCN] = None, # Specifically HeterogeneousGCN now for Option B
        device: Optional[str] = None,
        sentiment_weight: float = 0.5,
        rating_weight: float = 0.5,
        # embedding_model_name: str = "bert-base-multilingual-cased", # This was the original but changed it to distilbert-base-uncased
        embedding_model_name: str = "distilbert-base-uncased", # Ensure consistency with demo
        sentiment_model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        user_feature_dim: int = 64, # Add a parameter for user initial feature dimension for HeteroGCN
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
        self.user_feature_dim = user_feature_dim # Store this
        
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

        
        # Add storage for pre-computed job embeddings and initial user features
        self.all_job_bert_embeddings: Optional[torch.Tensor] = None
        self.all_user_initial_features: Optional[torch.Tensor] = None
        self.jobs_df_internal: Optional[pd.DataFrame] = None # To hold job texts
        self.training_interactions_df: Optional[pd.DataFrame] = None # To store interactions from train()
    
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
        jobs_df: pd.DataFrame, # <-- ADD jobs_df to access job texts for BERT
        user_col: str = 'user_id',
        job_col: str = 'job_id',
        rating_col: Optional[str] = 'rating',
        comment_col: Optional[str] = 'comment',
        rating_max: float = 5.0,
        rebuild_mappings: bool = True, # Default to True to ensure consistency for a new training run
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


        ##### NEW RecSys CODE #####
        logger.info("Preparing interaction data for Heterogeneous GCN...")
        self.jobs_df_internal = jobs_df.copy() # Store for later use (e.g., find_similar_jobs)

        if rebuild_mappings or not self.user_to_idx or not self.job_to_idx:
            logger.info("Rebuilding user and job mappings...")
            unique_users = interactions_df[user_col].unique()
            self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
            self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
            
            # Jobs mapping should come from jobs_df to ensure all jobs have features
            unique_jobs_in_jobs_df = jobs_df[job_col].unique()
            self.job_to_idx = {job: idx for idx, job in enumerate(unique_jobs_in_jobs_df)}
            self.idx_to_job = {idx: job for job, idx in self.job_to_idx.items()}
            logger.info(f"Created mappings for {len(self.user_to_idx)} users and {len(self.job_to_idx)} jobs.")

        # 1. Pre-compute BERT embeddings for ALL jobs from jobs_df
        # Ensure jobs_df is ordered by self.job_to_idx for the embeddings tensor
        ordered_job_ids = [self.idx_to_job[i] for i in range(len(self.idx_to_job))]
        job_texts_for_bert = []
        valid_ordered_job_ids_for_bert = []

        # Extract title and description, handling potential 'cleaned_' versions
        for j_id in ordered_job_ids:
            job_row = self.jobs_df_internal[self.jobs_df_internal[job_col] == j_id]
            if not job_row.empty:
                job_info = job_row.iloc[0]
                # Prefer cleaned versions if they exist (from process_job_descriptions)
                title = str(job_info.get('cleaned_title', job_info.get('title', '')))
                desc = str(job_info.get('cleaned_description', job_info.get('description', '')))
                text = (title + " " + desc).strip()
                if text:
                    job_texts_for_bert.append(text)
                    valid_ordered_job_ids_for_bert.append(j_id) # Keep track if some jobs had no text
                else:
                    # Handle jobs with no text: add zero embeddings or skip
                    logger.warning(f"Job {j_id} has no text for BERT embedding. Placeholder will be used.")
                    job_texts_for_bert.append("") # BERT will produce an embedding for empty string
                    valid_ordered_job_ids_for_bert.append(j_id) 
            else:
                 logger.error(f"Job ID {j_id} from mapping not found in jobs_df_internal. This should not happen.")
                 # Add placeholder to maintain order, though this indicates an issue
                 job_texts_for_bert.append("") 
                 valid_ordered_job_ids_for_bert.append(j_id)

        if not job_texts_for_bert:
            raise ValueError("No job texts found to generate BERT embeddings.")

        logger.info(f"Generating BERT embeddings for {len(job_texts_for_bert)} jobs...")
        bert_embeddings_np = self.embedding_model.batch_get_embeddings(job_texts_for_bert, batch_size=32)
        self.all_job_bert_embeddings = torch.tensor(bert_embeddings_np, dtype=torch.float).to(self.device)
        logger.info(f"Job BERT embeddings shape: {self.all_job_bert_embeddings.shape}")

        # 2. Prepare initial user features (e.g., random or learnable embeddings wrapper)
        # For HeteroGCN, users also need an initial '.x'. We can make them learnable via nn.Embedding
        # or just initialize them randomly if the GCN's first layer Linear will project them.
        # Let's create a learnable nn.Embedding for users that HeteroGCN will then project.
        if self.graph_model is None or create_new_model: # create_new_model is a param to self.train
             # This user_embedding will be part of HeteroGCN via node_embeddings dict.
             # The HeterogeneousGCN takes node_feature_dims for input 'x', then projects.
             # So self.all_user_initial_features can be indices for an nn.Embedding or random.
             # For simplicity with HeteroGCN's nn.Linear input projection, let's use random.
             # The HeteroGCN.node_embeddings['user'](data['user'].x) will handle it.
            self.all_user_initial_features = torch.randn(len(self.user_to_idx), self.user_feature_dim).to(self.device)
            logger.info(f"Initial user features shape: {self.all_user_initial_features.shape}")


        # 3. Process interactions
        # Filter interactions to only include users and jobs present in our new mappings
        valid_interaction_mask = interactions_df[user_col].isin(self.user_to_idx.keys()) & \
                                 interactions_df[job_col].isin(self.job_to_idx.keys())
        filtered_interactions_df = interactions_df[valid_interaction_mask].copy()

        if filtered_interactions_df.empty:
            raise ValueError("No valid interactions found after filtering with user/job mappings from jobs_df.")

        user_indices = torch.tensor([self.user_to_idx[user] for user in filtered_interactions_df[user_col]], dtype=torch.long)
        job_indices = torch.tensor([self.job_to_idx[job] for job in filtered_interactions_df[job_col]], dtype=torch.long)
        
        enhanced_ratings = self.calculate_batch_enhanced_ratings(
            filtered_interactions_df, # Use filtered
            rating_col=rating_col,
            comment_col=comment_col,
            normalize_ratings=True, # Assuming ratings in file are 1-5, normalized to 0-1 here
            rating_max=5.0 # Explicitly set this if needed, otherwise ensure normalize_ratings handles it
        )
        enhanced_ratings_tensor = torch.tensor(enhanced_ratings, dtype=torch.float)
        
        self.metadata["num_users"] = len(self.user_to_idx)
        self.metadata["num_jobs"] = len(self.job_to_idx)
        self.metadata["num_interactions"] = len(filtered_interactions_df)
        
        logger.info(f"Prepared {len(user_indices)} interaction edges.")
        return user_indices, job_indices, enhanced_ratings_tensor

        #####OLD RecSys CODE#####
        # # Create or update user/job mappings
        # if rebuild_mappings or not self.user_to_idx:
        #     unique_users = interactions_df[user_col].unique()
        #     self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        #     self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        # if rebuild_mappings or not self.job_to_idx:
        #     unique_jobs = interactions_df[job_col].unique()
        #     self.job_to_idx = {job: idx for idx, job in enumerate(unique_jobs)}
        #     self.idx_to_job = {idx: job for job, idx in self.job_to_idx.items()}
        
        # # Convert user and job IDs to indices
        # user_indices = torch.tensor([
        #     self.user_to_idx[user] for user in interactions_df[user_col]
        #     if user in self.user_to_idx  # Filter out users not in mapping
        # ], dtype=torch.long)
        
        # job_indices = torch.tensor([
        #     self.job_to_idx[job] for job in interactions_df[job_col]
        #     if job in self.job_to_idx  # Filter out jobs not in mapping
        # ], dtype=torch.long)
        
        # # Calculate enhanced ratings
        # enhanced_ratings = None
        # if rating_col in interactions_df.columns or comment_col in interactions_df.columns:
        #     enhanced_ratings = self.calculate_batch_enhanced_ratings(
        #         interactions_df,
        #         rating_col=rating_col,
        #         comment_col=comment_col,
        #         normalize_ratings=True,
        #         rating_max=rating_max
        #     )
        #     enhanced_ratings = torch.tensor(enhanced_ratings, dtype=torch.float)
        # else:
        #     # Default to neutral ratings if no rating information
        #     enhanced_ratings = torch.ones(len(interactions_df), dtype=torch.float) * 0.5
        
        # # Filter to include only users and jobs in the mappings
        # valid_indices = []
        # for i, (user, job) in enumerate(zip(interactions_df[user_col], interactions_df[job_col])):
        #     if user in self.user_to_idx and job in self.job_to_idx:
        #         valid_indices.append(i)
        
        # user_indices = torch.tensor([self.user_to_idx[interactions_df[user_col].iloc[i]] for i in valid_indices], dtype=torch.long)
        # job_indices = torch.tensor([self.job_to_idx[interactions_df[job_col].iloc[i]] for i in valid_indices], dtype=torch.long)
        # enhanced_ratings = enhanced_ratings[valid_indices]
        
        # # Update metadata
        # self.metadata["num_users"] = len(self.user_to_idx)
        # self.metadata["num_jobs"] = len(self.job_to_idx)
        # self.metadata["num_interactions"] = len(valid_indices)
        
        # return user_indices, job_indices, enhanced_ratings
    
    def build_graph(
        self, 
        user_indices: torch.Tensor, 
        job_indices: torch.Tensor,
        ratings: torch.Tensor
    ):  # Returns HeteroData
        
        ##### NEW RecSys CODE #####
        """Builds a HeteroData object for the HeterogeneousGCN."""
        logger.info("Building HeteroData graph...")
        if self.all_job_bert_embeddings is None or self.all_user_initial_features is None:
            raise ValueError("Job BERT embeddings or user features not prepared. Call prepare_interaction_data first.")

        hetero_graph = HeteroData()

        # Node features
        hetero_graph['user'].x = self.all_user_initial_features.to(self.device)
        hetero_graph['job'].x = self.all_job_bert_embeddings.to(self.device)
        
        # User-job interactions (edges)
        # Ensure indices are on the correct device if not already
        u_indices_tensor = user_indices.to(self.device)
        j_indices_tensor = job_indices.to(self.device)
        ratings_tensor = ratings.to(self.device)

        hetero_graph['user', 'rates', 'job'].edge_index = torch.stack([u_indices_tensor, j_indices_tensor], dim=0)
        hetero_graph['user', 'rates', 'job'].edge_attr = ratings_tensor # Store ratings as edge attributes

        # Add reverse edges for message passing in both directions (important for many GNNs)
        hetero_graph['job', 'rated_by', 'user'].edge_index = torch.stack([j_indices_tensor, u_indices_tensor], dim=0)
        hetero_graph['job', 'rated_by', 'user'].edge_attr = ratings_tensor # Share attributes or define new ones
        
        logger.info(f"HeteroData graph built: {hetero_graph}")
        self.graph = hetero_graph # Store the graph
        return hetero_graph
        

        ##### OLD RecSys CODE #####
        # """
        # Build the interaction graph for the recommendation model.
        
        # Args:
        #     user_indices: Tensor of user indices.
        #     job_indices: Tensor of job indices.
        #     ratings: Tensor of enhanced ratings.
            
        # Returns:
        #     torch_geometric.data.Data: The interaction graph.
        # """
        # # Build the interaction graph (user-job bipartite graph)
        # graph = build_interaction_graph(
        #     user_indices=user_indices,
        #     job_indices=job_indices,
        #     ratings=ratings,
        #     num_users=len(self.user_to_idx),
        #     num_jobs=len(self.job_to_idx)
        # )
        
        # self.graph = graph
        # return graph
    
    def create_gcn_model(
            
        ##### NEW RecSys CODE #####
        self,
        # Parameters for HeterogeneousGCN
        gcn_embedding_dim: int = 64, # This is the target embedding dim AFTER projection by HeteroGCN's first layer
        hidden_dim: int = 64,    # Hidden dim for GCN layers
        num_layers: int = 2,
        dropout: float = 0.2
        # conv_type is usually part of HeteroConv definition, let's assume GCNConv for now
            
        ##### OLD RecSys CODE #####
        # self,
        # embedding_dim: int = 64,
        # hidden_dim: int = 64,
        # num_layers: int = 2,
        # dropout: float = 0.2,
        # conv_type: str = 'gcn'
    ):
        
        ##### NEW RecSys CODE #####
        
        """Creates a HeterogeneousGCN model."""
        if self.all_job_bert_embeddings is None or self.all_user_initial_features is None:
            raise ValueError("Job/User features not prepared for GCN model creation.")

        bert_embedding_dim = self.all_job_bert_embeddings.shape[1]
        
        node_feature_dims = {
            'user': self.user_feature_dim, # Dimension of self.all_user_initial_features
            'job': bert_embedding_dim      # Dimension of BERT embeddings
        }
        
        self.graph_model = HeterogeneousGCN(
            node_types=['user', 'job'],
            edge_types=[
                ('user', 'rates', 'job'), 
                ('job', 'rated_by', 'user') # Important to include reverse edges if model uses them
            ],
            node_feature_dims=node_feature_dims,
            embedding_dim=gcn_embedding_dim, # Target dim after HeteroGCN's initial nn.Linear projection
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        logger.info(f"HeterogeneousGCN model created with node_feature_dims: {node_feature_dims}, target embedding_dim: {gcn_embedding_dim}")
        return self.graph_model
    

        # ##### OLD RecSys CODE #####
        # """
        # Create a new GCN recommendation model.
        
        # Args:
        #     embedding_dim: Dimension of the initial embeddings.
        #     hidden_dim: Dimension of hidden layers.
        #     num_layers: Number of graph convolution layers.
        #     dropout: Dropout rate for regularization.
        #     conv_type: Type of graph convolution ('gcn', 'sage', or 'gat').
            
        # Returns:
        #     GCNRecommender: The created GCN model.
        # """
        # self.graph_model = GCNRecommender(
        #     num_users=len(self.user_to_idx),
        #     num_jobs=len(self.job_to_idx),
        #     embedding_dim=embedding_dim,
        #     hidden_dim=hidden_dim,
        #     num_layers=num_layers,
        #     dropout=dropout,
        #     conv_type=conv_type
        # ).to(self.device)
        
        # return self.graph_model
    
    def train(
        self,
        interactions_df: pd.DataFrame,
        jobs_df: pd.DataFrame, # <-- ADD job_df to access job texts for BERT
        user_col: str = 'user_id',
        job_col: str = 'job_id',
        rating_col: Optional[str] = 'rating',
        comment_col: Optional[str] = 'comment',
        val_ratio: float = 0.2,
        epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        early_stop_patience: int = 50,
        rebuild_mappings: bool = False,
        create_new_model: bool = True,
        gcn_embedding_dim: int = 64, 
        hidden_dim_gcn: int = 64, # Renamed to avoid clash with BERT's hidden_dim if that term is used elsewhere
        num_layers_gcn: int = 2,
        dropout_gcn: float = 0.2,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = 'gcn'
    ) -> Dict:
        
        ###### NEW RecSys CODE #####
        
        """Train the HeterogeneousGCN recommendation model."""
        logger.info("Starting training process with HeterogeneousGCN...")
        self.training_interactions_df = interactions_df.copy() # Store it
        
        # Prepare interaction data, BERT embeddings for jobs, and initial user features
        user_indices, job_indices, ratings = self.prepare_interaction_data(
            interactions_df=interactions_df,
            jobs_df=jobs_df, # Pass it here
            user_col=user_col,
            job_col=job_col,
            rating_col=rating_col, # Make sure rating_col is defined or passed
            comment_col=comment_col, # Make sure comment_col is defined or passed
            rebuild_mappings=True # Usually true for a fresh train run
        )
        
        # Build the HeteroData graph object
        graph = self.build_graph(user_indices, job_indices, ratings) # self.graph is updated here
        
        if create_new_model or self.graph_model is None:
            self.create_gcn_model( # This now creates HeterogeneousGCN
                gcn_embedding_dim=gcn_embedding_dim,
                hidden_dim=hidden_dim_gcn,
                num_layers=num_layers_gcn,
                dropout=dropout_gcn
            )
        elif not isinstance(self.graph_model, HeterogeneousGCN):
            logger.warning("Existing graph_model is not HeterogeneousGCN. Creating a new one.")
            self.create_gcn_model(
                gcn_embedding_dim=gcn_embedding_dim,
                hidden_dim=hidden_dim_gcn,
                num_layers=num_layers_gcn,
                dropout=dropout_gcn
            )


        # Split interaction *edges* for training/validation
        # This uses the indices from the full interaction set
        num_samples = len(user_indices) # Number of edges
        indices = torch.randperm(num_samples)
        split_idx = int(num_samples * (1 - val_ratio)) # val_ratio is a train param
        
        train_mask = indices[:split_idx]
        val_mask = indices[split_idx:]
        
        # These are indices into the original user_indices, job_indices, ratings tensors
        train_user_indices_for_loss = user_indices[train_mask].to(self.device)
        train_job_indices_for_loss = job_indices[train_mask].to(self.device)
        train_ratings_for_loss = ratings[train_mask].to(self.device)
        
        val_user_indices_for_loss = user_indices[val_mask].to(self.device)
        val_job_indices_for_loss = job_indices[val_mask].to(self.device)
        val_ratings_for_loss = ratings[val_mask].to(self.device)

        # Optimizer and Loss
        optimizer = torch.optim.Adam(
            self.graph_model.parameters(), # Parameters of HeterogeneousGCN
            lr=learning_rate, # learning_rate is a train param
            weight_decay=weight_decay # weight_decay is a train param
        )
        loss_fn = torch.nn.MSELoss() # Still suitable for normalized 0-1 ratings

        # Training loop
        # ... (The rest of the training loop logic from your existing train method should be largely compatible)
        # Ensure that calls to self.graph_model use the correct inputs:
        # predictions = self.graph_model(graph.to(self.device), batch_user_indices, batch_job_indices)
        # The `graph` here is the full HeteroData graph.
        # `batch_user_indices` and `batch_job_indices` are the specific user/job indices for the current batch of *edges* whose ratings we want to predict.
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        # The graph itself is already on device if build_graph and feature prep put it there
        # Or ensure it is before the loop:
        graph = self.graph.to(self.device) 

        for epoch in range(epochs): # epochs is a train param
            self.graph_model.train()
            start_time = time.time()
            current_train_loss = 0.0
            
            # Process in batches of edges
            num_train_edges = len(train_user_indices_for_loss)
            for i in range(0, num_train_edges, batch_size): # batch_size is a train param
                batch_user_idx = train_user_indices_for_loss[i:i+batch_size]
                batch_job_idx = train_job_indices_for_loss[i:i+batch_size]
                batch_ratings_target = train_ratings_for_loss[i:i+batch_size]
                
                optimizer.zero_grad()
                # Pass the whole graph, and the specific user/job indices for edges in this batch
                predictions = self.graph_model(graph, batch_user_idx, batch_job_idx)
                loss = loss_fn(predictions, batch_ratings_target)
                loss.backward()
                optimizer.step()
                current_train_loss += loss.item() * len(batch_user_idx)
            
            current_train_loss /= num_train_edges
            history['train_loss'].append(current_train_loss)
            
            # Validation
            self.graph_model.eval()
            current_val_loss = 0.0
            num_val_edges = len(val_user_indices_for_loss)
            with torch.no_grad():
                for i in range(0, num_val_edges, batch_size):
                    batch_user_idx = val_user_indices_for_loss[i:i+batch_size]
                    batch_job_idx = val_job_indices_for_loss[i:i+batch_size]
                    batch_ratings_target = val_ratings_for_loss[i:i+batch_size]
                    
                    predictions = self.graph_model(graph, batch_user_idx, batch_job_idx)
                    loss = loss_fn(predictions, batch_ratings_target)
                    current_val_loss += loss.item() * len(batch_user_idx)
            
            current_val_loss /= num_val_edges
            history['val_loss'].append(current_val_loss)
            
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {current_train_loss:.6f}, Val Loss: {current_val_loss:.6f}, Time: {epoch_time:.2f}s")
            
            # Early stopping
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                # Optionally save best model here
            else:
                patience_counter += 1
            if patience_counter >= early_stop_patience: # early_stop_patience from train params
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
        self.metadata["last_trained"] = time.strftime("%Y-%m-%d %H:%M:%S")
        return history
        

        ##### OLD RecSys CODE #####
        # """
        # Train the recommendation model on interaction data.
        
        # Args:
        #     interactions_df: DataFrame with user-job interactions.
        #     user_col: Name of the user ID column.
        #     job_col: Name of the job ID column.
        #     rating_col: Name of the explicit rating column (optional).
        #     comment_col: Name of the comment text column (optional).
        #     val_ratio: Fraction of data to use for validation.
        #     epochs: Number of training epochs.
        #     batch_size: Training batch size.
        #     learning_rate: Learning rate for optimizer.
        #     weight_decay: L2 regularization strength.
        #     early_stop_patience: Number of epochs to wait before early stopping.
        #     rebuild_mappings: Whether to rebuild user and job ID mappings.
        #     create_new_model: Whether to create a new model or use an existing one.
        #     embedding_dim: Dimension of node embeddings.
        #     hidden_dim: Dimension of hidden layers.
        #     num_layers: Number of graph convolution layers.
        #     dropout: Dropout rate.
        #     conv_type: Type of graph convolution.
            
        # Returns:
        #     Dict: Training history.
        # """
        # # Prepare interaction data
        # user_indices, job_indices, ratings = self.prepare_interaction_data(
        #     interactions_df=interactions_df,
        #     user_col=user_col,
        #     job_col=job_col,
        #     rating_col=rating_col,
        #     comment_col=comment_col,
        #     rebuild_mappings=rebuild_mappings
        # )
        
        # # Build the interaction graph
        # self.build_graph(user_indices, job_indices, ratings)
        
        # # Create a new model or use existing
        # if create_new_model or self.graph_model is None:
        #     self.create_gcn_model(
        #         embedding_dim=embedding_dim,
        #         hidden_dim=hidden_dim,
        #         num_layers=num_layers,
        #         dropout=dropout,
        #         conv_type=conv_type
        #     )
        
        # # Split data into train and validation
        # num_samples = len(user_indices)
        # indices = torch.randperm(num_samples)
        # split_idx = int(num_samples * (1 - val_ratio))
        
        # train_mask = indices[:split_idx]
        # val_mask = indices[split_idx:]
        
        # train_user_indices = user_indices[train_mask]
        # train_job_indices = job_indices[train_mask]
        # train_ratings = ratings[train_mask]
        
        # val_user_indices = user_indices[val_mask]
        # val_job_indices = job_indices[val_mask]
        # val_ratings = ratings[val_mask]
        
        # # Move the graph to device
        # graph = self.graph.to(self.device)
        
        # # Setup optimizer
        # optimizer = torch.optim.Adam(
        #     self.graph_model.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay
        # )
        
        # # Setup loss function
        # loss_fn = torch.nn.MSELoss()
        
        # # Training loop
        # best_val_loss = float('inf')
        # patience_counter = 0
        # history = {'train_loss': [], 'val_loss': []}
        
        # for epoch in range(epochs):
        #     # Training
        #     self.graph_model.train()
        #     start_time = time.time()
        #     train_loss = 0.0
            
        #     # Process in batches
        #     for i in range(0, len(train_user_indices), batch_size):
        #         batch_user_indices = train_user_indices[i:i+batch_size].to(self.device)
        #         batch_job_indices = train_job_indices[i:i+batch_size].to(self.device)
        #         batch_ratings = train_ratings[i:i+batch_size].to(self.device)
                
        #         # Forward pass
        #         predictions = self.graph_model(
        #             graph,
        #             batch_user_indices,
        #             batch_job_indices
        #         )
                
        #         # Calculate loss
        #         loss = loss_fn(predictions, batch_ratings)
                
        #         # Backward pass and optimization
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
                
        #         train_loss += loss.item() * len(batch_ratings)
            
        #     # Calculate average training loss
        #     train_loss /= len(train_user_indices)
        #     history['train_loss'].append(train_loss)
            
        #     # Validation
        #     self.graph_model.eval()
        #     val_loss = 0.0
            
        #     with torch.no_grad():
        #         for i in range(0, len(val_user_indices), batch_size):
        #             batch_user_indices = val_user_indices[i:i+batch_size].to(self.device)
        #             batch_job_indices = val_job_indices[i:i+batch_size].to(self.device)
        #             batch_ratings = val_ratings[i:i+batch_size].to(self.device)
                    
        #             # Forward pass
        #             predictions = self.graph_model(
        #                 graph,
        #                 batch_user_indices,
        #                 batch_job_indices
        #             )
                
        #             # Calculate loss
        #             loss = loss_fn(predictions, batch_ratings)
        #             val_loss += loss.item() * len(batch_ratings)
            
        #     # Calculate average validation loss
        #     val_loss /= len(val_user_indices)
        #     history['val_loss'].append(val_loss)
            
        #     epoch_time = time.time() - start_time
        #     logger.info(f"Epoch {epoch+1}/{epochs} - "
        #                f"Train Loss: {train_loss:.6f}, "
        #                f"Val Loss: {val_loss:.6f}, "
        #                f"Time: {epoch_time:.2f}s")
            
        #     # Early stopping
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         patience_counter = 0
        #     else:
        #         patience_counter += 1
                
        #     if patience_counter >= early_stop_patience:
        #         logger.info(f"Early stopping triggered after {epoch+1} epochs")
        #         break
        
        # # Update metadata
        # self.metadata["last_trained"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # return history
    
    def recommend(
        self, 
        user_id: Any, 
        top_k: int = 10,
        exclude_rated: bool = True,
        rated_job_ids_for_user: Optional[List[Any]] = None 
    ) -> List[Dict[str, Any]]:
        """
        Recommend jobs for a professional user using explicit two-stage filtering (category, then location).
        """
        # 1. Retrieve professional's selected_categories and location
        users_df = getattr(self, 'users_df', None)
        jobs_df = getattr(self, 'jobs_df_internal', None)
        if users_df is None or jobs_df is None:
            raise ValueError("User and job data must be loaded and set on the recommender.")
        user_row = users_df[users_df['user_id'] == user_id]
        if user_row.empty:
            logger.warning(f"User {user_id} not found.")
            return []
        user_info = user_row.iloc[0]
        if user_info.get('user_type', '') != 'professional':
            logger.warning(f"User {user_id} is not a professional. No recommendations.")
            return []
        selected_cats = set(str(user_info.get('selected_categories', '')).split(';'))
        location = str(user_info.get('location', '')).strip().lower()
        # 2. Filter jobs by category
        jobs_filtered_cat = jobs_df[jobs_df['category'].isin(selected_cats)]
        # 3. Filter by location (case-insensitive)
        jobs_filtered = jobs_filtered_cat[jobs_filtered_cat['location'].str.lower() == location]
        # 4. Optionally exclude already rated jobs
        if exclude_rated and rated_job_ids_for_user is not None:
            jobs_filtered = jobs_filtered[~jobs_filtered['job_id'].isin(rated_job_ids_for_user)]
        # 5. Prepare output (by recency or random for now)
        jobs_filtered = jobs_filtered.sort_values('job_id', ascending=False) # Or shuffle
        recs = [
            {'job_id': row['job_id'], 'score': 1.0} # Score is 1.0 for now (all pass filter)
            for _, row in jobs_filtered.head(top_k).iterrows()
        ]
        return recs

    ##################### OLD RecSys FUNCTION #####################
    # def recommend_batch(
    #     self, 
    #     user_ids: List[Any], 
    #     top_k: int = 10,
    #     exclude_rated: bool = True,
    #     user_rated_jobs: Optional[Dict[Any, List[Any]]] = None
    # ) -> Dict[Any, List[Dict[str, Any]]]:
    #     """
    #     Generate job recommendations for multiple users.
        
    #     Args:
    #         user_ids: List of user IDs to recommend jobs for.
    #         top_k: Number of recommendations per user.
    #         exclude_rated: Whether to exclude jobs each user has already rated.
    #         user_rated_jobs: Dictionary mapping user IDs to lists of rated job IDs.
    #                          Required if exclude_rated is True.
            
    #     Returns:
    #         Dict[Any, List[Dict[str, Any]]]: 
    #             Dictionary mapping user IDs to their recommendations.
    #     """
    #     if self.graph is None or self.graph_model is None:
    #         raise ValueError("Model not trained. Call train() first.")
        
    #     # Convert user IDs to indices
    #     valid_user_ids = [user_id for user_id in user_ids if user_id in self.user_to_idx]
    #     user_indices = [self.user_to_idx[user_id] for user_id in valid_user_ids]
        
    #     # Convert rated job IDs to indices if needed
    #     user_rated_indices = None
    #     if exclude_rated and user_rated_jobs:
    #         user_rated_indices = {}
    #         for user_id in valid_user_ids:
    #             if user_id in user_rated_jobs:
    #                 rated_indices = []
    #                 for job_id in user_rated_jobs[user_id]:
    #                     if job_id in self.job_to_idx:
    #                         rated_indices.append(self.job_to_idx[job_id])
    #                 user_rated_indices[self.user_to_idx[user_id]] = rated_indices
        
    #     # Get recommendations for all users
    #     recommendations_dict = {}
        
    #     # Move graph to device
    #     graph = self.graph.to(self.device)
        
    #     for user_id in tqdm(valid_user_ids, desc="Generating recommendations"):
    #         user_idx = self.user_to_idx[user_id]
            
    #         rated_indices = user_rated_indices.get(user_idx, []) if user_rated_indices else None
            
    #         top_job_indices, top_scores = self.graph_model.recommend(
    #             graph=graph,
    #             user_id=user_idx,
    #             top_k=top_k,
    #             exclude_rated=exclude_rated,
    #             rated_indices=rated_indices
    #         )
            
    #         # Convert indices back to IDs and format results
    #         user_recommendations = []
    #         for job_idx, score in zip(top_job_indices, top_scores):
    #             job_id = self.idx_to_job[job_idx]
    #             user_recommendations.append({
    #                 'job_id': job_id,
    #                 'score': float(score)
    #             })
                
    #         recommendations_dict[user_id] = user_recommendations
        
    #     return recommendations_dict
    
    # In src/models/recommender.py, inside JobRecommender class:

    def evaluate(
        self,
        test_interactions: pd.DataFrame,
        jobs_df: pd.DataFrame, 
        user_col: str = 'user_id',
        job_col: str = 'job_id',
        rating_col: Optional[str] = 'rating',
        comment_col: Optional[str] = 'comment',
        top_k: int = 10,
        exclude_rated: bool = True, 
        # rating_threshold_for_relevance: float = 0.6,
    ) -> Dict[str, Any]:
        
        """Evaluate HeteroGCN model."""
        if self.graph is None or not isinstance(self.graph_model, HeterogeneousGCN):
            raise ValueError("HeteroGCN model not trained/graph not built.")

        # This block checks the scale of ratings in the test_interactions dataframe
        if rating_col in test_interactions.columns and not test_interactions.empty:
            # Take a small sample of ratings, ensuring we don't try to sample more than available
            num_to_sample = min(5, len(test_interactions[rating_col].dropna()))
            if num_to_sample > 0:
                sample_ratings = test_interactions[rating_col].dropna().sample(num_to_sample).tolist()
                logger.debug(f"EVAL_DEBUG_RATINGS: Sample (normalized) ratings from test_interactions['{rating_col}']: {sample_ratings}")
                logger.debug(f"EVAL_DEBUG_RATINGS: Min/Max (normalized) ratings in test_interactions['{rating_col}']: [{test_interactions[rating_col].min():.4f}, {test_interactions[rating_col].max():.4f}]")

        if self.jobs_df_internal is None: # Should be set during prepare_interaction_data called by train
         self.jobs_df_internal = jobs_df.copy()

        logger.info("Evaluating HeterogeneousGCN model (using 'is_relevant_for_ranking_eval' for ground truth)...")
        actual_ratings_for_mae_rmse = []
        predicted_ratings_for_mae_rmse = []
        all_recommendations_for_ranking = []
        all_ground_truth_for_ranking = []

        # For logging details of a few users AFTER the loop
        detailed_log_buffer = [] 
        users_logged_count = 0
        MAX_USERS_TO_LOG_DETAIL = 2 # Control how many users' details are logged

        test_users_with_interactions = test_interactions[user_col].unique()
        test_users_in_model = [u for u in test_users_with_interactions if u in self.user_to_idx]

        

        if not test_users_in_model:
            logger.warning("No test users known to the model. Returning empty metrics.")
            # Return a full default metrics dict
            metrics_keys = [f'precision@{k}' for k in [5, top_k]] + \
                        [f'recall@{k}' for k in [5, top_k]] + \
                        [f'ndcg@{k}' for k in [5, top_k]]
            empty_metrics = {key: 0.0 for key in metrics_keys}
            empty_metrics.update({'mae': float('nan'), 'rmse': float('nan'), 
                                'actual_ratings_array_for_plot': np.array([]), 
                                'predicted_ratings_array_for_plot': np.array([])})
            return empty_metrics
        
        for user_id in tqdm(test_users_in_model, desc="Evaluating Users", 
                   position=0, leave=True, ncols=100):
            # user_idx should be retrieved here if needed, but user_id is often used for lookups
            current_user_idx = self.user_to_idx[user_id] # Make sure to use this for GCN calls specific to the user
            current_user_test_interactions = test_interactions[test_interactions[user_col] == user_id]
                
            # --- NEW Ground Truth for Ranking based on the boolean flag ---
            if 'is_relevant_for_ranking_eval' not in current_user_test_interactions.columns:
                logger.error("'is_relevant_for_ranking_eval' column missing from test_interactions. Cannot determine ground truth for ranking.")
                # Potentially skip this user or return error metrics
                relevant_job_ids_for_user = [] # Default to empty if flag is missing
            else:
                relevant_job_ids_for_user = current_user_test_interactions[
                    current_user_test_interactions['is_relevant_for_ranking_eval'] == True
                ][job_col].tolist()
            
            relevant_job_ids_for_user = [jid for jid in relevant_job_ids_for_user if jid in self.job_to_idx]
            # --- END NEW Ground Truth ---

            # --- MODIFIED LOGGING FOR GROUND TRUTH ---
            logger.debug(f"EVAL LOG User {user_id}: Test items and their (normalized enhanced) ratings:")
                
            # logger.info(f"EVAL LOG User {user_id}: Found {len(relevant_job_ids_for_user)} 'is_relevant_for_ranking_eval'==True jobs in test set. True Relevant Job IDs (first 5): {relevant_job_ids_for_user[:5]}")
            tqdm.write(f"EVAL LOG User {user_id}: Found {len(relevant_job_ids_for_user)} 'is_relevant_for_ranking_eval'==True jobs in test set. True Relevant Job IDs (first 5): {relevant_job_ids_for_user[:5]}")
            all_ground_truth_for_ranking.append(relevant_job_ids_for_user)
                
            
            rated_for_exclusion = []
            if exclude_rated and self.training_interactions_df is not None:
                user_interactions_from_training = self.training_interactions_df[
                    self.training_interactions_df[user_col] == user_id
                ]
                if not user_interactions_from_training.empty:
                    rated_for_exclusion = user_interactions_from_training[job_col].unique().tolist()

            recs_for_user = self.recommend(
                user_id=user_id, top_k=top_k, 
                exclude_rated=exclude_rated, 
                rated_job_ids_for_user=rated_for_exclusion
            )
            

            current_recommendations = [rec['job_id'] for rec in recs_for_user]

            all_recommendations_for_ranking.append(current_recommendations)


            # --- Modification for LOG BLOCK 2 conditional logging ---
            # Determine if this user is one of the first few we're processing in this evaluation batch
            # A simple way is to use the length of all_recommendations_for_ranking before appending current user's
            should_log_details_for_this_user = all_recommendations_for_ranking.index(current_recommendations) < 3 
            # --- End of Modification ---

            # Collect detailed info for a few users instead of logging directly in the loop
            if users_logged_count < MAX_USERS_TO_LOG_DETAIL:
                user_detail = {"user_id": user_id, "top_recs": [], "true_relevant_scores": []}
                for rec_item in recs_for_user[:top_k]:
                    user_detail["top_recs"].append({"job_id": rec_item['job_id'], "score": rec_item['score']})
                if relevant_job_ids_for_user:
                    for true_relevant_job_id in relevant_job_ids_for_user:
                        if true_relevant_job_id in self.job_to_idx:
                            true_relevant_job_idx = self.job_to_idx[true_relevant_job_id]
                            user_idx_tensor = torch.tensor([current_user_idx], device=self.device)
                            job_idx_tensor = torch.tensor([true_relevant_job_idx], device=self.device)
                            self.graph_model.eval()
                            with torch.no_grad():
                                pred_score_for_true_relevant = self.graph_model(self.graph.to(self.device), user_idx_tensor, job_idx_tensor).item()
                            user_detail["true_relevant_scores"].append({"job_id": true_relevant_job_id, "score": pred_score_for_true_relevant})
                detailed_log_buffer.append(user_detail)
                users_logged_count += 1
            
            # all_recommendations_for_ranking.append([rec['job_id'] for rec in recs_for_user])

            for _, interaction_row in current_user_test_interactions.iterrows():
                rating_value_from_row = interaction_row.get(rating_col) 
                if rating_value_from_row is None or not isinstance(rating_value_from_row, (int, float)):
                    continue
                true_rating_normalized = float(rating_value_from_row) 

                job_id_eval = interaction_row[job_col]
                if job_id_eval in self.job_to_idx and user_id in self.user_to_idx:
                    user_idx_eval = self.user_to_idx[user_id] 
                    job_idx_eval = self.job_to_idx[job_id_eval]
                    self.graph_model.eval()
                    with torch.no_grad():
                        predicted_score_tensor = self.graph_model(
                            self.graph.to(self.device),
                            torch.tensor([user_idx_eval], device=self.device),
                            torch.tensor([job_idx_eval], device=self.device)
                        )
                    predicted_score_normalized = predicted_score_tensor.item()
                    actual_ratings_for_mae_rmse.append(true_rating_normalized)
                    predicted_ratings_for_mae_rmse.append(predicted_score_normalized)
        

        # Now, print the collected detailed logs for a few users AFTER the loop
        if detailed_log_buffer:
            logger.info("--- Detailed GCN Evaluation for Sample Users ---")
            for user_detail in detailed_log_buffer:
                logger.info(f"User {user_detail['user_id']} - Top {top_k} GCN Recs & Scores:")
                for rec_item in user_detail["top_recs"]:
                    logger.info(f"  Rec: {rec_item['job_id']}, Score: {rec_item['score']:.4f}")
                if user_detail["true_relevant_scores"]:
                    logger.info(f"User {user_detail['user_id']} - GCN Scores for THEIR True Relevant Test Jobs:")
                    for score_info in user_detail["true_relevant_scores"]:
                        logger.info(f"  True Relevant: {score_info['job_id']}, GCN Score: {score_info['score']:.4f}")
                else:
                    logger.info(f"User {user_detail['user_id']} - No 'is_relevant_for_ranking_eval'==True jobs found in test set to score.")
            logger.info("-------------------------------------------------")
        
        valid_k_values = sorted(list(set(k for k in [5, top_k] if k > 0 and k <= top_k)))
        if not valid_k_values and top_k > 0: valid_k_values = [top_k]
        if not valid_k_values : valid_k_values = [5] # Default if top_k was 0 or less


        metrics = calculate_recommendation_metrics(
            recommendations=all_recommendations_for_ranking,
            ground_truth=all_ground_truth_for_ranking,
            actual_ratings=np.array(actual_ratings_for_mae_rmse) if actual_ratings_for_mae_rmse else np.array([]), # Pass empty array if no ratings
            predicted_ratings=np.array(predicted_ratings_for_mae_rmse) if predicted_ratings_for_mae_rmse else np.array([]),
            k_values=valid_k_values # Use cleaned up valid_k_values
        )
        metrics['actual_ratings_array_for_plot'] = np.array(actual_ratings_for_mae_rmse) if actual_ratings_for_mae_rmse else np.array([])
        metrics['predicted_ratings_array_for_plot'] = np.array(predicted_ratings_for_mae_rmse) if predicted_ratings_for_mae_rmse else np.array([])
        
        return metrics
        





        # ####### OLD RecSys CODE #####
        # """
        # Evaluate the recommendation model on test data.
        
        # Args:
        #     test_interactions: DataFrame with test interactions.
        #     user_col: Name of the user ID column.
        #     job_col: Name of the job ID column.
        #     rating_col: Name of the rating column.
        #     comment_col: Name of the comment column.
        #     top_k: Number of recommendations to generate.
        #     exclude_rated: Whether to exclude rated items from recommendations.
            
        # Returns:
        #     Dict[str, float]: Evaluation metrics.
        # """
        # if self.graph is None or self.graph_model is None:
        #     raise ValueError("Model not trained. Call train() first.")
        
        # # Get unique users in test set
        # test_users = test_interactions[user_col].unique()
        
        # # Filter to only include users in our mapping
        # test_users = [user for user in test_users if user in self.user_to_idx]
        
        # if len(test_users) == 0:
        #     raise ValueError("No valid test users found.")
        
        # # Group test interactions by user
        # user_rated_jobs = {}
        # user_test_jobs = {}
        
        # for user in test_users:
        #     user_interactions = test_interactions[test_interactions[user_col] == user]
        #     user_rated_jobs[user] = list(user_interactions[job_col])
            
        #     # For enhanced ratings, combine explicit ratings and sentiment
        #     if rating_col in user_interactions.columns or comment_col in user_interactions.columns:
        #         enhanced_ratings = self.calculate_batch_enhanced_ratings(
        #             user_interactions,
        #             rating_col=rating_col,
        #             comment_col=comment_col
        #         )
        #         # Get jobs with positive ratings (above threshold)
        #         threshold = 0.6  # Consider ratings above 0.6 as positive
        #         positive_indices = np.where(enhanced_ratings > threshold)[0]
        #         user_test_jobs[user] = [
        #             user_interactions.iloc[i][job_col] for i in positive_indices
        #         ]
        #     else:
        #         # If no ratings, use all jobs as positive
        #         user_test_jobs[user] = list(user_interactions[job_col])
        
        # # Generate recommendations for all test users
        # recommendations = self.recommend_batch(
        #     user_ids=test_users,
        #     top_k=top_k,
        #     exclude_rated=exclude_rated,
        #     user_rated_jobs=user_rated_jobs if exclude_rated else None
        # )
        
        # # Prepare data for metric calculation
        # all_recommendations = []
        # all_ground_truth = []
        
        # for user in test_users:
        #     if user in recommendations and user in user_test_jobs:
        #         user_recs = [rec['job_id'] for rec in recommendations[user]]
        #         all_recommendations.append(user_recs)
        #         all_ground_truth.append(user_test_jobs[user])
        
        # # Préparation pour MAE/RMSE (corrigé)
        # actual_ratings = []
        # predicted_ratings = []
        # for user in test_users:
        #     user_interactions = test_interactions[test_interactions[user_col] == user]
        #     if rating_col in user_interactions.columns:
        #         for _, row in user_interactions.iterrows():
        #             job_id = row[job_col]
        #             if job_id in self.job_to_idx:
        #                 user_idx = self.user_to_idx[user]
        #                 job_idx = self.job_to_idx[job_id]
        #                 pred = self.graph_model(
        #                     self.graph.to(self.device),
        #                     torch.tensor([user_idx], device=self.device),
        #                     torch.tensor([job_idx], device=self.device)
        #                 )
        #                 actual_ratings.append(row[rating_col])
        #                 predicted_ratings.append(float(pred.cpu().detach().numpy()[0]))
        # if len(actual_ratings) == 0 or len(predicted_ratings) == 0:
        #     actual_ratings = None
        #     predicted_ratings = None
        # else:
        #     actual_ratings = np.array(actual_ratings)
        #     predicted_ratings = np.array(predicted_ratings)
        # # Calcul des métriques
        # metrics = calculate_recommendation_metrics(
        #     recommendations=all_recommendations,
        #     ground_truth=all_ground_truth,
        #     actual_ratings=actual_ratings,
        #     predicted_ratings=predicted_ratings,
        #     k_values=[5, 10]
        # )
        # # Add actual and predicted ratings to metrics for downstream use (e.g., plotting)
        # metrics['actual_ratings'] = actual_ratings
        # metrics['predicted_ratings'] = predicted_ratings
        # return metrics
    

    
    # `get_job_embeddings` should now return the GCN's output embeddings for jobs
    # which were produced using BERT embeddings as initial features.
    ##### NEW RecSys CODE #####
    def get_job_embeddings(self) -> Dict[Any, np.ndarray]:
        """Get learned GCN embeddings for all jobs (initialized with BERT)."""
        if not isinstance(self.graph_model, HeterogeneousGCN) or self.graph is None:
            raise ValueError("HeteroGCN Model not trained or graph not available.")
        
        self.graph_model.eval()
        with torch.no_grad():
            # This gets the dict of node features AFTER GCN layers
            # The forward pass of HeteroGCN up to the last conv layer
            x_dict = {}
            current_x_dict_features = {} # Features to input to first conv
            for node_type in self.graph_model.node_types: # self.graph_model.node_types=['user', 'job']
                # Apply initial type-specific linear projection
                current_x_dict_features[node_type] = self.graph_model.node_embeddings[node_type](self.graph[node_type].x.to(self.device))

            # Apply graph convolutions
            for i, conv_layer in enumerate(self.graph_model.convs):
                current_x_dict_features = conv_layer(current_x_dict_features, self.graph.edge_index_dict) # Pass edge_index_dict
                if i < len(self.graph_model.convs) - 1: # Not the last layer
                    for node_type in current_x_dict_features:
                        current_x_dict_features[node_type] = F.relu(current_x_dict_features[node_type])
                        # Dropout is usually not applied at inference for embeddings, but depends on model design
                        # current_x_dict_features[node_type] = self.graph_model.dropout(current_x_dict_features[node_type])
            
            # Final embeddings are in current_x_dict_features
            job_embs_tensor = current_x_dict_features['job'].cpu().numpy()
            
        job_embeddings_map = {
            self.idx_to_job[idx]: job_embs_tensor[idx] 
            for idx in range(job_embs_tensor.shape[0]) if idx in self.idx_to_job
        }
        return job_embeddings_map



    # ##### OLD RecSys CODE #####
    # def get_job_embeddings(self) -> Dict[Any, np.ndarray]:
    #     """
    #     Get learned embeddings for all jobs.
        
    #     Returns:
    #         Dict[Any, np.ndarray]: Dictionary mapping job IDs to their embeddings.
    #     """
    #     if self.graph is None or self.graph_model is None:
    #         raise ValueError("Model not trained. Call train() first.")
        
    #     # Extract job embeddings from the model
    #     self.graph_model.eval()
    #     with torch.no_grad():
    #         # Apply graph convolutions to get final node embeddings
    #         x = torch.cat([
    #             self.graph_model.user_embedding.weight,
    #             self.graph_model.job_embedding.weight
    #         ], dim=0)
            
    #         edge_index = self.graph.edge_index.to(self.device)
            
    #         for i, conv in enumerate(self.graph_model.convs):
    #             x = conv(x, edge_index)
    #             if i < len(self.graph_model.convs) - 1:
    #                 x = F.relu(x)
            
    #         # Extract job embeddings
    #         job_embs = x[self.graph_model.num_users:].cpu().numpy()
        
    #     # Map job indices to IDs
    #     job_embeddings = {
    #         self.idx_to_job[idx]: job_embs[idx]
    #         for idx in range(len(self.idx_to_job))
    #     }
        
    #     return job_embeddings
    



    ##### NEW RecSys CODE #####
    
    # `find_similar_jobs` can now use the BERT embeddings directly as modified in previous step,
    # OR it could use the GCN-processed embeddings from get_job_embeddings()
    # For PURE content similarity, stick to direct BERT. For hybrid, use GCN output.
    # The version you tested was for direct BERT, which is good.

    # `save_model` and `load_model` will need updates for HeteroGCN specifics
    # (e.g., saving/loading node_feature_dims for HeteroGCN.load)

    def save_model(self, path: str):
        """Saves the JobRecommender system, including HeteroGCN if used."""
        # ... (save embedding_model, sentiment_model as before) ...
        os.makedirs(path, exist_ok=True)
        
        embedding_path = os.path.join(path, 'bert_embeddings')
        os.makedirs(embedding_path, exist_ok=True)
        self.embedding_model.save_cache(embedding_path)
        
        sentiment_path = os.path.join(path, 'sentiment_model') # Ensure consistent naming
        os.makedirs(sentiment_path, exist_ok=True)
        self.sentiment_model.save_model(sentiment_path) # Check if this is the correct method

        if isinstance(self.graph_model, HeterogeneousGCN):
            graph_model_path = os.path.join(path, 'hetero_graph_model')
            os.makedirs(graph_model_path, exist_ok=True)
            # HeteroGCN save method already saves its config.
            # We need to save node_feature_dims alongside it or as part of its metadata for reload.
            # Let's assume HeteroGCN.save includes what it needs or we add it to its metadata.
            node_feature_dims_for_save = {
                'user': self.all_user_initial_features.shape[1] if self.all_user_initial_features is not None else self.user_feature_dim,
                'job': self.all_job_bert_embeddings.shape[1] if self.all_job_bert_embeddings is not None else 0 # Or a default BERT dim
            }
            # The HeteroGCN save method itself might need its config updated to store node_feature_dims
            # or we save it in JobRecommender's main config.json
            self.graph_model.save(graph_model_path, metadata={'node_feature_dims': node_feature_dims_for_save})


        if self.graph is not None: # self.graph is HeteroData
            torch.save(self.graph, os.path.join(path, 'interaction_hetero_graph.pt'))
        
        # ... (save mappings as before) ...
        # Convert keys to strings for JSON serialization more carefully
        serializable_mappings = {
            'user_to_idx': {str(k): int(v) for k, v in self.user_to_idx.items()},
            'idx_to_user': {str(int(k)): str(v) for k, v in self.idx_to_user.items()}, # Ensure key is string int
            'job_to_idx': {str(k): int(v) for k, v in self.job_to_idx.items()},
            'idx_to_job': {str(int(k)): str(v) for k, v in self.idx_to_job.items()}  # Ensure key is string int
        }
        with open(os.path.join(path, 'mappings.json'), 'w') as f:
            json.dump(serializable_mappings, f, indent=4)

        config_recommender = {
            'user_feature_dim': self.user_feature_dim,
            'sentiment_weight': self.sentiment_weight,
            'rating_weight': self.rating_weight,
            'metadata': self.metadata,
            'graph_model_type': 'HeterogeneousGCN' if isinstance(self.graph_model, HeterogeneousGCN) else 'None'
            # Potentially save GCN params here if not saved by GCN model itself in a separate config
        }
        with open(os.path.join(path, 'config_recommender.json'), 'w') as f: # Use a distinct name
            json.dump(config_recommender, f, indent=4)

    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None):
        """Loads a JobRecommender system."""
        # ... (device setup, load BERTEmbeddings, SentimentAnalysis as before) ...
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(os.path.join(path, 'config_recommender.json'), 'r') as f:
            config_recommender = json.load(f)
        
        # ... (load mappings carefully converting keys back as in save) ...
        with open(os.path.join(path, 'mappings.json'), 'r') as f:
            loaded_serializable_mappings = json.load(f)
        mappings = {
            'user_to_idx': {k: int(v) for k, v in loaded_serializable_mappings['user_to_idx'].items()},
            'idx_to_user': {int(k_str): v for k_str, v in loaded_serializable_mappings['idx_to_user'].items()},
            'job_to_idx': {k: int(v) for k, v in loaded_serializable_mappings['job_to_idx'].items()},
            'idx_to_job': {int(k_str): v for k_str, v in loaded_serializable_mappings['idx_to_job'].items()}
        }


        recommender = cls(
            embedding_model=BERTEmbeddings.load_from_cache(os.path.join(path, 'bert_embeddings'), device=device),
            sentiment_model=SentimentAnalysis.load_model(os.path.join(path, 'sentiment_model'), device=device), # Ensure path matches save
            device=device,
            user_feature_dim=config_recommender.get('user_feature_dim', 64),
            sentiment_weight=config_recommender.get('sentiment_weight',0.5), # from config_recommender
            rating_weight=config_recommender.get('rating_weight',0.5)   # from config_recommender
        )
        recommender.user_to_idx = mappings['user_to_idx']
        # ... (set other mappings and metadata) ...
        recommender.idx_to_user = mappings['idx_to_user']
        recommender.job_to_idx = mappings['job_to_idx']
        recommender.idx_to_job = mappings['idx_to_job']
        recommender.metadata = config_recommender.get('metadata', {})

        graph_model_type = config_recommender.get('graph_model_type')
        if graph_model_type == 'HeterogeneousGCN':
            graph_model_path = os.path.join(path, 'hetero_graph_model')
            if os.path.exists(graph_model_path):
                # HeteroGCN.load needs node_feature_dims. This should be saved with HeteroGCN model or here.
                # Assuming HeteroGCN.load can get it from its own saved config or metadata.
                # Let's retrieve it from HeteroGCN's saved metadata if present
                with open(os.path.join(graph_model_path, 'config.json'), 'r') as f_gcn_cfg:
                    gcn_config_saved = json.load(f_gcn_cfg)
                node_feature_dims_loaded = gcn_config_saved.get('metadata', {}).get('node_feature_dims')
                if not node_feature_dims_loaded:
                     # Fallback if not in metadata: Reconstruct from data that SHOULD be there.
                     # This is less ideal, assumes data exists.
                     bert_emb_example_file = os.path.join(path, 'bert_embeddings', 'pytorch_model.bin') # Path depends on AutoModel save
                     # Logic to get BERT dim would be needed
                     job_dim_from_bert = 768 # Example for distilbert
                     node_feature_dims_loaded = {'user': recommender.user_feature_dim, 'job': job_dim_from_bert}
                     logger.warning(f"node_feature_dims not found in HeteroGCN saved metadata, reconstructed as: {node_feature_dims_loaded}")

                #modified
                recommender.graph_model, _ = HeterogeneousGCN.load(graph_model_path, device=device)
        
        graph_path = os.path.join(path, 'interaction_hetero_graph.pt')
        if os.path.exists(graph_path):
            # Add weights_only=False for PyTorch 2.0+ if HeteroData pickle contains non-tensor Python objects
            try:
                recommender.graph = torch.load(graph_path, map_location=device, weights_only=False) # Try with False
            except Exception as e_load:
                logger.error(f"Failed to load graph with weights_only=False: {e_load}. Trying default.")
                recommender.graph = torch.load(graph_path, map_location=device)


        # Re-establish self.all_job_bert_embeddings and self.all_user_initial_features if graph_model is loaded and graph exists
        # These are needed if `train` is not called immediately after loading.
        if recommender.graph_model and recommender.graph and 'user' in recommender.graph and 'job' in recommender.graph :
            recommender.all_user_initial_features = recommender.graph['user'].x.to(device)
            recommender.all_job_bert_embeddings = recommender.graph['job'].x.to(device)
            logger.info("Restored all_user_initial_features and all_job_bert_embeddings from loaded graph.")

        return recommender
    

    ##### OLD RecSys Code #####
    # def save_model(self, path: str):
    #     """
    #     Save the entire recommendation system.
        
    #     Args:
    #         path: Directory path to save the model.
    #     """
    #     os.makedirs(path, exist_ok=True)
        
    #     # Save BERT embedding model
    #     embedding_path = os.path.join(path, 'bert_embeddings')
    #     os.makedirs(embedding_path, exist_ok=True)
    #     self.embedding_model.save_cache(embedding_path)
        
    #     # Save sentiment analysis model
    #     sentiment_path = os.path.join(path, 'sentiment')
    #     os.makedirs(sentiment_path, exist_ok=True)
    #     self.sentiment_model.save_model(sentiment_path)
        
    #     # Save graph model if available
    #     if self.graph_model is not None:
    #         graph_path = os.path.join(path, 'graph_model')
    #         os.makedirs(graph_path, exist_ok=True)
    #         self.graph_model.save(graph_path)
            
    #         # Save the graph
    #         if self.graph is not None:
    #             torch.save(self.graph, os.path.join(path, 'interaction_graph.pt'))
        
    #     # Save mappings and metadata
    #     mappings = {
    #         'user_to_idx': self.user_to_idx,
    #         'idx_to_user': self.idx_to_user,
    #         'job_to_idx': self.job_to_idx,
    #         'idx_to_job': self.idx_to_job
    #     }
        
    #     with open(os.path.join(path, 'mappings.json'), 'w') as f:
    #         # Convert keys to strings for JSON serialization
    #         serializable_mappings = {
    #             'user_to_idx': {str(k): v for k, v in self.user_to_idx.items()},
    #             'idx_to_user': {str(k): str(v) for k, v in self.idx_to_user.items()},
    #                         'job_to_idx': {str(k): v for k, v in self.job_to_idx.items()},
    #             'idx_to_job': {str(k): str(v) for k, v in self.idx_to_job.items()}
    #         }
    #         json.dump(serializable_mappings, f)
        
    #     # Save configuration and metadata
    #     config = {
    #         'sentiment_weight': self.sentiment_weight,
    #         'rating_weight': self.rating_weight,        'metadata': self.metadata
    #     }
        
    #     with open(os.path.join(path, 'config.json'), 'w') as f:
    #         json.dump(config, f)
    # @classmethod
    # def load_model(cls, path: str, device: Optional[str] = None):
    #     """
    #     Load a saved recommendation system.
        
    #     Args:
    #         path: Directory path where the model is saved.
    #         device: Device to load the models to ('cpu' or 'cuda').
            
    #     Returns:
    #         JobRecommender: Loaded recommendation system.
    #     """
    #     # Safety check for CUDA availability
    #     if device == 'cuda' and not torch.cuda.is_available():
    #         logger.warning("CUDA requested but not available. Falling back to CPU.")
    #         device = 'cpu'
            
    #     # Determine device
    #     device = device if device else ('cpu')
        
    #     # Autoriser les classes torch_geometric lors du chargement
    #     import torch.serialization
    #     torch.serialization.add_safe_globals(['torch_geometric.data.data.DataEdgeAttr'])
        
    #     # Load configuration
    #     with open(os.path.join(path, 'config.json'), 'r') as f:
    #         config = json.load(f)
        
    #     # Extract weights and metadata
    #     sentiment_weight = config.get('sentiment_weight', 0.5)
    #     rating_weight = config.get('rating_weight', 0.5)
    #     metadata = config.get('metadata', {})
        
    #     # Load mappings
    #     with open(os.path.join(path, 'mappings.json'), 'r') as f:
    #         serialized_mappings = json.load(f)
            
    #     # Convert string keys back to original types
    #     mappings = {
    #         'user_to_idx': {k: int(v) for k, v in serialized_mappings['user_to_idx'].items()},
    #         'idx_to_user': {int(k): v for k, v in serialized_mappings['idx_to_user'].items()},
    #         'job_to_idx': {k: int(v) for k, v in serialized_mappings['job_to_idx'].items()},
    #         'idx_to_job': {int(k): v for k, v in serialized_mappings['idx_to_job'].items()}
    #     }

    #     # Load models
    #     embedding_model = BERTEmbeddings.load_from_cache(
    #         os.path.join(path, 'bert_embeddings'),
    #         device=device
    #     )
        
    #     sentiment_model = SentimentAnalysis.load_model(
    #         os.path.join(path, 'sentiment'),
    #         device=device
    #     )
        
    #     # Create recommender instance
    #     recommender = cls(
    #         embedding_model=embedding_model,
    #         sentiment_model=sentiment_model,
    #         device=device,
    #         sentiment_weight=sentiment_weight,
    #         rating_weight=rating_weight
    #     )
        
    #     # Set mappings
    #     recommender.user_to_idx = mappings['user_to_idx']
    #     recommender.idx_to_user = mappings['idx_to_user']
    #     recommender.job_to_idx = mappings['job_to_idx']
    #     recommender.idx_to_job = mappings['idx_to_job']
        
    #     # Load graph model if available
    #     graph_model_path = os.path.join(path, 'graph_model')
    #     if os.path.exists(graph_model_path):
    #         from src.models.gcn import GCNRecommender
    #         try:
    #             recommender.graph_model, _ = GCNRecommender.load(graph_model_path, device=device)
    #         except Exception as e:
    #             logger.warning(f"Erreur lors du chargement du modèle GCN: {e}")
    #             logger.warning("Tentative de chargement manuel avec weights_only=False...")
                
    #             # Chargement manuel avec weights_only=False
    #             gcn_path = os.path.join(graph_model_path, 'model.pt')
    #             if os.path.exists(gcn_path):
    #                 # Obtenir les paramètres du modèle depuis config.json
    #                 with open(os.path.join(graph_model_path, 'config.json'), 'r') as f:
    #                     model_config = json.load(f)
                    
    #                 # Créer une nouvelle instance
    #                 recommender.graph_model = GCNRecommender(
    #                     embedding_dim=model_config.get('embedding_dim', 64),
    #                     hidden_dim=model_config.get('hidden_dim', 32),
    #                     num_users=model_config.get('num_users', len(recommender.user_to_idx)),
    #                     num_jobs=model_config.get('num_jobs', len(recommender.job_to_idx)),
    #                     num_layers=model_config.get('num_layers', 2),
    #                     dropout=model_config.get('dropout', 0.1),
    #                     conv_type=model_config.get('conv_type', 'GCNConv')
    #                 )
                
    #                 # Charger les poids
    #                 recommender.graph_model.load_state_dict(
    #                     torch.load(gcn_path, map_location=device, weights_only=False)
    #                 )
    #                 recommender.graph_model.to(device)
    #                 recommender.graph_model.eval()
    #       # Load the graph if available
    #     graph_path = os.path.join(path, 'interaction_graph.pt')
    #     if os.path.exists(graph_path):
    #         try:
    #             # Essayer d'abord avec weights_only=True (par défaut depuis PyTorch 2.6)
    #             recommender.graph = torch.load(graph_path, map_location=device)
    #         except Exception as e:
    #             logger.warning(f"Erreur lors du chargement du graphe: {e}")
    #             logger.warning("Tentative avec weights_only=False...")
    #             recommender.graph = torch.load(graph_path, map_location=device, weights_only=False)
        
    #     # Set metadata
    #     recommender.metadata = metadata
        
    #     return recommender

    def set_category_master_and_user_profile_features(self, categories_master, user_profile_features):
        """
        Set the canonical category list and user profile features (multi-hot vectors for professionals).
        """
        self.categories_master = categories_master
        self.user_profile_features = user_profile_features