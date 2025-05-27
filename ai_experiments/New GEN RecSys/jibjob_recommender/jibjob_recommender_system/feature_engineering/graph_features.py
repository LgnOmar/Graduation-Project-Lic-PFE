"""
Graph features module for preparing node and edge features for graph neural networks.
"""

import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import OneHotEncoder
from ..utils.helpers import cosine_similarity, batch_cosine_similarity

logger = logging.getLogger(__name__)

class GraphFeatures:
    """
    Class responsible for preparing node and edge features for graph neural networks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the GraphFeatures with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.graph_config = config['graph']
        self.model_config = config['model']
        
        # Content similarity threshold for creating similarity edges
        self.content_similarity_threshold = self.graph_config.get('content_similarity_threshold', 0.7)
        
        # Embedding dimension for node features
        self.embedding_dim = self.model_config.get('embedding_dim', 128)
        
    def prepare_professional_node_features(
        self,
        users_df: pd.DataFrame,
        professional_categories_df: pd.DataFrame,
        categories_df: pd.DataFrame,
        professional_embeddings: Dict[str, np.ndarray],
        location_features: np.ndarray = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare node features for professional users.
        
        Args:
            users_df: DataFrame containing user data.
            professional_categories_df: DataFrame containing professional category mappings.
            categories_df: DataFrame containing category data.
            professional_embeddings: Dictionary mapping user_id to text embedding.
            location_features: Optional array of location features.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping user_id to node features.
        """
        # Filter for professional users only
        professionals = users_df[users_df['user_type'] == 'professional']
        professional_ids = professionals['user_id'].tolist()
        
        # Create category one-hot encoding
        category_encoder = OneHotEncoder(sparse_output=False)
        category_encoder.fit(categories_df['category_id'].values.reshape(-1, 1))
        
        # Initialize features dictionary
        professional_features = {}
        
        # For each professional, combine:
        # 1. Text embeddings from profile_bio
        # 2. One-hot encoded categories
        # 3. Location features (if available)
        
        logger.info(f"Preparing node features for {len(professional_ids)} professionals")
        
        for idx, user_id in enumerate(professional_ids):
            # Get text embedding
            text_embedding = professional_embeddings.get(user_id, 
                                                      np.zeros(list(professional_embeddings.values())[0].shape[0] 
                                                              if professional_embeddings else 768))
            
            # Get categories for this professional
            user_categories = professional_categories_df[professional_categories_df['user_id'] == user_id]['category_id'].tolist()
            
            # One-hot encode categories
            if user_categories:
                categories_array = category_encoder.transform(np.array(user_categories).reshape(-1, 1))
                # Sum in case of multiple categories
                category_features = categories_array.sum(axis=0)
            else:
                category_features = np.zeros(len(category_encoder.categories_[0]))
                
            # Get location features if available
            loc_features = location_features[idx] if location_features is not None else np.array([0, 0])
            
            # Combine features
            # For dimensionality consistency, we might need to project these to the same dimensionality
            # or just concatenate and let the GNN handle the rest
            combined_features = np.concatenate([
                text_embedding,
                category_features,
                loc_features
            ])
            
            # Convert to tensor
            professional_features[user_id] = torch.FloatTensor(combined_features)
            
        logger.info(f"Prepared node features for {len(professional_features)} professionals")
        return professional_features
        
    def prepare_job_node_features(
        self,
        jobs_df: pd.DataFrame,
        categories_df: pd.DataFrame,
        job_embeddings: Dict[str, np.ndarray],
        location_features: np.ndarray = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare node features for jobs.
        
        Args:
            jobs_df: DataFrame containing job data.
            categories_df: DataFrame containing category data.
            job_embeddings: Dictionary mapping job_id to text embedding.
            location_features: Optional array of location features.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping job_id to node features.
        """
        job_ids = jobs_df['job_id'].tolist()
        
        # Create category one-hot encoding
        category_encoder = OneHotEncoder(sparse_output=False)
        category_encoder.fit(categories_df['category_id'].values.reshape(-1, 1))
        
        # Initialize features dictionary
        job_features = {}
        
        # For each job, combine:
        # 1. Text embeddings from title/description
        # 2. One-hot encoded category
        # 3. Location features (if available)
        
        logger.info(f"Preparing node features for {len(job_ids)} jobs")
        
        for idx, job_id in enumerate(job_ids):
            # Get text embedding
            text_embedding = job_embeddings.get(job_id, 
                                             np.zeros(list(job_embeddings.values())[0].shape[0] 
                                                     if job_embeddings else 768))
            
            # Get category for this job
            job_row = jobs_df[jobs_df['job_id'] == job_id]
            
            # One-hot encode category if available
            if 'required_category_id' in job_row.columns and not job_row['required_category_id'].isna().all():
                category_id = job_row['required_category_id'].iloc[0]
                if pd.notna(category_id):
                    try:
                        category_features = category_encoder.transform(np.array([category_id]).reshape(-1, 1))[0]
                    except:
                        # Category not in encoder categories
                        category_features = np.zeros(len(category_encoder.categories_[0]))
                else:
                    category_features = np.zeros(len(category_encoder.categories_[0]))
            else:
                category_features = np.zeros(len(category_encoder.categories_[0]))
                
            # Get location features if available
            loc_features = location_features[idx] if location_features is not None else np.array([0, 0])
            
            # Combine features
            combined_features = np.concatenate([
                text_embedding,
                category_features,
                loc_features
            ])
            
            # Convert to tensor
            job_features[job_id] = torch.FloatTensor(combined_features)
            
        logger.info(f"Prepared node features for {len(job_features)} jobs")
        return job_features
        
    def prepare_category_node_features(
        self,
        categories_df: pd.DataFrame,
        category_embeddings: Dict[str, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare node features for categories.
        
        Args:
            categories_df: DataFrame containing category data.
            category_embeddings: Dictionary mapping category_id to text embedding.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping category_id to node features.
        """
        category_ids = categories_df['category_id'].tolist()
        
        # Create one-hot encoding for categories
        category_encoder = OneHotEncoder(sparse_output=False)
        category_encoder.fit(categories_df['category_id'].values.reshape(-1, 1))
        
        # Initialize features dictionary
        category_features = {}
        
        logger.info(f"Preparing node features for {len(category_ids)} categories")
        
        for idx, category_id in enumerate(category_ids):
            # Get text embedding
            text_embedding = category_embeddings.get(category_id, 
                                                  np.zeros(list(category_embeddings.values())[0].shape[0] 
                                                          if category_embeddings else 768))
            
            # One-hot encode the category itself
            onehot_features = category_encoder.transform(np.array([category_id]).reshape(-1, 1))[0]
            
            # Combine features
            combined_features = np.concatenate([
                text_embedding,
                onehot_features
            ])
            
            # Convert to tensor
            category_features[category_id] = torch.FloatTensor(combined_features)
            
        logger.info(f"Prepared node features for {len(category_features)} categories")
        return category_features
        
    def prepare_location_node_features(
        self,
        locations_df: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare node features for locations.
        
        Args:
            locations_df: DataFrame containing location data.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary mapping location_id to node features.
        """
        location_ids = locations_df['location_id'].tolist()
        
        # Initialize features dictionary
        location_features = {}
        
        logger.info(f"Preparing node features for {len(location_ids)} locations")
        
        for idx, location_id in enumerate(location_ids):
            location_row = locations_df[locations_df['location_id'] == location_id]
            
            # Extract coordinates
            lat = location_row['latitude'].iloc[0] if not pd.isna(location_row['latitude'].iloc[0]) else 0
            lon = location_row['longitude'].iloc[0] if not pd.isna(location_row['longitude'].iloc[0]) else 0
            
            # Normalize coordinates
            norm_lat = lat / 90.0  # Maps to [-1, 1]
            norm_lon = lon / 180.0  # Maps to [-1, 1]
            
            # Create feature vector
            features = np.array([norm_lat, norm_lon])
            
            # Convert to tensor
            location_features[location_id] = torch.FloatTensor(features)
            
        logger.info(f"Prepared node features for {len(location_features)} locations")
        return location_features
        
    def find_content_similarity_edges(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        threshold: float = None
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of items with high content similarity for creating similarity edges.
        
        Args:
            embeddings_dict: Dictionary mapping item_id to embedding vector.
            threshold: Similarity threshold (if None, use the default).
            
        Returns:
            List[Tuple[str, str, float]]: List of (item1_id, item2_id, similarity) tuples.
        """
        if threshold is None:
            threshold = self.content_similarity_threshold
            
        item_ids = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[item_id] for item_id in item_ids])
        
        # Initialize list to store edges
        similarity_edges = []
        
        # Calculate similarities between all pairs
        for i, item1_id in enumerate(item_ids):
            # Get embedding for item1
            embedding1 = embeddings[i]
            
            # Calculate similarities to all other items
            similarities = batch_cosine_similarity(embedding1, embeddings)
            
            # Filter by threshold and avoid self-loops
            for j, (similarity, item2_id) in enumerate(zip(similarities, item_ids)):
                if i != j and similarity >= threshold:
                    similarity_edges.append((item1_id, item2_id, float(similarity)))
                    
        logger.info(f"Found {len(similarity_edges)} content similarity edges with threshold {threshold}")
        return similarity_edges
        
    def compute_similarity_matrix(
        self,
        professional_features: Dict[str, torch.Tensor],
        job_features: Dict[str, torch.Tensor]
    ) -> pd.DataFrame:
        """
        Compute a similarity matrix between professionals and jobs.
        
        Args:
            professional_features: Dictionary mapping user_id to professional node features.
            job_features: Dictionary mapping job_id to job node features.
            
        Returns:
            pd.DataFrame: DataFrame with professional_id, job_id, and similarity score.
        """
        professional_ids = list(professional_features.keys())
        job_ids = list(job_features.keys())
        
        # Convert features to numpy arrays
        professional_embeddings = np.array([professional_features[p_id].numpy() for p_id in professional_ids])
        job_embeddings = np.array([job_features[j_id].numpy() for j_id in job_ids])
        
        # Initialize list to store similarities
        similarities = []
        
        # Calculate similarities between all professional-job pairs
        for i, p_id in enumerate(professional_ids):
            # Get embedding for professional
            p_embedding = professional_embeddings[i]
            
            # Calculate similarities to all jobs
            job_similarities = batch_cosine_similarity(p_embedding, job_embeddings)
            
            # Add to list
            for j, (similarity, j_id) in enumerate(zip(job_similarities, job_ids)):
                similarities.append({
                    'user_id': p_id,
                    'job_id': j_id,
                    'similarity': float(similarity)
                })
                
        # Convert to DataFrame
        similarities_df = pd.DataFrame(similarities)
        
        logger.info(f"Computed {len(similarities_df)} professional-job similarity scores")
        return similarities_df
        
    def compute_distance_weighted_edges(
        self,
        distances_df: pd.DataFrame,
        max_distance_km: float = 100.0,
        weight_function: str = 'inverse'
    ) -> pd.DataFrame:
        """
        Compute edge weights based on distance between nodes.
        
        Args:
            distances_df: DataFrame with distance information.
            max_distance_km: Maximum distance to consider (beyond this, weight is 0).
            weight_function: How to convert distance to weight ('inverse' or 'exp_decay').
            
        Returns:
            pd.DataFrame: DataFrame with source, target, and weight columns.
        """
        # Filter by maximum distance
        filtered_df = distances_df[distances_df['distance_km'] <= max_distance_km].copy()
        
        # Convert distance to weight
        if weight_function == 'inverse':
            # Inverse distance (closer = higher weight)
            filtered_df['weight'] = 1.0 / (filtered_df['distance_km'] + 1.0)
        elif weight_function == 'exp_decay':
            # Exponential decay (closer = higher weight)
            # Scale factor: smaller values make the weight decay faster with distance
            scale_factor = 10.0
            filtered_df['weight'] = np.exp(-filtered_df['distance_km'] / scale_factor)
        else:
            # Default: inverse distance
            filtered_df['weight'] = 1.0 / (filtered_df['distance_km'] + 1.0)
            
        # Normalize weights to [0, 1]
        max_weight = filtered_df['weight'].max()
        if max_weight > 0:
            filtered_df['weight'] = filtered_df['weight'] / max_weight
            
        logger.info(f"Computed {len(filtered_df)} distance-weighted edges with {weight_function} weighting")
        return filtered_df
        
    def combine_similarity_and_distance(
        self,
        similarity_df: pd.DataFrame,
        distance_df: pd.DataFrame,
        similarity_weight: float = 0.7,
        distance_weight: float = 0.3
    ) -> pd.DataFrame:
        """
        Combine content similarity and distance weights into a single edge weight.
        
        Args:
            similarity_df: DataFrame with content similarity weights.
            distance_df: DataFrame with distance-based weights.
            similarity_weight: Weight factor for content similarity.
            distance_weight: Weight factor for distance.
            
        Returns:
            pd.DataFrame: DataFrame with combined weights.
        """
        # Merge the two DataFrames
        merged_df = similarity_df.merge(
            distance_df[['user_id', 'job_id', 'weight']],
            on=['user_id', 'job_id'],
            how='inner',
            suffixes=('_similarity', '_distance')
        )
        
        # Combine weights
        merged_df['combined_weight'] = (
            similarity_weight * merged_df['similarity'] + 
            distance_weight * merged_df['weight_distance']
        )
        
        logger.info(f"Combined {len(merged_df)} similarity and distance weights")
        return merged_df
        
    def compute_category_edge_weights(
        self,
        professional_categories_df: pd.DataFrame,
        jobs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute edge weights based on category matches between professionals and jobs.
        
        Args:
            professional_categories_df: DataFrame with professional-category mappings.
            jobs_df: DataFrame with job information including categories.
            
        Returns:
            pd.DataFrame: DataFrame with professional_id, job_id, and weight columns.
        """
        # Get unique professional and job IDs
        professional_ids = professional_categories_df['user_id'].unique()
        job_ids = jobs_df['job_id'].unique()
        
        # Initialize list to store matches
        category_matches = []
        
        # For each professional-job pair, check if there's a category match
        for p_id in professional_ids:
            # Get categories for this professional
            user_categories = professional_categories_df[
                professional_categories_df['user_id'] == p_id
            ]['category_id'].tolist()
            
            for j_id in job_ids:
                # Get category for this job
                job_row = jobs_df[jobs_df['job_id'] == j_id]
                
                if 'required_category_id' in job_row.columns and not job_row['required_category_id'].isna().all():
                    job_category = job_row['required_category_id'].iloc[0]
                    
                    # Check if there's a match
                    if pd.notna(job_category) and job_category in user_categories:
                        category_matches.append({
                            'user_id': p_id,
                            'job_id': j_id,
                            'category_match': 1.0  # Full match
                        })
                    else:
                        # No direct match, but could add partial matching logic here
                        category_matches.append({
                            'user_id': p_id,
                            'job_id': j_id,
                            'category_match': 0.0  # No match
                        })
        
        # Convert to DataFrame
        matches_df = pd.DataFrame(category_matches)
        
        logger.info(f"Computed {len(matches_df)} category-based edge weights")
        return matches_df
        
    def create_edge_features(
        self,
        similarities_df: pd.DataFrame,
        distances_df: pd.DataFrame,
        category_matches_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive edge features by combining multiple sources.
        
        Args:
            similarities_df: DataFrame with content similarity information.
            distances_df: DataFrame with distance information.
            category_matches_df: DataFrame with category match information.
            
        Returns:
            pd.DataFrame: DataFrame with comprehensive edge features.
        """
        # Merge all features
        edge_features = similarities_df.merge(
            distances_df[['user_id', 'job_id', 'distance_km']],
            on=['user_id', 'job_id'],
            how='left'
        )
        
        edge_features = edge_features.merge(
            category_matches_df[['user_id', 'job_id', 'category_match']],
            on=['user_id', 'job_id'],
            how='left'
        )
        
        # Fill NaN values
        edge_features['distance_km'] = edge_features['distance_km'].fillna(-1)  # Missing distance
        edge_features['category_match'] = edge_features['category_match'].fillna(0)  # No category match
        
        # Create combined score (example: weighted average)
        edge_features['combined_score'] = (
            0.5 * edge_features['similarity'] + 
            0.3 * (1 / (edge_features['distance_km'] + 1)) + 
            0.2 * edge_features['category_match']
        )
        
        logger.info(f"Created {len(edge_features)} edge features")
        return edge_features
        
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize feature vectors to have unit norm.
        
        Args:
            features: Feature array with shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Normalized features.
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms
