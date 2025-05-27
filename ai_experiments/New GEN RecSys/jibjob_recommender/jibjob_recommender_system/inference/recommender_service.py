"""
Recommender service module for providing job recommendations.
"""

import logging
import os
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from ..models.gcn_recommender import GCNRecommender, HeteroGCNRecommender
from ..feature_engineering.text_embedder import TextEmbedder
from ..feature_engineering.location_features import LocationFeatures
from ..utils.helpers import load_object

logger = logging.getLogger(__name__)

class RecommenderService:
    """
    Service class for providing job recommendations to professionals.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RecommenderService with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.model_config = config['model']
        
        # Directory paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.models_dir = os.path.join(self.base_dir, 'saved_models')
        self.features_dir = os.path.join(self.base_dir, 'saved_features')
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model selection
        self.use_heterogeneous = self.model_config.get('use_heterogeneous', True)
        
        # Initialize model (will be loaded later)
        self.model = None
        
        # Initialize auxiliary components
        self.text_embedder = TextEmbedder(config)
        self.location_features = LocationFeatures(config)
        
        # Mappings (will be loaded as part of the model)
        self.professional_id_to_idx = {}
        self.job_id_to_idx = {}
        self.professional_idx_to_id = {}
        self.job_idx_to_id = {}
        
        # Job metadata (will be loaded from features)
        self.job_metadata = None
        self.location_lookup = None
        
    def load_model(self, model_filename: str = None) -> bool:
        """
        Load a trained recommendation model.
        
        Args:
            model_filename: Name of the model file to load (if None, use the default).
            
        Returns:
            bool: True if successful, False otherwise.
        """
        # If no filename provided, use default
        if model_filename is None:
            if self.use_heterogeneous:
                model_filename = 'heterogcn_model.pt'
            else:
                model_filename = 'gcn_model.pt'
                
        model_path = os.path.join(self.models_dir, model_filename)
        
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            # Load model state dict and metadata
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Create model based on the checkpoint
            if self.use_heterogeneous:
                self.model = HeteroGCNRecommender(self.config)
            else:
                self.model = GCNRecommender(self.config)
                
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load mappings
            self.professional_id_to_idx = checkpoint.get('professional_id_to_idx', {})
            self.job_id_to_idx = checkpoint.get('job_id_to_idx', {})
            self.professional_idx_to_id = checkpoint.get('professional_idx_to_id', {})
            self.job_idx_to_id = checkpoint.get('job_idx_to_id', {})
            
            # Load additional metadata
            self._load_metadata()
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
            
    def _load_metadata(self) -> None:
        """
        Load job metadata and other necessary data.
        """
        try:
            # Load job metadata
            jobs_with_loc_path = os.path.join(self.features_dir, 'jobs_with_loc.pkl')
            if os.path.exists(jobs_with_loc_path):
                self.job_metadata = load_object(jobs_with_loc_path)
                logger.info("Job metadata loaded successfully")
                
            # Load location lookup
            location_lookup_path = os.path.join(self.features_dir, 'location_lookup.pkl')
            if os.path.exists(location_lookup_path):
                self.location_lookup = load_object(location_lookup_path)
                logger.info("Location lookup loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            
    def recommend_jobs(
        self,
        user_id: str,
        top_k: int = 10,
        filter_by_location: bool = True,
        max_distance_km: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Get job recommendations for a specific professional user.
        
        Args:
            user_id: ID of the professional user.
            top_k: Number of recommendations to return.
            filter_by_location: Whether to filter recommendations by distance.
            max_distance_km: Maximum distance for filtering (if filter_by_location is True).
            
        Returns:
            List[Dict[str, Any]]: List of recommended job dictionaries with scores.
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []
            
        if user_id not in self.professional_id_to_idx:
            logger.warning(f"User ID {user_id} not found in model's known users")
            return self._get_fallback_recommendations(user_id, top_k)
            
        try:
            # Get user's node index in the graph
            user_idx = self.professional_id_to_idx[user_id]
            
            # Get model predictions
            with torch.no_grad():
                predictions = self.model.predict_for_user(user_idx)
                
            # Convert predictions to numpy
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.cpu().numpy()
                
            # Map job indices to IDs and create score dictionary
            job_scores = {}
            for job_idx, score in enumerate(predictions):
                if job_idx in self.job_idx_to_id:
                    job_id = self.job_idx_to_id[job_idx]
                    job_scores[job_id] = float(score)
                    
            # Filter by location if needed
            if filter_by_location and self.job_metadata is not None:
                job_scores = self._filter_by_distance(user_id, job_scores, max_distance_km)
                
            # Sort jobs by score and get top-k
            sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Create result list with job metadata
            recommendations = []
            for job_id, score in sorted_jobs:
                job_info = self._get_job_info(job_id)
                job_info['score'] = score
                recommendations.append(job_info)
                
            logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
            return self._get_fallback_recommendations(user_id, top_k)
            
    def recommend_jobs_batch(
        self,
        user_ids: List[str],
        top_k: int = 10,
        filter_by_location: bool = True,
        max_distance_km: float = 50.0
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get job recommendations for multiple professional users.
        
        Args:
            user_ids: List of professional user IDs.
            top_k: Number of recommendations to return per user.
            filter_by_location: Whether to filter recommendations by distance.
            max_distance_km: Maximum distance for filtering (if filter_by_location is True).
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping user_id to list of recommended jobs.
        """
        results = {}
        
        for user_id in user_ids:
            recommendations = self.recommend_jobs(
                user_id, top_k, filter_by_location, max_distance_km)
            results[user_id] = recommendations
            
        return results
        
    def recommend_for_new_user(
        self,
        profile_text: str,
        categories: List[str],
        location: Dict[str, Any] = None,
        top_k: int = 10,
        filter_by_location: bool = True,
        max_distance_km: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Get job recommendations for a new user not present in the training data.
        
        Args:
            profile_text: Text description of the user's profile.
            categories: List of category IDs the user is interested in.
            location: Dictionary with 'latitude' and 'longitude' keys.
            top_k: Number of recommendations to return.
            filter_by_location: Whether to filter recommendations by distance.
            max_distance_km: Maximum distance for filtering (if filter_by_location is True).
            
        Returns:
            List[Dict[str, Any]]: List of recommended job dictionaries with scores.
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return []
            
        try:
            # Generate embeddings for the profile text
            self.text_embedder.load_model()
            profile_embedding = self.text_embedder.generate_embedding(profile_text)
            
            # Get job embeddings (for similarity calculation)
            job_embeddings_path = os.path.join(self.features_dir, 'job_embeddings.pkl')
            job_embeddings = load_object(job_embeddings_path)
            
            if job_embeddings is None:
                logger.error("Could not load job embeddings")
                return []
                
            # Calculate similarity between profile and jobs
            job_scores = {}
            for job_id, job_embedding in job_embeddings.items():
                similarity = np.dot(profile_embedding, job_embedding) / (
                    np.linalg.norm(profile_embedding) * np.linalg.norm(job_embedding))
                job_scores[job_id] = float(similarity)
                
            # Filter by categories
            job_scores = self._filter_by_categories(job_scores, categories)
            
            # Filter by location if needed
            if filter_by_location and location is not None and self.job_metadata is not None:
                job_scores = self._filter_by_coordinates(
                    location.get('latitude'), location.get('longitude'),
                    job_scores, max_distance_km)
                
            # Sort jobs by score and get top-k
            sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Create result list with job metadata
            recommendations = []
            for job_id, score in sorted_jobs:
                job_info = self._get_job_info(job_id)
                job_info['score'] = score
                recommendations.append(job_info)
                
            logger.info(f"Generated {len(recommendations)} recommendations for new user")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations for new user: {str(e)}")
            return []
            
    def _get_job_info(self, job_id: str) -> Dict[str, Any]:
        """
        Get job information from metadata.
        
        Args:
            job_id: ID of the job.
            
        Returns:
            Dict[str, Any]: Dictionary with job information.
        """
        if self.job_metadata is not None and job_id in self.job_metadata['job_id'].values:
            job_row = self.job_metadata[self.job_metadata['job_id'] == job_id].iloc[0]
            
            # Create job info dictionary
            job_info = {
                'job_id': job_id,
                'title': job_row.get('title', 'Unknown Title'),
                'description': job_row.get('description', ''),
                'category_id': job_row.get('required_category_id', None),
                'location_id': job_row.get('location_id', None),
                'latitude': job_row.get('latitude', None),
                'longitude': job_row.get('longitude', None)
            }
            
            return job_info
        else:
            return {'job_id': job_id}
            
    def _filter_by_distance(
        self,
        user_id: str,
        job_scores: Dict[str, float],
        max_distance_km: float
    ) -> Dict[str, float]:
        """
        Filter job scores by distance from user's location.
        
        Args:
            user_id: ID of the professional user.
            job_scores: Dictionary mapping job_id to score.
            max_distance_km: Maximum distance in kilometers.
            
        Returns:
            Dict[str, float]: Filtered job scores.
        """
        if self.job_metadata is None:
            return job_scores
            
        # Find user's location
        user_location = None
        if user_id in self.job_metadata['user_id'].values:
            user_row = self.job_metadata[self.job_metadata['user_id'] == user_id].iloc[0]
            if 'latitude' in user_row and 'longitude' in user_row:
                user_location = {
                    'latitude': user_row['latitude'],
                    'longitude': user_row['longitude']
                }
                
        if user_location is None:
            logger.warning(f"Location not found for user {user_id}, skipping distance filtering")
            return job_scores
            
        # Filter jobs by distance
        filtered_scores = {}
        for job_id, score in job_scores.items():
            if job_id in self.job_metadata['job_id'].values:
                job_row = self.job_metadata[self.job_metadata['job_id'] == job_id].iloc[0]
                
                if 'latitude' in job_row and 'longitude' in job_row:
                    # Calculate distance between user and job
                    distance = self.location_features.calculate_distance(
                        user_location['latitude'], user_location['longitude'],
                        job_row['latitude'], job_row['longitude']
                    )
                    
                    # Include job if within max distance
                    if distance <= max_distance_km:
                        filtered_scores[job_id] = score
                        
        return filtered_scores
        
    def _filter_by_coordinates(
        self,
        latitude: float,
        longitude: float,
        job_scores: Dict[str, float],
        max_distance_km: float
    ) -> Dict[str, float]:
        """
        Filter job scores by distance from given coordinates.
        
        Args:
            latitude: Latitude coordinate.
            longitude: Longitude coordinate.
            job_scores: Dictionary mapping job_id to score.
            max_distance_km: Maximum distance in kilometers.
            
        Returns:
            Dict[str, float]: Filtered job scores.
        """
        if self.job_metadata is None or latitude is None or longitude is None:
            return job_scores
            
        # Filter jobs by distance
        filtered_scores = {}
        for job_id, score in job_scores.items():
            if job_id in self.job_metadata['job_id'].values:
                job_row = self.job_metadata[self.job_metadata['job_id'] == job_id].iloc[0]
                
                if 'latitude' in job_row and 'longitude' in job_row:
                    # Calculate distance between coordinates and job
                    distance = self.location_features.calculate_distance(
                        latitude, longitude,
                        job_row['latitude'], job_row['longitude']
                    )
                    
                    # Include job if within max distance
                    if distance <= max_distance_km:
                        filtered_scores[job_id] = score
                        
        return filtered_scores
        
    def _filter_by_categories(
        self,
        job_scores: Dict[str, float],
        categories: List[str]
    ) -> Dict[str, float]:
        """
        Filter and boost job scores by categories.
        
        Args:
            job_scores: Dictionary mapping job_id to score.
            categories: List of category IDs the user is interested in.
            
        Returns:
            Dict[str, float]: Filtered and boosted job scores.
        """
        if not categories or self.job_metadata is None:
            return job_scores
            
        # Filter and boost jobs by category
        adjusted_scores = {}
        for job_id, score in job_scores.items():
            if job_id in self.job_metadata['job_id'].values:
                job_row = self.job_metadata[self.job_metadata['job_id'] == job_id].iloc[0]
                
                if 'required_category_id' in job_row and job_row['required_category_id'] in categories:
                    # Boost score for category match
                    adjusted_scores[job_id] = score * 1.5  # 50% boost
                else:
                    adjusted_scores[job_id] = score
                    
        return adjusted_scores
        
    def _get_fallback_recommendations(
        self,
        user_id: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get fallback recommendations when user is not found or an error occurs.
        
        Args:
            user_id: ID of the professional user.
            top_k: Number of recommendations to return.
            
        Returns:
            List[Dict[str, Any]]: List of fallback job recommendations.
        """
        logger.info(f"Using fallback recommendations for user {user_id}")
        
        # If job metadata is available, recommend most recent jobs
        if self.job_metadata is not None:
            # Sort by date if available, otherwise just take first ones
            if 'created_at' in self.job_metadata.columns:
                sorted_jobs = self.job_metadata.sort_values('created_at', ascending=False)
            else:
                sorted_jobs = self.job_metadata
                
            # Take top_k jobs
            fallback_jobs = sorted_jobs.head(top_k)
            
            # Format recommendations
            recommendations = []
            for _, job_row in fallback_jobs.iterrows():
                job_id = job_row['job_id']
                job_info = self._get_job_info(job_id)
                job_info['score'] = 0.5  # Default score
                recommendations.append(job_info)
                
            return recommendations
                
        return []
