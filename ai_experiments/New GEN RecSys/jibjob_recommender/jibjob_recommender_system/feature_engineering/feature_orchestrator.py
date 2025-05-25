"""
Feature orchestrator module for coordinating feature engineering components.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

from .text_embedder import TextEmbedder
from .location_features import LocationFeatures
from .graph_features import GraphFeatures
from ..utils.helpers import save_object, load_object

logger = logging.getLogger(__name__)

class FeatureOrchestrator:
    """
    Class responsible for coordinating feature engineering components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the FeatureOrchestrator with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        
        # Initialize feature processors
        self.text_embedder = TextEmbedder(config)
        self.location_features = LocationFeatures(config)
        self.graph_features = GraphFeatures(config)
        
        # Create a directory for saving features
        self.features_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_features')
        os.makedirs(self.features_dir, exist_ok=True)
        
    def process_and_save_all_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Process all features and save them to disk.
        
        Args:
            data_dict: Dictionary of DataFrames.
            
        Returns:
            Dict[str, Any]: Dictionary containing all processed features.
        """
        logger.info("Starting feature processing pipeline")
        
        # Process locations
        locations_df = self.location_features.process_locations(data_dict['locations'])
        location_lookup = self.location_features.create_location_lookup(locations_df)
        
        # Add location data to users and jobs
        users_with_loc = self.location_features.add_location_data_to_users(
            data_dict['users'], location_lookup)
        jobs_with_loc = self.location_features.add_location_data_to_jobs(
            data_dict['jobs'], location_lookup)
            
        # Extract location features
        user_loc_features = self.location_features.get_location_features(users_with_loc)
        job_loc_features = self.location_features.get_location_features(jobs_with_loc)
        
        # Calculate distances between professionals and jobs
        distances_df = self.location_features.calculate_user_job_distances(users_with_loc, jobs_with_loc)
        
        # Save location features
        save_object(location_lookup, os.path.join(self.features_dir, 'location_lookup.pkl'))
        save_object(users_with_loc, os.path.join(self.features_dir, 'users_with_loc.pkl'))
        save_object(jobs_with_loc, os.path.join(self.features_dir, 'jobs_with_loc.pkl'))
        save_object(distances_df, os.path.join(self.features_dir, 'distances_df.pkl'))
        
        logger.info("Location feature processing complete")
        
        # Load text embedder model
        self.text_embedder.load_model()
        
        # Generate text embeddings
        professional_embeddings = self.text_embedder.generate_professional_embeddings(users_with_loc)
        job_embeddings = self.text_embedder.generate_job_embeddings(jobs_with_loc)
        category_embeddings = self.text_embedder.generate_category_embeddings(data_dict['categories'])
        
        # Save text embeddings
        save_object(professional_embeddings, os.path.join(self.features_dir, 'professional_embeddings.pkl'))
        save_object(job_embeddings, os.path.join(self.features_dir, 'job_embeddings.pkl'))
        save_object(category_embeddings, os.path.join(self.features_dir, 'category_embeddings.pkl'))
        
        logger.info("Text embedding generation complete")
        
        # Generate node features
        professional_features = self.graph_features.prepare_professional_node_features(
            users_with_loc,
            data_dict['professional_categories'],
            data_dict['categories'],
            professional_embeddings,
            user_loc_features
        )
        
        job_features = self.graph_features.prepare_job_node_features(
            jobs_with_loc,
            data_dict['categories'],
            job_embeddings,
            job_loc_features
        )
        
        category_features = self.graph_features.prepare_category_node_features(
            data_dict['categories'],
            category_embeddings
        )
        
        location_features = self.graph_features.prepare_location_node_features(
            locations_df
        )
        
        # Generate content similarity edges
        professional_similarity_edges = self.graph_features.find_content_similarity_edges(professional_embeddings)
        job_similarity_edges = self.graph_features.find_content_similarity_edges(job_embeddings)
        
        # Calculate professional-job similarity scores
        similarity_df = self.graph_features.compute_similarity_matrix(professional_features, job_features)
        
        # Save node features and edges
        save_object(professional_features, os.path.join(self.features_dir, 'professional_features.pkl'))
        save_object(job_features, os.path.join(self.features_dir, 'job_features.pkl'))
        save_object(category_features, os.path.join(self.features_dir, 'category_features.pkl'))
        save_object(location_features, os.path.join(self.features_dir, 'location_node_features.pkl'))
        save_object(professional_similarity_edges, os.path.join(self.features_dir, 'professional_similarity_edges.pkl'))
        save_object(job_similarity_edges, os.path.join(self.features_dir, 'job_similarity_edges.pkl'))
        save_object(similarity_df, os.path.join(self.features_dir, 'professional_job_similarity.pkl'))
        
        logger.info("Graph feature processing complete")
        
        # Create result dictionary
        result = {
            'users_with_loc': users_with_loc,
            'jobs_with_loc': jobs_with_loc,
            'location_lookup': location_lookup,
            'distances_df': distances_df,
            'professional_embeddings': professional_embeddings,
            'job_embeddings': job_embeddings,
            'category_embeddings': category_embeddings,
            'professional_features': professional_features,
            'job_features': job_features,
            'category_features': category_features,
            'location_features': location_features,
            'professional_similarity_edges': professional_similarity_edges,
            'job_similarity_edges': job_similarity_edges,
            'similarity_df': similarity_df
        }
        
        logger.info("All features processed and saved successfully")
        return result
        
    def load_all_features(self) -> Dict[str, Any]:
        """
        Load all previously saved features from disk.
        
        Returns:
            Dict[str, Any]: Dictionary containing all loaded features.
        """
        logger.info("Loading features from disk")
        
        result = {}
        
        # List of feature files to load
        feature_files = [
            'location_lookup.pkl',
            'users_with_loc.pkl',
            'jobs_with_loc.pkl',
            'distances_df.pkl',
            'professional_embeddings.pkl',
            'job_embeddings.pkl',
            'category_embeddings.pkl',
            'professional_features.pkl',
            'job_features.pkl',
            'category_features.pkl',
            'location_node_features.pkl',
            'professional_similarity_edges.pkl',
            'job_similarity_edges.pkl',
            'professional_job_similarity.pkl'
        ]
        
        # Load each feature file
        for file in feature_files:
            file_path = os.path.join(self.features_dir, file)
            if os.path.exists(file_path):
                feature_name = os.path.splitext(file)[0]
                result[feature_name] = load_object(file_path)
                logger.info(f"Loaded {feature_name}")
            else:
                logger.warning(f"Feature file {file} not found")
                
        return result
