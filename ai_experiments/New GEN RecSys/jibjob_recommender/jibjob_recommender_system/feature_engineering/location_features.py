"""
Location features module for processing geographic data.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from ..utils.helpers import calculate_haversine_distance, batch_calculate_distances

logger = logging.getLogger(__name__)

class LocationFeatures:
    """
    Class responsible for processing location data and generating location-based features.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LocationFeatures with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.features_config = config['features']['location']
        self.max_distance_km = self.features_config.get('max_distance_km', 50.0)
        
    def process_locations(self, locations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process location data to prepare for feature engineering.
        
        Args:
            locations_df: DataFrame containing location data.
            
        Returns:
            pd.DataFrame: Processed location data.
        """
        # Copy to avoid modifying the original
        df = locations_df.copy()
        
        # Ensure latitude and longitude are numeric
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Check for invalid coordinates
        invalid_coords = df[(df['latitude'] < -90) | (df['latitude'] > 90) | 
                           (df['longitude'] < -180) | (df['longitude'] > 180) |
                           df['latitude'].isna() | df['longitude'].isna()]
                           
        if not invalid_coords.empty:
            logger.warning(f"Found {len(invalid_coords)} locations with invalid coordinates")
            
            # Set invalid coordinates to NaN
            df.loc[invalid_coords.index, ['latitude', 'longitude']] = np.nan
            
        logger.info(f"Processed {len(df)} location records")
        return df
        
    def calculate_distance_matrix(self, locations_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate a distance matrix between all locations.
        
        Args:
            locations_df: DataFrame containing location data.
            
        Returns:
            np.ndarray: Distance matrix of shape (n_locations, n_locations).
        """
        n_locations = len(locations_df)
        distance_matrix = np.zeros((n_locations, n_locations))
        
        lats = locations_df['latitude'].values
        lons = locations_df['longitude'].values
        
        for i in range(n_locations):
            # Calculate distances from location i to all other locations
            distances = batch_calculate_distances(
                lats[i], lons[i], lats, lons
            )
            distance_matrix[i, :] = distances
            
        logger.info(f"Calculated distance matrix of shape {distance_matrix.shape}")
        return distance_matrix
        
    def create_location_lookup(self, locations_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Create a lookup dictionary for location data.
        
        Args:
            locations_df: DataFrame containing location data.
            
        Returns:
            Dict[str, Dict[str, float]]: Dictionary mapping location_id to location data.
        """
        location_lookup = {}
        
        for _, row in locations_df.iterrows():
            if pd.isna(row['latitude']) or pd.isna(row['longitude']):
                continue
                
            location_lookup[row['location_id']] = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'name': row['name'],
                'wilaya_name': row['wilaya_name']
            }
            
        logger.info(f"Created location lookup for {len(location_lookup)} locations")
        return location_lookup
        
    def add_location_data_to_users(self, users_df: pd.DataFrame, 
                                 location_lookup: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Add latitude and longitude to users DataFrame based on location_id.
        
        Args:
            users_df: DataFrame containing user data.
            location_lookup: Dictionary mapping location_id to location data.
            
        Returns:
            pd.DataFrame: Users DataFrame with added location data.
        """
        # Copy to avoid modifying the original
        df = users_df.copy()
        
        # Add empty columns for latitude and longitude
        df['latitude'] = np.nan
        df['longitude'] = np.nan
        df['location_name'] = ""
        
        # Fill in location data
        for idx, row in df.iterrows():
            location_id = row['location_id']
            if location_id in location_lookup:
                df.at[idx, 'latitude'] = location_lookup[location_id]['latitude']
                df.at[idx, 'longitude'] = location_lookup[location_id]['longitude']
                df.at[idx, 'location_name'] = location_lookup[location_id]['name']
                
        # Count missing locations
        missing_locations = df[df['latitude'].isna()]['location_id'].nunique()
        if missing_locations > 0:
            logger.warning(f"Could not find location data for {missing_locations} unique location IDs in users")
            
        logger.info(f"Added location data to {len(df)} user records")
        return df
        
    def add_location_data_to_jobs(self, jobs_df: pd.DataFrame,
                                location_lookup: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Add latitude and longitude to jobs DataFrame based on location_id.
        
        Args:
            jobs_df: DataFrame containing job data.
            location_lookup: Dictionary mapping location_id to location data.
            
        Returns:
            pd.DataFrame: Jobs DataFrame with added location data.
        """
        # Copy to avoid modifying the original
        df = jobs_df.copy()
        
        # Add empty columns for latitude and longitude
        df['latitude'] = np.nan
        df['longitude'] = np.nan
        df['location_name'] = ""
        
        # Fill in location data
        for idx, row in df.iterrows():
            location_id = row['location_id']
            if location_id in location_lookup:
                df.at[idx, 'latitude'] = location_lookup[location_id]['latitude']
                df.at[idx, 'longitude'] = location_lookup[location_id]['longitude']
                df.at[idx, 'location_name'] = location_lookup[location_id]['name']
                
        # Count missing locations
        missing_locations = df[df['latitude'].isna()]['location_id'].nunique()
        if missing_locations > 0:
            logger.warning(f"Could not find location data for {missing_locations} unique location IDs in jobs")
            
        logger.info(f"Added location data to {len(df)} job records")
        return df
        
    def calculate_user_job_distances(self, users_df: pd.DataFrame, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate distances between professionals and jobs.
        
        Args:
            users_df: DataFrame containing user data with latitude and longitude.
            jobs_df: DataFrame containing job data with latitude and longitude.
            
        Returns:
            pd.DataFrame: DataFrame with user_id, job_id, and distance.
        """
        # Filter for professional users only
        professionals = users_df[users_df['user_type'] == 'professional'].copy()
        
        # Filter out records with missing coordinates
        valid_professionals = professionals[~professionals['latitude'].isna() & ~professionals['longitude'].isna()]
        valid_jobs = jobs_df[~jobs_df['latitude'].isna() & ~jobs_df['longitude'].isna()]
        
        # Check if we have valid data
        if valid_professionals.empty or valid_jobs.empty:
            logger.warning("No valid professional or job coordinates available for distance calculation")
            return pd.DataFrame(columns=['user_id', 'job_id', 'distance_km'])
            
        # Initialize list to store results
        distances = []
        
        # Calculate distances from each professional to each job
        for _, prof in valid_professionals.iterrows():
            job_distances = batch_calculate_distances(
                prof['latitude'], prof['longitude'],
                valid_jobs['latitude'].values, valid_jobs['longitude'].values
            )
            
            for j, (_, job) in enumerate(valid_jobs.iterrows()):
                distances.append({
                    'user_id': prof['user_id'],
                    'job_id': job['job_id'],
                    'distance_km': job_distances[j]
                })
                
        # Convert to DataFrame
        distances_df = pd.DataFrame(distances)
        
        logger.info(f"Calculated {len(distances_df)} professional-job distances")
        return distances_df
        
    def normalize_coordinates(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Normalize coordinates to a suitable range for ML models.
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
            
        Returns:
            Tuple[float, float]: Normalized latitude and longitude (-1 to 1)
        """
        norm_lat = lat / 90.0  # Maps to [-1, 1]
        norm_lon = lon / 180.0  # Maps to [-1, 1]
        
        return norm_lat, norm_lon
        
    def get_location_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract normalized location features from a DataFrame.
        
        Args:
            df: DataFrame with latitude and longitude columns.
            
        Returns:
            np.ndarray: Array of normalized location features.
        """
        # Handle missing coordinates with zeros (center of the world)
        df_filled = df.copy()
        df_filled['latitude'] = df_filled['latitude'].fillna(0)
        df_filled['longitude'] = df_filled['longitude'].fillna(0)
        
        # Normalize coordinates
        normalized = np.array([
            self.normalize_coordinates(lat, lon)
            for lat, lon in zip(df_filled['latitude'], df_filled['longitude'])
        ])
        
        return normalized
