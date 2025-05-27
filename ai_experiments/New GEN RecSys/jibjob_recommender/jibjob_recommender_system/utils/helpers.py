"""
Helper utility functions for the JibJob recommendation system.
"""

import os
import pickle
import math
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth using the Haversine formula.
    
    Args:
        lat1: Latitude of first point in degrees
        lon1: Longitude of first point in degrees
        lat2: Latitude of second point in degrees
        lon2: Longitude of second point in degrees
        
    Returns:
        float: Distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth radius in kilometers
    radius = 6371
    
    # Calculate distance
    distance = radius * c
    
    return distance

def batch_calculate_distances(origin_lat: float, origin_lon: float, 
                             destination_lats: List[float], destination_lons: List[float]) -> np.ndarray:
    """
    Calculate distances from one origin point to multiple destination points.
    Optimized for batch processing.
    
    Args:
        origin_lat: Latitude of origin point
        origin_lon: Longitude of origin point
        destination_lats: List of destination latitudes
        destination_lons: List of destination longitudes
        
    Returns:
        np.ndarray: Array of distances in kilometers
    """
    # Convert to radians and numpy arrays
    origin_lat, origin_lon = map(math.radians, [origin_lat, origin_lon])
    destination_lats = np.radians(destination_lats)
    destination_lons = np.radians(destination_lons)
    
    # Haversine formula vectorized
    dlat = destination_lats - origin_lat
    dlon = destination_lons - origin_lon
    
    a = np.sin(dlat/2)**2 + np.cos(origin_lat) * np.cos(destination_lats) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth radius in kilometers
    radius = 6371
    
    # Calculate distances
    distances = radius * c
    
    return distances

def save_object(obj: Any, filepath: str) -> bool:
    """
    Save a Python object to disk using pickle.
    
    Args:
        obj: Object to save
        filepath: Path to save the object to
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
            
        logger.info(f"Object saved successfully to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving object to {filepath}: {str(e)}")
        return False
        
def load_object(filepath: str) -> Optional[Any]:
    """
    Load a Python object from disk using pickle.
    
    Args:
        filepath: Path to load the object from
        
    Returns:
        Optional[Any]: The loaded object if successful, None otherwise
    """
    try:
        with open(filepath, 'rb') as f:
            obj = pickle.load(f)
            
        logger.info(f"Object loaded successfully from {filepath}")
        return obj
        
    except Exception as e:
        logger.error(f"Error loading object from {filepath}: {str(e)}")
        return None
        
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        float: Cosine similarity (between -1 and 1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return dot_product / (norm1 * norm2)
    
def batch_cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between a vector and each row of a matrix.
    
    Args:
        vec: Vector to compare
        matrix: Matrix where each row is a vector to compare against
        
    Returns:
        np.ndarray: Array of cosine similarities
    """
    # Normalize the vector
    vec_norm = vec / np.linalg.norm(vec)
    
    # Normalize each row of the matrix
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1  # Avoid division by zero
    normalized_matrix = matrix / matrix_norms
    
    # Calculate similarities
    similarities = np.dot(normalized_matrix, vec_norm)
    
    return similarities
    
def create_id_mapping(ids: List[str]) -> Dict[str, int]:
    """
    Create a mapping from string IDs to integer indices.
    
    Args:
        ids: List of string IDs
        
    Returns:
        Dict[str, int]: Mapping from string ID to integer index
    """
    return {id_str: idx for idx, id_str in enumerate(ids)}

def filter_by_distance(origin_lat: float, origin_lon: float, 
                      destination_lats: List[float], destination_lons: List[float],
                      max_distance_km: float) -> List[int]:
    """
    Filter destinations by maximum distance from origin.
    
    Args:
        origin_lat: Latitude of origin point
        origin_lon: Longitude of origin point
        destination_lats: List of destination latitudes
        destination_lons: List of destination longitudes
        max_distance_km: Maximum distance in kilometers
        
    Returns:
        List[int]: Indices of destinations within the maximum distance
    """
    distances = batch_calculate_distances(origin_lat, origin_lon, destination_lats, destination_lons)
    return [i for i, distance in enumerate(distances) if distance <= max_distance_km]
