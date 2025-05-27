"""
User type definitions for JibJob recommendation system.
This module provides the basic definitions and utilities for working with different user types.
"""

from typing import List, Dict, Optional, Any
from enum import Enum

class UserType(str, Enum):
    """Enum representing user types in the JibJob platform."""
    CLIENT = "client"
    PROFESSIONAL = "professional"

class JobCategory(str, Enum):
    """Enum representing job categories available in JibJob."""
    PLUMBING = "Plumbing"
    PAINTING = "Painting"
    GARDENING = "Gardening"
    ASSEMBLY = "Assembly"
    TECH_SUPPORT = "Tech Support"
    CLEANING = "Cleaning"
    MOVING = "Moving"
    ELECTRICAL = "Electrical"
    TUTORING = "Tutoring"
    DELIVERY = "Delivery"
    MECHANIC = "Auto mechanic"
    TRANSLATOR = "Translator"
    HOUSEKEEPER = "Housekeeper"
    CONSTRUCTION = "Construction worker"
    CARPENTRY = "Carpenter"
    WELDING = "Welder"
    
    @classmethod
    def all_values(cls) -> List[str]:
        """Get all category values as strings."""
        return [c.value for c in cls]

def calculate_location_distance(loc1: str, loc2: str) -> float:
    """
    Calculate a simple distance score between two locations.
    
    This is a simplified version that treats different location strings 
    as either matching (distance=0) or not matching (distance=1).
    In a real system, this would use geocoding and actual distance calculations.
    
    Args:
        loc1: First location string
        loc2: Second location string
        
    Returns:
        float: Distance score (0 if locations match, 1 otherwise)
    """
    # Simple exact match for demo purposes
    # In a real system, this would use geographic coordinates and distance formulas
    if not loc1 or not loc2:
        return 1.0  # Maximum distance if either location is missing
    
    return 0.0 if loc1.strip().lower() == loc2.strip().lower() else 1.0

class ProfessionalProfile:
    """Represents a professional's profile with their categories and preferences."""
    
    def __init__(
        self, 
        user_id: Any,
        categories: List[str], 
        location: Optional[str] = None,
        max_travel_distance: float = 50.0  # in km, not used in simple location matching
    ):
        """
        Initialize a professional profile.
        
        Args:
            user_id: Unique identifier for the professional
            categories: List of job categories the professional is interested in
            location: The professional's location
            max_travel_distance: Maximum distance the professional is willing to travel
        """
        self.user_id = user_id
        self.categories = categories
        self.location = location
        self.max_travel_distance = max_travel_distance
        
    def matches_job(self, job_category: str, job_location: str) -> bool:
        """
        Check if job category and location match the professional's profile.
        
        Args:
            job_category: Category of the job
            job_location: Location of the job
            
        Returns:
            bool: True if both category and location match, False otherwise
        """
        # Check category match
        category_match = job_category in self.categories
        
        # Check location match 
        location_match = (not self.location or not job_location or 
                         calculate_location_distance(self.location, job_location) == 0.0)
        
        return category_match and location_match
