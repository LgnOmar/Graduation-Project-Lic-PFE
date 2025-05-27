import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class SimpleRecommenderDemo:
    """
    A simple demonstration of JibJob recommendation functionality without using the full GCN model.
    This demonstrates the core recommendation concepts based on category matching and location.
    """
    def __init__(self, data_dir='./sample_data'):
        self.data_dir = data_dir
        self.load_data()
        
    def load_data(self):
        """Load all required data files"""
        # Load locations
        with open(os.path.join(self.data_dir, 'locations.json'), 'r') as f:
            self.locations = json.load(f)
        
        # Create a dictionary for quick location lookups
        self.location_dict = {loc['location_id']: loc for loc in self.locations}
        
        # Load categories
        self.categories = pd.read_csv(os.path.join(self.data_dir, 'categories.csv'))
        
        # Load users
        self.users = pd.read_csv(os.path.join(self.data_dir, 'users.csv'))
        self.professionals = self.users[self.users['user_type'] == 'professional']
        
        # Load professional categories
        self.professional_categories = pd.read_csv(os.path.join(self.data_dir, 'professional_categories.csv'))
        
        # Load jobs
        self.jobs = pd.read_csv(os.path.join(self.data_dir, 'jobs.csv'))
        
        # Optional: Load job applications if available
        try:
            self.job_applications = pd.read_csv(os.path.join(self.data_dir, 'job_applications.csv'))
            self.has_applications = True
        except:
            self.has_applications = False
            
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in kilometers"""
        R = 6371  # Earth radius in kilometers
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    def get_user_categories(self, user_id):
        """Get categories selected by a professional user"""
        user_categories = self.professional_categories[self.professional_categories['user_id'] == user_id]
        return user_categories['category_id'].tolist()
    
    def get_recommendations(self, user_id, top_k=10, filter_location=True, max_distance=50.0):
        """
        Get job recommendations for a professional user.
        
        Args:
            user_id: ID of the professional user
            top_k: Number of recommendations to return
            filter_location: Whether to filter by location
            max_distance: Maximum distance in kilometers
            
        Returns:
            List of recommended job dictionaries
        """
        # Check if user exists
        if user_id not in self.professionals['user_id'].values:
            raise ValueError(f"User {user_id} not found or is not a professional")
        
        # Get user information
        user = self.professionals[self.professionals['user_id'] == user_id].iloc[0]
        user_location_id = user['location_id']
        user_location = self.location_dict[user_location_id]
        user_categories = self.get_user_categories(user_id)
        
        # Filter jobs by category
        if user_categories:
            matching_jobs = self.jobs[self.jobs['required_category_id'].isin(user_categories)]
        else:
            matching_jobs = self.jobs  # If no categories, consider all jobs
        
        # Calculate distances for all matching jobs
        job_scores = []
        
        for _, job in matching_jobs.iterrows():
            job_location_id = job['location_id']
            job_location = self.location_dict[job_location_id]
            
            # Calculate distance
            distance = self.haversine_distance(
                float(user_location['latitude']), float(user_location['longitude']),
                float(job_location['latitude']), float(job_location['longitude'])
            )
            
            # Calculate category match score (1 if match, 0 if not)
            category_match = 1 if job['required_category_id'] in user_categories else 0
            
            # Calculate distance score (inversely proportional to distance)
            distance_score = 1 / (1 + distance/10)  # Normalized to 0-1 range
            
            # Calculate final score - simple weighted sum
            final_score = 0.7 * category_match + 0.3 * distance_score
              # Only add to results if it's a category match or reasonably close
            if category_match == 1 or distance <= max_distance * 1.5:
                job_scores.append({
                    'job_id': job['job_id'],
                    'title': job['title'],
                    'description': job['description'],
                    'distance': distance,
                    'category_match': category_match == 1,
                    'score': final_score
                })
        
        # Filter by distance if requested
        if filter_location:
            job_scores = [job for job in job_scores if job['distance'] <= max_distance]
        
        # Sort jobs by score
        job_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k recommendations
        recommendations = job_scores[:top_k]
        
        return recommendations

def main():
    # Initialize the recommender
    recommender = SimpleRecommenderDemo()
    
    # Get a list of professional users
    professionals = recommender.professionals['user_id'].tolist()
    
    if not professionals:
        print("No professional users found in the dataset!")
        return
    
    # Pick a random professional
    user_id = professionals[0]
    
    # Get the professional's categories
    user_categories = recommender.get_user_categories(user_id)
    user_info = recommender.professionals[recommender.professionals['user_id'] == user_id].iloc[0]
    
    print(f"Generating recommendations for user: {user_id}")
    print(f"Username: {user_info['username']}")
    print(f"Bio: {user_info['profile_bio']}")
    print(f"Categories: {user_categories}")
    print("=" * 50)
    
    # Get recommendations
    try:
        recommendations = recommender.get_recommendations(
            user_id=user_id, 
            top_k=5, 
            filter_location=True,
            max_distance=50.0
        )
        
        # Print recommendations
        print(f"Top 5 job recommendations for {user_id}:")
        print("=" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Job: {rec['title']} (ID: {rec['job_id']})")
            print(f"   Distance: {rec['distance']:.2f} km")
            print(f"   Category match: {'Yes' if rec['category_match'] else 'No'}")
            print(f"   Score: {rec['score']:.4f}")
            print("-" * 40)
            
        # Save recommendations to file
        with open('simple_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\nRecommendations saved to simple_recommendations.json")
        
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    main()
