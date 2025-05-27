import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class JibJobRecommender:
    """
    A simplified version of the JibJob recommendation system.
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
        self.location_dict = {loc['location_id']: loc for loc in self.locations}
        
        # Load categories
        self.categories = pd.read_csv(os.path.join(self.data_dir, 'categories.csv'))
        self.category_dict = dict(zip(self.categories['category_id'], self.categories['category_name']))
        
        # Load users
        self.users = pd.read_csv(os.path.join(self.data_dir, 'users.csv'))
        self.professionals = self.users[self.users['user_type'] == 'professional']
        
        # Load professional categories
        self.professional_categories = pd.read_csv(os.path.join(self.data_dir, 'professional_categories.csv'))
        
        # Create a dictionary of professional -> categories
        self.prof_to_categories = {}
        for _, row in self.professional_categories.iterrows():
            if row['user_id'] not in self.prof_to_categories:
                self.prof_to_categories[row['user_id']] = []
            self.prof_to_categories[row['user_id']].append(row['category_id'])
        
        # Load jobs
        self.jobs = pd.read_csv(os.path.join(self.data_dir, 'jobs.csv'))
        
        # Load job applications
        self.job_applications = pd.read_csv(os.path.join(self.data_dir, 'job_applications.csv'))
        
        print(f"Loaded {len(self.professionals)} professionals")
        print(f"Loaded {len(self.jobs)} jobs")
        print(f"Loaded {len(self.professional_categories)} professional-category associations")
            
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
        return self.prof_to_categories.get(user_id, [])
    
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
        user_location = self.location_dict.get(user_location_id, {'latitude': 0, 'longitude': 0})
        user_categories = self.get_user_categories(user_id)
        
        print(f"User categories: {user_categories}")
        print(f"User location: {user_location_id}")
        
        # Job scores list
        job_scores = []
        
        # Process all jobs
        for _, job in self.jobs.iterrows():
            # Get job information
            job_location_id = job['location_id']
            job_location = self.location_dict.get(job_location_id, {'latitude': 0, 'longitude': 0})
            job_category = job['required_category_id']
            
            # Calculate distance
            distance = self.haversine_distance(
                float(user_location['latitude']), float(user_location['longitude']),
                float(job_location['latitude']), float(job_location['longitude'])
            )
            
            # Check if this job category matches any of the user's categories
            category_match = 1.0 if job_category in user_categories else 0.0
            
            # Distance score (inversely proportional to distance)
            # Closer jobs get higher scores
            distance_score = 1.0 / (1.0 + distance/50.0)
            
            # Calculate final score - weighted sum
            final_score = 0.7 * category_match + 0.3 * distance_score
            
            # Add to job scores
            job_scores.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'description': job['description'],
                'category_id': job['required_category_id'],
                'category_name': self.category_dict.get(job['required_category_id'], ''),
                'distance': distance,
                'distance_km': f"{distance:.2f} km",
                'category_match': bool(category_match),
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

    def show_recommendations_for_user(self, user_id, top_k=5):
        """Show recommendations for a specific user"""
        try:
            # Get user info
            user = self.professionals[self.professionals['user_id'] == user_id].iloc[0]
            
            print(f"\n{'='*60}")
            print(f"RECOMMENDATIONS FOR USER: {user_id}")
            print(f"Username: {user['username']}")
            print(f"Profile: {user['profile_bio']}")
            
            # Get user categories
            user_categories = self.get_user_categories(user_id)
            category_names = [self.category_dict.get(cat_id, cat_id) for cat_id in user_categories]
            print(f"Selected Categories: {', '.join(category_names)}")
            print(f"{'='*60}\n")
            
            # Get recommendations
            recommendations = self.get_recommendations(
                user_id=user_id,
                top_k=top_k,
                filter_location=True,
                max_distance=50.0
            )
            
            if not recommendations:
                print("No matching jobs found for this user!")
                return
            
            # Print recommendations
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']} (ID: {rec['job_id']})")
                print(f"   Category: {rec['category_name']} ({rec['category_id']})")
                print(f"   Distance: {rec['distance_km']}")
                print(f"   Category Match: {'✓' if rec['category_match'] else '✗'}")
                print(f"   Score: {rec['score']:.4f}")
                print(f"   Description: {rec['description'][:100]}...")
                print(f"{'-'*50}")
            
            # Save to file
            output_file = f"recommendations_{user_id}.json"
            with open(output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
            print(f"\nRecommendations saved to {output_file}")
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")


def main():
    # Initialize the recommender
    recommender = JibJobRecommender()
    
    # Get sample professionals
    sample_professionals = recommender.professionals['user_id'].tolist()[:3]
    
    # Show recommendations for each professional
    for prof_id in sample_professionals:
        recommender.show_recommendations_for_user(prof_id, top_k=5)


if __name__ == "__main__":
    main()
