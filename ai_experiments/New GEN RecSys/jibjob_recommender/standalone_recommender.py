"""
Standalone JibJob recommendation system that doesn't depend on the original modules.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

class StandaloneRecommender:
    """
    A standalone version of the JibJob recommendation system.
    """
    def __init__(self, data_dir='./sample_data'):
        self.data_dir = data_dir
        
        # Load data
        self.load_data()
        
        print(f"Loaded {len(self.professionals)} professionals")
        print(f"Loaded {len(self.jobs)} jobs")
        print(f"Loaded {len(self.categories)} categories")

    def load_data(self):
        """Load all required data files"""
        # Load locations
        with open(os.path.join(self.data_dir, 'locations.json'), 'r') as f:
            self.locations = json.load(f)
        
        # Create a dictionary for quick location lookups
        self.location_dict = {loc['location_id']: loc for loc in self.locations}
        
        # Load categories
        self.categories = pd.read_csv(os.path.join(self.data_dir, 'categories.csv'))
        self.category_dict = dict(zip(self.categories['category_id'], self.categories['category_name']))
        
        # Load users
        self.users = pd.read_csv(os.path.join(self.data_dir, 'users.csv'))
        self.professionals = self.users[self.users['user_type'] == 'professional']
        self.clients = self.users[self.users['user_type'] == 'client']
        
        # Load professional categories
        self.professional_categories = pd.read_csv(os.path.join(self.data_dir, 'professional_categories.csv'))
          # Create a dictionary of professional -> categories
        self.prof_to_categories = {}
        for _, row in self.professional_categories.iterrows():
            # Handle possible format differences in user IDs and category IDs
            user_id = row['user_id']
            # Convert "cat_XXX" to int if necessary
            try:
                category_id = int(row['category_id'].replace('cat_', '')) if isinstance(row['category_id'], str) and 'cat_' in row['category_id'] else row['category_id']
            except (ValueError, AttributeError):
                category_id = row['category_id']
                
            if user_id not in self.prof_to_categories:
                self.prof_to_categories[user_id] = []
            self.prof_to_categories[user_id].append(category_id)
            
            # Also add mapping for "user_XXXX" format if the current ID is "prof_XXX"
            if isinstance(user_id, str) and user_id.startswith('prof_'):
                # Create alternative user ID format (user_XXXX)
                alt_user_id = 'user_' + user_id.split('_')[1].zfill(4)
                if alt_user_id not in self.prof_to_categories:
                    self.prof_to_categories[alt_user_id] = []
                self.prof_to_categories[alt_user_id].append(category_id)
        
        # Load jobs
        self.jobs = pd.read_csv(os.path.join(self.data_dir, 'jobs.csv'))
        
        # Optional: load job applications if available
        try:
            self.job_applications = pd.read_csv(os.path.join(self.data_dir, 'job_applications.csv'))
            self.has_applications = True
        except Exception:
            self.has_applications = False
            self.job_applications = None

    def get_user_categories(self, user_id):
        """Get categories for a professional user"""
        return self.prof_to_categories.get(user_id, [])
      def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in kilometers"""
        R = 6371  # Earth radius in kilometers
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    def get_job_recommendations(self, user_id, top_k=10, max_distance=50.0):
        """
        Get job recommendations for a professional user
        
        Args:
            user_id: ID of the professional user
            top_k: Number of recommendations to return
            max_distance: Maximum distance in km
            
        Returns:
            List of recommended jobs
        """
        # Check if user exists directly in users.csv
        if user_id in self.professionals['user_id'].values:
            # Using the format from users.csv
            user = self.professionals[self.professionals['user_id'] == user_id].iloc[0]
            prof_id_for_categories = user_id
        else:
            # Maybe the user is using the ID format from professional_categories.csv (prof_XXX)
            # Try mapping it to the users.csv format
            # Assuming prof_050 would be user_0050
            if user_id.startswith("prof_"):
                # Extract number part and create user_id equivalent
                number_part = user_id.split("_")[1]
                equivalent_user_id = f"user_{number_part.zfill(4)}"
                if equivalent_user_id in self.professionals['user_id'].values:
                    user = self.professionals[self.professionals['user_id'] == equivalent_user_id].iloc[0]
                    prof_id_for_categories = user_id  # Use prof_XXX format for categories
                else:
                    raise ValueError(f"User {user_id} (mapped to {equivalent_user_id}) not found or is not a professional")
            else:
                raise ValueError(f"User {user_id} not found or is not a professional")
        
        # Get user information
        user = self.professionals[self.professionals['user_id'] == user_id].iloc[0]
        user_loc_id = user['location_id']
        user_categories = self.get_user_categories(user_id)
        
        # Get user location
        user_location = self.location_dict.get(user_loc_id, {'latitude': 0, 'longitude': 0})
        
        # Calculate job scores
        job_scores = []
        
        print(f"User {user_id} has categories: {user_categories}")
        print(f"User location: {user_loc_id}")
        
        for _, job in self.jobs.iterrows():
            # Get job location
            job_loc_id = job['location_id']
            job_location = self.location_dict.get(job_loc_id, {'latitude': 0, 'longitude': 0})
            
            # Calculate distance
            try:
                distance = self.calculate_distance(
                    user_location['latitude'], user_location['longitude'],
                    job_location['latitude'], job_location['longitude']
                )
            except (ValueError, TypeError) as e:
                print(f"Error calculating distance for job {job['job_id']}: {e}")
                distance = float('inf')  # Set to infinity if we can't calculate
            
            # Check category match
            category_match = 1.0 if job['required_category_id'] in user_categories else 0.0
            
            # Calculate score components
            category_score = category_match * 0.6
            distance_score = 0.4 / (1 + distance / 20.0) if distance < float('inf') else 0  # Closer is better
            
            # Combine scores
            total_score = category_score + distance_score
            
            # Skip jobs that are too far away
            if distance > max_distance:
                continue
                
            # Get category name
            category_name = self.category_dict.get(job['required_category_id'], 'Unknown')
                
            # Add to job scores
            job_scores.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'description': job['description'],
                'required_category': job['required_category_id'],
                'category_name': category_name,
                'distance': float(distance),
                'score': float(total_score),
                'category_match': bool(category_match)
            })
        
        # Sort by score
        job_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k
        return job_scores[:top_k]

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run standalone JibJob recommendation')
    parser.add_argument('--user-id', type=str, required=True, help='ID of professional user')
    parser.add_argument('--top-k', type=int, default=5, help='Number of recommendations')
    args = parser.parse_args()
    
    # Create recommender
    recommender = StandaloneRecommender()
    
    try:
        # Get recommendations
        print(f"\nGenerating recommendations for user: {args.user_id}")
        recommendations = recommender.get_job_recommendations(args.user_id, args.top_k)
        
        # Display recommendations
        print(f"\nTop {args.top_k} job recommendations:")
        print("="*50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} (ID: {rec['job_id']})")
            print(f"   Required category: {rec['required_category']} ({rec['category_name']})")
            print(f"   Distance: {rec['distance']:.2f} km")
            print(f"   Category match: {'Yes' if rec['category_match'] else 'No'}")
            print(f"   Score: {rec['score']:.4f}")
            print(f"   Description: {rec['description'][:100]}...")
            print("-"*40)
        
        # Save recommendations to file
        output_file = f"recommendations_{args.user_id}.json"
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"\nRecommendations saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
