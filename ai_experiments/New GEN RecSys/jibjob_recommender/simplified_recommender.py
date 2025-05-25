"""
Simplified version of the JibJob recommendation system that doesn't rely on torch_geometric
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import math
from typing import Dict, List, Any, Optional

import torch
from transformers import DistilBertModel, DistilBertTokenizer

# Add current dir to path
sys.path.insert(0, os.path.abspath('.'))

class SimplifiedRecommender:
    """
    A simplified version of the JibJob recommendation system.
    """    def __init__(self, data_dir='./sample_data'):
        self.data_dir = data_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data directly from files
        try:
            # Load locations
            with open(os.path.join(data_dir, 'locations.json'), 'r') as f:
                self.locations = json.load(f)
            
            # Load categories
            self.categories = pd.read_csv(os.path.join(data_dir, 'categories.csv'))
            
            # Load users
            self.users = pd.read_csv(os.path.join(data_dir, 'users.csv'))
            self.professionals = self.users[self.users['user_type'] == 'professional']
            self.clients = self.users[self.users['user_type'] == 'client']
            
            # Load professional categories
            self.professional_categories = pd.read_csv(os.path.join(data_dir, 'professional_categories.csv'))
            
            # Load jobs
            self.jobs = pd.read_csv(os.path.join(data_dir, 'jobs.csv'))
            
            # Try to load job applications if available
            try:
                self.job_applications = pd.read_csv(os.path.join(data_dir, 'job_applications.csv'))
            except:
                self.job_applications = None
                
        except Exception as e:
            print(f"Error loading data: {e}")
          # Create a dictionary for quick location lookups
        self.location_dict = {loc['location_id']: loc for loc in self.locations}
        
        # Create a dictionary of professional -> categories
        self.prof_to_categories = {}
        for _, row in self.professional_categories.iterrows():
            if row['user_id'] not in self.prof_to_categories:
                self.prof_to_categories[row['user_id']] = []
            self.prof_to_categories[row['user_id']].append(row['category_id'])
        
        print(f"Loaded {len(self.professionals)} professionals")
        print(f"Loaded {len(self.jobs)} jobs")
        print(f"Loaded {len(self.categories)} categories")    def get_user_categories(self, user_id):
        """Get categories for a professional user"""
        return self.prof_to_categories.get(user_id, [])
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in kilometers"""
        R = 6371  # Earth radius in kilometers
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        
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
        # Check if user exists
        if user_id not in self.professionals['user_id'].values:
            raise ValueError(f"User {user_id} not found or is not a professional")
          # Get user information
        user = self.professionals[self.professionals['user_id'] == user_id].iloc[0]
        user_loc_id = user['location_id']
        user_categories = self.get_user_categories(user_id)
        
        print(f"User {user_id} has categories: {user_categories}")
        print(f"User location: {user_loc_id}")
        
        # Get user location
        user_location = self.location_dict.get(user_loc_id, {'latitude': 0, 'longitude': 0})
        
        # Calculate job scores
        job_scores = []
        
        for _, job in self.jobs.iterrows():
            # Get job location
            job_loc_id = job['location_id']
            job_location = self.location_dict.get(job_loc_id, {'latitude': 0, 'longitude': 0})
            if not job_location:
                print(f"Warning: Location {job_loc_id} not found for job {job['job_id']}")
                job_location = {'latitude': 0, 'longitude': 0}
            
            # Calculate distance
            distance = self.location_features.calculate_distance(
                user_location['latitude'], user_location['longitude'],
                job_location['latitude'], job_location['longitude']
            )
            
            # Check category match
            category_match = 1.0 if job['required_category_id'] in user_categories else 0.0
            
            # Calculate text relevance (simplified)
            # In the full system, this would involve comparing embeddings
            text_relevance = 0.5  # Placeholder
            
            # Calculate score components
            category_score = category_match * 0.6
            distance_score = 0.4 / (1 + distance / 20.0)  # Closer is better
            
            # Combine scores
            total_score = category_score + distance_score
            
            # Skip jobs that are too far away
            if distance > max_distance:
                continue
                
            # Add to job scores
            job_scores.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'description': job['description'],
                'required_category': job['required_category_id'],
                'distance': distance,
                'score': total_score,
                'category_match': bool(category_match)
            })
        
        # Sort by score
        job_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k
        return job_scores[:top_k]

def main():
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run simplified JibJob recommendation')
    parser.add_argument('--user-id', type=str, required=True, help='ID of professional user')
    parser.add_argument('--top-k', type=int, default=5, help='Number of recommendations')
    args = parser.parse_args()
    
    # Create recommender
    recommender = SimplifiedRecommender()
    
    try:
        # Get recommendations
        print(f"\nGenerating recommendations for user: {args.user_id}")
        recommendations = recommender.get_job_recommendations(args.user_id, args.top_k)
        
        # Display recommendations
        print(f"\nTop {args.top_k} job recommendations:")
        print("="*50)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']} (ID: {rec['job_id']})")
            print(f"   Required category: {rec['required_category']}")
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
