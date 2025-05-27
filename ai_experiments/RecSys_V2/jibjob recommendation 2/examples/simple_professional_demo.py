"""
Simple test script to demonstrate the professional job matching functionality.
This simplified version avoids complex model training and focuses on category matching.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

class SimpleJobRecommender:
    """
    A simplified job recommender that matches professionals with jobs based on categories and location.
    """
    
    def __init__(self):
        self.jobs_df = None
        self.users_df = None
    
    def load_data(self, jobs_df, users_df):
        """Load jobs and users data."""
        self.jobs_df = jobs_df
        self.users_df = users_df
        print(f"Loaded {len(jobs_df)} jobs and {len(users_df)} users")
    
    def recommend_for_professional(self, user_id, top_k=10):
        """
        Recommend jobs for a professional based on their categories.
        
        Args:
            user_id: The ID of the professional user
            top_k: Maximum number of recommendations to return
        
        Returns:
            List of recommended jobs
        """
        # Get professional data
        user_data = self.users_df[self.users_df['user_id'] == user_id]
        if user_data.empty or user_data['user_type'].iloc[0] != 'professional':
            print(f"User {user_id} not found or not a professional")
            return []
        
        # Get professional categories and location
        professional_categories = user_data['categories'].iloc[0]
        professional_location = user_data['location'].iloc[0] if 'location' in user_data.columns else None
        
        # Find matching jobs
        matching_jobs = self.jobs_df[self.jobs_df['category'].isin(professional_categories)]
        
        # Simple sorting (in practice, you would use more sophisticated ranking)
        matching_jobs = matching_jobs.sort_values(by='job_id').head(top_k)
        
        # Prepare recommendations
        recommendations = []
        for _, job in matching_jobs.iterrows():
            category_match = job['category'] in professional_categories
            
            recommendations.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'category': job['category'],
                'location': job.get('location', ''),
                'description': job.get('description', '')[:100] + '...',
                'category_match': category_match,
                'match_score': 1.0 if category_match else 0.5,
                'location_match': job.get('location', '') == professional_location
            })
        
        return recommendations

def generate_sample_data(n_users=10, n_jobs=20):
    """
    Generate sample data for testing.
    
    Returns:
        Tuple of (users_df, jobs_df)
    """
    # Define job categories
    job_categories = ['Plumbing', 'Painting', 'Cleaning', 'Carpentry', 'Electrical']
    
    # Generate users
    users = []
    for i in range(n_users):
        user_type = 'professional' if i < n_users * 0.7 else 'client'
        categories = np.random.choice(job_categories, size=np.random.randint(1, 4), replace=False).tolist() if user_type == 'professional' else []
        users.append({
            'user_id': f"user_{i}",
            'user_type': user_type,
            'name': f"User {i}",
            'categories': categories,
            'location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'])
        })
    
    # Generate jobs
    jobs = []
    for i in range(n_jobs):
        jobs.append({
            'job_id': f"job_{i}",
            'title': f"Job {i}",
            'description': f"This is job {i} description with details about the work required.",
            'category': np.random.choice(job_categories),
            'location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'])
        })
    
    return pd.DataFrame(users), pd.DataFrame(jobs)

def main():
    print("=" * 60)
    print("Professional Job Recommendation Demo")
    print("=" * 60)
    
    # Generate sample data
    users_df, jobs_df = generate_sample_data(n_users=20, n_jobs=50)
    
    # Initialize recommender
    recommender = SimpleJobRecommender()
    recommender.load_data(jobs_df, users_df)
    
    # Find a professional user
    professionals = users_df[users_df['user_type'] == 'professional']
    if professionals.empty:
        print("No professional users found in sample data")
        return
    
    test_professional = professionals.iloc[0]
    professional_id = test_professional['user_id']
    
    print(f"\nGenerating recommendations for professional: {professional_id}")
    print(f"Categories: {test_professional['categories']}")
    print(f"Location: {test_professional['location']}")
    
    # Get recommendations
    recommendations = recommender.recommend_for_professional(professional_id, top_k=5)
    
    if not recommendations:
        print("No recommendations found")
        return
    
    # Display recommendations
    print("\nTop recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['title']} (Category: {rec['category']}, Location: {rec['location']})")
        print(f"   Match score: {rec['match_score']:.2f}")
        print(f"   Category match: {'Yes' if rec['category_match'] else 'No'}, Location match: {'Yes' if rec['location_match'] else 'No'}")
        print(f"   Description: {rec['description']}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
