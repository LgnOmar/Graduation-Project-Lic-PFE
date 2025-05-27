"""
Demonstration script for professional job matching in JibJob recommendation system.

This script shows how to:
1. Generate synthetic professional and client users
2. Generate job listings with categories
3. Use the professional-job matching feature to recommend jobs to professionals
4. Compare different recommendation approaches

Usage:
    python professional_recommendation_demo.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from datetime import datetime

# Add the parent directory to the path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.recommender import JobRecommender
from src.data.preprocessing import process_job_descriptions, normalize_ratings
from src.utils.sample_data import generate_users, generate_jobs, generate_interactions
from src.data.user_types import UserType, JobCategory, ProfessionalProfile

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("JibJob Professional Recommendation System Demo")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    n_users = 100
    n_jobs = 200
    n_interactions = 500
    
    # Generate users with professional/client distinction
    users_df = generate_users(n_users, professional_ratio=0.7)
    print(f"Generated {len(users_df)} users ({users_df['user_type'].value_counts()['professional']} professionals, "
          f"{users_df['user_type'].value_counts()['client']} clients)")
    
    # Generate jobs
    jobs_df = generate_jobs(n_jobs)
    print(f"Generated {len(jobs_df)} jobs across {jobs_df['category'].nunique()} categories")
    
    # Generate interactions
    interactions_df = generate_interactions(users_df, jobs_df, n_interactions)
    print(f"Generated {len(interactions_df)} interactions")
    
    # Initialize recommender
    print("\n2. Initializing recommender system...")
    recommender = JobRecommender()
    
    # Train the recommender
    print("\n3. Training recommender system...")
    history = recommender.train(
        interactions_df=interactions_df,
        jobs_df=jobs_df,
        users_df=users_df,
        user_col='user_id',
        job_col='job_id',
        rating_col='rating',
        comment_col='comment',
        user_type_col='user_type',
        categories_col='categories',
        epochs=50,
        batch_size=64,
        early_stop_patience=10
    )
    
    # Select a professional for demonstration
    professionals = users_df[users_df['user_type'] == 'professional']
    demo_professional = professionals.iloc[0]
    professional_id = demo_professional['user_id']
    professional_categories = demo_professional['categories']
    professional_location = demo_professional['location']
    
    print(f"\n4. Generating recommendations for professional: {professional_id}")
    print(f"   Categories: {professional_categories}")
    print(f"   Location: {professional_location}")
      # Generate recommendations using category and location matching
    recommendations = recommender.recommend_for_professional(
        professional_id=professional_id,
        professional_categories=professional_categories,
        professional_location=professional_location,
        top_k=10,
        require_category_match=True,
        max_location_distance=50.0,  # Allow some distance for better matching
        exclude_rated=True
    )
      # Display recommendations
    print("\nTop recommendations:")
    if not recommendations:
        print("No recommendations found! Check for configuration issues.")
    else:
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec['title']} (Category: {rec['category']}, Location: {rec['location']})")
            print(f"   Match score: {rec.get('match_score', rec.get('score', 0.0)):.2f}")
            print(f"   Category match: {'Yes' if rec.get('category_match', False) else 'No'}, Location match: {'Yes' if rec.get('location_match', False) else 'No'}")
            print(f"   Description: {rec['description'][:100]}...")
    
    # Compare with standard recommendations
    print("\n5. Comparing with standard recommendation approach...")
    standard_recommendations = recommender.recommend(
        user_id=professional_id,
        top_k=10,
        exclude_rated=True
    )
    
    print("\nStandard recommendations:")
    for i, rec in enumerate(standard_recommendations):
        job_id = rec['job_id']
        job_info = jobs_df[jobs_df['job_id'] == job_id].iloc[0]
        print(f"{i+1}. {job_info['title']} (Category: {job_info['category']}, Location: {job_info['location']})")
        print(f"   Score: {rec.get('score', 'N/A')}")
    
    # Compare category match rates between approaches
    category_match_count_professional = sum(1 for rec in recommendations if rec['category_match'])
    category_match_rate_professional = category_match_count_professional / len(recommendations) if recommendations else 0
    
    category_match_count_standard = sum(
        1 for rec in standard_recommendations 
        if jobs_df[jobs_df['job_id'] == rec['job_id']].iloc[0]['category'] in professional_categories
    )
    category_match_rate_standard = category_match_count_standard / len(standard_recommendations) if standard_recommendations else 0
    
    print("\n6. Comparison summary:")
    print(f"   Professional recommendation category match rate: {category_match_rate_professional:.2%}")
    print(f"   Standard recommendation category match rate: {category_match_rate_standard:.2%}")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
