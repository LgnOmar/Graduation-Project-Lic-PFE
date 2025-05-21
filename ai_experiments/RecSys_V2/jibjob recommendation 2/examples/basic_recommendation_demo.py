"""
Basic demonstration of the JibJob recommendation system.

This script demonstrates how to:
1. Initialize the recommendation system
2. Load and preprocess data
3. Train the recommendation model
4. Generate recommendations for users
5. Find similar jobs
6. Analyze sentiment in comments

Usage:
    python basic_recommendation_demo.py
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.recommender import JobRecommender
from src.data.preprocessing import process_job_descriptions, normalize_ratings
from src.utils.visualization import plot_recommendation_quality


# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data(n_users=50, n_jobs=100, n_interactions=500):
    """Generate sample data for demonstration purposes."""
    logger.info(f"Generating sample data with {n_users} users, {n_jobs} jobs, and {n_interactions} interactions")
    
    # Create sample users
    users = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(1, n_users+1)],
        'username': [f"User {i}" for i in range(1, n_users+1)],
        'location': np.random.choice(['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida'], n_users)
    })
    
    # Create sample jobs
    job_categories = ['Plumbing', 'Painting', 'Gardening', 'Assembly', 'Tech Support', 
                      'Cleaning', 'Moving', 'Electrical', 'Tutoring', 'Delivery']
    
    job_titles = [
        # Plumbing titles
        "Fix leaking sink", "Repair bathroom plumbing", "Install new faucet",
        "Fix water heater", "Unclog toilet", "Fix pipe leak", "Install new shower",
        
        # Painting titles
        "Paint living room", "Paint exterior of house", "Paint bedroom walls",
        "Repaint kitchen cabinets", "Paint fence", "Wallpaper installation",
        
        # Gardening titles
        "Lawn mowing and trimming", "Plant new garden", "Tree pruning",
        "Garden maintenance", "Weed removal", "Build raised garden bed",
        
        # Assembly titles
        "Assemble furniture", "Assemble desk", "Assemble bookshelf",
        "Assemble bed frame", "Assemble exercise equipment", "Assemble kitchen cabinets",
        
        # Tech Support titles
        "Computer repair", "Printer setup", "Home network setup",
        "Smartphone troubleshooting", "Software installation", "Smart home device setup",
        
        # Cleaning titles
        "Deep clean apartment", "Clean windows", "House cleaning",
        "Office cleaning", "Carpet cleaning", "Post-construction cleanup",
        
        # Moving titles
        "Help moving heavy furniture", "Apartment moving assistance", "Moving boxes to storage",
        "Pack and move items", "Furniture rearrangement", "Help with moving truck",
        
        # Electrical titles
        "Light fixture installation", "Outlet repair", "Ceiling fan installation",
        "Electrical troubleshooting", "Install light switch", "Wiring repair",
        
        # Tutoring titles
        "Math tutoring", "Language lessons", "Programming tutoring",
        "Physics homework help", "Chemistry tutoring", "Essay writing assistance",
        
        # Delivery titles
        "Package pickup", "Grocery delivery", "Food delivery",
        "Medication pickup", "Deliver documents", "Furniture delivery"
    ]
    
    # Create jobs with categories
    job_data = []
    for i in range(1, n_jobs+1):
        category_idx = (i - 1) % len(job_categories)
        title_idx = (i - 1) % len(job_titles)
        
        category = job_categories[category_idx]
        title = job_titles[title_idx]
        description = f"Looking for someone to help with {title.lower()} in my home/office."
        
        job_data.append({
            'job_id': f"job_{i}",
            'title': title,
            'description': description,
            'category': category,
            'location': np.random.choice(['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida'])
        })
    
    jobs = pd.DataFrame(job_data)
    
    # Create sample interactions with some patterns to learn from
    interaction_data = []
    
    # Create user preferences for categories
    user_preferences = {}
    for user_id in users['user_id']:
        # Each user likes 2-3 categories more than others
        preferred_categories = np.random.choice(job_categories, size=np.random.randint(2, 4), replace=False)
        user_preferences[user_id] = preferred_categories
    
    # Generate interactions based on preferences
    for _ in range(n_interactions):
        user_id = np.random.choice(users['user_id'])
        
        # 80% of interactions are with preferred categories
        if np.random.random() < 0.8:
            preferred_cats = user_preferences[user_id]
            category = np.random.choice(preferred_cats)
            
            # Filter jobs by category and pick one
            category_jobs = jobs[jobs['category'] == category]['job_id'].values
            job_id = np.random.choice(category_jobs)
            
            # Higher ratings for preferred categories (4-5 stars)
            rating = np.random.uniform(4.0, 5.0)
            sentiment = "positive"
        else:
            # Random job from non-preferred categories
            non_preferred_cats = [c for c in job_categories if c not in user_preferences[user_id]]
            category = np.random.choice(non_preferred_cats)
            
            # Filter jobs by category and pick one
            category_jobs = jobs[jobs['category'] == category]['job_id'].values
            job_id = np.random.choice(category_jobs)
            
            # Lower ratings for non-preferred categories (1-3 stars)
            rating = np.random.uniform(1.0, 3.0)
            sentiment = "negative" if rating < 2.0 else "neutral"
        
        # Generate a comment based on rating
        if sentiment == "positive":
            comments = [
                "Great service, very professional!",
                "Excellent work, completed on time.",
                "Very satisfied with the quality of work.",
                "Highly recommended, will hire again.",
                "Perfect job, exceeded expectations."
            ]
            comment = np.random.choice(comments)
        elif sentiment == "neutral":
            comments = [
                "The job was done as requested.",
                "Acceptable service, but could be better.",
                "Completed the task adequately.",
                "Reasonable quality for the price.",
                "The work was satisfactory."
            ]
            comment = np.random.choice(comments)
        else:
            comments = [
                "Poor service, not recommended.",
                "Did not finish the job properly.",
                "Disappointed with the quality.",
                "Would not hire again.",
                "Work was below expectations."
            ]
            comment = np.random.choice(comments)
        
        interaction_data.append({
            'user_id': user_id,
            'job_id': job_id,
            'rating': rating,
            'comment': comment
        })
    
    # Convert to DataFrame
    interactions = pd.DataFrame(interaction_data)
    
    return users, jobs, interactions


def main():
    # Generate sample data
    users_df, jobs_df, interactions_df = generate_sample_data()
    
    logger.info(f"Generated {len(users_df)} users, {len(jobs_df)} jobs, and {len(interactions_df)} interactions")
    
    # Initialize the recommendation system
    logger.info("Initializing recommendation system...")
    recommender = JobRecommender(
        embedding_model_name="distilbert-base-uncased",  # Using a faster model for demo
        sentiment_model_name="nlptown/bert-base-multilingual-uncased-sentiment",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Process job descriptions
    logger.info("Processing job descriptions...")
    jobs_df = process_job_descriptions(
        jobs_df,
        text_columns=['title', 'description'],
        remove_stopwords=True
    )
    
    # Split data into train and test sets (80/20)
    from sklearn.model_selection import train_test_split
    train_interactions, test_interactions = train_test_split(
        interactions_df, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {len(train_interactions)} interactions")
    logger.info(f"Test set: {len(test_interactions)} interactions")
    
    # Train the recommendation model
    logger.info("Training recommendation model...")
    recommender.train(
        interactions_df=train_interactions,
        epochs=5,
        batch_size=64,
        embedding_dim=32,
        hidden_dim=32
    )
    
    # Evaluate the model
    logger.info("Evaluating recommendation model...")
    metrics = recommender.evaluate(
        test_interactions=test_interactions,
        top_k=10
    )
    
    logger.info(f"Evaluation metrics: {metrics}")
    
    # Plot recommendation quality
    logger.info("Plotting recommendation quality...")
    plot_recommendation_quality(
        actual_ratings=metrics['actual_ratings'],
        predicted_ratings=metrics['predicted_ratings'],
        title="JibJob Recommendation Quality"
    )
    plt.savefig("recommendation_quality.png")
    
    # Generate recommendations for a few users
    logger.info("Generating recommendations for sample users...")
    sample_users = np.random.choice(users_df['user_id'].values, 5, replace=False)
    
    for user_id in sample_users:
        logger.info(f"Recommendations for {user_id}:")
        # Get jobs the user has already rated
        rated_job_ids = interactions_df[interactions_df['user_id'] == user_id]['job_id'].tolist() if True else None
        recommendations = recommender.recommend(
            user_id=user_id,
            top_k=5,
            exclude_rated=True,
            rated_job_ids=rated_job_ids
        )
        # Get the user's interactions
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        print(f"\nUser {user_id} has rated {len(user_interactions)} jobs:")
        for _, row in user_interactions.iterrows():
            job_info = jobs_df[jobs_df['job_id'] == row['job_id']].iloc[0]
            print(f"  - {job_info['title']} ({job_info['category']}): {row['rating']:.1f}/5.0")
        print(f"\nTop 5 recommendations for {user_id}:")
        for rec in recommendations:
            job_id = rec['job_id']
            score = rec['score']
            job_info = jobs_df[jobs_df['job_id'] == job_id].iloc[0]
            print(f"  - {job_info['title']} ({job_info['category']}): score = {score:.4f}")
        print()
    
    # Find similar jobs for a sample job
    sample_job = np.random.choice(jobs_df['job_id'].values)
    sample_job_info = jobs_df[jobs_df['job_id'] == sample_job].iloc[0]
    
    logger.info(f"Finding similar jobs to {sample_job}: {sample_job_info['title']}")
    similar_jobs = recommender.find_similar_jobs(sample_job, top_k=5)
    
    print(f"\nSimilar jobs to {sample_job_info['title']} ({sample_job_info['category']}):")
    for job_id, similarity in similar_jobs:
        job_info = jobs_df[jobs_df['job_id'] == job_id].iloc[0]
        print(f"  - {job_info['title']} ({job_info['category']}): similarity = {similarity:.4f}")
    
    # Analyze sentiment in some comments
    logger.info("Analyzing sentiment in sample comments...")
    sample_comments = [
        "The service was excellent and completed on time!",
        "The job was done but not to my satisfaction.",
        "Terrible service, would not recommend.",
        "Very professional and skilled worker."
    ]
    
    print("\nSentiment Analysis:")
    for comment in sample_comments:
        sentiment = recommender.sentiment_model.analyze_sentiment(comment)
        print(f"  - '{comment}': {sentiment:.4f}")
    
    # Save the trained model
    logger.info("Saving the trained model...")
    recommender.save_model("jibjob_recommender_model")
    
    logger.info("Demo completed successfully!")


if __name__ == "__main__":
    import torch
    main()
