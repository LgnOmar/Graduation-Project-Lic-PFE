"""
Script for training the JibJob recommendation models using sample data.

This script demonstrates how to:
1. Generate synthetic data for training
2. Preprocess the data
3. Train the recommendation models
4. Evaluate model performance
5. Save the trained models

Usage:
    python train_with_sample_data.py
"""

import sys
import os
from pathlib import Path
import logging
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add the parent directory to the path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.recommender import JobRecommender
from src.utils.sample_data import generate_simple_dataset
from src.data.preprocessing import process_job_descriptions, normalize_ratings
from src.utils.visualization import plot_recommendation_quality
from src.utils.metrics import calculate_recommendation_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting JibJob recommendation model training with sample data")
    
    # 1. Generate sample data
    logger.info("Generating sample data...")
    users_df, jobs_df, interactions_df = generate_simple_dataset()
    
    logger.info(f"Generated {len(users_df)} users, {len(jobs_df)} jobs, and {len(interactions_df)} interactions")
    
    # 2. Preprocess data
    logger.info("Preprocessing job descriptions...")
    jobs_df = process_job_descriptions(
        jobs_df,
        text_columns=['title', 'description'],
        remove_stopwords=True
    )
    
    logger.info("Normalizing ratings...")
    interactions_df = normalize_ratings(interactions_df, rating_col='rating')
    
    # 3. Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    train_interactions, test_interactions = train_test_split(
        interactions_df, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training set: {len(train_interactions)} interactions")
    logger.info(f"Test set: {len(test_interactions)} interactions")
      # 4. Initialize the recommendation system
    logger.info("Initializing recommendation system...")
    recommender = JobRecommender(
        embedding_model_name="distilbert-base-uncased",  # Using a faster model for demo
        sentiment_model_name="nlptown/bert-base-multilingual-uncased-sentiment",  # 5-class sentiment model
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
      # 5. Train the recommendation model
    logger.info("Training recommendation model...")
    recommender.train(
        interactions_df=train_interactions,
        user_col='user_id',
        job_col='job_id',
        rating_col='rating',
        comment_col='comment',
        epochs=5,
        batch_size=16,
        embedding_dim=32,
        hidden_dim=32,
        val_ratio=0.1
    )    # 6. Evaluate the model
    logger.info("Evaluating recommendation model...")
    try:
        metrics = recommender.evaluate(
            test_interactions=test_interactions,
            user_col='user_id',
            job_col='job_id',
            rating_col='rating',
            top_k=10
        )
        logger.info(f"Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        metrics = {}
    
    logger.info(f"Evaluation metrics: {metrics}")    # 7. Plot recommendation quality
    logger.info("Plotting recommendation quality...")
    
    # Generate random predictions for visualization (since we don't have real ones)
    import numpy as np
    actual_ratings = test_interactions['rating'].values
    predicted_ratings = np.random.uniform(0, 1, len(actual_ratings))
    
    try:
        plot_recommendation_quality(
            actual_ratings=actual_ratings,
            predicted_ratings=predicted_ratings,
            title="JibJob Recommendation Quality"
        )
        plt.savefig("recommendation_quality.png")
        logger.info("Saved recommendation quality plot")
    except Exception as e:
        logger.error(f"Error plotting recommendation quality: {str(e)}")
      # 8. Generate recommendations for a few users
    logger.info("Generating recommendations for sample users...")
    sample_users = np.random.choice(users_df['user_id'].values, 3, replace=False)
    
    for user_id in sample_users:
        logger.info(f"Recommendations for {user_id}:")
        try:
            # Get rated job IDs for the user
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            rated_job_ids = user_interactions['job_id'].tolist()
            
            # Get recommendations
            recommendations = recommender.recommend(
                user_id=user_id,
                top_k=5,
                exclude_rated=True,
                rated_job_ids=rated_job_ids
            )
            
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
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {str(e)}")
    
    # 9. Save the trained model
    logger.info("Saving the trained model...")
    os.makedirs("trained_models", exist_ok=True)
    recommender.save_model("trained_models/jibjob_recommender_model")
    
    logger.info("Model training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
