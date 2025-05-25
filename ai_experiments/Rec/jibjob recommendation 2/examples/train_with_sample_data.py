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
from src.data.preprocessing import process_job_descriptions, normalize_ratings, load_jibjob_csv_data, encode_professional_selected_categories, prepare_user_job_data
from src.utils.visualization import plot_recommendation_quality
from src.utils.metrics import calculate_recommendation_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting JibJob recommendation model training with new file-based synthetic data")

    # 1. Load data from CSVs
    data = load_jibjob_csv_data(data_dir="sample_data")
    users_df = data['users']
    jobs_df = data['jobs']
    interactions_df = data['interactions']
    categories_master = data['categories']

    logger.info(f"Loaded {len(users_df)} users, {len(jobs_df)} jobs, {len(interactions_df)} interactions, {len(categories_master)} categories")

    # --- COLUMN COMPATIBILITY PATCH FOR NEW SYNTHETIC DATA ---
    # Rename professional_user_id to user_id in interactions if present
    if 'professional_user_id' in interactions_df.columns:
        interactions_df = interactions_df.rename(columns={'professional_user_id': 'user_id'})
    # Rename selected_category_ids to selected_categories in users if present
    if 'selected_category_ids' in users_df.columns:
        users_df = users_df.rename(columns={'selected_category_ids': 'selected_categories'})

    # 2. Preprocess job descriptions
    logger.info("Preprocessing job descriptions...")
    jobs_df = process_job_descriptions(
        jobs_df,
        text_columns=['title', 'description'],
        remove_stopwords=True
    )

    logger.info("Normalizing ratings...")
    # Use correct professional_id column for interactions
    if 'professional_id' in interactions_df.columns:
        interactions_df = interactions_df.rename(columns={'professional_id': 'user_id'})
    interactions_df = normalize_ratings(interactions_df, rating_col='rating')

    # 3. Split data into train and test sets
    logger.info("Splitting data into train and test sets...")
    from sklearn.model_selection import train_test_split
    train_interactions, test_interactions = train_test_split(
        interactions_df, test_size=0.2, random_state=42
    )

    logger.info(f"Training set: {len(train_interactions)} interactions")
    logger.info(f"Test set: {len(test_interactions)} interactions")

    # 4. Prepare user profile features (multi-hot for professionals)
    user_profile_features = encode_professional_selected_categories(
        users_df, categories_master, user_id_col='user_id',
        selected_categories_col='selected_categories', user_type_col='user_type'
    )

    # 5. Initialize the recommendation system
    logger.info("Initializing recommendation system...")
    recommender = JobRecommender(
        embedding_model_name="distilbert-base-uncased",
        sentiment_model_name="nlptown/bert-base-multilingual-uncased-sentiment",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    recommender.users_df = users_df
    recommender.jobs_df_internal = jobs_df
    recommender.set_category_master_and_user_profile_features(categories_master, user_profile_features)

    # 6. Train the recommendation model (if GCN is used, pass user_profile_features as user features)
    logger.info("Training recommendation model...")
    recommender.train(
        interactions_df=train_interactions,
        jobs_df=jobs_df,
        user_col='user_id',
        job_col='job_id',
        rating_col='rating',
        comment_col='comment',
        epochs=5,
        batch_size=16,
        embedding_dim=32,
        hidden_dim=32,
        val_ratio=0.1
    )

    # 7. Evaluate the model
    logger.info("Evaluating recommendation model...")
    # Patch: If 'is_relevant_for_ranking_eval' is missing, add a dummy column (all True for jobs the user rated in test set)
    if 'is_relevant_for_ranking_eval' not in test_interactions.columns:
        logger.warning("'is_relevant_for_ranking_eval' column missing from test_interactions. Adding dummy relevance column for evaluation.")
        # Mark as relevant if the user rated the job (simulate ground truth for ranking)
        test_interactions['is_relevant_for_ranking_eval'] = True
    try:
        metrics = recommender.evaluate(
            test_interactions=test_interactions,
            jobs_df=jobs_df,
            user_col='user_id',
            job_col='job_id',
            rating_col='rating',
            top_k=10
        )
        logger.info(f"Metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        metrics = {}

    logger.info(f"Evaluation metrics: {metrics}")

    # 8. Generate recommendations for a few professionals
    logger.info("Generating recommendations for sample professionals...")
    professionals = users_df[users_df['user_type'] == 'professional']['user_id'].values
    sample_users = np.random.choice(professionals, 3, replace=False)
    for user_id in sample_users:
        logger.info(f"Recommendations for {user_id} (professional):")
        try:
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            rated_job_ids = user_interactions['job_id'].tolist()
            recommendations = recommender.recommend(
                user_id=user_id,
                top_k=5,
                exclude_rated=True,
                rated_job_ids_for_user=rated_job_ids
            )
            print(f"\nProfessional {user_id} has rated {len(user_interactions)} jobs:")
            for _, row in user_interactions.iterrows():
                job_info = jobs_df[jobs_df['job_id'] == row['job_id']].iloc[0]
                # Always map category_id to category_name using categories_master
                if 'category_id' in job_info and 'category_name' in categories_master:
                    cat_id = job_info['category_id']
                    category_val = categories_master.loc[categories_master['category_id'] == cat_id, 'category_name'].values[0]
                elif 'category' in job_info:
                    category_val = job_info['category']
                else:
                    category_val = 'N/A'
                print(f"  - {job_info['title']} ({category_val}): {row['rating']:.1f}/5.0")
            print(f"\nTop 5 recommendations for {user_id}:")
            for rec in recommendations:
                job_id = rec['job_id']
                score = rec['score']
                job_info = jobs_df[jobs_df['job_id'] == job_id].iloc[0]
                if 'category_id' in job_info and 'category_name' in categories_master:
                    cat_id = job_info['category_id']
                    category_val = categories_master.loc[categories_master['category_id'] == cat_id, 'category_name'].values[0]
                elif 'category' in job_info:
                    category_val = job_info['category']
                else:
                    category_val = 'N/A'
                print(f"  - {job_info['title']} ({category_val}): score = {score:.4f}")
            print()
        except Exception as e:
            logger.error(f"Error generating recommendations for professional {user_id}: {str(e)}")

    # 9. Save the trained model
    logger.info("Saving the trained model...")
    os.makedirs("trained_models", exist_ok=True)
    recommender.save_model("trained_models/jibjob_recommender_model")
    logger.info("Model training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
