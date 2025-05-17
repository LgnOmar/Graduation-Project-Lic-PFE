"""
Module for generating and saving simulated data for the JibJob recommendation system.
"""
import pandas as pd
import numpy as np
from typing import Tuple

def generate_sample_data(
    n_users: int = 1000,
    n_jobs: int = 500,
    n_interactions: int = 2000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate sample data for users, jobs, and their interactions.
    
    Args:
        n_users: Number of users to generate
        n_jobs: Number of jobs to generate
        n_interactions: Number of user-job interactions to generate
    
    Returns:
        Tuple of (users_df, jobs_df, interactions_df)
    """
    # Generate users
    users_df = pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(n_users)],
        'description_profil_utilisateur_anglais': [
            f'Profile description for user {i}' for i in range(n_users)
        ]
    })
    
    # Generate jobs
    categories = ['Cleaning', 'Gardening', 'Moving', 'Painting', 'Teaching', 'Pet Care']
    jobs_df = pd.DataFrame({
        'job_id': [f'job_{i}' for i in range(n_jobs)],
        'description_mission_anglais': [
            f'Detailed job description for job {i}' for i in range(n_jobs)
        ],
        'categorie_mission': np.random.choice(categories, size=n_jobs)
    })
    
    # Generate interactions
    user_ids = np.random.choice(users_df['user_id'], size=n_interactions)
    job_ids = np.random.choice(jobs_df['job_id'], size=n_interactions)
    
    interactions_df = pd.DataFrame({
        'user_id': user_ids,
        'job_id': job_ids,
        'rating_explicite': np.random.choice(
            [np.nan] + list(range(1, 6)),
            size=n_interactions,
            p=[0.3, 0.1, 0.15, 0.15, 0.15, 0.15]
        ),
        'commentaire_texte_anglais': [
            np.random.choice([np.nan, f'Comment for job {i}'], p=[0.7, 0.3])
            for i in range(n_interactions)
        ]
    })
    
    return users_df, jobs_df, interactions_df

def save_dataframes(
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    output_dir: str = 'data'
) -> None:
    """
    Save the generated DataFrames to CSV files.
    
    Args:
        users_df: DataFrame containing user information
        jobs_df: DataFrame containing job information
        interactions_df: DataFrame containing user-job interactions
        output_dir: Directory to save the CSV files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    users_df.to_csv(f'{output_dir}/users_df.csv', index=False)
    jobs_df.to_csv(f'{output_dir}/jobs_df.csv', index=False)
    interactions_df.to_csv(f'{output_dir}/interactions_df.csv', index=False)

if __name__ == '__main__':
    users_df, jobs_df, interactions_df = generate_sample_data()
    save_dataframes(users_df, jobs_df, interactions_df)
