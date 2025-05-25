"""
Utility for generating synthetic data for demos and testing.

This module provides functions to create:
1. Synthetic users with basic profiles
2. Synthetic job listings with titles, descriptions, and categories
3. Synthetic interactions between users and jobs, including ratings and comments
4. Synthetic graph data for testing graph-based models
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, HeteroData
from typing import Tuple, Dict, List, Optional


def generate_users(n_users: int = 100) -> pd.DataFrame:
    """
    Generate synthetic user data.
    
    Args:
        n_users: Number of users to generate
        
    Returns:
        DataFrame containing user data
    """
    locations = ['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida', 'Setif', 'Batna', 'Djelfa']
    
    users = pd.DataFrame({
        'user_id': [f"user_{i}" for i in range(1, n_users+1)],
        'username': [f"User {i}" for i in range(1, n_users+1)],
        'location': np.random.choice(locations, n_users),
        'join_date': pd.date_range(start='2022-01-01', periods=n_users),
        'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_users)
    })
    
    return users


def generate_jobs(n_jobs: int = 200) -> pd.DataFrame:
    """
    Generate synthetic job data.
    
    Args:
        n_jobs: Number of jobs to generate
        
    Returns:
        DataFrame containing job data
    """
    job_categories = ['Plumbing', 'Painting', 'Gardening', 'Assembly', 'Tech Support', 
                      'Cleaning', 'Moving', 'Electrical', 'Tutoring', 'Delivery']
    
    locations = ['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida', 'Setif', 'Batna', 'Djelfa']
    
    # Basic titles for each category
    titles_by_category = {
        'Plumbing': ["Fix leaking sink", "Repair bathroom plumbing", "Install new faucet"],
        'Painting': ["Paint living room", "Paint exterior of house", "Paint bedroom walls"],
        'Gardening': ["Lawn mowing and trimming", "Plant new garden", "Tree pruning"],
        'Assembly': ["Assemble furniture", "Assemble desk", "Assemble bookshelf"],
        'Tech Support': ["Computer repair", "Printer setup", "Home network setup"],
        'Cleaning': ["Deep clean apartment", "Clean windows", "House cleaning"],
        'Moving': ["Help moving heavy furniture", "Moving assistance", "Moving boxes to storage"],
        'Electrical': ["Light fixture installation", "Outlet repair", "Electrical troubleshooting"],
        'Tutoring': ["Math tutoring", "Language lessons", "Programming tutoring"],
        'Delivery': ["Package pickup", "Grocery delivery", "Food delivery"]
    }
    
    # Generate jobs
    jobs = []
    for i in range(1, n_jobs+1):
        category = np.random.choice(job_categories)
        title = np.random.choice(titles_by_category[category])
        description = f"Looking for someone to help with {title.lower()} in my home/office."
        
        jobs.append({
            'job_id': f"job_{i}",
            'title': title,
            'description': description,
            'category': category,
            'location': np.random.choice(locations),
            'posting_date': pd.Timestamp('now') - pd.Timedelta(days=np.random.randint(0, 90)),
            'budget': np.random.randint(1000, 10000)
        })
    
    return pd.DataFrame(jobs)


def generate_interactions(
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    n_interactions: int = 1000,
    preference_strength: float = 0.8
) -> pd.DataFrame:
    """
    Generate synthetic interactions between users and jobs.
    
    Args:
        users_df: DataFrame of users
        jobs_df: DataFrame of jobs
        n_interactions: Number of interactions to generate
        preference_strength: Probability that a user interacts with their preferred job categories
        
    Returns:
        DataFrame containing interaction data
    """
    # Create user preferences for categories
    job_categories = jobs_df['category'].unique()
    user_preferences = {}
    
    for user_id in users_df['user_id']:
        # Each user likes 2-3 categories more than others
        preferred_categories = np.random.choice(job_categories, size=np.random.randint(2, 4), replace=False)
        user_preferences[user_id] = preferred_categories
    
    # Generate positive comment templates
    positive_comments = [
        "Great service, very professional!",
        "Excellent work, completed on time.",
        "Very satisfied with the quality of work.",
        "Highly recommended, will hire again.",
        "Perfect job, exceeded expectations.",
        "The worker was punctual and efficient.",
        "Very skilled and knowledgeable.",
        "Excellent communication and service."
    ]
    
    # Generate neutral comment templates
    neutral_comments = [
        "The job was done as requested.",
        "Acceptable service, but could be better.",
        "Completed the task adequately.",
        "Reasonable quality for the price.",
        "The work was satisfactory.",
        "Got the job done, nothing special.",
        "Average service quality.",
        "Meets basic expectations."
    ]
    
    # Generate negative comment templates
    negative_comments = [
        "Poor service, not recommended.",
        "Did not finish the job properly.",
        "Disappointed with the quality.",
        "Would not hire again.",
        "Work was below expectations.",
        "Late and unprofessional.",
        "Poor communication throughout.",
        "Did not follow instructions."
    ]
    
    # Generate interactions based on preferences
    interactions = []
    
    for _ in range(n_interactions):
        user_id = np.random.choice(users_df['user_id'].values)
        
        # Determine if this interaction is with a preferred category
        if np.random.random() < preference_strength:
            preferred_cats = user_preferences[user_id]
            category = np.random.choice(preferred_cats)
            
            # Filter jobs by category and pick one
            category_jobs = jobs_df[jobs_df['category'] == category]['job_id'].values
            if len(category_jobs) > 0:
                job_id = np.random.choice(category_jobs)
                
                # Higher ratings for preferred categories (4-5 stars)
                rating = np.random.uniform(4.0, 5.0)
                comment = np.random.choice(positive_comments)
            else:
                continue
        else:
            # Random job from non-preferred categories
            non_preferred_cats = [c for c in job_categories if c not in user_preferences[user_id]]
            if not non_preferred_cats:
                continue
                
            category = np.random.choice(non_preferred_cats)
            
            # Filter jobs by category and pick one
            category_jobs = jobs_df[jobs_df['category'] == category]['job_id'].values
            if len(category_jobs) > 0:
                job_id = np.random.choice(category_jobs)
                
                # Lower ratings for non-preferred categories (1-3 stars)
                rating = np.random.uniform(1.0, 3.0)
                
                if rating < 2.0:
                    comment = np.random.choice(negative_comments)
                else:
                    comment = np.random.choice(neutral_comments)
            else:
                continue
        
        interactions.append({
            'user_id': user_id,
            'job_id': job_id,
            'rating': rating,
            'comment': comment,
            'timestamp': pd.Timestamp('now') - pd.Timedelta(days=np.random.randint(0, 30))
        })
    
    return pd.DataFrame(interactions)


def generate_graph_data(n_users: int = 50, n_jobs: int = 100, n_edges: int = 200) -> Data:
    """
    Generate synthetic graph data for testing graph models.
    
    Args:
        n_users: Number of user nodes
        n_jobs: Number of job nodes
        n_edges: Number of user-job interactions
        
    Returns:
        PyTorch Geometric Data object
    """
    # Generate random edges (user-job interactions)
    user_indices = torch.randint(0, n_users, (n_edges,))
    job_indices = torch.randint(0, n_jobs, (n_edges,))
    
    # Create edge weights (ratings)
    edge_weights = torch.rand(n_edges) * 4 + 1  # Ratings between 1 and 5
    
    # Create edge indices in the format expected by PyG
    edge_index = torch.stack([
        user_indices,
        job_indices
    ])
    
    # Create graph data object
    data = Data(
        edge_index=edge_index,
        edge_weight=edge_weights,
        num_users=n_users,
        num_jobs=n_jobs
    )
    
    return data


def generate_hetero_graph_data(n_users: int = 50, n_jobs: int = 100, 
                              n_categories: int = 10, embedding_dim: int = 32) -> HeteroData:
    """
    Generate heterogeneous graph data for testing heterogeneous graph models.
    
    Args:
        n_users: Number of user nodes
        n_jobs: Number of job nodes
        n_categories: Number of category nodes
        embedding_dim: Dimension of node features
        
    Returns:
        PyTorch Geometric HeteroData object
    """
    # Create heterogeneous graph data object
    data = HeteroData()
    
    # Generate node features
    data['user'].x = torch.randn(n_users, embedding_dim)
    data['job'].x = torch.randn(n_jobs, embedding_dim)
    data['category'].x = torch.randn(n_categories, embedding_dim)
    
    # Generate user-job interactions (ratings)
    n_ratings = min(n_users * 3, n_users * n_jobs // 2)  # Each user rates multiple jobs
    user_indices = torch.randint(0, n_users, (n_ratings,))
    job_indices = torch.randint(0, n_jobs, (n_ratings,))
    
    data['user', 'rates', 'job'].edge_index = torch.stack([user_indices, job_indices])
    data['user', 'rates', 'job'].edge_attr = torch.rand(n_ratings) * 4 + 1  # Ratings 1-5
    
    # Generate job-category relationships
    # Each job belongs to one category
    job_indices = torch.arange(n_jobs)
    category_indices = torch.randint(0, n_categories, (n_jobs,))
    
    data['job', 'belongs_to', 'category'].edge_index = torch.stack([job_indices, category_indices])
    
    # Generate category-job relationships (reverse of job-category)
    # Maps from each category to all jobs in that category
    edge_list = []
    for cat_idx in range(n_categories):
        jobs_in_category = torch.where(category_indices == cat_idx)[0]
        if len(jobs_in_category) > 0:
            cat_indices = torch.full((len(jobs_in_category),), cat_idx)
            edge_list.append(torch.stack([cat_indices, jobs_in_category]))
    
    if edge_list:
        data['category', 'has_job', 'job'].edge_index = torch.cat(edge_list, dim=1)
    
    return data


def generate_simple_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate a simple dataset for testing and demos.
    
    Returns:
        Tuple of (users_df, jobs_df, interactions_df)
    """
    users_df = generate_users(n_users=50)
    jobs_df = generate_jobs(n_jobs=100)
    interactions_df = generate_interactions(users_df, jobs_df, n_interactions=200)
    
    return users_df, jobs_df, interactions_df
