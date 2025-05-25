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


def generate_and_save_full_dataset(
    n_professionals: int = 50,
    n_clients: int = 20,
    n_jobs: int = 100,
    n_interactions: int = 300,
    output_dir: str = "sample_data"
):
    """
    Generate and save a full synthetic dataset with the required schema and strong profile-category/location correlation.
    Creates users.csv, jobs.csv, interactions.csv, job_categories_master_list.csv in output_dir.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Canonical job categories
    job_categories = [
        'Plumbing', 'Painting', 'Gardening', 'Assembly', 'Tech Support',
        'Cleaning', 'Moving', 'Electrical', 'Tutoring', 'Delivery'
    ]
    job_categories_master = pd.DataFrame({'category_name': job_categories})
    job_categories_master.to_csv(os.path.join(output_dir, 'job_categories_master_list.csv'), index=False)

    # Generate professionals
    locations = ['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida', 'Setif', 'Batna', 'Djelfa']
    professionals = []
    for i in range(1, n_professionals + 1):
        selected_cats = np.random.choice(job_categories, size=np.random.randint(1, 4), replace=False)
        professionals.append({
            'user_id': f'pro_{i}',
            'location': np.random.choice(locations),
            'user_type': 'professional',
            'selected_categories': ';'.join(selected_cats)
        })
    professionals_df = pd.DataFrame(professionals)

    # Generate clients
    clients = []
    for i in range(1, n_clients + 1):
        clients.append({
            'user_id': f'client_{i}',
            'location': np.random.choice(locations),
            'user_type': 'client',
            'selected_categories': ''
        })
    clients_df = pd.DataFrame(clients)

    users_df = pd.concat([professionals_df, clients_df], ignore_index=True)
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)

    # Generate jobs (all posted by clients)
    jobs = []
    for i in range(1, n_jobs + 1):
        category = np.random.choice(job_categories)
        location = np.random.choice(locations)
        posted_by_client_id = np.random.choice(clients_df['user_id'])
        jobs.append({
            'job_id': f'job_{i}',
            'title': f"{category} job #{i}",
            'description': f"Need help with {category.lower()} at {location}.",
            'category': category,
            'location': location,
            'posted_by_client_id': posted_by_client_id
        })
    jobs_df = pd.DataFrame(jobs)
    jobs_df.to_csv(os.path.join(output_dir, 'jobs.csv'), index=False)

    # Generate interactions: only professionals interact, and ratings correlate with profile match
    interactions = []
    for _ in range(n_interactions):
        pro = professionals_df.sample(1).iloc[0]
        pro_id = pro['user_id']
        pro_cats = set(pro['selected_categories'].split(';'))
        pro_loc = pro['location']
        # 80%: match both category and location, 10%: match category only, 10%: random
        p = np.random.rand()
        if p < 0.8:
            # Find jobs matching both
            candidates = jobs_df[(jobs_df['category'].isin(pro_cats)) & (jobs_df['location'].str.lower() == pro_loc.lower())]
            if not candidates.empty:
                job = candidates.sample(1).iloc[0]
                rating = np.random.uniform(4.0, 5.0)
            else:
                job = jobs_df.sample(1).iloc[0]
                rating = np.random.uniform(2.0, 3.5)
        elif p < 0.9:
            # Match category only
            candidates = jobs_df[jobs_df['category'].isin(pro_cats)]
            if not candidates.empty:
                job = candidates.sample(1).iloc[0]
                rating = np.random.uniform(3.0, 4.2)
            else:
                job = jobs_df.sample(1).iloc[0]
                rating = np.random.uniform(2.0, 3.5)
        else:
            # Random
            job = jobs_df.sample(1).iloc[0]
            rating = np.random.uniform(1.0, 3.5)
        interactions.append({
            'professional_id': pro_id,
            'job_id': job['job_id'],
            'rating': round(rating, 2)
        })
    interactions_df = pd.DataFrame(interactions)
    interactions_df.to_csv(os.path.join(output_dir, 'interactions.csv'), index=False)

    print(f"Synthetic dataset saved to {output_dir}/ (users.csv, jobs.csv, interactions.csv, job_categories_master_list.csv)")


def generate_master_lists(
    n_categories: int = 100,
    n_cities: int = 20,
    districts_per_city: int = 3,
    output_dir: str = "sample_data"
):
    """
    Generate and save master lists for job categories and locations.
    - job_categories_master_list.csv: category_id, category_name
    - locations_master_list.csv: location_id, city_name, district_name
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Generate granular job categories
    base_categories = [
        "Plumbing", "Painting", "Gardening", "Assembly", "Tech Support", "Cleaning", "Moving", "Electrical", "Tutoring", "Delivery",
        "Carpentry", "Roofing", "Flooring", "Locksmith", "Pest Control", "Landscaping", "HVAC", "Auto Repair", "Photography", "Translation",
        "Writing", "Graphic Design", "Web Development", "App Development", "Data Entry", "Accounting", "Legal Advice", "Marketing", "Event Planning", "Catering",
        "Pet Care", "Childcare", "Elderly Care", "Fitness Training", "Music Lessons", "Language Lessons", "Math Tutoring", "Science Tutoring", "Test Prep", "Resume Writing",
        "Career Coaching", "Business Consulting", "Interior Design", "Home Staging", "Sewing", "Tailoring", "Laundry", "Grocery Shopping", "Personal Assistant", "Security"
    ]
    # Expand with sub-specialties
    granular_categories = []
    for i, base in enumerate(base_categories):
        if base in ["Plumbing", "Painting", "Gardening", "Tutoring", "Delivery", "Cleaning", "Electrical", "Assembly"]:
            granular_categories.append(f"Residential {base}")
            granular_categories.append(f"Commercial {base}")
        elif base == "Tutoring":
            granular_categories.append("Mathematics Tutoring - High School")
            granular_categories.append("Mathematics Tutoring - University")
            granular_categories.append("French Language Tutoring")
            granular_categories.append("English Language Tutoring")
        elif base == "Delivery":
            granular_categories.append("Local Food Delivery")
            granular_categories.append("Parcel Delivery")
        elif base == "Assembly":
            granular_categories.append("Furniture Assembly - IKEA")
            granular_categories.append("Furniture Assembly - Custom")
        else:
            granular_categories.append(base)
    # Ensure we have at least n_categories
    while len(granular_categories) < n_categories:
        granular_categories.append(f"Special Service {len(granular_categories)+1}")
    granular_categories = granular_categories[:n_categories]
    job_categories_master = pd.DataFrame({
        'category_id': range(1, n_categories+1),
        'category_name': granular_categories
    })
    job_categories_master.to_csv(os.path.join(output_dir, 'job_categories_master_list.csv'), index=False)

    # Generate locations (city + district)
    city_names = [
        "Algiers", "Oran", "Constantine", "Annaba", "Blida", "Setif", "Batna", "Djelfa", "Tlemcen", "Bejaia",
        "Tizi Ouzou", "Sidi Bel Abbes", "Biskra", "Skikda", "Mostaganem", "Boumerdes", "El Oued", "Tiaret", "Bechar", "Ghardaia"
    ][:n_cities]
    locations = []
    location_id = 1
    for city in city_names:
        for d in range(1, districts_per_city+1):
            district = f"{city} District {d}"
            locations.append({
                'location_id': location_id,
                'city_name': city,
                'district_name': district
            })
            location_id += 1
    locations_master = pd.DataFrame(locations)
    locations_master.to_csv(os.path.join(output_dir, 'locations_master_list.csv'), index=False)
    print(f"Master lists saved to {output_dir}/ (job_categories_master_list.csv, locations_master_list.csv)")


def generate_and_save_full_dataset_v2(
    n_professionals: int = 8000,
    n_clients: int = 2000,
    n_jobs: int = 20000,
    n_interactions: int = 100000,
    output_dir: str = "sample_data"
):
    """
    Generate and save a full synthetic dataset using master lists (IDs only).
    - users.csv: user_id, user_type, location_id, selected_category_ids
    - jobs.csv: job_id, title, description, category_id, location_id, posted_by_user_id, posting_date
    - interactions.csv: professional_user_id, job_id, rating, application_status, interaction_timestamp
    """
    import os
    import random
    from datetime import datetime, timedelta
    os.makedirs(output_dir, exist_ok=True)

    # Load master lists
    categories = pd.read_csv(os.path.join(output_dir, 'job_categories_master_list.csv'))
    locations = pd.read_csv(os.path.join(output_dir, 'locations_master_list.csv'))
    n_categories = len(categories)
    n_locations = len(locations)

    # --- USERS ---
    professionals = []
    for i in range(1, n_professionals+1):
        user_id = f"prof_{i:04d}"
        # Cluster: pick a main category, then sample 1-5 in same cluster
        main_cat = random.randint(1, n_categories)
        cluster = [main_cat]
        # Add up to 4 more, close in ID (simulate related skills)
        for _ in range(random.randint(0, 4)):
            offset = random.choice([-2, -1, 1, 2, 3, -3])
            new_cat = main_cat + offset
            if 1 <= new_cat <= n_categories:
                cluster.append(new_cat)
        cluster = list(sorted(set(cluster)))
        selected_category_ids = ";".join(str(cid) for cid in cluster)
        # More users in first 5 cities
        if i <= int(n_professionals*0.5):
            loc_id = random.randint(1, 5*3)  # 5 cities × 3 districts
        else:
            loc_id = random.randint(1, n_locations)
        professionals.append({
            'user_id': user_id,
            'user_type': 'professional',
            'location_id': loc_id,
            'selected_category_ids': selected_category_ids
        })
    clients = []
    for i in range(1, n_clients+1):
        user_id = f"client_{i:04d}"
        if i <= int(n_clients*0.5):
            loc_id = random.randint(1, 5*3)
        else:
            loc_id = random.randint(1, n_locations)
        clients.append({
            'user_id': user_id,
            'user_type': 'client',
            'location_id': loc_id,
            'selected_category_ids': ''
        })
    users_df = pd.DataFrame(professionals + clients)
    users_df.to_csv(os.path.join(output_dir, 'users.csv'), index=False)

    # --- JOBS ---
    jobs = []
    for i in range(1, n_jobs+1):
        job_id = f"job_{i:05d}"
        category_id = random.randint(1, n_categories)
        location_id = random.randint(1, n_locations)
        posted_by_user_id = f"client_{random.randint(1, n_clients):04d}"
        cat_name = categories.loc[categories['category_id'] == category_id, 'category_name'].values[0]
        loc_row = locations.loc[locations['location_id'] == location_id].iloc[0]
        city = loc_row['city_name']
        district = loc_row['district_name']
        # Title/desc with keywords
        title = f"{cat_name} needed in {district}, {city}"
        description = f"Looking for a professional skilled in {cat_name.lower()} for a job in {district}, {city}. Immediate start preferred."
        posting_date = datetime.now() - timedelta(days=random.randint(0, 90))
        jobs.append({
            'job_id': job_id,
            'title': title,
            'description': description,
            'category_id': category_id,
            'location_id': location_id,
            'posted_by_user_id': posted_by_user_id,
            'posting_date': posting_date.strftime('%Y-%m-%d %H:%M:%S')
        })
    jobs_df = pd.DataFrame(jobs)
    jobs_df.to_csv(os.path.join(output_dir, 'jobs.csv'), index=False)

    # --- INTERACTIONS ---
    application_statuses = [
        'viewed', 'applied', 'hired', 'completed_positive', 'completed_negative'
    ]
    interactions = []
    for _ in range(n_interactions):
        prof_row = professionals[random.randint(0, n_professionals-1)]
        prof_id = prof_row['user_id']
        prof_loc = prof_row['location_id']
        prof_cats = [int(cid) for cid in prof_row['selected_category_ids'].split(';')]
        # 80%: match both category and location, 10%: match category only, 10%: random
        p = random.random()
        if p < 0.8:
            candidates = jobs_df[(jobs_df['category_id'].isin(prof_cats)) & (jobs_df['location_id'] == prof_loc)]
            if not candidates.empty:
                job = candidates.sample(1).iloc[0]
                status = random.choices(
                    ['applied', 'hired', 'completed_positive'], [0.5, 0.2, 0.3]
                )[0]
                rating = round(random.uniform(4.0, 5.0), 2) if status == 'completed_positive' else round(random.uniform(3.0, 4.5), 2)
            else:
                job = jobs_df.sample(1).iloc[0]
                status = 'viewed'
                rating = round(random.uniform(2.0, 3.5), 2)
        elif p < 0.9:
            candidates = jobs_df[jobs_df['category_id'].isin(prof_cats)]
            if not candidates.empty:
                job = candidates.sample(1).iloc[0]
                status = random.choices(
                    ['applied', 'hired', 'completed_negative'], [0.6, 0.2, 0.2]
                )[0]
                rating = round(random.uniform(2.0, 4.2), 2) if status == 'completed_negative' else round(random.uniform(3.0, 4.0), 2)
            else:
                job = jobs_df.sample(1).iloc[0]
                status = 'viewed'
                rating = round(random.uniform(2.0, 3.5), 2)
        else:
            job = jobs_df.sample(1).iloc[0]
            status = random.choices(
                ['viewed', 'applied', 'completed_negative'], [0.7, 0.2, 0.1]
            )[0]
            rating = round(random.uniform(1.0, 3.5), 2)
        timestamp = datetime.now() - timedelta(days=random.randint(0, 90), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        interactions.append({
            'professional_user_id': prof_id,
            'job_id': job['job_id'],
            'rating': rating,
            'application_status': status,
            'interaction_timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })
    interactions_df = pd.DataFrame(interactions)
    interactions_df.to_csv(os.path.join(output_dir, 'interactions.csv'), index=False)
    print(f"Full synthetic dataset saved to {output_dir}/ (users.csv, jobs.csv, interactions.csv)")
