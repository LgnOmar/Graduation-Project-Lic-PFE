import pandas as pd
import json
import os
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Output directory
output_dir = './sample_data/'
os.makedirs(output_dir, exist_ok=True)

# Create locations data
n_locations = 50
locations = [
    {
        'location_id': f'loc_{i:03d}',
        'post_code': f'{np.random.randint(10000, 99999):05d}',
        'name': f'City {i}',
        'wilaya_name': f'Region {i//10}',
        'longitude': float(np.random.uniform(-180, 180)),
        'latitude': float(np.random.uniform(-90, 90))
    } for i in range(n_locations)
]
with open(os.path.join(output_dir, 'locations.json'), 'w') as f:
    json.dump(locations, f)

# Create categories
n_categories = 20
categories = pd.DataFrame({
    'category_id': [f'cat_{i:03d}' for i in range(n_categories)],
    'category_name': [f'Category {i}' for i in range(n_categories)]
})
categories.to_csv(os.path.join(output_dir, 'categories.csv'), index=False)

# Create users
n_professionals = 100
n_clients = 50
users = pd.DataFrame({
    'user_id': [f'prof_{i:03d}' for i in range(n_professionals)] + [f'client_{i:03d}' for i in range(n_clients)],
    'username': [f'pro_user_{i}' for i in range(n_professionals)] + [f'client_user_{i}' for i in range(n_clients)],
    'user_type': ['professional'] * n_professionals + ['client'] * n_clients,
    'location_id': [f'loc_{np.random.randint(0, n_locations):03d}' for _ in range(n_professionals + n_clients)],
    'profile_bio': [f'Professional with skills in categories {np.random.choice(n_categories, 3, replace=False)}' for _ in range(n_professionals)] + 
                  [f'Client looking for services in categories {np.random.choice(n_categories, 2, replace=False)}' for _ in range(n_clients)]
})
users.to_csv(os.path.join(output_dir, 'users.csv'), index=False)

# Create professional_categories relations
avg_categories_per_professional = 3
professional_categories = []
for i in range(n_professionals):
    n_categories_for_prof = np.random.randint(1, 6)
    categories_for_prof = np.random.choice(n_categories, size=n_categories_for_prof, replace=False)
    for cat in categories_for_prof:
        professional_categories.append({
            'user_id': f'prof_{i:03d}',
            'category_id': f'cat_{cat:03d}'
        })
professional_categories_df = pd.DataFrame(professional_categories)
professional_categories_df.to_csv(os.path.join(output_dir, 'professional_categories.csv'), index=False)

# Create jobs
n_jobs = 200
jobs = pd.DataFrame({
    'job_id': [f'job_{i:03d}' for i in range(n_jobs)],
    'title': [f'Job title {i}' for i in range(n_jobs)],
    'description': [f'Detailed description for job {i} requiring skills in category {np.random.randint(0, n_categories)}' for i in range(n_jobs)],
    'location_id': [f'loc_{np.random.randint(0, n_locations):03d}' for _ in range(n_jobs)],
    'posted_by_user_id': [f'client_{np.random.randint(0, n_clients):03d}' for _ in range(n_jobs)],
    'required_category_id': [f'cat_{np.random.randint(0, n_categories):03d}' for _ in range(n_jobs)]
})
jobs.to_csv(os.path.join(output_dir, 'jobs.csv'), index=False)

# Create job_applications
n_applications = 300
start_date = datetime.now() - timedelta(days=90)
job_applications = pd.DataFrame({
    'application_id': [f'app_{i:03d}' for i in range(n_applications)],
    'job_id': [f'job_{np.random.randint(0, n_jobs):03d}' for _ in range(n_applications)],
    'professional_user_id': [f'prof_{np.random.randint(0, n_professionals):03d}' for _ in range(n_applications)],
    'application_status': np.random.choice(['applied', 'viewed', 'accepted', 'rejected'], n_applications),
    'timestamp': [(start_date + timedelta(days=np.random.randint(0, 90))).isoformat() for _ in range(n_applications)]
})
job_applications.to_csv(os.path.join(output_dir, 'job_applications.csv'), index=False)

print('Sample data generated successfully!')
