import os
import sys
import json
import pandas as pd
import numpy as np

# Load the data
data_dir = './sample_data'

# Load categories
categories = pd.read_csv(os.path.join(data_dir, 'categories.csv'))
category_dict = dict(zip(categories['category_id'], categories['category_name']))

# Load professional categories
professional_categories = pd.read_csv(os.path.join(data_dir, 'professional_categories.csv'))

# Load jobs
jobs = pd.read_csv(os.path.join(data_dir, 'jobs.csv'))

# Find professionals with the most matching jobs
prof_category_counts = professional_categories['user_id'].value_counts()
top_professionals = prof_category_counts.head(10).index.tolist()

print("Top professionals by number of categories:")
for prof_id in top_professionals:
    prof_cats = professional_categories[professional_categories['user_id'] == prof_id]['category_id'].tolist()
    print(f"{prof_id}: {len(prof_cats)} categories - {prof_cats}")

print("\nLooking for jobs matching these categories...")

# For each professional, find matching jobs
for prof_id in top_professionals:
    prof_cats = professional_categories[professional_categories['user_id'] == prof_id]['category_id'].tolist()
    matching_jobs = jobs[jobs['required_category_id'].isin(prof_cats)]
    
    if len(matching_jobs) > 0:
        print(f"\n{prof_id} has {len(matching_jobs)} matching jobs:")
        for cat_id in prof_cats:
            cat_jobs = jobs[jobs['required_category_id'] == cat_id]
            if len(cat_jobs) > 0:
                print(f"  - Category {cat_id} ({category_dict.get(cat_id)}): {len(cat_jobs)} jobs")
        break
