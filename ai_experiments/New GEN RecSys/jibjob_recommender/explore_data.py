import os
import json
import pandas as pd

# Load data
data_dir = './sample_data'

# Load users
users_df = pd.read_csv(os.path.join(data_dir, 'users.csv'))
print(f"Loaded {len(users_df)} users")
print(f"User columns: {users_df.columns.tolist()}")

# Load categories
categories_df = pd.read_csv(os.path.join(data_dir, 'categories.csv'))
print(f"Loaded {len(categories_df)} categories")
print(f"Category columns: {categories_df.columns.tolist()}")

# Load jobs
jobs_df = pd.read_csv(os.path.join(data_dir, 'jobs.csv'))
print(f"Loaded {len(jobs_df)} jobs")
print(f"Job columns: {jobs_df.columns.tolist()}")

# Sample data exploration
print("\n--- Sample User ---")
sample_user = users_df.iloc[0]
print(sample_user)

print("\n--- Sample Category ---")
sample_category = categories_df.iloc[0]
print(sample_category)

print("\n--- Sample Job ---")
sample_job = jobs_df.iloc[0]
print(sample_job)

# Let's analyze the first few users and their categories
print("\n--- First 3 Users and Their Categories ---")
for i, user in users_df.iloc[:3].iterrows():
    print(f"User ID: {user['user_id']}")
    print(f"Username: {user['username']}")
    print(f"User Type: {user['user_type']}")
    
    # Check if the user has selected categories
    if 'selected_category_ids' in user and not pd.isna(user['selected_category_ids']):
        category_ids = [cat_id.strip() for cat_id in str(user['selected_category_ids']).split(';')]
        print(f"Selected Category IDs: {category_ids}")
        
        # Look up category names
        if 'id' in categories_df.columns:
            # Try to match by ID column
            for cat_id in category_ids:
                try:
                    cat_id_int = int(cat_id)
                    category = categories_df[categories_df['id'] == cat_id_int]
                    if not category.empty:
                        cat_name = category.iloc[0]['name']
                        print(f"  - Category {cat_id}: {cat_name}")
                except:
                    print(f"  - Category {cat_id}: (Unknown)")
    else:
        print("No selected categories")
    print("-" * 50)
