import os
import json
import pandas as pd
import numpy as np

# Load data from sample_data directory
data_dir = './sample_data'

# Load user
prof_id = 'prof_050'

# Load locations
with open(os.path.join(data_dir, 'locations.json'), 'r') as f:
    locations = json.load(f)
location_dict = {loc['location_id']: loc for loc in locations}

# Load users
users = pd.read_csv(os.path.join(data_dir, 'users.csv'))
professional = users[users['user_id'] == prof_id].iloc[0]
prof_location_id = professional['location_id']
prof_location = location_dict[prof_location_id]

print(f"Professional {prof_id} location: {prof_location_id}")
print(f"Latitude: {prof_location['latitude']}, Longitude: {prof_location['longitude']}")

# Load categories
categories = pd.read_csv(os.path.join(data_dir, 'categories.csv'))
category_dict = dict(zip(categories['category_id'], categories['category_name']))

# Load professional categories
professional_categories = pd.read_csv(os.path.join(data_dir, 'professional_categories.csv'))
prof_cats = professional_categories[professional_categories['user_id'] == prof_id]['category_id'].tolist()

print(f"\nProfessional categories: {prof_cats}")
for cat_id in prof_cats:
    print(f"  - {cat_id}: {category_dict.get(cat_id)}")

# Load jobs
jobs = pd.read_csv(os.path.join(data_dir, 'jobs.csv'))

# Find matching jobs
matching_jobs = jobs[jobs['required_category_id'].isin(prof_cats)]
print(f"\nTotal jobs matching categories: {len(matching_jobs)}")

# Find jobs with the same location
same_location_jobs = jobs[jobs['location_id'] == prof_location_id]
print(f"Total jobs in the same location: {len(same_location_jobs)}")

# Find matching jobs in the same location
matching_same_location = matching_jobs[matching_jobs['location_id'] == prof_location_id]
print(f"Jobs matching categories in the same location: {len(matching_same_location)}")

if len(matching_same_location) > 0:
    print("\nMatching jobs in the same location:")
    for _, job in matching_same_location.iterrows():
        print(f"  - {job['job_id']}: {job['title']} (Category: {job['required_category_id']})")
else:
    print("\nNo matching jobs found in the same location!")

# Find jobs with different category in the same location
diff_cat_same_loc = same_location_jobs[~same_location_jobs['required_category_id'].isin(prof_cats)]
if len(diff_cat_same_loc) > 0:
    print("\nJobs in the same location but different categories:")
    for _, job in diff_cat_same_loc.iterrows():
        print(f"  - {job['job_id']}: {job['title']} (Category: {job['required_category_id']})")
