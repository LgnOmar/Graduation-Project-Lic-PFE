import os
import pandas as pd
import json

# Load data from sample_data directory
data_dir = './sample_data'

# Load categories
categories = pd.read_csv(os.path.join(data_dir, 'categories.csv'))
print(f"Loaded {len(categories)} categories")

# Load users
users = pd.read_csv(os.path.join(data_dir, 'users.csv'))
professionals = users[users['user_type'] == 'professional']
clients = users[users['user_type'] == 'client']
print(f"Loaded {len(professionals)} professionals and {len(clients)} clients")

# Load professional categories
professional_categories = pd.read_csv(os.path.join(data_dir, 'professional_categories.csv'))
print(f"Loaded {len(professional_categories)} professional-category associations")

# Load jobs
jobs = pd.read_csv(os.path.join(data_dir, 'jobs.csv'))
print(f"Loaded {len(jobs)} jobs")

# Load job applications
job_applications = pd.read_csv(os.path.join(data_dir, 'job_applications.csv'))
print(f"Loaded {len(job_applications)} job applications")

# Get example data
print("\nEXAMPLE DATA:")
print("-" * 50)

# Example professional
print("Example Professional:")
example_prof = professionals.iloc[0]
print(example_prof)
print()

# Get categories for this professional
prof_categories = professional_categories[professional_categories['user_id'] == example_prof['user_id']]
print("Professional's Categories:")
print(prof_categories)
print()

# Find jobs matching this professional's categories
matching_jobs = pd.merge(
    prof_categories,
    jobs,
    left_on='category_id',
    right_on='required_category_id',
    how='inner'
)
print(f"Found {len(matching_jobs)} jobs matching the professional's categories")
print()

# Show top 5 matching jobs
if len(matching_jobs) > 0:
    print("Sample Matching Jobs:")
    for i, job in matching_jobs.head().iterrows():
        print(f"Job ID: {job['job_id']}")
        print(f"Title: {job['title']}")
        print(f"Category: {job['category_id']}")
        print("-" * 30)
else:
    print("No matching jobs found!")

# Check to see if there are any issues with the data
print("\nDATASET VALIDATION:")
print("-" * 50)

# Check if any professionals have no categories
profs_without_categories = set(professionals['user_id']) - set(professional_categories['user_id'])
print(f"Professionals with no categories: {len(profs_without_categories)}")

# Check if any jobs have invalid category IDs
invalid_job_categories = set(jobs['required_category_id']) - set(categories['category_id'])
print(f"Jobs with invalid category IDs: {len(invalid_job_categories)}")

# Check if we have any job applications
print(f"Total job applications: {len(job_applications)}")

# For diagnostic purposes, print out the distribution of category IDs in jobs
job_category_counts = jobs['required_category_id'].value_counts().head(10)
print("\nTop 10 job categories:")
print(job_category_counts)

# For diagnostic purposes, print out the distribution of category IDs in professional preferences
prof_category_counts = professional_categories['category_id'].value_counts().head(10)
print("\nTop 10 professional categories:")
print(prof_category_counts)
