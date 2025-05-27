"""
jobs_generator.py

Purpose:
    Generate synthetic jobs and their required categories for the JibJob platform.

Key Functions:
    - generate_jobs(num_jobs: int, avg_categories: int, clients_path: str, categories_path: str, output_dir: str, random_seed: int)
        - Creates 'jobs.csv' and 'job_required_categories.csv'.
        - Assigns each job a client and 1-3 required categories.
        - Generates realistic job titles and descriptions using category keywords.

Inputs:
    - num_jobs: Number of jobs to generate.
    - avg_categories: Average number of categories per job.
    - clients_path: Path to clients.csv.
    - categories_path: Path to categories.csv.
    - output_dir: Output directory for CSVs.
    - random_seed: For reproducibility.

Outputs:
    - jobs.csv, job_required_categories.csv

High-Level Logic:
    1. Generate jobs with unique IDs, titles, and descriptions.
    2. Assign each job 1-3 required categories (random, weighted).
    3. Save all to CSVs.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_jobs(num_jobs: int, avg_categories_per_job: int, clients_path: str, categories_path: str, output_dir: str, random_seed: int = 42, min_categories: int = 1, max_categories: int = 3):
    random.seed(random_seed)
    np.random.seed(random_seed)
    clients = pd.read_csv(clients_path)
    categories = pd.read_csv(categories_path)
    cat_ids = categories['category_id'].tolist()
    jobs = []
    job_cats = []
    weights = np.random.dirichlet(np.ones(len(cat_ids)), size=1)[0]
    
    # Helper function to safely access keywords
    def get_keyword(keywords_list, index, default_suffix="related task"):
        """Helper to safely get a keyword or a default value by cycling if index is out of bounds."""
        if not keywords_list:
            return f"a general {default_suffix}"
        return keywords_list[index % len(keywords_list)]

    desc_templates = [
        lambda n, k: (
            f"{n} required for immediate work involving {get_keyword(k, 0)} and {get_keyword(k, 1)}. "
            f"Candidates should be adept at {get_keyword(k, 2)} and able to handle {get_keyword(k, 3, 'related duties')}. "
            f"Flexibility and reliability are essential."
        ),
        lambda n, k: (
            f"Looking for a skilled {n} to assist with {get_keyword(k, 0)}. "
            f"Experience in {get_keyword(k, 1)} and {get_keyword(k, 2)} is a plus. "
            f"Must be able to manage {get_keyword(k, 3, 'miscellaneous tasks')}."
        ),
        lambda n, k: (
            f"Seeking {n} for projects involving {get_keyword(k, 0)}. "
            f"Knowledge of {get_keyword(k, 1)} and {get_keyword(k, 2)} required. "
            f"Ability to handle {get_keyword(k, 3, 'varied responsibilities')} independently is preferred."
        ),
    ]
    for i in range(num_jobs):
        job_id = 5001 + i
        client_id = int(clients.sample(1)['client_id'])
        k = int(np.clip(int(np.random.normal(avg_categories_per_job, 1)), min_categories, max_categories))
        selected = np.random.choice(cat_ids, size=k, replace=False, p=weights)
        cat_names = categories[categories['category_id'].isin(selected)]['category_name'].tolist()
        cat_keywords = []
        for cid in selected:
            cat_keywords.extend(categories[categories['category_id'] == cid].iloc[0]['category_name'].split())
        title = f"{'Urgent' if random.random() < 0.5 else 'Need'}: {' and '.join(cat_names)} expert needed"
        desc = random.choice(desc_templates)(" and ".join(cat_names), cat_keywords)
        posting_date = (datetime.now() - timedelta(days=random.randint(0, 60))).strftime('%Y-%m-%d')
        jobs.append({
            "job_id": job_id,
            "client_id": client_id,
            "title": title,
            "description": desc,
            "posting_date": posting_date
        })
        for cat in selected:
            job_cats.append({
                "job_id": job_id,
                "category_id": cat
            })
    pd.DataFrame(jobs).to_csv(f"{output_dir}/jobs.csv", index=False)
    pd.DataFrame(job_cats).to_csv(f"{output_dir}/job_required_categories.csv", index=False)
