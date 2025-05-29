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
import yaml

def generate_jobs(num_jobs: int, avg_categories_per_job: int, clients_path: str, categories_path: str, output_dir: str, base_category_definitions: list, random_seed: int = 42, min_categories: int = 1, max_categories: int = 3):
    random.seed(random_seed)
    np.random.seed(random_seed)
    clients = pd.read_csv(clients_path)
    categories = pd.read_csv(categories_path)
    cat_ids = categories['category_id'].tolist()
    jobs = []
    job_cats = []
    weights = np.random.dirichlet(np.ones(len(cat_ids)), size=1)[0]
    # Build a mapping from category_name to keywords using passed base_category_definitions
    catname_to_keywords = {name: keywords for name, keywords in base_category_definitions}

    # Helper function to safely access keywords
    def get_keyword(keywords_list, index, default_suffix="related task"):
        """Helper to safely get a keyword or a default value by cycling if index is out of bounds."""
        if not keywords_list:
            return f"a general {default_suffix}"
        return keywords_list[index % len(keywords_list)]

    for i in range(num_jobs):
        job_id = 5001 + i
        client_id = int(clients.sample(1)['client_id'])
        k = int(np.clip(int(np.random.normal(avg_categories_per_job, 1)), min_categories, max_categories))
        selected = np.random.choice(cat_ids, size=k, replace=False, p=weights)
        cat_names = categories[categories['category_id'].isin(selected)]['category_name'].tolist()
        # --- Use richer keywords from config for all selected categories ---
        cat_keywords = []
        for cname in cat_names:
            cat_keywords.extend(catname_to_keywords.get(cname, []))
        # More diverse title templates
        title_templates = [
            lambda names, kws: f"Urgent: {' and '.join(names)} expert needed for {get_keyword(kws, 0)}",
            lambda names, kws: f"Seeking {' & '.join(names)} Specialist - Focus on {get_keyword(kws, 1)}",
            lambda names, kws: f"Part-time {names[0]} Opportunity: {get_keyword(kws, 2)} skills required",
            lambda names, kws: f"Experienced {names[0]} for {get_keyword(kws, 0)} and {get_keyword(kws, 1)} projects"
        ]
        title = random.choice(title_templates)(cat_names, cat_keywords)
        # More diverse description templates
        desc_templates = [
            lambda n, k: (
                f"We are seeking a {n[0]} with strong skills in {get_keyword(k, 0)} and {get_keyword(k, 1)}. "
                f"Experience with {n[1] if len(n)>1 else n[0]} tasks, especially {get_keyword(k, 2)}, would be a significant advantage. "
                f"This role involves {get_keyword(k, 3, 'varied responsibilities')} and collaboration across teams."
            ),
            lambda n, k: (
                f"Looking for a {n[0]} professional to handle {get_keyword(k, 0)} and {get_keyword(k, 1)}. "
                f"Ability to manage {get_keyword(k, 2)} and support {n[1] if len(n)>1 else n[0]}-related projects is required. "
                f"Attention to detail and reliability are essential."
            ),
            lambda n, k: (
                f"Join our team as a {n[0]} expert. Key tasks include {get_keyword(k, 0)}, {get_keyword(k, 1)}, and {get_keyword(k, 2)}. "
                f"Knowledge of {n[1] if len(n)>1 else n[0]} and experience with {get_keyword(k, 3, 'miscellaneous tasks')} are a plus."
            ),
            lambda n, k: (
                f"Exciting opportunity for a {n[0]} with expertise in {get_keyword(k, 0)}. "
                f"The ideal candidate will also be familiar with {get_keyword(k, 1)} and {get_keyword(k, 2)}. "
                f"Role includes {get_keyword(k, 3, 'cross-functional duties')}."
            )
        ]
        desc = random.choice(desc_templates)(cat_names, cat_keywords)
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
