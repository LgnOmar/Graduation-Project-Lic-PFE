"""
interactions_generator.py

Purpose:
    Generate synthetic professional-job interactions for the JibJob platform, simulating applications or likes.

Key Functions:
    - generate_interactions(professionals_path: str, jobs_path: str, professional_categories_path: str, job_categories_path: str, output_path: str, num_interactions_per_professional: int, random_seed: int)
        - Creates 'interactions.csv'.
        - For each professional, finds jobs with high category overlap and simulates interactions.
        - Adds randomness and avoids unrealistic interaction counts.

Inputs:
    - professionals_path: Path to professionals.csv.
    - jobs_path: Path to jobs.csv.
    - professional_categories_path: Path to professional_selected_categories.csv.
    - job_categories_path: Path to job_required_categories.csv.
    - output_path: Path to save interactions.csv.
    - num_interactions_per_professional: Approximate number of interactions per professional.
    - random_seed: For reproducibility.

Outputs:
    - interactions.csv

High-Level Logic:
    1. For each professional, find jobs with high Jaccard similarity to their categories.
    2. Simulate interactions, add randomness, and avoid unrealistic counts.
    3. Save to CSV.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

def generate_interactions(professionals_path: str, jobs_path: str, professional_categories_path: str, job_categories_path: str, output_path: str, num_interactions_per_professional: int, random_seed: int = 42, jaccard_prob_offset: float = 0.1, unrelated_prob: float = 0.05):
    random.seed(random_seed)
    np.random.seed(random_seed)
    professionals = pd.read_csv(professionals_path)
    jobs = pd.read_csv(jobs_path)
    prof_cats = pd.read_csv(professional_categories_path)
    job_cats = pd.read_csv(job_categories_path)
    interactions = []
    prof_id_to_cats = prof_cats.groupby('professional_id')['category_id'].apply(set).to_dict()
    job_id_to_cats = job_cats.groupby('job_id')['category_id'].apply(set).to_dict()
    for _, prof in professionals.iterrows():
        prof_id = prof['professional_id']
        prof_cat_set = prof_id_to_cats.get(prof_id, set())
        job_scores = []
        for _, job in jobs.iterrows():
            job_id = job['job_id']
            job_cat_set = job_id_to_cats.get(job_id, set())
            score = jaccard(prof_cat_set, job_cat_set)
            job_scores.append((job_id, score))
        job_scores.sort(key=lambda x: x[1], reverse=True)
        n = min(num_interactions_per_professional, len(job_scores))
        chosen = set()
        for job_id, score in job_scores:
            if score == 0 and random.random() > unrelated_prob:
                continue
            if len(chosen) >= n:
                break
            if random.random() < (score + jaccard_prob_offset):
                chosen.add(job_id)
                timestamp = (datetime.now() - timedelta(days=random.randint(0, 60))).strftime('%Y-%m-%d')
                interactions.append({
                    "professional_id": prof_id,
                    "job_id": job_id,
                    "interaction_type": "applied",
                    "interaction_timestamp": timestamp,
                    "interaction_score": 1.0
                })
    pd.DataFrame(interactions).to_csv(output_path, index=False)
