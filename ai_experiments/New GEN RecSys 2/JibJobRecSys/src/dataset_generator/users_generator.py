"""
users_generator.py

Purpose:
    Generate synthetic professionals, their selected categories, and clients for the JibJob platform.

Key Functions:
    - generate_professionals(num_professionals: int, avg_categories: int, categories_path: str, output_path: str, random_seed: int)
        - Creates 'professionals.csv' and 'professional_selected_categories.csv'.
        - Assigns each professional a random (weighted) set of categories.
    - generate_clients(num_clients: int, output_path: str, random_seed: int)
        - Creates 'clients.csv'.

Inputs:
    - num_professionals: Number of professionals to generate.
    - avg_categories: Average number of categories per professional.
    - categories_path: Path to categories.csv.
    - output_path: Output directory for CSVs.
    - random_seed: For reproducibility.

Outputs:
    - professionals.csv, professional_selected_categories.csv, clients.csv

High-Level Logic:
    1. Generate professionals with unique IDs and names.
    2. Assign each professional 1-5 categories (weighted random).
    3. Generate clients with unique IDs and names.
    4. Save all to CSVs.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List

def generate_professionals(num_professionals: int, avg_categories: int, categories_path: str, output_dir: str, random_seed: int = 42, min_categories: int = 1, max_categories: int = 5):
    random.seed(random_seed)
    np.random.seed(random_seed)
    categories = pd.read_csv(categories_path)
    cat_ids = categories['category_id'].tolist()
    # Weighted probabilities for category popularity
    weights = np.random.dirichlet(np.ones(len(cat_ids)), size=1)[0]
    professionals = []
    prof_cats = []
    for i in range(num_professionals):
        prof_id = 1001 + i
        name = f"Prof_User_{prof_id}"
        creation_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
        professionals.append({
            "professional_id": prof_id,
            "professional_name": name,
            "profile_creation_date": creation_date
        })
        k = int(np.clip(int(np.random.normal(avg_categories, 1)), min_categories, max_categories))
        selected = np.random.choice(cat_ids, size=k, replace=False, p=weights)
        for cat in selected:
            prof_cats.append({
                "professional_id": prof_id,
                "category_id": cat
            })
    pd.DataFrame(professionals).to_csv(f"{output_dir}/professionals.csv", index=False)
    pd.DataFrame(prof_cats).to_csv(f"{output_dir}/professional_selected_categories.csv", index=False)

def generate_clients(num_clients: int, output_dir: str, random_seed: int = 42):
    random.seed(random_seed)
    clients = []
    for i in range(num_clients):
        client_id = 2001 + i
        name = f"Client_User_{client_id}"
        creation_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d')
        clients.append({
            "client_id": client_id,
            "client_name": name,
            "profile_creation_date": creation_date
        })
    pd.DataFrame(clients).to_csv(f"{output_dir}/clients.csv", index=False)
