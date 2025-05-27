"""
categories_generator.py

Purpose:
    Generate a synthetic set of job categories for the JibJob platform, each with a unique ID, name, and detailed description suitable for BERT embedding.

Key Functions:
    - generate_categories(num_categories: int, output_path: str, random_seed: int)
        - Creates 'categories.csv' with columns: category_id, category_name, category_description.
        - Uses templates and seed keywords for diversity and realism.
        - Ensures descriptions are distinct and informative.

Inputs:
    - num_categories: Number of categories to generate.
    - output_path: Path to save the CSV.
    - random_seed: For reproducibility.

Outputs:
    - CSV file at output_path.

High-Level Logic:
    1. Define base category names and associated keywords.
    2. For each category, generate a name and a templated description.
    3. Save all categories to CSV.
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict

def generate_categories(num_categories: int, output_path: str, base_category_definitions: list, random_seed: int = 42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    base_categories = base_category_definitions
    templates = [
        lambda n, k: f"This category encompasses tasks such as {', '.join(k[:3])}, and general {n.lower()} work. Professionals in this field often handle {k[0]}-related issues and possess skills in {k[1]}. Expertise in {k[2]} is highly valued. Clients typically seek help for {k[3]} and similar needs. The {n} category is essential for maintaining quality and safety.",
        lambda n, k: f"This field is critical for {k[0]} and demands proficiency in {k[1]}. Key services include {k[2]} and troubleshooting related to {k[3]}. {n} professionals are known for their expertise and reliability.",
        lambda n, k: f"{n} covers a wide range of tasks, including {k[0]}, {k[1]}, and {k[2]}. Clients often require help with {k[3]}. Success in this category requires attention to detail and strong problem-solving skills."
    ]
    categories = []
    for i in range(num_categories):
        base_idx = i % len(base_categories)
        name, keywords = base_categories[base_idx]
        if i >= len(base_categories):
            name = f"{name} - Specialization {i // len(base_categories) + 1}"
        desc = random.choice(templates)(name, keywords)
        categories.append({
            "category_id": i + 1,
            "category_name": name,
            "category_description": desc
        })
    df = pd.DataFrame(categories)
    df.to_csv(output_path, index=False)
