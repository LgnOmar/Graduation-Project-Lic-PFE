"""
Demonstration script for the sample data generation utilities.

This script shows how to:
1. Generate synthetic users, jobs, and interactions
2. Generate graph data for GCN models
3. Save the generated data for future use
4. Visualize the generated data

Usage:
    python sample_data_demo.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from datetime import datetime

# Add the parent directory to the path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.sample_data import (
    generate_users, 
    generate_jobs, 
    generate_interactions,
    generate_graph_data,
    generate_hetero_graph_data,
    generate_simple_dataset
)
from src.utils.visualization import plot_graph, plot_rating_distribution
from src.data.graph_builder import build_interaction_graph
from src.models.gcn import GCNRecommender
from src.utils.metrics import ndcg_at_k as calculate_ndcg, hit_rate_at_k as calculate_hit_rate


def main():
    print("=" * 80)
    print("JibJob Recommendation System - Sample Data Demo")
    print("=" * 80)
    
    # Generate simple dataset
    print("\n1. Generating simple dataset...")
    users_df, jobs_df, interactions_df = generate_simple_dataset()
    
    print(f"Generated {len(users_df)} users, {len(jobs_df)} jobs, and {len(interactions_df)} interactions")
    
    # Display sample data
    print("\nSample users:")
    print(users_df.head(3))
    
    print("\nSample jobs:")
    print(jobs_df.head(3))
    
    print("\nSample interactions:")
    print(interactions_df.head(3))
    
    # Visualize rating distribution
    print("\n2. Visualizing rating distribution...")
    plt.figure(figsize=(8, 5))
    interactions_df['rating'].hist(bins=10)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig('rating_distribution.png')
    print(f"Rating distribution saved to rating_distribution.png")
    
    # Generate graph data for GCN
    print("\n3. Generating graph data for GCN...")
    graph_data = generate_graph_data(n_users=50, n_jobs=100, n_edges=200)
    print(f"Generated graph with {graph_data.num_users} users, {graph_data.num_jobs} jobs, and {graph_data.edge_index.size(1)} edges")
    
    # Create a GCN model and get recommendations
    print("\n4. Creating GCN model and getting recommendations...")
    gcn_model = GCNRecommender(
        num_users=50,
        num_jobs=100,
        embedding_dim=32,
        hidden_dim=32,
        num_layers=2
    )
    
    sample_user_id = 10
    top_k = 5
    
    job_ids, scores = gcn_model.recommend(
        graph=graph_data,
        user_id=sample_user_id,
        top_k=top_k
    )
    
    print(f"Recommendations for user {sample_user_id}:")
    for job_id, score in zip(job_ids, scores):
        print(f"  - Job {job_id}: score = {score:.4f}")
    
    # Generate heterogeneous graph data
    print("\n5. Generating heterogeneous graph data...")
    hetero_data = generate_hetero_graph_data(
        n_users=30, 
        n_jobs=50, 
        n_categories=5, 
        embedding_dim=32
    )
    
    print(f"Generated heterogeneous graph with:")
    print(f"  - {hetero_data['user'].x.size(0)} users")
    print(f"  - {hetero_data['job'].x.size(0)} jobs")
    print(f"  - {hetero_data['category'].x.size(0)} categories")
    print(f"  - {hetero_data['user', 'rates', 'job'].edge_index.size(1)} user-job interactions")
    print(f"  - {hetero_data['job', 'belongs_to', 'category'].edge_index.size(1)} job-category relationships")
    
    # Save generated data
    print("\n6. Saving generated data to files...")
    # Create output directory
    output_dir = "sample_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataframes
    users_df.to_csv(f"{output_dir}/sample_users.csv", index=False)
    jobs_df.to_csv(f"{output_dir}/sample_jobs.csv", index=False)
    interactions_df.to_csv(f"{output_dir}/sample_interactions.csv", index=False)
    
    # Save metadata
    metadata = {
        "num_users": len(users_df),
        "num_jobs": len(jobs_df),
        "num_interactions": len(interactions_df),
        "rating_mean": float(interactions_df['rating'].mean()),
        "rating_std": float(interactions_df['rating'].std()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Sample data saved to {output_dir}/ directory")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
