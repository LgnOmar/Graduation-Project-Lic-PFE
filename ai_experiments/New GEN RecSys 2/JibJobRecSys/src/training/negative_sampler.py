"""
negative_sampler.py

Purpose:
    Sample negative (Professional, Job) pairs for link prediction training.

Key Functions:
    - sample_negative_edges(positive_edges, num_job_nodes, num_neg_per_pos, all_positive_interactions_map)
        - For each professional, samples jobs not in their positive set.

Inputs:
    - positive_edges: Tuple of (src, dst) for positive edges.
    - num_job_nodes: Total number of job nodes.
    - num_neg_per_pos: Number of negatives per positive.
    - all_positive_interactions_map: Dict[int, Set[int]] mapping professional idx to set of positive job idxs.

Outputs:
    - neg_src, neg_dst: Negative edge indices.

High-Level Logic:
    1. For each professional in positive_edges, sample jobs not in their positive set.
    2. Return negative edge indices.
"""

import torch
import random
from typing import Dict, Set

def sample_negative_edges(positive_edges, num_job_nodes, num_neg_per_pos, all_positive_interactions_map: Dict[int, Set[int]]):
    src, dst = positive_edges
    neg_src = []
    neg_dst = []
    max_attempts = num_job_nodes * 2
    for s in src.tolist():
        true_positives = all_positive_interactions_map.get(s, set())
        current_neg_samples = set()
        attempts = 0
        while len(current_neg_samples) < num_neg_per_pos and attempts < max_attempts:
            candidate = random.randint(0, num_job_nodes - 1)
            if candidate not in true_positives:
                current_neg_samples.add(candidate)
            attempts += 1
        neg_src.extend([s] * len(current_neg_samples))
        neg_dst.extend(list(current_neg_samples))
    return torch.tensor(neg_src), torch.tensor(neg_dst)
