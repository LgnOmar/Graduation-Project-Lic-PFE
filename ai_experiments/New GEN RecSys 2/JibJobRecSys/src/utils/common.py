"""
common.py

Purpose:
    Utility functions for config loading, reproducibility, and general helpers.

Key Functions:
    - set_seed(seed: int)
    - load_config(path: str)

High-Level Logic:
    1. Set random, numpy, and torch seeds for reproducibility.
    2. Load config from YAML or JSON.
"""

import random
import numpy as np
import torch
import json
import os

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(path: str):
    ext = os.path.splitext(path)[-1]
    with open(path, 'r') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            import yaml
            return yaml.safe_load(f)
        else:
            raise ValueError('Unsupported config file type')
