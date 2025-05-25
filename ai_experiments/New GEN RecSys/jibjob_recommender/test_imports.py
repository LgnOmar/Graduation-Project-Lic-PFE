"""
A simple script to test imports from the JibJob recommender system
"""
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.abspath('.'))

# Try importing from various modules
try:
    from jibjob_recommender_system import __init__
    print("Successfully imported root package")
except Exception as e:
    print(f"Error importing root package: {e}")

try:
    from jibjob_recommender_system.config import config_loader
    print("Successfully imported config_loader")
except Exception as e:
    print(f"Error importing config_loader: {e}")

try:
    from jibjob_recommender_system.data_handling import data_loader
    print("Successfully imported data_loader")
except Exception as e:
    print(f"Error importing data_loader: {e}")

try:
    from jibjob_recommender_system.feature_engineering import text_embedder
    print("Successfully imported text_embedder")
except Exception as e:
    print(f"Error importing text_embedder: {e}")

try:
    from jibjob_recommender_system.graph_construction import graph_builder
    print("Successfully imported graph_builder")
except Exception as e:
    print(f"Error importing graph_builder: {e}")

try:
    from jibjob_recommender_system.models import gcn_recommender
    print("Successfully imported gcn_recommender")
except Exception as e:
    print(f"Error importing gcn_recommender: {e}")

try:
    from jibjob_recommender_system.training import trainer
    print("Successfully imported trainer")
except Exception as e:
    print(f"Error importing trainer: {e}")

print("\nImport test completed.")
