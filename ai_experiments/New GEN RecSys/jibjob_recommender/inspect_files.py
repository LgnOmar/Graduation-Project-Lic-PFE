import os
import pandas as pd
import json

# Define data directory
data_dir = './sample_data'

# List CSV files
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print(f"CSV files found: {csv_files}")

# List JSON files
json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
print(f"JSON files found: {json_files}")

# Print top rows of each CSV file
print("\n=== CSV FILES ===")
for file in csv_files:
    try:
        df = pd.read_csv(os.path.join(data_dir, file))
        print(f"\n--- {file} (Rows: {len(df)}, Columns: {len(df.columns)}) ---")
        print(df.head(2).to_string())
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Print samples from each JSON file
print("\n=== JSON FILES ===")
for file in json_files:
    try:
        with open(os.path.join(data_dir, file), 'r') as f:
            data = json.load(f)
            print(f"\n--- {file} ---")
            if isinstance(data, list):
                print(f"List with {len(data)} items")
                if len(data) > 0:
                    print("First item:")
                    print(data[0])
            elif isinstance(data, dict):
                print("Dictionary with keys:", list(data.keys()))
    except Exception as e:
        print(f"Error reading {file}: {e}")
