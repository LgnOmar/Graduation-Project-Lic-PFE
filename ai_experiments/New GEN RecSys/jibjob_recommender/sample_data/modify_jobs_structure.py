import pandas as pd

# --- Configuration ---
source_csv_path = 'jobs.csv'  # The file WITH THE ACTUAL DATA
target_csv_path = 'jobs_weak.csv' # The name for your new, correctly structured file

# Define the mapping from source column names (from jobs.csv)
# to target column names (structure from jobs_weak.csv / your screenshot).
# In this case, we assume the column names you want to keep from 'jobs.csv'
# are EXACTLY the same as the target column names.
# If a column in 'jobs.csv' has a slightly different name but means the same thing,
# you would change the KEY here. E.g., if jobs.csv had 'job_identifier' instead of 'job_id':
# 'job_identifier': 'job_id',
column_mapping = {
    'job_id': 'job_id',
    'title': 'title',
    'description': 'description',
    'location_id': 'location_id',
    'posted_by_user_id': 'posted_by_user_id',
    'required_category_id': 'required_category_id'
    # Any column in 'jobs.csv' NOT listed as a KEY here will be DROPPED.
}

# Define the desired order of columns in the output file.
# These names MUST match the VALUES from the column_mapping above.
output_column_order = [
    'job_id',
    'title',
    'description',
    'location_id',
    'posted_by_user_id',
    'required_category_id'
]

# --- Script ---
try:
    # 1. Read the source CSV file (jobs.csv)
    print(f"Reading source file: {source_csv_path}")
    source_df = pd.read_csv(source_csv_path)
    print("\nSource DataFrame (from jobs.csv) columns:")
    print(source_df.columns.tolist())
    print("\nSource DataFrame head:")
    print(source_df.head())

    # Check if essential mapping keys are present in source_df
    missing_source_columns = [sc for sc in column_mapping.keys() if sc not in source_df.columns]
    if missing_source_columns:
        print(f"\nWARNING: The following source columns specified in 'column_mapping' were NOT FOUND in '{source_csv_path}': {missing_source_columns}")
        print("These columns will be missing or empty in the output. Please check your 'column_mapping' keys or the column names in 'jobs.csv'.")

    # 2. Create a new DataFrame for the transformed data
    transformed_df = pd.DataFrame()

    # 3. Select and rename columns based on the mapping
    print("\nApplying column mapping...")
    for source_col, target_col in column_mapping.items():
        if source_col in source_df.columns:
            transformed_df[target_col] = source_df[source_col]
            print(f"  Mapped source column '{source_col}' to target column '{target_col}'")
        else:
            # This case is now partially handled by the warning above
            print(f"  Warning: Source column '{source_col}' (for target '{target_col}') not found in '{source_csv_path}'. Output column will be empty/NaN.")
            transformed_df[target_col] = pd.NA # Fill with Not Available / NaN

    # 4. Ensure the columns are in the desired order and only these columns are present
    #    This reindex step is crucial. It will:
    #    - Order the columns as specified in output_column_order.
    #    - Add any column from output_column_order that wasn't in transformed_df yet (e.g., if source_col was missing), filling it with NaN.
    #    - Drop any columns that were in transformed_df but are NOT in output_column_order (though our mapping logic prevents this).
    print(f"\nReordering columns to match: {output_column_order}")
    transformed_df = transformed_df.reindex(columns=output_column_order)

    print("\nTransformed DataFrame head:")
    print(transformed_df.head())
    print("\nTransformed DataFrame columns:")
    print(transformed_df.columns.tolist())

    # 5. Write the transformed DataFrame to a new CSV file
    transformed_df.to_csv(target_csv_path, index=False) # index=False prevents writing pandas index
    print(f"\nTransformed data saved to '{target_csv_path}'")

except FileNotFoundError:
    print(f"Error: The source file '{source_csv_path}' was not found.")
    print("Please ensure the file name is correct and it's in the same directory as the script, or provide the full path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")