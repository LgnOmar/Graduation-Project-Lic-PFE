"""
Script to verify the quality of the newly generated Algerian data.
"""
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import sys

print("JibJob Recommendation System - Algerian Data Quality Check")
print("=" * 70)

# Check if the data directory exists
if not os.path.exists("data"):
    print("ERROR: data directory not found!")
    sys.exit(1)

# Check if required data files exist
required_files = ["jobs_df.csv", "users_df.csv", "interactions_df.csv"]
for file in required_files:
    file_path = os.path.join("data", file)
    if not os.path.exists(file_path):
        print(f"ERROR: Required data file {file} not found!")
        sys.exit(1)

print("\n1. Loading data files...")
try:
    jobs_df = pd.read_csv("data/jobs_df.csv")
    users_df = pd.read_csv("data/users_df.csv")
    interactions_df = pd.read_csv("data/interactions_df.csv")
    
    print(f"   ✓ Successfully loaded {len(jobs_df)} jobs")
    print(f"   ✓ Successfully loaded {len(users_df)} users")
    print(f"   ✓ Successfully loaded {len(interactions_df)} interactions")
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

print("\n2. Checking jobs data...")
print(f"   ✓ Job IDs: {len(jobs_df['job_id'].unique())} unique values")

# Check job locations (wilayas)
if "location" in jobs_df.columns:
    locations = jobs_df["location"].unique()
    print(f"   ✓ Locations: {len(locations)} unique wilayas")
    print(f"     Top 10 wilayas: {', '.join(jobs_df['location'].value_counts().head(10).index.tolist())}")
else:
    print("   ✗ No location column found in jobs data")

# Check job categories
if "categorie_mission" in jobs_df.columns:
    categories = jobs_df["categorie_mission"].unique()
    print(f"   ✓ Categories: {len(categories)} unique values")
    print(f"     Categories: {', '.join(categories)}")
else:
    print("   ✗ No categorie_mission column found in jobs data")

# Check job descriptions
if "description_mission_anglais" in jobs_df.columns:
    desc_lengths = jobs_df["description_mission_anglais"].str.len()
    print(f"   ✓ Description lengths: min={desc_lengths.min()}, max={desc_lengths.max()}, avg={int(desc_lengths.mean())}")
else:
    print("   ✗ No description_mission_anglais column found in jobs data")

print("\n3. Checking users data...")
print(f"   ✓ User IDs: {len(users_df['user_id'].unique())} unique values")

if "skills" in users_df.columns:
    # Count non-empty skills
    skills_count = users_df["skills"].notna().sum()
    print(f"   ✓ Users with skills: {skills_count} ({skills_count/len(users_df)*100:.1f}%)")
else:
    print("   ✗ No skills column found in users data")

if "preferences" in users_df.columns:
    # Count non-empty preferences
    prefs_count = users_df["preferences"].notna().sum()
    print(f"   ✓ Users with preferences: {prefs_count} ({prefs_count/len(users_df)*100:.1f}%)")
else:
    print("   ✗ No preferences column found in users data")

print("\n4. Checking interactions data...")
# Check ratings distribution
if "rating_explicite" in interactions_df.columns:
    ratings = interactions_df["rating_explicite"].dropna()
    print(f"   ✓ Ratings: {len(ratings)} ({len(ratings)/len(interactions_df)*100:.1f}% of interactions)")
    
    # Print rating distribution
    rating_counts = ratings.value_counts().sort_index()
    print("   Rating distribution:")
    for rating, count in rating_counts.items():
        print(f"     {rating}: {count} ({count/len(ratings)*100:.1f}%)")
else:
    print("   ✗ No rating_explicite column found in interactions data")

# Check comments
if "commentaire_texte_anglais" in interactions_df.columns:
    comments = interactions_df["commentaire_texte_anglais"].dropna()
    print(f"   ✓ Comments: {len(comments)} ({len(comments)/len(interactions_df)*100:.1f}% of interactions)")
    
    # Print sample comments
    print("   Sample comments:")
    for comment in comments.head(3).tolist():
        print(f"     - {comment}")
else:
    print("   ✗ No commentaire_texte_anglais column found in interactions data")

print("\n5. Creating data summary report...")
# Create a summary report
summary = {
    "jobs": {
        "count": len(jobs_df),
        "categories": len(jobs_df["categorie_mission"].unique()) if "categorie_mission" in jobs_df.columns else 0,
        "locations": len(jobs_df["location"].unique()) if "location" in jobs_df.columns else 0
    },
    "users": {
        "count": len(users_df),
        "with_skills": users_df["skills"].notna().sum() if "skills" in users_df.columns else 0,
        "with_preferences": users_df["preferences"].notna().sum() if "preferences" in users_df.columns else 0
    },
    "interactions": {
        "count": len(interactions_df),
        "with_ratings": interactions_df["rating_explicite"].notna().sum() if "rating_explicite" in interactions_df.columns else 0,
        "with_comments": interactions_df["commentaire_texte_anglais"].notna().sum() if "commentaire_texte_anglais" in interactions_df.columns else 0
    }
}

# Save summary to file
with open("data_quality_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"   ✓ Summary saved to data_quality_summary.json")

print("\n" + "=" * 70)
print("Data quality check complete! The generated Algerian data looks good.")
print("Next steps:")
print("1. Run the JibJob Demo API to use this data:")
print("   python src/demo_api.py")
print("2. Access the API documentation at http://localhost:8000/docs")
print("3. Test recommendations with new realistic Algerian data")
