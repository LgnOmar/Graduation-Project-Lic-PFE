"""
Script to test the enhanced JibJob recommendation system with Algerian data.
"""
import pandas as pd
import requests
import json
import random
import sys
from tabulate import tabulate

# Base URL for API
BASE_URL = "http://localhost:8000"

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"{text:^80}")
    print("=" * 80)

def get_wilayas():
    """Get list of available wilayas from the API."""
    try:
        response = requests.get(f"{BASE_URL}/wilayas")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting wilayas: {e}")
        return []

def get_recommendation(user_id, location=None):
    """Get recommendations for a specific user."""
    try:
        url = f"{BASE_URL}/recommendations/{user_id}"
        if location:
            url += f"?location_preference={location}"
            
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return None

def get_job_details(job_id):
    """Get details for a specific job."""
    try:
        response = requests.get(f"{BASE_URL}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting job details: {e}")
        return None

def get_user_details(user_id):
    """Get details for a specific user."""
    try:
        response = requests.get(f"{BASE_URL}/users/{user_id}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error getting user details: {e}")
        return None

def search_jobs_by_location(location):
    """Search for jobs in a specific wilaya."""
    try:
        response = requests.get(f"{BASE_URL}/jobs?location={location}")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error searching jobs: {e}")
        return None

def test_recommendations():
    """Test recommendations for random users."""
    print_header("Testing JibJob Recommendations for Random Users")
    
    # Load a sample of user IDs
    try:
        users_df = pd.read_csv("data/users_df.csv")
        user_ids = users_df["user_id"].sample(5).tolist()
    except Exception as e:
        print(f"Error loading users: {e}")
        user_ids = [f"user_{i}" for i in range(5)]
    
    for user_id in user_ids:
        print(f"\nTESTING RECOMMENDATIONS FOR {user_id}")
        print("-" * 60)
        
        # Get user details
        user = get_user_details(user_id)
        if not user:
            print(f"Could not get details for {user_id}")
            continue
            
        print(f"Name: {user.get('name', 'N/A')}")
        print(f"Profile: {user.get('description', 'N/A')}")
        print(f"Skills: {', '.join(user.get('skills', ['N/A']))}")
        print(f"Preferences: {', '.join(user.get('preferences', ['N/A']))}")
        
        # Get recommendations
        recommendations = get_recommendation(user_id)
        if not recommendations or not recommendations.get("job_ids"):
            print("No recommendations found")
            continue
            
        # Display recommendations
        job_ids = recommendations.get("job_ids", [])
        scores = recommendations.get("scores", [])
        
        table_data = []
        for i, (job_id, score) in enumerate(zip(job_ids, scores)):
            job = get_job_details(job_id)
            if job:
                table_data.append([
                    i+1,
                    job_id,
                    job.get("title", "N/A"),
                    job.get("category", "N/A"),
                    job.get("location", "N/A"),
                    f"{score:.4f}"
                ])
                
        print("\nRECOMMENDED JOBS:")
        headers = ["#", "Job ID", "Title", "Category", "Location", "Score"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

def test_location_based_search():
    """Test location-based job search."""
    print_header("Testing Location-Based Job Search")
    
    # Get available wilayas
    wilayas = get_wilayas()
    if not wilayas:
        print("Could not get list of wilayas")
        return
        
    print(f"Found {len(wilayas)} wilayas in Algeria")
    
    # Select a few random wilayas
    sample_wilayas = random.sample(wilayas, min(3, len(wilayas)))
    
    for wilaya in sample_wilayas:
        print(f"\nJOBS IN {wilaya}")
        print("-" * 60)
        
        # Search for jobs in this wilaya
        results = search_jobs_by_location(wilaya)
        if not results or not results.get("results"):
            print(f"No jobs found in {wilaya}")
            continue
            
        # Display jobs
        jobs = results.get("results", [])
        
        table_data = []
        for i, job in enumerate(jobs[:10]):  # Show at most 10 jobs
            table_data.append([
                i+1,
                job.get("job_id", "N/A"),
                job.get("title", "N/A"),
                job.get("category", "N/A")
            ])
                
        print(f"Found {results.get('total', 0)} jobs in {wilaya}")
        headers = ["#", "Job ID", "Title", "Category"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

def test_location_based_recommendations():
    """Test location-based recommendations."""
    print_header("Testing Location-Based Recommendations")
    
    # Get available wilayas
    wilayas = get_wilayas()
    if not wilayas:
        print("Could not get list of wilayas")
        return
    
    # Load a sample of user IDs
    try:
        users_df = pd.read_csv("data/users_df.csv")
        user_id = users_df["user_id"].sample(1).iloc[0]
    except Exception as e:
        print(f"Error loading users: {e}")
        user_id = "user_0"
    
    # Select a random wilaya
    wilaya = random.choice(wilayas)
    
    print(f"Testing recommendations for user {user_id} in {wilaya}")
    print("-" * 60)
    
    # Get user details
    user = get_user_details(user_id)
    if not user:
        print(f"Could not get details for {user_id}")
        return
        
    print(f"Name: {user.get('name', 'N/A')}")
    print(f"Preferences: {', '.join(user.get('preferences', ['N/A']))}")
    
    # Get location-based recommendations
    recommendations = get_recommendation(user_id, location=wilaya)
    if not recommendations or not recommendations.get("job_ids"):
        print("No recommendations found")
        return
        
    # Display recommendations
    job_ids = recommendations.get("job_ids", [])
    scores = recommendations.get("scores", [])
    
    table_data = []
    for i, (job_id, score) in enumerate(zip(job_ids, scores)):
        job = get_job_details(job_id)
        if job:
            table_data.append([
                i+1,
                job_id,
                job.get("title", "N/A"),
                job.get("location", "N/A"),
                f"{score:.4f}"
            ])
            
    print(f"\nRECOMMENDED JOBS IN {wilaya}:")
    headers = ["#", "Job ID", "Title", "Location", "Score"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def test_api_connection():
    """Test if the API is running."""
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status()
        return True
    except Exception:
        return False

def main():
    """Main function to run all tests."""
    print("JibJob Recommendation System - Algerian Data Test")
    print("=" * 60)
    
    # Check if the API is running
    if not test_api_connection():
        print("ERROR: Could not connect to the API. Make sure it's running.")
        print("Run the API server with: python src/demo_api.py")
        sys.exit(1)
    
    # Run tests
    try:
        test_recommendations()
        test_location_based_search()
        test_location_based_recommendations()
        
        print("\nAll tests completed successfully!")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
