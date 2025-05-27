"""
Demonstration of the JibJob recommendation API.

This script demonstrates how to:
1. Start the FastAPI server
2. Load a pre-trained model
3. Make API calls to the recommendation service

Usage:
    # First start the API server in one terminal:
    uvicorn src.api.main:app --reload
    
    # Then run this script in another terminal:
    python api_demo.py
"""

import sys
import requests
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API base URL
API_URL = "http://localhost:8000"


def test_api_health():
    """Test the API health check endpoint."""
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        logger.info("API is healthy: " + response.json()["status"])
        return True
    else:
        logger.error(f"API health check failed: {response.status_code} - {response.text}")
        return False


def load_model():
    """Send request to load the recommendation model."""
    logger.info("Sending request to load model...")
    response = requests.post(
        f"{API_URL}/model/load",
        json={"model_path": "jibjob_recommender_model"}
    )
    
    if response.status_code == 200:
        logger.info("Model loaded successfully")
        return True
    else:
        logger.error(f"Failed to load model: {response.status_code} - {response.text}")
        return False


def get_recommendations(user_id, top_k=5):
    """Get recommendations for a user."""
    logger.info(f"Getting recommendations for user {user_id}...")
    
    response = requests.post(
        f"{API_URL}/recommend",
        json={"user_id": user_id, "top_k": top_k, "exclude_rated": True}
    )
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Got {len(result['recommendations'])} recommendations in {result['processing_time']:.3f}s")
        return result
    else:
        logger.error(f"Failed to get recommendations: {response.status_code} - {response.text}")
        return None


def get_batch_recommendations(user_ids, top_k=5):
    """Get recommendations for multiple users."""
    logger.info(f"Getting batch recommendations for {len(user_ids)} users...")
    
    response = requests.post(
        f"{API_URL}/recommend/batch",
        json={"user_ids": user_ids, "top_k": top_k, "exclude_rated": True}
    )
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Got batch recommendations in {result['processing_time']:.3f}s")
        return result
    else:
        logger.error(f"Failed to get batch recommendations: {response.status_code} - {response.text}")
        return None


def find_similar_jobs(job_id, top_k=5):
    """Find similar jobs to a given job."""
    logger.info(f"Finding jobs similar to {job_id}...")
    
    response = requests.post(
        f"{API_URL}/jobs/similar",
        json={"job_id": job_id, "top_k": top_k}
    )
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Found {len(result['similar_jobs'])} similar jobs in {result['processing_time']:.3f}s")
        return result
    else:
        logger.error(f"Failed to find similar jobs: {response.status_code} - {response.text}")
        return None


def analyze_job_text(title, description, top_k=5):
    """Analyze job text and find similar existing jobs."""
    logger.info(f"Analyzing job text: {title}...")
    
    response = requests.post(
        f"{API_URL}/jobs/analyze",
        json={"title": title, "description": description, "top_k": top_k}
    )
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Analyzed job text in {result['processing_time']:.3f}s")
        return result
    else:
        logger.error(f"Failed to analyze job text: {response.status_code} - {response.text}")
        return None


def analyze_sentiment(comment):
    """Analyze sentiment in a comment."""
    logger.info(f"Analyzing sentiment: {comment[:50]}...")
    
    response = requests.post(
        f"{API_URL}/analyze/sentiment",
        json={"text": comment}
    )
    
    if response.status_code == 200:
        result = response.json()
        logger.info(f"Sentiment score: {result['sentiment_score']:.3f} (in {result['processing_time']:.3f}s)")
        return result
    else:
        logger.error(f"Failed to analyze sentiment: {response.status_code} - {response.text}")
        return None


def main():
    """Run the API demo."""
    logger.info("Starting JibJob Recommendation API demo...")
    
    # Check if API is running
    if not test_api_health():
        logger.error("API is not running. Please start the server with: uvicorn src.api.main:app --reload")
        return
    
    # Load the recommendation model
    if not load_model():
        logger.error("Failed to load model. Make sure you've trained and saved the model first.")
        return
    
    # Sample users from our generated data
    # In a real system, these would be actual user IDs from your database
    sample_users = ["user_1", "user_2", "user_3", "user_4", "user_5"]
    
    # Get recommendations for a single user
    user_recs = get_recommendations(sample_users[0], top_k=5)
    
    if user_recs:
        print(f"\nRecommendations for {user_recs['user_id']}:")
        for rec in user_recs['recommendations']:
            print(f"  - Job {rec['job_id']}: Score = {rec['score']:.4f}")
    
    # Get batch recommendations for multiple users
    batch_recs = get_batch_recommendations(sample_users, top_k=3)
    
    if batch_recs:
        print("\nBatch Recommendations:")
        for user_id, recs in batch_recs['results'].items():
            print(f"\n  User {user_id}:")
            for rec in recs:
                print(f"    - Job {rec['job_id']}: Score = {rec['score']:.4f}")
    
    # Find similar jobs
    # In a real system, this would be an actual job ID from your database
    similar_jobs_result = find_similar_jobs("job_1", top_k=5)
    
    if similar_jobs_result:
        print(f"\nJobs similar to {similar_jobs_result['job_id']}:")
        for rec in similar_jobs_result['similar_jobs']:
            print(f"  - Job {rec['job_id']}: Similarity = {rec['similarity']:.4f}")
    
    # Analyze job text
    job_analysis = analyze_job_text(
        title="Fix kitchen plumbing",
        description="Need a plumber to fix leaking pipes under the kitchen sink. Available on weekends.",
        top_k=5
    )
    
    if job_analysis:
        print("\nAnalyzed job text and found similar jobs:")
        for rec in job_analysis['similar_jobs']:
            print(f"  - Job {rec['job_id']}: Similarity = {rec['similarity']:.4f}")
    
    # Analyze sentiment in comments
    sample_comments = [
        "The service was excellent and completed on time!",
        "The job was done but not to my satisfaction.",
        "Terrible service, would not recommend.",
        "Very professional and skilled worker."
    ]
    
    print("\nSentiment Analysis Results:")
    for comment in sample_comments:
        sentiment_result = analyze_sentiment(comment)
        if sentiment_result:
            score = sentiment_result['sentiment_score']
            print(f"  - '{comment}': {score:.3f} ({get_sentiment_label(score)})")
    
    logger.info("API demo completed successfully!")


def get_sentiment_label(score):
    """Convert a sentiment score to a human-readable label."""
    if score >= 0.75:
        return "Very Positive"
    elif score >= 0.6:
        return "Positive"
    elif score >= 0.4:
        return "Neutral"
    elif score >= 0.25:
        return "Negative"
    else:
        return "Very Negative"


if __name__ == "__main__":
    main()
