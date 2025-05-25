"""
Interactive demo for the JibJob recommendation system.
"""
import os
import sys
import pandas as pd
import numpy as np
import pickle
import logging
from typing import List, Dict, Tuple
import requests
from time import sleep
import random
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JibJobDemo:
    def __init__(self, api_url: str = "http://localhost:8000"):
        """Initialize the demo client."""
        self.api_url = api_url
        self.users_df = None
        self.jobs_df = None
        self.api_available = False
        
    def load_data(self) -> bool:
        """Load necessary data for the demo."""
        try:
            # Load data
            self.users_df = pd.read_csv('data/users_df.csv')
            self.jobs_df = pd.read_csv('data/jobs_df.csv')
            
            # Sample information
            logger.info(f"Loaded {len(self.users_df)} users and {len(self.jobs_df)} jobs.")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False
            
    def check_api(self) -> bool:
        """Check if API is running."""
        try:
            response = requests.get(f"{self.api_url}/docs", timeout=2)
            self.api_available = response.status_code < 400
            if self.api_available:
                logger.info("API is running and available.")
            else:
                logger.error(f"API returned status code {response.status_code}")
            return self.api_available
        except requests.exceptions.RequestException:
            logger.error("API is not running. Please start the API server first.")
            logger.info("Run: python src/api.py")
            return False
            
    def get_user_info(self, user_id: str) -> Dict:
        """Get user information."""
        user = self.users_df[self.users_df['user_id'] == user_id]
        if len(user) == 0:
            return {"error": "User not found"}
            
        return user.iloc[0].to_dict()
        
    def get_job_info(self, job_id: str) -> Dict:
        """Get job information."""
        job = self.jobs_df[self.jobs_df['job_id'] == job_id]
        if len(job) == 0:
            return {"error": "Job not found"}
            
        return job.iloc[0].to_dict()
        
    def get_recommendations(self, user_id: str, top_n: int = 5) -> List[Dict]:
        """Get recommendations for a user."""
        if not self.api_available:
            return [{"error": "API not available"}]
            
        try:
            response = requests.get(f"{self.api_url}/recommendations/{user_id}?top_n={top_n}")
            if response.status_code == 200:
                data = response.json()
                
                # Enrich with job details
                result = []
                for job_id, score in zip(data['job_ids'], data['scores']):
                    job_info = self.get_job_info(job_id)
                    job_info['score'] = score
                    result.append(job_info)
                return result
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return [{"error": f"API returned status {response.status_code}"}]
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return [{"error": str(e)}]
            
    def simulate_user_session(self, user_id: str = None) -> None:
        """Simulate a user session with recommendations and interactions."""
        # Pick random user if not specified
        if user_id is None:
            user_id = random.choice(self.users_df['user_id'].tolist())
            
        # Get user info
        user_info = self.get_user_info(user_id)
        print("\n" + "="*50)
        print(f"User Session: {user_id}")
        print("="*50)
        print(f"- Age: {user_info.get('age', 'N/A')}")
        print(f"- Ville: {user_info.get('ville', 'N/A')}")
        print(f"- Experience: {user_info.get('experience', 'N/A')}")
        print("="*50)
        
        # Get recommendations
        print("\nGetting recommendations...")
        recommendations = self.get_recommendations(user_id, top_n=5)
        
        print("\n📋 Recommended Jobs:")
        for i, job in enumerate(recommendations):
            print(f"\n{i+1}. {job.get('titre_emploi', 'Job Title N/A')} (Score: {job.get('score', 'N/A'):.4f})")
            print(f"   📍 {job.get('ville', 'Location N/A')} | {job.get('categorie', 'Category N/A')}")
            print(f"   💼 Description: {job.get('description_mission_anglais', '')[:100]}...")
            
        # Simulate interaction
        print("\n🔄 Simulating user interaction...")
        sleep(1.5)  # Simulate thinking time
        
        # Select random job and interaction
        selected_job = random.choice(recommendations)
        interaction = random.choice(['applied', 'saved', 'viewed', 'ignored'])
        job_id = selected_job.get('job_id')
        
        print(f"\nUser {interaction} job: {selected_job.get('titre_emploi', job_id)}")
        
        # In a real system, we would log this interaction to update the model
        print("\n✅ Interaction recorded. This would be used to improve future recommendations.")
        
    def run_demo(self) -> None:
        """Run the interactive demo."""
        print("\n🚀 JibJob Recommendation System Demo")
        
        if not self.load_data():
            print("❌ Failed to load data. Exiting demo.")
            return
            
        if not self.check_api():
            print("❌ API is not available. Exiting demo.")
            return
        
        # Main demo loop
        while True:
            print("\n" + "="*50)
            print("Options:")
            print("1. Get recommendations for a random user")
            print("2. Get recommendations for a specific user")
            print("3. Get detailed job information")
            print("4. Exit demo")
            print("="*50)
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                self.simulate_user_session()
            elif choice == '2':
                user_ids = self.users_df['user_id'].tolist()
                print(f"\nAvailable user IDs (first 5): {user_ids[:5]}")
                user_id = input("\nEnter a user ID: ")
                self.simulate_user_session(user_id)
            elif choice == '3':
                job_id = input("\nEnter a job ID (e.g., job_1): ")
                job_info = self.get_job_info(job_id)
                if "error" in job_info:
                    print(f"\n❌ {job_info['error']}")
                else:
                    print("\n" + "="*50)
                    print(f"Job Details: {job_id}")
                    print("="*50)
                    print(f"Title: {job_info.get('titre_emploi', 'N/A')}")
                    print(f"Category: {job_info.get('categorie', 'N/A')}")
                    print(f"Location: {job_info.get('ville', 'N/A')}")
                    print(f"\nDescription:\n{job_info.get('description_mission_anglais', 'N/A')}")
                    print("="*50)
            elif choice == '4':
                print("\n👋 Thank you for using the JibJob Recommendation Demo!")
                break
            else:
                print("\n❌ Invalid choice. Please try again.")
                
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    demo = JibJobDemo()
    demo.run_demo()
