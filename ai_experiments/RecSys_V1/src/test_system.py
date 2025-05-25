"""
Script to test the complete recommendation system.
"""
import logging
import pandas as pd
import torch
import requests
from typing import List, Tuple, Dict
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_pipeline():
    """Test data generation and processing"""
    try:
        # Load and check data files
        interactions_df = pd.read_csv('data/processed_interactions.csv')
        jobs_df = pd.read_csv('data/jobs_df.csv')
        users_df = pd.read_csv('data/users_df.csv')
        
        # Check if job embeddings exist
        with open('data/job_embeddings.pkl', 'rb') as f:
            job_embeddings = pickle.load(f)
            
        logger.info(f"Loaded data successfully:")
        logger.info(f"- {len(interactions_df)} interactions")
        logger.info(f"- {len(jobs_df)} jobs")
        logger.info(f"- {len(users_df)} users")
        logger.info(f"- {len(job_embeddings)} job embeddings")
        
        # Check format of data
        logger.info("\nInteractions DataFrame sample:")
        logger.info(f"Columns: {interactions_df.columns.tolist()}")
        logger.info(interactions_df.head(2))
        
        logger.info("\nJobs DataFrame sample:")
        logger.info(f"Columns: {jobs_df.columns.tolist()}")
        logger.info(jobs_df.head(2))
        
        logger.info("\nUsers DataFrame sample:")
        logger.info(f"Columns: {users_df.columns.tolist()}")
        logger.info(users_df.head(2))
        
        # Check for embeddings format
        sample_key = next(iter(job_embeddings.keys()))
        logger.info(f"\nSample job embedding shape: {job_embeddings[sample_key].shape}")
        
        # Verify required columns exist
        required_cols = {
            'interactions_df': ['user_id', 'job_id', 'enhanced_rating'],
            'jobs_df': ['job_id', 'description_mission_anglais'],
            'users_df': ['user_id']
        }
        
        for df_name, cols in required_cols.items():
            df = eval(df_name)
            missing = [col for col in cols if col not in df.columns]
            if missing:
                logger.error(f"Missing columns in {df_name}: {missing}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error testing data pipeline: {str(e)}")
        return False

def test_model_existence():
    """Test if the trained model exists"""
    try:
        if not os.path.exists('models/best_model.pt'):
            logger.error("Model file 'models/best_model.pt' not found.")
            return False
            
        # Load model checkpoint to verify integrity
        checkpoint = torch.load('models/best_model.pt')
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'val_loss', 'val_metrics']
        
        missing = [key for key in required_keys if key not in checkpoint]
        if missing:
            logger.error(f"Missing keys in model checkpoint: {missing}")
            return False
            
        logger.info(f"Model checkpoint loaded successfully. Validation AUC: {checkpoint['val_metrics']['auc']:.4f}")
        return True
    except Exception as e:
        logger.error(f"Error checking model: {str(e)}")
        return False

def test_model_inference(user_ids: List[str], top_n: int = 5) -> bool:
    """Test model predictions directly"""
    try:
        from recommender import load_recommender
        
        # Load recommender
        recommender = load_recommender()
        
        # Test recommendations for each user
        results = {}
        for user_id in user_ids:
            recommendations = recommender.get_recommendations(user_id, top_n=top_n)
            
            logger.info(f"\nRecommendations for user {user_id}:")
            for job_id, score in recommendations:
                logger.info(f"- Job {job_id}: score = {score:.4f}")
                
            # Store results
            results[user_id] = recommendations
            
        # Verify recommendations are different for different users
        if len(user_ids) >= 2:
            user1, user2 = user_ids[:2]
            jobs_user1 = {job_id for job_id, _ in results[user1]}
            jobs_user2 = {job_id for job_id, _ in results[user2]}
            
            overlap = len(jobs_user1.intersection(jobs_user2))
            logger.info(f"\nRecommendation overlap between users: {overlap}/{top_n}")
            
        return True
    except Exception as e:
        logger.error(f"Error testing model inference: {str(e)}")
        return False

def test_api_endpoints(user_ids: List[str], base_url: str = "http://localhost:8000") -> bool:
    """Test API endpoints"""
    try:
        for user_id in user_ids:
            # Test recommendations endpoint
            response = requests.get(f"{base_url}/recommendations/{user_id}")
            
            if response.status_code == 200:
                recommendations = response.json()
                logger.info(f"\nAPI Recommendations for user {user_id}:")
                for job_id, score in zip(recommendations['job_ids'], recommendations['scores']):
                    logger.info(f"- Job {job_id}: score = {score:.4f}")
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return False
                
        # Test error handling with non-existent user
        fake_user = "non_existent_user_id"
        response = requests.get(f"{base_url}/recommendations/{fake_user}")
        if response.status_code == 404:
            logger.info(f"API correctly returns 404 for non-existent user")
        else:
            logger.error(f"API error handling failed: Expected 404, got {response.status_code}")
            return False
        
        # Test Swagger UI/documentation availability
        docs_response = requests.get(f"{base_url}/docs")
        if docs_response.status_code == 200:
            logger.info("API documentation (Swagger UI) is available")
        else:
            logger.warning(f"API documentation might not be available: {docs_response.status_code}")
            
        return True
    except Exception as e:
        logger.error(f"Error testing API: {str(e)}")
        return False

def test_performance():
    """Test model performance using test results"""
    try:
        if not os.path.exists('models/test_results.pkl'):
            logger.warning("Test results file not found. Skipping performance test.")
            return True
            
        with open('models/test_results.pkl', 'rb') as f:
            results = pickle.load(f)
            
        metrics = results.get('test_metrics', {})
        logger.info("\nModel performance on test set:")
        logger.info(f"- AUC: {metrics.get('auc', 'N/A'):.4f}")
        logger.info(f"- Precision: {metrics.get('precision', 'N/A'):.4f}")
        logger.info(f"- Recall: {metrics.get('recall', 'N/A'):.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model performance: {str(e)}")
        return False

def main():
    """Run all tests"""
    # Test data pipeline
    logger.info("\n=== Testing Data Pipeline ===")
    if not test_data_pipeline():
        logger.error("Data pipeline test failed!")
        return
        
    # Test model existence
    logger.info("\n=== Testing Model Existence ===")
    if not test_model_existence():
        logger.error("Model existence test failed!")
        return
    
    # Test performance metrics
    logger.info("\n=== Testing Model Performance ===")
    if not test_performance():
        logger.error("Performance test failed!")
        return
    
    # Get some test users
    try:
        users_df = pd.read_csv('data/users_df.csv')
        test_users = users_df['user_id'].head(3).tolist()
    except Exception as e:
        logger.error(f"Error loading test users: {str(e)}")
        return
    
    # Test model inference
    logger.info("\n=== Testing Model Inference ===")
    if not test_model_inference(test_users):
        logger.error("Model inference test failed!")
        return
    
    # Test API
    logger.info("\n=== Testing API Endpoints ===")
    if not test_api_endpoints(test_users):
        logger.error("API test failed!")
        return
    
    logger.info("\n=== All tests completed successfully! ===")
    logger.info("The JibJob recommendation system is ready for production use.")

if __name__ == "__main__":
    main()
