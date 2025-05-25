"""
Test script to verify sentiment analysis and feature engineering.
"""
import os
import sys
import logging
import pandas as pd
import traceback

# Set up logging with both file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_modules.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Print diagnostic info
logger.info("Test modules starting")
logger.info(f"Current directory: {os.getcwd()}")

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logger.info(f"Path after update: {sys.path}")

try:
    logger.info("Importing sentiment analysis module...")
    from sentiment_analysis_module import SentimentAnalyzer
    logger.info("Sentiment analysis module imported successfully")
    
    logger.info("Importing feature engineering module...")
    from feature_engineering import FeatureEngineer
    logger.info("Feature engineering module imported successfully")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def test_sentiment_analyzer():
    logger.info("Testing Sentiment Analyzer...")
    try:
        analyzer = SentimentAnalyzer()
        test_text = "This job opportunity is amazing!"
        result = analyzer.predict_sentiment(test_text)
        logger.info(f"Test text: {test_text}")
        logger.info(f"Sentiment result: {result}")
        return True
    except Exception as e:
        logger.error(f"Error in sentiment analysis test: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def test_feature_engineer():
    logger.info("\nTesting Feature Engineer...")
    try:
        engineer = FeatureEngineer()
        
        # Load sample data
        logger.info("Loading sample data...")
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        logger.info(f"Data directory: {data_dir}")
        
        # Check if data files exist
        interactions_path = os.path.join(data_dir, 'interactions_df.csv')
        jobs_path = os.path.join(data_dir, 'jobs_df.csv')
        logger.info(f"Interactions file exists: {os.path.exists(interactions_path)}")
        logger.info(f"Jobs file exists: {os.path.exists(jobs_path)}")
        
        interactions_df = pd.read_csv(interactions_path)
        jobs_df = pd.read_csv(jobs_path)
        
        logger.info(f"Interactions shape: {interactions_df.shape}")
        logger.info(f"Jobs shape: {jobs_df.shape}")
        
        logger.info("Processing one interaction...")
        if not interactions_df.empty and 'commentaire_texte_anglais' in interactions_df.columns:
            sample = interactions_df.iloc[0]
            if pd.notna(sample['commentaire_texte_anglais']):
                sentiment = engineer.sentiment_analyzer.predict_sentiment(sample['commentaire_texte_anglais'])
                logger.info(f"Sample comment: {sample['commentaire_texte_anglais']}")
                logger.info(f"Sentiment: {sentiment}")
        
        logger.info("Generating one job embedding...")
        if not jobs_df.empty and 'description_mission_anglais' in jobs_df.columns:
            sample_job = jobs_df.iloc[0]
            embedding = engineer.get_bert_embedding(sample_job['description_mission_anglais'])
            logger.info(f"Sample job description: {sample_job['description_mission_anglais']}")
            logger.info(f"Embedding shape: {embedding.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error in feature engineering test: {str(e)}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting tests...")
    
    # Test sentiment analyzer
    if test_sentiment_analyzer():
        logger.info("Sentiment analyzer test passed!")
    else:
        logger.error("Sentiment analyzer test failed!")
        sys.exit(1)
    
    # Test feature engineer
    if test_feature_engineer():
        logger.info("Feature engineer test passed!")
    else:
        logger.error("Feature engineer test failed!")
        sys.exit(1)
    
    logger.info("All tests completed successfully!")
