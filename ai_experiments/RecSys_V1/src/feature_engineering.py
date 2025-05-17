"""
Module for feature engineering, including sentiment analysis and BERT embeddings.
"""
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List
import logging
from sentiment_analysis_module import SentimentAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize the feature engineer.
        
        Args:
            model_name: Name of the BERT model to use for embeddings
        """
        logger.info(f"Initializing FeatureEngineer with model: {model_name}")
        try:
            self.bert_model = AutoModel.from_pretrained(model_name)
            self.bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_analyzer = SentimentAnalyzer()
            logger.info("Successfully initialized all components")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
            
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.bert_model = self.bert_model.to(self.device)
        
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Generate BERT embedding for a text.
        
        Args:
            text: Input text
            
        Returns:
            numpy array of embedding
        """
        # Tokenize and prepare input
        inputs = self.bert_tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding[0]  # Return as 1D array
        
    def process_interactions(
        self,
        interactions_df: pd.DataFrame,
        rating_weight: float = 0.7,
        sentiment_weight: float = 0.3
    ) -> pd.DataFrame:
        """
        Process interactions to add sentiment scores and enhanced ratings.
        
        Args:
            interactions_df: DataFrame of user-job interactions
            rating_weight: Weight for explicit rating in enhanced rating
            sentiment_weight: Weight for sentiment score in enhanced rating
            
        Returns:
            Processed DataFrame with new columns
        """
        # Copy DataFrame to avoid modifying original
        df = interactions_df.copy()
        
        # Add sentiment scores
        logger.info("Calculating sentiment scores...")
        sentiments = df['commentaire_texte_anglais'].apply(
            lambda x: self.sentiment_analyzer.predict_sentiment(x)
        )
        df['sentiment_score'] = sentiments.apply(lambda x: x['score'])
        df['sentiment_label'] = sentiments.apply(lambda x: x['label'])
        
        # Normalize explicit ratings to [0, 1]
        df['normalized_rating'] = (df['rating_explicite'] - 1) / 4
        
        # Normalize sentiment scores to [0, 1]
        df['normalized_sentiment'] = (df['sentiment_score'] + 1) / 2
        
        # Calculate enhanced rating
        df['enhanced_rating'] = (
            rating_weight * df['normalized_rating'].fillna(0.5) +
            sentiment_weight * df['normalized_sentiment'].fillna(0.5)
        )
        
        logger.info("Finished processing interactions")
        return df
        
    def process_jobs(self, jobs_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate BERT embeddings for job descriptions.
        
        Args:
            jobs_df: DataFrame containing job information
            
        Returns:
            Dictionary mapping job_id to embedding array
        """
        embeddings = {}
        logger.info("Generating BERT embeddings for job descriptions...")
        for _, row in jobs_df.iterrows():
            job_id = row['job_id']
            description = row['description_mission_anglais']
            embedding = self.get_bert_embedding(description)
            embeddings[job_id] = embedding
            
        logger.info("Finished generating job embeddings")
        return embeddings

def process_and_save_features(
    interactions_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    output_dir: str = 'data'
) -> None:
    """
    Process and save features for the recommendation system.
    
    Args:
        interactions_df: DataFrame of user-job interactions
        jobs_df: DataFrame of job information
        output_dir: Directory to save processed features
    """
    import os
    import pickle
    
    # Initialize feature engineer
    logger.info("Initializing feature engineer...")
    engineer = FeatureEngineer()
    
    # Process interactions
    logger.info("Processing interactions...")
    processed_interactions = engineer.process_interactions(interactions_df)
    
    # Generate job embeddings
    logger.info("Generating job embeddings...")
    job_embeddings = engineer.process_jobs(jobs_df)
    
    # Save processed data
    logger.info(f"Saving processed data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    processed_interactions.to_csv(f'{output_dir}/processed_interactions.csv', index=False)
    
    with open(f'{output_dir}/job_embeddings.pkl', 'wb') as f:
        pickle.dump(job_embeddings, f)
    logger.info("Feature processing complete!")

if __name__ == "__main__":
    # Load raw data
    logger.info("Loading raw data...")
    interactions_df = pd.read_csv('data/interactions_df.csv')
    jobs_df = pd.read_csv('data/jobs_df.csv')
    
    # Process and save features
    process_and_save_features(interactions_df, jobs_df)
