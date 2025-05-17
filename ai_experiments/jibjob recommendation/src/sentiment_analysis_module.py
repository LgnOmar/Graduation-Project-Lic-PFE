"""
Simple sentiment analysis module using transformers.
"""
from transformers import pipeline
import torch
import pandas as pd
from typing import Dict

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with a pre-trained model."""
        print("Initializing sentiment analysis model...")
        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def predict_sentiment(self, text: str) -> Dict[str, float]:
        """
        Predict sentiment for a given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing sentiment score and label
        """
        if pd.isna(text) or not text.strip():
            return {'score': 0.0, 'label': 'NEUTRAL'}
            
        try:
            result = self.pipeline(text)[0]
            
            # Convert POSITIVE/NEGATIVE to numeric score between -1 and 1
            score = result['score']
            if result['label'] == 'NEGATIVE':
                score = -score
                
            return {
                'score': score,
                'label': result['label']
            }
        except Exception as e:
            print(f"Error predicting sentiment: {e}")
            return {'score': 0.0, 'label': 'NEUTRAL'}
