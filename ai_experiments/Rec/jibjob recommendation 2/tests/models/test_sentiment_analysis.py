"""
Tests for the sentiment analysis model.
"""

import pytest
import numpy as np
from src.models.sentiment_analysis import SentimentAnalysis


class TestSentimentAnalysis:
    """Tests for the SentimentAnalysis class."""
    
    @pytest.fixture
    def sentiment_model(self):
        """Create a test sentiment analysis model."""
        return SentimentAnalysis(model_name="distilbert-base-uncased-finetuned-sst-2-english", device="cpu")
    
    def test_init(self, sentiment_model):
        """Test initialization of the model."""
        assert sentiment_model is not None
        assert sentiment_model.model is not None
        assert sentiment_model.tokenizer is not None

    def test_analyze_sentiment(self, sentiment_model):
        """Test sentiment analysis on individual texts."""
        # Positive text
        positive_text = "The service was excellent and the worker completed the task efficiently."
        positive_score = sentiment_model.analyze_sentiment(positive_text)
        
        # Negative text
        negative_text = "The worker was late and did a poor job. I'm very disappointed."
        negative_score = sentiment_model.analyze_sentiment(negative_text)
        
        # Neutral text
        neutral_text = "The work was completed as requested."
        neutral_score = sentiment_model.analyze_sentiment(neutral_text)
        
        # Test with return_scores=True
        pos_score_with_raw, raw_scores = sentiment_model.analyze_sentiment(positive_text, return_scores=True)
        
        # Check that scores are between the expected range
        assert 0 <= positive_score <= 1
        assert 0 <= negative_score <= 1
        assert 0 <= neutral_score <= 1
        
        # Check that the positive text has a higher sentiment score than the negative one
        assert positive_score > negative_score
        
        # Check that neutral score is between positive and negative (or close to it)
        assert (neutral_score > negative_score) or pytest.approx(neutral_score, 0.2) == negative_score

    def test_batch_analyze_sentiment(self, sentiment_model):
        """Test sentiment analysis on a batch of texts."""
        texts = [
            "The service was excellent and prompt.",
            "The worker was late and did a bad job.",
            "The job was completed as expected."
        ]
        
        scores = sentiment_model.batch_analyze_sentiment(texts)
        
        # Check that we got the right number of scores
        assert len(scores) == len(texts)
        
        # Check that all scores are between 0 and 1
        for score in scores:
            assert 0 <= score <= 1
            
        # Check that the positive text has a higher sentiment score than the negative one
        assert scores[0] > scores[1]

    def test_fine_tune(self, sentiment_model):
        """Test the fine-tuning process with a tiny dataset."""
        texts = [
            "The service was excellent.",
            "The worker was professional.",
            "The job was terrible.",
            "Very disappointed with the service."
        ]
        labels = [1, 1, 0, 0]  # 1: positive, 0: negative
        
        # Mock fine-tuning with a very small dataset and just 1 epoch
        history = sentiment_model.fine_tune(
            texts=texts,
            labels=labels,
            epochs=1,
            batch_size=2
        )
        
        # Check that history contains expected metrics
        assert 'train_loss' in history
        assert 'train_acc' in history
        assert len(history['train_loss']) == 1  # One epoch
    
    def test_save_and_load_model(self, sentiment_model):
        """Test saving and loading the model."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model to temp directory
            sentiment_model.save_model(temp_dir)
            
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "config.json"))
            
            # Load model from saved files
            loaded_model = SentimentAnalysis.load_model(temp_dir, device="cpu")
            
            # Test the loaded model
            text = "This is a test review."
            original_sentiment = sentiment_model.analyze_sentiment(text)
            loaded_sentiment = loaded_model.analyze_sentiment(text)
            
            # Sentiments should be identical
            assert original_sentiment == loaded_sentiment
