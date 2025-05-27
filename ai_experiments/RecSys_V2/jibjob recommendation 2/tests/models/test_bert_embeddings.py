"""
Tests for the BERT embeddings model.
"""

import pytest
import torch
import numpy as np
from src.models.bert_embeddings import BERTEmbeddings


class TestBERTEmbeddings:
    """Tests for the BERTEmbeddings class."""
    
    @pytest.fixture
    def embedding_model(self):
        """Create a test BERT embeddings model."""
        return BERTEmbeddings(model_name="distilbert-base-uncased", device="cpu")

    def test_init(self, embedding_model):
        """Test initialization of the model."""
        assert embedding_model is not None
        assert embedding_model.device == "cpu"
        assert embedding_model.model is not None
        assert embedding_model.tokenizer is not None

    def test_get_embeddings_single(self, embedding_model):
        """Test getting embeddings for a single text."""
        text = "This is a test job description for plumbing work in Algiers."
        embedding = embedding_model.get_embeddings(text)
        
        # Check if the embedding has the correct shape and type
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 2  # Should be a 2D array with 1 row
        assert embedding.shape[0] == 1  # One text input = one row
        assert embedding.shape[1] > 0  # Should have non-zero dimensions

    def test_get_embeddings_batch(self, embedding_model):
        """Test getting embeddings for multiple texts."""
        texts = [
            "This is a test job description for plumbing work.",
            "Looking for someone to help with gardening tasks.",
            "Need assistance with moving furniture to a new apartment."
        ]
        
        embeddings = embedding_model.get_embeddings(texts)
        
        # Check if embeddings have the correct shape and type
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2  # Should be a 2D array
        assert embeddings.shape[0] == len(texts)  # Should have as many rows as input texts
        assert embeddings.shape[1] > 0  # Should have non-zero dimensions in embedding space
    
    def test_batch_get_embeddings(self, embedding_model):
        """Test batch processing of embeddings."""
        texts = [
            "This is a test job description for plumbing work.",
            "Looking for someone to help with gardening tasks.",
            "Need assistance with moving furniture to a new apartment.",
            "Require a professional electrician for house wiring."
        ]
        
        embeddings = embedding_model.batch_get_embeddings(texts, batch_size=2)
        
        # Check if embeddings have the correct shape and type
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0

    def test_calculate_text_similarity(self, embedding_model):
        """Test semantic similarity calculation."""
        text1 = "Need help with plumbing repair in the kitchen."
        text2 = "Looking for a plumber to fix my kitchen sink."
        text3 = "Require assistance with gardening and lawn mowing."
        
        # Calculate similarities
        sim12 = embedding_model.calculate_text_similarity(text1, text2)
        sim13 = embedding_model.calculate_text_similarity(text1, text3)
        
        # Similarity should be a float
        assert isinstance(sim12, float)
        assert isinstance(sim13, float)
        
        # Similarity between similar texts should be higher than between dissimilar texts
        assert sim12 > sim13
        
        # Test with different pooling and similarity metrics
        sim12_cls = embedding_model.calculate_text_similarity(
            text1, text2, pooling_strategy='cls'
        )
        sim12_dot = embedding_model.calculate_text_similarity(
            text1, text2, similarity_metric='dot'
        )
        
        assert isinstance(sim12_cls, float)
        assert isinstance(sim12_dot, float)
    
    def test_save_and_load_cache(self, embedding_model):
        """Test saving and loading model cache."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model to temp directory
            embedding_model.save_cache(temp_dir)
            
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, "config.json"))
            
            # Load model from cache
            loaded_model = BERTEmbeddings.load_from_cache(temp_dir, device="cpu")
            
            # Test the loaded model
            text = "This is a test job description."
            original_emb = embedding_model.get_embeddings(text)
            loaded_emb = loaded_model.get_embeddings(text)
            
            # Embeddings should be very similar or identical
            np.testing.assert_allclose(original_emb, loaded_emb, rtol=1e-5)

    def test_invalid_pooling_strategy(self, embedding_model):
        """Test that invalid pooling strategy raises an error."""
        with pytest.raises(ValueError):
            embedding_model.get_embeddings("Test text", pooling_strategy='invalid')
    
    def test_invalid_similarity_metric(self, embedding_model):
        """Test that invalid similarity metric raises an error."""
        with pytest.raises(ValueError):
            embedding_model.calculate_text_similarity(
                "Text 1", "Text 2", similarity_metric='invalid'
            )
