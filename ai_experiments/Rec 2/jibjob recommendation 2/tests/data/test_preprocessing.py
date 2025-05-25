"""
Tests for the data preprocessing utilities.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import (
    clean_text,
    process_job_descriptions,
    extract_features,
    normalize_ratings,
    split_train_test,
    create_user_job_matrices,
    create_interaction_tensors
)


class TestPreprocessing:
    """Tests for the preprocessing functions."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test basic cleaning
        text = "This is a TEST job description with numbers: 123, and punctuation!!!"
        cleaned = clean_text(text, remove_numbers=True, remove_punctuation=True, lowercase=True)
        
        assert "123" not in cleaned
        assert "!!!" not in cleaned
        assert "TEST" not in cleaned
        assert "test" in cleaned
        
        # Test stopword removal
        text = "This is a test with stopwords like and or the"
        cleaned = clean_text(text, remove_stopwords=True)
        
        assert "is" not in cleaned.split()
        assert "a" not in cleaned.split()
        assert "the" not in cleaned.split()
        assert "test" in cleaned
        assert "stopwords" in cleaned
        
        # Test stemming
        text = "Testing running and jumps"
        cleaned = clean_text(text, stemming=True)
        
        assert "testing" not in cleaned
        assert "running" not in cleaned
        assert "jumps" not in cleaned
        assert "test" in cleaned
        assert "run" in cleaned
        assert "jump" in cleaned
    
    def test_process_job_descriptions(self):
        """Test processing of job descriptions."""
        # Create sample data
        jobs_df = pd.DataFrame({
            'job_id': [1, 2, 3],
            'title': ['Plumber needed', 'Garden work', 'House painting'],
            'description': [
                'Need a plumber to fix sink',
                'Looking for gardening help',
                'Need to paint living room walls'
            ]
        })
        
        # Process descriptions
        processed_df = process_job_descriptions(
            jobs_df,
            text_columns=['title', 'description'],
            remove_stopwords=True,
            stemming=True
        )
        
        # Check that new columns were created
        assert 'processed_title' in processed_df.columns
        assert 'processed_description' in processed_df.columns
        assert 'combined_text' in processed_df.columns
        
        # Check that processing was applied
        for _, row in processed_df.iterrows():
            # Check that stopwords were removed
            assert "a" not in row['processed_description'].split()
            assert "the" not in row['processed_description'].split()
            
            # Check that combined text includes both title and description
            assert len(row['combined_text']) > len(row['processed_title'])
            assert len(row['combined_text']) > len(row['processed_description'])
    
    def test_extract_features(self):
        """Test feature extraction from text."""
        # Create sample data
        texts = [
            "Need a plumber for kitchen sink repair",
            "Looking for gardening assistance with lawn",
            "Need help with house painting living room",
            "Plumbing services required for bathroom"
        ]
        
        # Extract TF-IDF features
        features = extract_features(texts, method='tfidf', max_features=10)
        
        # Check result format
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(texts)
        assert features.shape[1] <= 10  # Max features
        
        # Test with different method
        features_count = extract_features(texts, method='count', max_features=10)
        assert isinstance(features_count, np.ndarray)
        assert features_count.shape[0] == len(texts)
        
        # Test with n-grams
        features_ngram = extract_features(
            texts, method='tfidf', max_features=10, ngram_range=(1, 2)
        )
        assert isinstance(features_ngram, np.ndarray)
        assert features_ngram.shape[0] == len(texts)
    
    def test_normalize_ratings(self):
        """Test rating normalization."""
        # Create sample data
        ratings = np.array([1.0, 3.0, 5.0, 2.5, 4.0])
        
        # Normalize to 0-1 range
        normalized = normalize_ratings(ratings)
        
        # Check result
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized[0] == 0.0    # 1.0 -> 0.0
        assert normalized[2] == 1.0    # 5.0 -> 1.0
        assert normalized[3] == 0.375  # 2.5 -> (2.5-1)/(5-1) = 0.375
        
        # Test with custom range
        normalized_custom = normalize_ratings(ratings, new_min=-1, new_max=1)
        assert normalized_custom.min() == -1.0
        assert normalized_custom.max() == 1.0
    
    def test_split_train_test(self):
        """Test train-test splitting."""
        # Create sample data
        interactions_df = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4],
            'job_id': [101, 102, 103, 104, 105, 106, 107],
            'rating': [4.5, 3.0, 5.0, 4.0, 3.5, 2.0, 4.5]
        })
        
        # Split data
        train_df, test_df = split_train_test(interactions_df, test_size=0.3)
        
        # Check result
        assert len(train_df) + len(test_df) == len(interactions_df)
        assert len(test_df) / len(interactions_df) <= 0.4  # Approximately 0.3
        assert len(test_df) / len(interactions_df) >= 0.2  # Approximately 0.3
        
        # Check that user-job pairs are unique across train and test sets
        train_pairs = set(zip(train_df['user_id'], train_df['job_id']))
        test_pairs = set(zip(test_df['user_id'], test_df['job_id']))
        assert len(train_pairs.intersection(test_pairs)) == 0
    
    def test_create_user_job_matrices(self):
        """Test creation of user-job matrices."""
        # Create sample data
        interactions_df = pd.DataFrame({
            'user_id': [101, 101, 102, 103],
            'job_id': [201, 202, 201, 203],
            'rating': [4.5, 3.0, 5.0, 4.0]
        })
        
        # Create matrices
        user_job_matrix, user_mapping, job_mapping = create_user_job_matrices(
            interactions_df,
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating'
        )
        
        # Check result format
        assert isinstance(user_job_matrix, np.ndarray)
        assert user_job_matrix.shape == (3, 3)  # 3 users x 3 jobs
        assert isinstance(user_mapping, dict)
        assert isinstance(job_mapping, dict)
        
        # Check mappings
        assert set(user_mapping.keys()) == {101, 102, 103}
        assert set(job_mapping.keys()) == {201, 202, 203}
        
        # Check values in matrix
        assert user_job_matrix[user_mapping[101], job_mapping[201]] == 4.5
        assert user_job_matrix[user_mapping[101], job_mapping[202]] == 3.0
        assert user_job_matrix[user_mapping[102], job_mapping[201]] == 5.0
        assert user_job_matrix[user_mapping[103], job_mapping[203]] == 4.0
        
        # Unrated should be 0
        assert user_job_matrix[user_mapping[102], job_mapping[202]] == 0.0
    
    def test_create_interaction_tensors(self):
        """Test creation of interaction tensors for PyTorch models."""
        # Create sample data
        interactions_df = pd.DataFrame({
            'user_id': [1, 1, 2, 3],
            'job_id': [101, 102, 101, 103],
            'rating': [4.5, 3.0, 5.0, 4.0]
        })
        
        # Create mapping dictionaries
        user_mapping = {1: 0, 2: 1, 3: 2}
        job_mapping = {101: 0, 102: 1, 103: 2}
        
        # Create tensors
        user_indices, job_indices, ratings = create_interaction_tensors(
            interactions_df,
            user_mapping,
            job_mapping,
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating'
        )
        
        # Check result format
        assert isinstance(user_indices, torch.Tensor)
        assert isinstance(job_indices, torch.Tensor)
        assert isinstance(ratings, torch.Tensor)
        
        # Check dimensions
        assert user_indices.shape == (4,)  # 4 interactions
        assert job_indices.shape == (4,)
        assert ratings.shape == (4,)
        
        # Check values
        assert user_indices[0].item() == 0  # user_id 1 -> 0
        assert job_indices[0].item() == 0  # job_id 101 -> 0
        assert ratings[0].item() == 4.5
