"""
Tests for the main JobRecommender system.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from src.models.recommender import JobRecommender
from src.models.bert_embeddings import BERTEmbeddings
from src.models.sentiment_analysis import SentimentAnalysis
from src.models.gcn import GCNRecommender
import tempfile
import os


class TestJobRecommender:
    """Tests for the JobRecommender class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample user data
        users = pd.DataFrame({
            'user_id': [1, 2, 3, 4, 5],
            'username': ['user1', 'user2', 'user3', 'user4', 'user5'],
            'location': ['Algiers', 'Oran', 'Constantine', 'Annaba', 'Blida']
        })
        
        # Create sample job data
        jobs = pd.DataFrame({
            'job_id': [101, 102, 103, 104, 105],
            'title': [
                'Plumbing repair needed',
                'House painting service',
                'Gardening assistance required',
                'Furniture assembly help',
                'Computer setup and configuration'
            ],
            'description': [
                'Need a plumber to fix kitchen sink leak',
                'Looking for someone to paint living room walls',
                'Need help with garden maintenance and planting',
                'Need assistance assembling new furniture',
                'Need help setting up new computer and software'
            ],
            'category': ['Plumbing', 'Painting', 'Gardening', 'Assembly', 'Tech'],
            'location': ['Algiers', 'Oran', 'Algiers', 'Constantine', 'Blida']
        })
        
        # Create sample interactions
        interactions = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 4, 5],
            'job_id': [101, 103, 102, 104, 103, 105, 101],
            'rating': [4.5, 3.0, 5.0, 4.0, 4.5, 3.5, 2.0],
            'comment': [
                'Good service, fixed the issue quickly',
                'Did an okay job but was a bit late',
                'Excellent work, very professional',
                'Good assembly work, but took longer than expected',
                'Great gardening service, will hire again',
                'Helped with computer setup but missed some software',
                'Plumbing work was not satisfactory'
            ]
        })
        
        return {'users': users, 'jobs': jobs, 'interactions': interactions}
    
    @pytest.fixture
    def recommender(self):
        """Create a test recommender with minimal model initialization."""
        # Use smaller/faster models for testing
        return JobRecommender(
            embedding_model_name="distilbert-base-uncased",
            sentiment_model_name="distilbert-base-uncased-finetuned-sst-2-english",
            device="cpu"
        )
    
    def test_init(self, recommender):
        """Test initialization of the recommender system."""
        assert recommender is not None
        assert isinstance(recommender.embedding_model, BERTEmbeddings)
        assert isinstance(recommender.sentiment_model, SentimentAnalysis)
        assert recommender.graph_model is None  # Should be None until trained
    
    def test_process_job_texts(self, recommender, sample_data):
        """Test processing of job text data."""
        jobs_df = sample_data['jobs']
        
        # Process job texts
        job_embeddings = recommender.process_job_texts(
            jobs_df,
            text_columns=['title', 'description'],
            job_id_col='job_id'
        )
        
        # Check result structure
        assert isinstance(job_embeddings, dict)
        assert len(job_embeddings) == len(jobs_df)
        
        # Check embeddings format
        for job_id, embedding in job_embeddings.items():
            assert job_id in jobs_df['job_id'].values
            assert isinstance(embedding, np.ndarray)
            assert embedding.ndim == 1
            assert embedding.shape[0] > 0
    
    def test_process_user_comments(self, recommender, sample_data):
        """Test sentiment analysis of user comments."""
        interactions_df = sample_data['interactions']
        
        # Process comments
        sentiment_scores = recommender.process_user_comments(
            interactions_df,
            comment_col='comment',
            user_id_col='user_id',
            job_id_col='job_id'
        )
        
        # Check result structure
        assert isinstance(sentiment_scores, dict)
        
        # Check that we have sentiment scores for all interactions with comments
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            job_id = row['job_id']
            
            if user_id in sentiment_scores and job_id in sentiment_scores[user_id]:
                score = sentiment_scores[user_id][job_id]
                assert isinstance(score, float)
                assert 0 <= score <= 1
    
    def test_calculate_enhanced_ratings(self, recommender, sample_data):
        """Test calculation of enhanced ratings."""
        interactions_df = sample_data['interactions']
        
        # Create dummy sentiment scores (normally would come from process_user_comments)
        sentiment_scores = {}
        for _, row in interactions_df.iterrows():
            user_id = row['user_id']
            job_id = row['job_id']
            if user_id not in sentiment_scores:
                sentiment_scores[user_id] = {}
            sentiment_scores[user_id][job_id] = 0.7  # Dummy sentiment score
        
        # Calculate enhanced ratings
        enhanced_df = recommender.calculate_enhanced_ratings(
            interactions_df,
            sentiment_scores,
            rating_col='rating',
            user_id_col='user_id',
            job_id_col='job_id',
            rating_weight=0.6,
            sentiment_weight=0.4
        )
        
        # Check result structure
        assert isinstance(enhanced_df, pd.DataFrame)
        assert len(enhanced_df) == len(interactions_df)
        assert 'enhanced_rating' in enhanced_df.columns
        
        # Check that enhanced ratings are between 0 and 5 (assuming original scale was 0-5)
        assert enhanced_df['enhanced_rating'].min() >= 0
        assert enhanced_df['enhanced_rating'].max() <= 5
        
        # Check calculation for a specific row
        for _, row in enhanced_df.iterrows():
            user_id = row['user_id']
            job_id = row['job_id']
            
            # Get original rating and sentiment
            original_rating = row['rating']
            sentiment = sentiment_scores[user_id][job_id]
            
            # Normalize the rating to 0-1 scale for comparison
            normalized_rating = original_rating / 5.0
            
            # Calculate expected enhanced rating
            expected_enhanced = (0.6 * normalized_rating + 0.4 * sentiment) * 5.0
            
            # Check that our calculation matches
            assert abs(row['enhanced_rating'] - expected_enhanced) < 0.01
    
    def test_train_and_recommend(self, recommender, sample_data):
        """Test the end-to-end training and recommendation process."""
        # Prepare data
        users_df = sample_data['users']
        jobs_df = sample_data['jobs']
        interactions_df = sample_data['interactions']
        
        # Train the recommender (with minimal settings for testing)
        recommender.train(
            users_df=users_df,
            jobs_df=jobs_df,
            interactions_df=interactions_df,
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating',
            comment_col='comment',
            epochs=1,
            batch_size=2,
            embedding_dim=16,
            hidden_dim=16
        )
        
        # Check that the graph model was created
        assert recommender.graph_model is not None
        assert isinstance(recommender.graph_model, GCNRecommender)
        
        # Get recommendations
        recommendations = recommender.recommend(
            user_ids=[1, 2],
            top_k=2,
            exclude_rated=True
        )
        
        # Check recommendations structure
        assert isinstance(recommendations, dict)
        assert 1 in recommendations
        assert 2 in recommendations
        
        # Check recommendation format
        for user_id, recs in recommendations.items():
            assert isinstance(recs, list)
            assert len(recs) <= 2  # May be less if user has rated many jobs
            
            for rec in recs:
                assert isinstance(rec, tuple)
                job_id, score = rec
                assert job_id in jobs_df['job_id'].values
                assert isinstance(score, float)
    
    def test_save_and_load_model(self, recommender, sample_data):
        """Test saving and loading the recommender model."""
        # Train a simple model first
        users_df = sample_data['users']
        jobs_df = sample_data['jobs']
        interactions_df = sample_data['interactions']
        
        recommender.train(
            users_df=users_df,
            jobs_df=jobs_df,
            interactions_df=interactions_df,
            user_id_col='user_id',
            job_id_col='job_id',
            rating_col='rating',
            comment_col='comment',
            epochs=1,
            embedding_dim=16,
            hidden_dim=16
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            model_path = os.path.join(temp_dir, "recommender")
            recommender.save_model(model_path)
            
            # Check that files were created
            assert os.path.exists(os.path.join(model_path, "config.json"))
            assert os.path.exists(os.path.join(model_path, "graph_model.pt"))
            
            # Create new recommender and load saved model
            new_recommender = JobRecommender(device="cpu")
            new_recommender.load_model(model_path)
            
            # Check that loading worked
            assert new_recommender.graph_model is not None
            assert new_recommender.user_mapping is not None
            assert new_recommender.job_mapping is not None
            
            # Check recommendations
            orig_recs = recommender.recommend([1], top_k=2)
            new_recs = new_recommender.recommend([1], top_k=2)
            
            # Recommendations should be the same
            assert orig_recs[1][0][0] == new_recs[1][0][0]  # Same job ID
            assert abs(orig_recs[1][0][1] - new_recs[1][0][1]) < 0.01  # Similar score
