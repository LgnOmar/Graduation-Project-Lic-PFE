"""
Tests for the JibJob recommendation API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
import numpy as np
import os

from src.api.main import app


class TestRecommendationAPI:
    """Tests for the FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "JibJob Recommendation API" in response.json()["message"]
    
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    @patch("src.api.main.recommender")
    def test_get_recommendations(self, mock_recommender, client):
        """Test the recommendation endpoint."""
        # Mock the recommender's recommend method
        mock_recommender.recommend.return_value = {
            "user123": [(101, 0.95), (102, 0.85), (103, 0.75)]
        }
        
        # Make a request
        response = client.post(
            "/recommend",
            json={"user_id": "user123", "top_k": 3}
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user123"
        assert len(data["recommendations"]) == 3
        assert data["recommendations"][0]["job_id"] == 101
        assert data["recommendations"][0]["score"] == 0.95
    
    @patch("src.api.main.recommender")
    def test_batch_recommendations(self, mock_recommender, client):
        """Test the batch recommendation endpoint."""
        # Mock the recommender's batch_recommend method
        mock_recommender.recommend.return_value = {
            "user1": [(101, 0.9), (102, 0.8)],
            "user2": [(103, 0.9), (104, 0.8)]
        }
        
        # Make a request
        response = client.post(
            "/recommend/batch",
            json={"user_ids": ["user1", "user2"], "top_k": 2}
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"]["user1"][0]["job_id"] == 101
        assert data["results"]["user2"][0]["job_id"] == 103
    
    @patch("src.api.main.recommender")
    def test_similar_jobs(self, mock_recommender, client):
        """Test the similar jobs endpoint."""
        # Mock the recommender's find_similar_jobs method
        mock_recommender.find_similar_jobs.return_value = [
            (102, 0.9), (103, 0.8), (104, 0.7)
        ]
        
        # Make a request
        response = client.post(
            "/jobs/similar",
            json={"job_id": 101, "top_k": 3}
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == 101
        assert len(data["similar_jobs"]) == 3
        assert data["similar_jobs"][0]["job_id"] == 102
        assert data["similar_jobs"][0]["similarity"] == 0.9
    
    @patch("src.api.main.recommender")
    def test_analyze_job_text(self, mock_recommender, client):
        """Test the job text analysis endpoint."""
        # Mock the recommender's analyze_job_text method
        mock_embeddings = np.random.rand(768).tolist()
        mock_recommender.embedding_model.get_embeddings.return_value = np.array([mock_embeddings])
        mock_recommender.find_similar_jobs_by_embedding.return_value = [
            (101, 0.9), (102, 0.8), (103, 0.7)
        ]
        
        # Make a request
        response = client.post(
            "/jobs/analyze",
            json={
                "title": "Plumbing repair",
                "description": "Need help with a sink leak",
                "top_k": 3
            }
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "embedding" in data
        assert len(data["embedding"]) == 768
        assert "similar_jobs" in data
        assert len(data["similar_jobs"]) == 3
    
    @patch("src.api.main.recommender")
    def test_analyze_sentiment(self, mock_recommender, client):
        """Test the sentiment analysis endpoint."""
        # Mock the recommender's sentiment_model.analyze_sentiment method
        mock_recommender.sentiment_model.analyze_sentiment.return_value = 0.85
        
        # Make a request
        response = client.post(
            "/analyze/sentiment",
            json={"text": "The service was excellent and the worker was very professional."}
        )
        
        # Check response
        assert response.status_code == 200
        data = response.json()
        assert "sentiment_score" in data
        assert data["sentiment_score"] == 0.85
    
    def test_error_handling(self, client):
        """Test error handling with invalid requests."""
        # Test with missing required parameter
        response = client.post(
            "/recommend",
            json={"top_k": 3}  # Missing user_id
        )
        assert response.status_code == 422  # Unprocessable Entity
        
        # Test with invalid parameter value
        response = client.post(
            "/recommend",
            json={"user_id": "user123", "top_k": -1}  # Invalid top_k
        )
        assert response.status_code == 422  # Unprocessable Entity
