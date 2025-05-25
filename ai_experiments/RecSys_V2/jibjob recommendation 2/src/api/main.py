"""
Main API entry point for JibJob recommendation system.
This module provides the FastAPI application for serving recommendations.
"""

from fastapi import FastAPI, HTTPException, Query, Depends, Body, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from typing import List, Dict, Any, Optional
import json
import os
from pydantic import BaseModel, Field
import torch
import numpy as np
import torch.nn.functional as F

from src.models.recommender import JobRecommender
from src.utils.config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="JibJob Recommendation API",
    description="API for JibJob intelligent job recommendation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global object to store the loaded recommendation model
recommender: Optional[JobRecommender] = None

# Pydantic models for request/response validation
class JobRecommendationRequest(BaseModel):
    user_id: Any
    top_k: int = Field(default=5, ge=1, le=100)
    exclude_rated: bool = Field(default=True)
    rated_job_ids: Optional[List[Any]] = None

class BatchRecommendationRequest(BaseModel):
    user_ids: List[Any]
    top_k: int = Field(default=5, ge=1, le=100)
    exclude_rated: bool = Field(default=True)
    user_rated_jobs: Optional[Dict[str, List[Any]]] = None

class JobData(BaseModel):
    job_id: Any
    score: float

class RecommendationResponse(BaseModel):
    user_id: Any
    recommendations: List[JobData]
    processing_time: float

class BatchRecommendationResponse(BaseModel):
    results: Dict[str, List[JobData]]
    processing_time: float

class JobSimilarityRequest(BaseModel):
    job_id: Any
    top_k: int = Field(default=5, ge=1, le=100)

class SentimentAnalysisRequest(BaseModel):
    text: str

class SentimentAnalysisResponse(BaseModel):
    sentiment_score: float
    processing_time: float

class ErrorResponse(BaseModel):
    error: str
    details: Optional[Any] = None

# Dependency to ensure model is loaded
async def get_recommender() -> JobRecommender:
    """
    Get the loaded recommendation model.
    
    Returns:
        JobRecommender: Loaded model.
    
    Raises:
        HTTPException: If model is not loaded.
    """
    if recommender is None:
        raise HTTPException(
            status_code=503, 
            detail="Recommendation model not loaded. Call /load-model endpoint first."
        )
    return recommender

@app.on_event("startup")
async def startup_event():
    """Initialize API service on startup."""
    logger.info("Starting JibJob Recommendation API")
    
    # Attempt to load model if path specified in config
    config = get_config()
    model_path = config.get("paths_model_dir")
    
    if model_path and os.path.exists(model_path):
        try:
            logger.info(f"Attempting to load model from: {model_path}")
            await load_model_from_path(model_path)
        except Exception as e:
            logger.error(f"Failed to load model automatically: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down JibJob Recommendation API")
    # Nothing to clean up for now


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "JibJob Recommendation API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": recommender is not None
    }


@app.post("/load-model", response_model=Dict[str, Any])
async def load_model(model_path: str = Body(..., embed=True)):
    """
    Load a recommendation model from disk.
    
    Args:
        model_path: Path to the saved model.
    
    Returns:
        Dict[str, Any]: Status message.
    """
    return await load_model_from_path(model_path)


async def load_model_from_path(model_path: str) -> Dict[str, Any]:
    """
    Load a model from the given path.
    
    Args:
        model_path: Path to the saved model.
    
    Returns:
        Dict[str, Any]: Status message.
    
    Raises:
        HTTPException: If model loading fails.
    """
    global recommender
    
    start_time = time.time()
    
    try:
        # Force CPU device to avoid CUDA issues
        device = "cpu"
        
        # Autoriser les classes torch_geometric lors du chargement
        import torch.serialization
        torch.serialization.add_safe_globals(['torch_geometric.data.data.DataEdgeAttr'])
        
        # Load model
        logger.info(f"Loading model from {model_path} on {device}")
        recommender = JobRecommender.load_model(model_path, device=device)
        
        loading_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {loading_time:.2f} seconds")
        
        # Get model metadata
        metadata = recommender.metadata
        
        return {
            "status": "success",
            "message": "Model loaded successfully",
            "loading_time": loading_time,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@app.get("/model-info", response_model=Dict[str, Any])
async def get_model_info(rec: JobRecommender = Depends(get_recommender)):
    """
    Get information about the loaded recommendation model.
    
    Returns:
        Dict[str, Any]: Model information.
    """
    return {
        "metadata": rec.metadata,
        "num_users": len(rec.user_to_idx),
        "num_jobs": len(rec.job_to_idx),
        "device": rec.device,
        "model_type": type(rec.graph_model).__name__ if rec.graph_model else None
    }


@app.post(
    "/recommend", 
    response_model=RecommendationResponse,
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}
)
async def recommend_jobs(
    request: JobRecommendationRequest,
    rec: JobRecommender = Depends(get_recommender)
):
    """
    Get job recommendations for a user.
    
    Args:
        request: Recommendation request parameters.
    
    Returns:
        RecommendationResponse: Recommended jobs with scores.
    
    Raises:
        HTTPException: If user ID is not found or other errors occur.
    """
    start_time = time.time()
    
    try:
        # Convert user_id to string if it's numeric to match how it's stored
        user_id = request.user_id
        
        recommendations = rec.recommend(
            user_id=user_id,
            top_k=request.top_k,
            exclude_rated=request.exclude_rated,
            rated_job_ids=request.rated_job_ids
        )
        
        processing_time = time.time() - start_time
        
        return RecommendationResponse(
            user_id=user_id,
            recommendations=[JobData(**rec) for rec in recommendations],
            processing_time=processing_time
        )
        
    except ValueError as e:
        if "Unknown user ID" in str(e):
            raise HTTPException(
                status_code=404,
                detail=f"User not found: {request.user_id}"
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=str(e)
            )
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.post(
    "/recommend-batch", 
    response_model=BatchRecommendationResponse,
    responses={503: {"model": ErrorResponse}}
)
async def recommend_jobs_batch(
    request: BatchRecommendationRequest,
    rec: JobRecommender = Depends(get_recommender)
):
    """
    Get job recommendations for multiple users.
    
    Args:
        request: Batch recommendation request parameters.
    
    Returns:
        BatchRecommendationResponse: Recommendations for each user.
    
    Raises:
        HTTPException: If errors occur during processing.
    """
    start_time = time.time()
    
    try:
        recommendations = rec.recommend_batch(
            user_ids=request.user_ids,
            top_k=request.top_k,
            exclude_rated=request.exclude_rated,
            user_rated_jobs=request.user_rated_jobs
        )
        
        # Convert to response format
        results = {}
        for user_id, recs in recommendations.items():
            results[str(user_id)] = [JobData(job_id=r["job_id"], score=r["score"]) for r in recs]
        
        processing_time = time.time() - start_time
        
        return BatchRecommendationResponse(
            results=results,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error generating batch recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate batch recommendations: {str(e)}"
        )


@app.post(
    "/analyze-sentiment", 
    response_model=SentimentAnalysisResponse,
    responses={503: {"model": ErrorResponse}}
)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    rec: JobRecommender = Depends(get_recommender)
):
    """
    Analyze sentiment in the provided text.
    
    Args:
        request: Text to analyze.
    
    Returns:
        SentimentAnalysisResponse: Sentiment score.
    
    Raises:
        HTTPException: If errors occur during processing.
    """
    start_time = time.time()
    
    try:
        sentiment_score = rec.sentiment_model.analyze_sentiment(request.text)
        processing_time = time.time() - start_time
        
        return SentimentAnalysisResponse(
            sentiment_score=float(sentiment_score),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze sentiment: {str(e)}"
        )


@app.get(
    "/job-similarity/{job_id}", 
    response_model=List[JobData],
    responses={404: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}
)
async def get_job_similarity(
    job_id: str,
    top_k: int = Query(5, ge=1, le=100),
    rec: JobRecommender = Depends(get_recommender)
):
    """
    Get similar jobs based on embeddings.
    
    Args:
        job_id: ID of the job to find similarities for.
        top_k: Number of similar jobs to return.
    
    Returns:
        List[JobData]: Similar jobs with similarity scores.
    
    Raises:
        HTTPException: If job ID is not found or other errors occur.
    """
    try:
        # Get all job embeddings
        job_embeddings = rec.get_job_embeddings()
        
        if job_id not in job_embeddings:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        
        # Calculate similarities
        target_embedding = job_embeddings[job_id]
        similarities = []
        
        for other_id, embedding in job_embeddings.items():
            if other_id != job_id:
                # Cosine similarity
                similarity = np.dot(target_embedding, embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(embedding)
                )
                similarities.append((other_id, float(similarity)))
        
        # Sort by similarity (descending) and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_k]
        
        return [JobData(job_id=j_id, score=score) for j_id, score in top_similar]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar jobs: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to find similar jobs: {str(e)}"
        )


@app.get(
    "/health", 
    response_model=Dict[str, Any],
    responses={503: {"model": ErrorResponse}}
)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Dict[str, Any]: Service status information.
    """
    status = {
        "status": "healthy",
        "model_loaded": recommender is not None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if recommender is not None:
        status["model_info"] = {
            "num_users": len(recommender.user_to_idx),
            "num_jobs": len(recommender.job_to_idx),
            "device": recommender.device
        }
    
    return status


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for the API."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)}
    )


def run_api():
    """
    Run the API server.
    
    This function is called when running the module directly.
    """
    import uvicorn
    
    # Load configuration
    config = get_config()
    host = config.get("api_host", "0.0.0.0")
    port = int(config.get("api_port", 8000))
    workers = int(config.get("api_workers", 1))
    debug = bool(config.get("api_debug", False))
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    run_api()
