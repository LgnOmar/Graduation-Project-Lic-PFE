"""
FastAPI application for serving job recommendations.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn

from recommender import load_recommender

app = FastAPI(title="JibJob Recommendation API")

# Load the recommender system at startup
recommender = load_recommender()

class RecommendationResponse(BaseModel):
    job_ids: List[str]
    scores: List[float]
    
@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    top_n: int = 10
) -> RecommendationResponse:
    """
    Get job recommendations for a user.
    
    Args:
        user_id: ID of the user to get recommendations for
        top_n: Number of recommendations to return
        
    Returns:
        RecommendationResponse containing job IDs and scores
    """
    try:
        recommendations = recommender.get_recommendations(user_id, top_n=top_n)
        job_ids, scores = zip(*recommendations)
        return RecommendationResponse(job_ids=list(job_ids), scores=list(scores))
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
