from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="JibJob API")

class Gig(BaseModel):
    title: str
    description: str
    location: dict
    price_range: dict
    category: str
    skills_required: List[str]
    
@app.get("/")
async def root():
    return {"message": "Welcome to JibJob API"}

@app.post("/gigs/")
async def create_gig(gig: Gig):
    # TODO: Implementation
    return {"message": "Gig created", "gig": gig}