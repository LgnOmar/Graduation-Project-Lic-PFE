"""
Demonstration API server that simulates the JibJob recommendation system.
This version does not require a pre-trained model file.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import random
import pandas as pd
import os

# Create the FastAPI app
app = FastAPI(title="JibJob Recommendation API Demo")

print("Loading JibJob Algerian job market data...")

# Load job data if available
job_data = {}
try:
    if os.path.exists("data/jobs_df.csv"):
        jobs_df = pd.read_csv("data/jobs_df.csv")
        for _, row in jobs_df.iterrows():
            job_id = row["job_id"]
            job_data[job_id] = {
                "title": row.get("title", f"Job {job_id}"),
                "description": row.get("description_mission_anglais", "No description available"),
                "category": row.get("categorie_mission", "General"),
                "location": row.get("location", "Algiers")
            }
        print(f"Loaded {len(job_data)} jobs from data file")
    else:
        raise FileNotFoundError("jobs_df.csv not found")
except Exception as e:
    print(f"Error loading job data: {e}")
    print("Creating dummy Algerian job data...")
    
    # Algerian wilayas (provinces)
    algerian_wilayas = [
        "Adrar", "Chlef", "Laghouat", "Batna", "Béjaïa", "Biskra", 
        "Blida", "Bouira", "Tlemcen", "Tiaret", "Tizi Ouzou", "Alger", 
        "Djelfa", "Jijel", "Sétif", "Skikda", "Annaba", "Constantine", 
        "Médéa", "Mostaganem", "Oran", "Boumerdès", "El Oued", "Tipaza"
    ]
    
    # Algerian job categories
    algerian_categories = {
        "Home Services": ["Plumbing repairs", "Electrical installations", "Housekeeping", "Painting services"],
        "Teaching": ["Math tutoring", "French language lessons", "Arabic lessons", "Science tutoring"],
        "Transportation": ["Moving assistance", "Furniture transport", "Delivery services"],
        "Digital Services": ["Website development", "Graphic design", "Social media management"],
        "Handcrafts": ["Traditional carpet making", "Pottery services", "Leather crafting"],
        "Maintenance": ["Smartphone repair", "Computer maintenance", "Home appliance repair"]
    }
    
    for i in range(1, 101):
        job_id = f"job_{i}"
        category = random.choice(list(algerian_categories.keys()))
        job_type = random.choice(algerian_categories[category])
        location = random.choice(algerian_wilayas)
        job_data[job_id] = {
            "title": f"{job_type} in {location}",
            "description": f"Need professional for {job_type} service in {location}. Competitive pay and flexible hours.",
            "category": category,
            "location": location
        }
    print(f"Created {len(job_data)} dummy Algerian jobs")

# Load user data if available
user_data = {}
try:
    if os.path.exists("data/users_df.csv"):
        users_df = pd.read_csv("data/users_df.csv")
        for _, row in users_df.iterrows():
            user_id = row["user_id"]
            skills_str = row.get("skills", "")
            prefs_str = row.get("preferences", "")
            
            # Handle different data formats
            if isinstance(skills_str, str):
                skills = [s.strip() for s in skills_str.split(",") if s.strip()]
            else:
                skills = []
                
            if isinstance(prefs_str, str):
                preferences = [p.strip() for p in prefs_str.split(",") if p.strip()]
            else:
                preferences = []
                
            user_data[user_id] = {
                "name": row.get("name", f"User {user_id}"),
                "description": row.get("description_profil_utilisateur_anglais", ""),
                "skills": skills,
                "preferences": preferences
            }
        print(f"Loaded {len(user_data)} users from data file")
    else:
        raise FileNotFoundError("users_df.csv not found")
except Exception as e:
    print(f"Error loading user data: {e}")
    print("Creating dummy Algerian user profiles...")
    
    algerian_names = [
        "Mohamed", "Ahmed", "Ali", "Karim", "Youcef", "Omar", "Amine", "Sofiane", 
        "Fatima", "Amina", "Yasmine", "Meriem", "Amel", "Leila", "Samira", "Nadia"
    ]
    
    algerian_skills = [
        "programming", "design", "teaching", "electrical repair", "plumbing", "driving",
        "woodworking", "cooking", "car repair", "web development", "language instruction",
        "accounting", "customer service", "project management", "social media", "office skills"
    ]
    
    for i in range(1, 51):
        user_id = f"user_{i}"
        name = f"{random.choice(algerian_names)} {chr(65 + random.randint(0, 25))}."
        user_skills = random.sample(algerian_skills, k=random.randint(1, 4))
        user_preferences = random.sample(list(algerian_categories.keys()), k=random.randint(1, 3))
        
        user_data[user_id] = {
            "name": name,
            "description": f"Professional based in {random.choice(algerian_wilayas)} with experience in {', '.join(user_skills)}.",
            "skills": user_skills,
            "preferences": user_preferences
        }
    print(f"Created {len(user_data)} dummy Algerian user profiles")

class RecommendationResponse(BaseModel):
    job_ids: List[str]
    scores: List[float]
    
@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {
        "message": "Welcome to JibJob Recommendation API Demo",
        "endpoints": {
            "GET /recommendations/{user_id}": "Get job recommendations for a user"
        }
    }

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    top_n: int = 10,
    location_preference: str = None
) -> RecommendationResponse:
    """
    Get job recommendations for a user based on Algerian context.
    
    Args:
        user_id: ID of the user to get recommendations for
        top_n: Number of recommendations to return
        location_preference: Optional filter for job location (wilaya)
        
    Returns:
        RecommendationResponse containing job IDs and scores
    """
    try:
        # Check if user exists
        if user_id not in user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = user_data[user_id]
        
        # Extract user preferences and skills
        user_categories = user.get("preferences", [])
        user_skills = user.get("skills", [])
        user_description = user.get("description", "")
        
        # If no preferences found, extract potential preferences from description
        if not user_categories:
            # Common categories to look for in the description
            all_categories = set()
            for job_id, job in job_data.items():
                category = job.get("category")
                if category:
                    all_categories.add(category)
            
            # Check for category mentions in user description
            for category in all_categories:
                if category.lower() in user_description.lower():
                    user_categories.append(category)
        
        # If still no preferences, use "General" as default
        if not user_categories:
            user_categories = ["General"]
        
        # Find all potentially matching jobs
        all_potential_jobs = []
        for job_id, job in job_data.items():
            job_score = 0.5  # Start with base score
            
            # Score factors:
            # 1. Category match
            job_category = job.get("category", "General")
            if job_category in user_categories:
                job_score += 0.25
            
            # 2. Location match if specified
            job_location = job.get("location", "")
            if location_preference and job_location == location_preference:
                job_score += 0.2
            
            # 3. Skills match - check for skill mentions in job description
            job_description = job.get("description", "").lower()
            skill_matches = sum(1 for skill in user_skills if skill.lower() in job_description)
            if skill_matches > 0:
                job_score += min(0.2, skill_matches * 0.05)  # Cap skill bonus at 0.2
            
            # 4. Add slight randomness to prevent identical scores
            job_score += random.uniform(0, 0.05)
            
            # Cap score at 0.99
            final_score = min(0.99, job_score)
            
            all_potential_jobs.append((job_id, job, final_score))
        
        # Sort by score (descending) and take top_n
        all_potential_jobs.sort(key=lambda x: x[2], reverse=True)
        top_recommendations = all_potential_jobs[:top_n]
        
        # Extract job IDs and scores
        job_ids = [rec[0] for rec in top_recommendations]
        scores = [rec[2] for rec in top_recommendations]
        
        return RecommendationResponse(job_ids=job_ids, scores=scores)
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}")
async def get_job_details(job_id: str) -> Dict[str, Any]:
    """
    Get details for a specific Algerian job.
    
    Args:
        job_id: ID of the job to get details for
        
    Returns:
        Dictionary containing job details
    """
    try:
        if job_id not in job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = job_data[job_id]
        
        # Get any interactions/ratings for this job
        try:
            if os.path.exists("data/interactions_df.csv"):
                interactions_df = pd.read_csv("data/interactions_df.csv")
                job_interactions = interactions_df[interactions_df['job_id'] == job_id]
                
                # Add rating statistics if available
                ratings = job_interactions['rating_explicite'].dropna()
                if len(ratings) > 0:
                    job['avg_rating'] = float(ratings.mean())
                    job['rating_count'] = int(len(ratings))
                
                # Add sample comments if available
                comments = job_interactions['commentaire_texte_anglais'].dropna()
                if len(comments) > 0:
                    job['sample_comments'] = comments.tolist()[:3]  # Include up to 3 sample comments
        except Exception as e:
            print(f"Error loading interactions: {e}")
            
        return job
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}")
async def get_user_details(user_id: str) -> Dict[str, Any]:
    """
    Get details for a specific Algerian user.
    
    Args:
        user_id: ID of the user to get details for
        
    Returns:
        Dictionary containing user details
    """
    try:
        if user_id not in user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        user = user_data[user_id].copy()
        
        # Get recent interactions for this user
        try:
            if os.path.exists("data/interactions_df.csv"):
                interactions_df = pd.read_csv("data/interactions_df.csv")
                user_interactions = interactions_df[interactions_df['user_id'] == user_id]
                
                # Add recent job interactions
                if len(user_interactions) > 0:
                    recent_jobs = []
                    for _, interaction in user_interactions.head(5).iterrows():
                        job_id = interaction['job_id']
                        if job_id in job_data:
                            job_info = {
                                'job_id': job_id,
                                'title': job_data[job_id].get('title', ''),
                                'rating': float(interaction['rating_explicite']) if pd.notna(interaction['rating_explicite']) else None
                            }
                            recent_jobs.append(job_info)
                    
                    user['recent_interactions'] = recent_jobs
        except Exception as e:
            print(f"Error loading user interactions: {e}")
            
        return user
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def search_jobs(
    location: str = None,
    category: str = None,
    q: str = None,
    page: int = 1,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Search for jobs based on various criteria.
    
    Args:
        location: Filter jobs by location (wilaya)
        category: Filter jobs by category
        q: Search query in job title or description
        page: Page number for pagination
        limit: Number of results per page
        
    Returns:
        Dictionary containing job search results
    """
    try:
        matching_jobs = []
        
        # Filter jobs based on criteria
        for job_id, job in job_data.items():
            # Location filter
            if location and job.get("location", "") != location:
                continue
                
            # Category filter
            if category and job.get("category", "") != category:
                continue
                
            # Search query filter
            if q:
                q_lower = q.lower()
                title = job.get("title", "").lower()
                description = job.get("description", "").lower()
                if q_lower not in title and q_lower not in description:
                    continue
            
            # Add job to results
            matching_jobs.append({
                "job_id": job_id,
                **job
            })
        
        # Sort results by title
        matching_jobs.sort(key=lambda j: j.get("title", ""))
        
        # Paginate results
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_jobs = matching_jobs[start_idx:end_idx]
        
        # Get available locations and categories for filtering
        all_locations = sorted(set(job.get("location", "") for job in job_data.values() if job.get("location")))
        all_categories = sorted(set(job.get("category", "") for job in job_data.values() if job.get("category")))
        
        return {
            "total": len(matching_jobs),
            "page": page,
            "limit": limit,
            "total_pages": (len(matching_jobs) + limit - 1) // limit,
            "results": paginated_jobs,
            "available_filters": {
                "locations": all_locations,
                "categories": all_categories
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/wilayas")
async def get_wilayas() -> List[str]:
    """
    Get list of Algerian wilayas represented in the job data.
    
    Returns:
        List of wilaya names
    """
    wilayas = set()
    for job in job_data.values():
        location = job.get("location")
        if location:
            wilayas.add(location)
    
    return sorted(list(wilayas))

if __name__ == "__main__":
    print("Starting JibJob Recommendation API Demo...")
    print("This version doesn't require a pre-trained model file")
    print("\nAvailable endpoints:")
    print("  - GET /recommendations/{user_id}")
    print("  - GET /jobs/{job_id}")
    print("  - GET /users/{user_id}")
    print("  - GET /jobs")
    print("  - GET /wilayas")
    print("\nStarting server on http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
