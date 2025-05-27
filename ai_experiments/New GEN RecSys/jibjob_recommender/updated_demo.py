import os
import json
import pandas as pd
import numpy as np

class JibJobRecommender:
    """
    A simplified version of the JibJob recommendation system.
    This demonstrates the core recommendation concepts based on category matching and location.
    """
    def __init__(self, data_dir='./sample_data'):
        self.data_dir = data_dir
        self.load_data()
        
    def load_data(self):
        """Load all required data files"""
        # Load locations
        with open(os.path.join(self.data_dir, 'locations.json'), 'r') as f:
            self.locations = json.load(f)
        
        # Create a dictionary for quick location lookups
        self.location_dict = {}
        for loc in self.locations:
            # Handle different location JSON structures
            if 'location_id' in loc:
                self.location_dict[loc['location_id']] = loc
            elif 'id' in loc:
                self.location_dict[loc['id']] = loc
        
        # Load categories
        self.categories = pd.read_csv(os.path.join(self.data_dir, 'categories.csv'))
        
        # Check if we have a numeric or string category_id
        if 'category_id' in self.categories.columns:
            self.category_dict = dict(zip(self.categories['category_id'], self.categories['category_name']))
        else:
            # Assume we have id and name columns instead
            self.category_dict = dict(zip(self.categories['id'], self.categories['name']))
        
        # Load users
        self.users = pd.read_csv(os.path.join(self.data_dir, 'users.csv'))
        self.professionals = self.users[self.users['user_type'] == 'professional']
        
        print(f"Loaded {len(self.professionals)} professionals")
        
        # Process categories for professionals
        self.prof_to_categories = {}
        
        # Check if we have a separate professional_categories file or if categories are in users.csv
        if 'selected_category_ids' in self.professionals.columns:
            # Categories are in the users.csv file as semicolon-separated values
            for _, user in self.professionals.iterrows():
                user_id = user['user_id']
                
                if pd.isna(user['selected_category_ids']):
                    self.prof_to_categories[user_id] = []
                else:
                    # Parse semicolon-separated category IDs
                    try:
                        category_ids = [int(cat_id.strip()) for cat_id in user['selected_category_ids'].split(';') if cat_id.strip()]
                        self.prof_to_categories[user_id] = category_ids
                    except:
                        # Handle if they're not numeric
                        category_ids = [cat_id.strip() for cat_id in user['selected_category_ids'].split(';') if cat_id.strip()]
                        self.prof_to_categories[user_id] = category_ids
        else:
            # Try to load from professional_categories.csv
            try:
                professional_categories = pd.read_csv(os.path.join(self.data_dir, 'professional_categories.csv'))
                
                # Create a dictionary of professional -> categories
                for _, row in professional_categories.iterrows():
                    if row['user_id'] not in self.prof_to_categories:
                        self.prof_to_categories[row['user_id']] = []
                    self.prof_to_categories[row['user_id']].append(row['category_id'])
                    
                print(f"Loaded {len(professional_categories)} professional-category associations")
            except:
                print("Could not load professional_categories.csv")
        
        # Load jobs
        self.jobs = pd.read_csv(os.path.join(self.data_dir, 'jobs.csv'))
        print(f"Loaded {len(self.jobs)} jobs")
        
        # Check which column contains the category ID in jobs
        if 'required_category_id' in self.jobs.columns:
            self.jobs_category_column = 'required_category_id'
        elif 'category_id' in self.jobs.columns:
            self.jobs_category_column = 'category_id'
        else:
            self.jobs_category_column = 'category'
            
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in kilometers"""
        R = 6371  # Earth radius in kilometers
        
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance
    
    def get_user_categories(self, user_id):
        """Get categories selected by a professional user"""
        return self.prof_to_categories.get(user_id, [])
    
    def get_recommendations(self, user_id, top_k=10, filter_location=True, max_distance=50.0):
        """
        Get job recommendations for a professional user.
        
        Args:
            user_id: ID of the professional user
            top_k: Number of recommendations to return
            filter_location: Whether to filter by location
            max_distance: Maximum distance in kilometers
            
        Returns:
            List of recommended job dictionaries
        """
        # Check if user exists
        if user_id not in self.professionals['user_id'].values:
            raise ValueError(f"User {user_id} not found or is not a professional")
        
        # Get user information
        user = self.professionals[self.professionals['user_id'] == user_id].iloc[0]
        user_location_id = user['location_id']
        
        # Try to get location from dictionary
        user_location = self.location_dict.get(user_location_id, {'latitude': 0, 'longitude': 0})
        
        # Handle different location JSON structures
        user_lat = user_location.get('latitude', user_location.get('lat', 0))
        user_lon = user_location.get('longitude', user_location.get('lng', 0))
        
        user_categories = self.get_user_categories(user_id)
        
        print(f"User categories: {user_categories}")
        print(f"User location: {user_location_id}")
        
        # Job scores list
        job_scores = []
        
        # Process all jobs
        for _, job in self.jobs.iterrows():
            # Get job information
            job_location_id = job['location_id']
            job_location = self.location_dict.get(job_location_id, {'latitude': 0, 'longitude': 0})
            
            # Handle different location JSON structures
            job_lat = job_location.get('latitude', job_location.get('lat', 0))
            job_lon = job_location.get('longitude', job_location.get('lng', 0))
            
            job_category = job[self.jobs_category_column]
            
            # Calculate distance
            try:
                distance = self.haversine_distance(
                    float(user_lat), float(user_lon),
                    float(job_lat), float(job_lon)
                )
            except:
                distance = 0  # Default if calculation fails
            
            # Check if this job category matches any of the user's categories
            category_match = 1.0 if job_category in user_categories else 0.0
            
            # Distance score (inversely proportional to distance)
            # Closer jobs get higher scores
            distance_score = 1.0 / (1.0 + distance/50.0)
            
            # Calculate final score - weighted sum
            final_score = 0.7 * category_match + 0.3 * distance_score
            
            # Add to job scores
            job_scores.append({
                'job_id': job['job_id'],
                'title': job['title'],
                'description': job['description'],
                'category_id': job_category,
                'category_name': self.category_dict.get(job_category, str(job_category)),
                'distance': distance,
                'distance_km': f"{distance:.2f} km",
                'category_match': bool(category_match),
                'score': final_score
            })
        
        # Filter by distance if requested
        if filter_location:
            job_scores = [job for job in job_scores if job['distance'] <= max_distance]
        
        # Sort jobs by score
        job_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k recommendations
        recommendations = job_scores[:top_k]
        
        return recommendations

    def show_recommendations_for_user(self, user_id, top_k=5):
        """Show recommendations for a specific user"""
        try:
            # Get user info
            user = self.professionals[self.professionals['user_id'] == user_id].iloc[0]
            
            print(f"\n{'='*60}")
            print(f"RECOMMENDATIONS FOR USER: {user_id}")
            print(f"Username: {user['username']}")
            print(f"Profile: {user['profile_bio']}")
            
            # Get user categories
            user_categories = self.get_user_categories(user_id)
            category_names = []
            for cat_id in user_categories:
                cat_name = self.category_dict.get(cat_id, str(cat_id))
                category_names.append(cat_name)
            
            if category_names:
                print(f"Selected Categories: {', '.join(str(name) for name in category_names)}")
            else:
                print("No categories selected")
                
            print(f"{'='*60}\n")
            
            # Get recommendations
            recommendations = self.get_recommendations(
                user_id=user_id,
                top_k=top_k,
                filter_location=True,
                max_distance=50.0
            )
            
            if not recommendations:
                print("No matching jobs found for this user!")
                return
            
            # Print recommendations
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']} (ID: {rec['job_id']})")
                print(f"   Category: {rec['category_name']} ({rec['category_id']})")
                print(f"   Distance: {rec['distance_km']}")
                print(f"   Category Match: {'✓' if rec['category_match'] else '✗'}")
                print(f"   Score: {rec['score']:.4f}")
                print(f"   Description: {rec['description'][:100]}...")
                print(f"{'-'*50}")
            
            # Save to file
            output_file = f"recommendations_{user_id}.json"
            with open(output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
            print(f"\nRecommendations saved to {output_file}")
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")


def main():
    # Initialize the recommender
    recommender = JibJobRecommender()
    
    # Get sample professionals
    sample_professionals = recommender.professionals['user_id'].tolist()[:3]
    
    # Show recommendations for each professional
    for prof_id in sample_professionals:
        recommender.show_recommendations_for_user(prof_id, top_k=5)


if __name__ == "__main__":
    main()
