    def recommend_for_professional(
        self,
        professional_id: Any,
        professional_categories: List[str],
        professional_location: Optional[str] = None,
        top_k: int = 10,
        require_category_match: bool = True,
        max_location_distance: float = 0.0,
        exclude_rated: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Recommend jobs for professional users based on category and location.
        
        Args:
            professional_id: ID of the professional user
            professional_categories: List of categories the professional is interested in
            professional_location: Location of the professional (optional)
            top_k: Maximum number of recommendations to return
            require_category_match: Whether to only recommend jobs that match the professional's categories
            max_location_distance: Maximum distance for location-based recommendations
            exclude_rated: Whether to exclude jobs the user has already rated
            
        Returns:
            List[Dict[str, Any]]: List of recommended jobs with scores
        """
        logger.info(f"Generating professional recommendations for user {professional_id}...")
        
        # Find jobs that match the professional's categories
        if require_category_match:
            category_matched_jobs = self.jobs_df_internal[
                self.jobs_df_internal['category'].isin(professional_categories)
            ]
        else:
            category_matched_jobs = self.jobs_df_internal.copy() # All jobs if no specific category match required
        
        # Further filter by location if needed
        if max_location_distance > 0.0 and professional_location and 'location' in self.jobs_df_internal.columns:
            # Assuming professional_location is a string like "City, Country"
            # and location in jobs_df_internal is also in the same format
            location_filtered_jobs = category_matched_jobs[
                category_matched_jobs['location'].apply(
                    lambda loc: calculate_location_distance(professional_location, loc) <= max_location_distance if loc else False
                )
            ]
        else:
            location_filtered_jobs = category_matched_jobs
        
        # Exclude jobs already rated by the user
        if exclude_rated and self.training_interactions_df is not None:
            rated_job_ids = self.training_interactions_df[self.training_interactions_df['user_id'] == professional_id]['job_id'].values
            location_filtered_jobs = location_filtered_jobs[~location_filtered_jobs['job_id'].isin(rated_job_ids)]
        
        # Sort by some relevance score - here we just use a placeholder
        # In practice, you might have a learned model to rank these
        location_filtered_jobs = location_filtered_jobs.sort_values(by='job_id') # Placeholder: sort by job_id
        
        # Limit to top_k jobs
        top_jobs = location_filtered_jobs.head(top_k)
        
        recommendations = []
        for _, job_row in top_jobs.iterrows():
            # Check if job category matches any of the professional categories
            category_match = job_row.get('category', '') in professional_categories
            
            # Check for location match if location info available
            location_match = False
            if professional_location and 'location' in job_row and job_row['location']:
                distance = calculate_location_distance(professional_location, job_row['location'])
                location_match = distance <= max_location_distance if max_location_distance > 0 else True
            
            # Calculate a match score based on category and location
            match_score = 1.0 if category_match else 0.5
            if professional_location and location_match:
                match_score += 0.5
            
            recommendations.append({
                'job_id': job_row['job_id'],
                'score': match_score,  # Match score
                'match_score': match_score,  # For demo consistency
                'title': job_row.get('title', ''),
                'description': job_row.get('description', ''),
                'category': job_row.get('category', ''),
                'location': job_row.get('location', ''),
                'category_match': category_match,
                'location_match': location_match
            })
        
        logger.info(f"Found {len(recommendations)} recommendations for professional user {professional_id}.")
        return recommendations

