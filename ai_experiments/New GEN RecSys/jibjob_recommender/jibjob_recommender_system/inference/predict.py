"""
Command-line interface for making job recommendations.
"""

import os
import sys
import argparse
import logging
import json
import pandas as pd
from typing import Dict, List, Any, Optional

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jibjob_recommender_system.config.config_loader import ConfigLoader
from jibjob_recommender_system.utils.logging_config import setup_logging
from jibjob_recommender_system.inference.recommender_service import RecommenderService
from jibjob_recommender_system.data_handling.data_loader import DataLoader

def setup_argparser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="JibJob Recommendation System - CLI for job recommendations")
        
    # User identification (existing or new)
    user_group = parser.add_mutually_exclusive_group(required=True)
    user_group.add_argument('--user-id', type=str, help='ID of the professional user to get recommendations for')
    user_group.add_argument('--new-user', action='store_true', help='Generate recommendations for a new user')
    
    # New user options
    parser.add_argument('--profile-text', type=str,
                      help='Profile text description for a new user')
    parser.add_argument('--categories', type=str, nargs='+',
                      help='Category IDs the new user is interested in')
    parser.add_argument('--latitude', type=float,
                      help='Latitude coordinate for the new user')
    parser.add_argument('--longitude', type=float,
                      help='Longitude coordinate for the new user')
                      
    # Recommendation options
    parser.add_argument('--top-k', type=int, default=10,
                      help='Number of recommendations to return')
    parser.add_argument('--filter-location', action='store_true', default=True,
                      help='Filter recommendations by location')
    parser.add_argument('--max-distance', type=float, default=50.0,
                      help='Maximum distance in kilometers for location filtering')
    parser.add_argument('--batch-file', type=str,
                      help='Path to a file containing user IDs for batch processing')
                      
    # Output options
    parser.add_argument('--output', type=str, default='recommendations.json',
                      help='Path to save the recommendations')
    parser.add_argument('--format', type=str, choices=['json', 'csv'], default='json',
                      help='Output format for recommendations')
                      
    # Model options
    parser.add_argument('--model', type=str,
                      help='Path to the model file to load')
    parser.add_argument('--config', type=str, default='../../config/settings.yaml',
                      help='Path to the configuration file')
                      
    # Logging options
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
                      
    return parser

def validate_new_user_args(args: argparse.Namespace) -> bool:
    """
    Validate command-line arguments for a new user.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        bool: True if valid, False otherwise.
    """
    if args.new_user:
        if not args.profile_text and not args.categories:
            logging.error("New user requires profile text or categories")
            return False
            
    return True

def load_batch_users(batch_file: str) -> Optional[List[str]]:
    """
    Load user IDs from a batch file.
    
    Args:
        batch_file: Path to the batch file.
        
    Returns:
        Optional[List[str]]: List of user IDs, or None if file not found.
    """
    if not os.path.exists(batch_file):
        logging.error(f"Batch file not found: {batch_file}")
        return None
        
    try:
        with open(batch_file, 'r') as f:
            # Attempt to parse as JSON first
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'user_ids' in data:
                    return data['user_ids']
            except json.JSONDecodeError:
                # If not JSON, try as CSV or plain text
                return [line.strip() for line in f.readlines() if line.strip()]
                
    except Exception as e:
        logging.error(f"Error loading batch file: {str(e)}")
        return None

def save_recommendations(
    recommendations: Any,
    output_file: str,
    output_format: str = 'json'
) -> bool:
    """
    Save recommendations to a file.
    
    Args:
        recommendations: Recommendations data (dict or list).
        output_file: Path to save the recommendations.
        output_format: Format to save the recommendations in.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        if output_format == 'json':
            with open(output_file, 'w') as f:
                json.dump(recommendations, f, indent=2)
                
        elif output_format == 'csv':
            # Convert to DataFrame and save as CSV
            if isinstance(recommendations, dict):
                # Flatten the dictionary of lists
                flat_data = []
                for user_id, user_recs in recommendations.items():
                    for rec in user_recs:
                        rec_copy = rec.copy()
                        rec_copy['user_id'] = user_id
                        flat_data.append(rec_copy)
                        
                df = pd.DataFrame(flat_data)
                
            else:
                df = pd.DataFrame(recommendations)
                
            df.to_csv(output_file, index=False)
            
        logging.info(f"Recommendations saved to {output_file}")
        return True
        
    except Exception as e:
        logging.error(f"Error saving recommendations: {str(e)}")
        return False

def main() -> None:
    """
    Main function for the recommendation CLI.
    """
    # Set up argument parser
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()
    
    if config is None:
        logging.error(f"Failed to load configuration from {args.config}")
        sys.exit(1)
        
    # Validate arguments
    if not validate_new_user_args(args):
        parser.print_help()
        sys.exit(1)
        
    # Initialize recommender service
    recommender = RecommenderService(config)
    
    # Load model
    model_loaded = recommender.load_model(args.model)
    if not model_loaded:
        logging.error("Failed to load recommendation model")
        sys.exit(1)
        
    # Generate recommendations
    if args.batch_file:
        # Batch processing
        user_ids = load_batch_users(args.batch_file)
        if user_ids is None:
            sys.exit(1)
            
        logging.info(f"Generating recommendations for {len(user_ids)} users")
        recommendations = recommender.recommend_jobs_batch(
            user_ids,
            top_k=args.top_k,
            filter_by_location=args.filter_location,
            max_distance_km=args.max_distance
        )
        
    elif args.new_user:
        # New user
        logging.info("Generating recommendations for new user")
        
        # Prepare location dictionary if coordinates provided
        location = None
        if args.latitude is not None and args.longitude is not None:
            location = {
                'latitude': args.latitude,
                'longitude': args.longitude
            }
            
        recommendations = recommender.recommend_for_new_user(
            profile_text=args.profile_text or "",
            categories=args.categories or [],
            location=location,
            top_k=args.top_k,
            filter_by_location=args.filter_location,
            max_distance_km=args.max_distance
        )
        
    else:
        # Single user
        logging.info(f"Generating recommendations for user {args.user_id}")
        recommendations = recommender.recommend_jobs(
            args.user_id,
            top_k=args.top_k,
            filter_by_location=args.filter_location,
            max_distance_km=args.max_distance
        )
        
    # Save recommendations
    save_success = save_recommendations(
        recommendations,
        args.output,
        args.format
    )
    
    if save_success:
        logging.info("Recommendation process completed successfully")
    else:
        logging.error("Failed to save recommendations")
        sys.exit(1)

if __name__ == "__main__":
    main()
