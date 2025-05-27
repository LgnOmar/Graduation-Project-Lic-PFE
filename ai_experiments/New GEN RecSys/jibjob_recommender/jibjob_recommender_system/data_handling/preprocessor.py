"""
Data preprocessor module for JibJob recommendation system.
Preprocesses data for feature engineering and model training.
"""

import re
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import datetime

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class responsible for preprocessing data before feature engineering and model training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataPreprocessor with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        
    def preprocess_all_data(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all datasets in the data dictionary.
        
        Args:
            data_dict: Dictionary of DataFrames to preprocess.
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of preprocessed DataFrames.
        """
        processed_data_dict = {}
        
        # Process each dataset
        if 'locations' in data_dict:
            processed_data_dict['locations'] = self.preprocess_locations(data_dict['locations'])
            
        if 'categories' in data_dict:
            processed_data_dict['categories'] = self.preprocess_categories(data_dict['categories'])
            
        if 'users' in data_dict:
            processed_data_dict['users'] = self.preprocess_users(data_dict['users'])
            
        if 'professional_categories' in data_dict:
            processed_data_dict['professional_categories'] = self.preprocess_professional_categories(data_dict['professional_categories'])
            
        if 'jobs' in data_dict:
            processed_data_dict['jobs'] = self.preprocess_jobs(data_dict['jobs'])
            
        if 'job_applications' in data_dict:
            processed_data_dict['job_applications'] = self.preprocess_job_applications(data_dict['job_applications'])
            
        logger.info("All data preprocessed successfully")
        return processed_data_dict
    
    def preprocess_locations(self, locations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the locations DataFrame.
        
        Args:
            locations_df: DataFrame containing location data.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = locations_df.copy()
        
        # Ensure latitude and longitude are float
        df['latitude'] = df['latitude'].astype(float)
        df['longitude'] = df['longitude'].astype(float)
        
        # Clean location names
        df['name'] = df['name'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df['wilaya_name'] = df['wilaya_name'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Fill any missing values with appropriate defaults
        if df['post_code'].isna().any():
            logger.warning("Found missing post_code values in locations data, filling with 'unknown'")
            df['post_code'] = df['post_code'].fillna('unknown')
            
        logger.info(f"Preprocessed {len(df)} location records")
        return df
    
    def preprocess_categories(self, categories_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the categories DataFrame.
        
        Args:
            categories_df: DataFrame containing category data.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = categories_df.copy()
        
        # Clean category names
        df['category_name'] = df['category_name'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        logger.info(f"Preprocessed {len(df)} category records")
        return df
        
    def preprocess_users(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the users DataFrame.
        
        Args:
            users_df: DataFrame containing user data.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = users_df.copy()
        
        # Clean text fields
        df['username'] = df['username'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Process profile_bio text
        if 'profile_bio' in df.columns:
            df['profile_bio'] = df['profile_bio'].apply(self._clean_text)
            
            # Fill empty profile_bio for professionals with a default value
            professionals_mask = df['user_type'] == 'professional'
            empty_bio_mask = df['profile_bio'].isna() | (df['profile_bio'] == '')
            
            if (professionals_mask & empty_bio_mask).any():
                logger.warning("Found professionals with empty profile_bio, filling with default value")
                df.loc[professionals_mask & empty_bio_mask, 'profile_bio'] = 'No profile information provided'
                
        # Ensure user_type is lowercase for consistency
        df['user_type'] = df['user_type'].str.lower()
        
        logger.info(f"Preprocessed {len(df)} user records")
        return df
        
    def preprocess_professional_categories(self, professional_categories_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the professional categories DataFrame.
        
        Args:
            professional_categories_df: DataFrame containing professional category mappings.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = professional_categories_df.copy()
        
        # No specific preprocessing needed for this simple mapping table
        # Could add additional processing here if needed in the future
        
        logger.info(f"Preprocessed {len(df)} professional category records")
        return df
        
    def preprocess_jobs(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the jobs DataFrame.
        
        Args:
            jobs_df: DataFrame containing job data.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = jobs_df.copy()
        
        # Clean text fields
        df['title'] = df['title'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        df['description'] = df['description'].apply(self._clean_text)
        
        # Add a combined 'text_content' field for easier text processing later
        df['text_content'] = df['title'] + ' ' + df['description']
        
        # Add job posting timestamp if not present (useful for temporal analysis)
        if 'posted_timestamp' not in df.columns:
            logger.info("Adding synthetic posted_timestamp to jobs data")
            df['posted_timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%SZ")
            
        logger.info(f"Preprocessed {len(df)} job records")
        return df
        
    def preprocess_job_applications(self, job_applications_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the job applications DataFrame.
        
        Args:
            job_applications_df: DataFrame containing job application data.
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        df = job_applications_df.copy()
        
        # Process timestamp
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except ValueError:
                logger.warning("Could not parse timestamp values in job_applications, leaving as is")
                
        # Normalize application status values
        if 'application_status' in df.columns:
            df['application_status'] = df['application_status'].str.lower()
            
            # Define a list of valid status values
            valid_statuses = ['applied', 'viewed', 'accepted', 'rejected']
            
            # Check for unknown values
            unknown_statuses = df[~df['application_status'].isin(valid_statuses)]['application_status'].unique()
            if len(unknown_statuses) > 0:
                logger.warning(f"Found unknown application status values: {unknown_statuses}")
                
        # Add a binary indicator for successful applications (useful for modeling)
        df['is_positive_interaction'] = df['application_status'].isin(['applied', 'accepted'])
        
        logger.info(f"Preprocessed {len(df)} job application records")
        return df
        
    def _clean_text(self, text: Optional[str]) -> str:
        """
        Clean a text string by removing special characters, excess whitespace, etc.
        
        Args:
            text: Text to clean.
            
        Returns:
            str: Cleaned text.
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags if any
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove special characters but keep spaces and normal punctuation for readability
        text = re.sub(r'[^\w\s\.,;:!?-]', '', text)
        
        # Replace multiple spaces/newlines with a single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
