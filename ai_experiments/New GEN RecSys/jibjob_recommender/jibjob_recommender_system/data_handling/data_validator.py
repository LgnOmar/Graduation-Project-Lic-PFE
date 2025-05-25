"""
Data validator module for JibJob recommendation system.
Validates the data loaded from various sources.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Class responsible for validating loaded data against expected schemas.
    """
    
    def __init__(self):
        """
        Initialize the DataValidator.
        """
        pass
        
    def validate_all_data(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate all datasets in the data dictionary.
        
        Args:
            data_dict: Dictionary of DataFrames to validate.
            
        Returns:
            bool: True if all validations pass, False otherwise.
        """
        validation_results = []
        
        # Validate each dataset according to its schema
        validation_results.append(self.validate_locations(data_dict.get('locations')))
        validation_results.append(self.validate_categories(data_dict.get('categories')))
        validation_results.append(self.validate_users(data_dict.get('users')))
        validation_results.append(self.validate_professional_categories(
            data_dict.get('professional_categories'),
            data_dict.get('users'),
            data_dict.get('categories')
        ))
        validation_results.append(self.validate_jobs(
            data_dict.get('jobs'),
            data_dict.get('locations'),
            data_dict.get('users'),
            data_dict.get('categories')
        ))
        
        # Validate job applications if present
        if 'job_applications' in data_dict:
            validation_results.append(self.validate_job_applications(
                data_dict.get('job_applications'),
                data_dict.get('jobs'),
                data_dict.get('users')
            ))
        
        # Check cross-dataset relationships
        validation_results.append(self.validate_cross_dataset_relationships(data_dict))
        
        return all(validation_results)
        
    def validate_locations(self, locations_df: Optional[pd.DataFrame]) -> bool:
        """
        Validate the locations DataFrame.
        
        Args:
            locations_df: DataFrame containing location data.
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        if locations_df is None:
            logger.error("Locations DataFrame is None")
            return False
            
        required_columns = ['location_id', 'post_code', 'name', 'wilaya_name', 'longitude', 'latitude']
        if not self._check_required_columns(locations_df, required_columns, 'locations'):
            return False
            
        # Check unique constraint for location_id
        if not self._check_unique_constraint(locations_df, 'location_id', 'locations'):
            return False
            
        # Check data types for latitude and longitude
        try:
            locations_df['longitude'] = locations_df['longitude'].astype(float)
            locations_df['latitude'] = locations_df['latitude'].astype(float)
        except ValueError as e:
            logger.error(f"Invalid values for latitude or longitude in locations data: {str(e)}")
            return False
            
        # Check for valid latitude/longitude ranges
        valid_lat_mask = (locations_df['latitude'] >= -90) & (locations_df['latitude'] <= 90)
        valid_long_mask = (locations_df['longitude'] >= -180) & (locations_df['longitude'] <= 180)
        
        if not valid_lat_mask.all():
            logger.error("Invalid latitude values (outside -90 to 90 range) in locations data")
            return False
            
        if not valid_long_mask.all():
            logger.error("Invalid longitude values (outside -180 to 180 range) in locations data")
            return False
            
        logger.info("Locations data validation passed")
        return True
        
    def validate_categories(self, categories_df: Optional[pd.DataFrame]) -> bool:
        """
        Validate the categories DataFrame.
        
        Args:
            categories_df: DataFrame containing category data.
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        if categories_df is None:
            logger.error("Categories DataFrame is None")
            return False
            
        required_columns = ['category_id', 'category_name']
        if not self._check_required_columns(categories_df, required_columns, 'categories'):
            return False
            
        # Check unique constraint for category_id
        if not self._check_unique_constraint(categories_df, 'category_id', 'categories'):
            return False
            
        logger.info("Categories data validation passed")
        return True
        
    def validate_users(self, users_df: Optional[pd.DataFrame]) -> bool:
        """
        Validate the users DataFrame.
        
        Args:
            users_df: DataFrame containing user data.
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        if users_df is None:
            logger.error("Users DataFrame is None")
            return False
            
        required_columns = ['user_id', 'username', 'user_type', 'location_id']
        if not self._check_required_columns(users_df, required_columns, 'users'):
            return False
            
        # Check unique constraint for user_id
        if not self._check_unique_constraint(users_df, 'user_id', 'users'):
            return False
            
        # Check user_type enum values
        valid_user_types = ['professional', 'client']
        invalid_types = users_df[~users_df['user_type'].isin(valid_user_types)]
        
        if not invalid_types.empty:
            logger.error(f"Invalid user_type values found: {invalid_types['user_type'].unique()}. "
                         f"Valid values are {valid_user_types}")
            return False
            
        logger.info("Users data validation passed")
        return True
        
    def validate_professional_categories(self, 
                                         prof_cat_df: Optional[pd.DataFrame],
                                         users_df: Optional[pd.DataFrame],
                                         categories_df: Optional[pd.DataFrame]) -> bool:
        """
        Validate the professional categories DataFrame and its relationships.
        
        Args:
            prof_cat_df: DataFrame containing professional category mappings.
            users_df: DataFrame containing user data.
            categories_df: DataFrame containing category data.
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        if prof_cat_df is None:
            logger.error("Professional Categories DataFrame is None")
            return False
            
        required_columns = ['user_id', 'category_id']
        if not self._check_required_columns(prof_cat_df, required_columns, 'professional_categories'):
            return False
            
        # Check foreign key relationships if the related DataFrames are available
        if users_df is not None:
            # Check that user_id references professionals
            prof_users = users_df[users_df['user_type'] == 'professional']['user_id'].unique()
            invalid_users = prof_cat_df[~prof_cat_df['user_id'].isin(prof_users)]
            
            if not invalid_users.empty:
                logger.warning(f"Found {len(invalid_users)} professional_categories records with user_id not referencing "
                              f"a professional user or user doesn't exist")
                
        if categories_df is not None:
            # Check that category_id exists in categories
            valid_categories = categories_df['category_id'].unique()
            invalid_categories = prof_cat_df[~prof_cat_df['category_id'].isin(valid_categories)]
            
            if not invalid_categories.empty:
                logger.warning(f"Found {len(invalid_categories)} professional_categories records with category_id "
                              f"that doesn't exist in categories data")
                
        # Check for duplicate user_id, category_id combinations
        duplicates = prof_cat_df.duplicated(subset=['user_id', 'category_id'], keep=False)
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate user_id, category_id pairs in professional_categories")
            
        logger.info("Professional categories data validation passed")
        return True
        
    def validate_jobs(self, 
                      jobs_df: Optional[pd.DataFrame],
                      locations_df: Optional[pd.DataFrame],
                      users_df: Optional[pd.DataFrame],
                      categories_df: Optional[pd.DataFrame]) -> bool:
        """
        Validate the jobs DataFrame and its relationships.
        
        Args:
            jobs_df: DataFrame containing job data.
            locations_df: DataFrame containing location data.
            users_df: DataFrame containing user data.
            categories_df: DataFrame containing category data.
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        if jobs_df is None:
            logger.error("Jobs DataFrame is None")
            return False
            
        required_columns = ['job_id', 'title', 'description', 'location_id', 'posted_by_user_id']
        if not self._check_required_columns(jobs_df, required_columns, 'jobs'):
            return False
            
        # Check unique constraint for job_id
        if not self._check_unique_constraint(jobs_df, 'job_id', 'jobs'):
            return False
            
        # Check foreign key relationships if the related DataFrames are available
        if locations_df is not None:
            valid_locations = locations_df['location_id'].unique()
            invalid_locations = jobs_df[~jobs_df['location_id'].isin(valid_locations)]
            
            if not invalid_locations.empty:
                logger.warning(f"Found {len(invalid_locations)} jobs with location_id that doesn't exist in locations data")
                
        if users_df is not None:
            # Check that posted_by_user_id references clients
            client_users = users_df[users_df['user_type'] == 'client']['user_id'].unique()
            invalid_users = jobs_df[~jobs_df['posted_by_user_id'].isin(client_users)]
            
            if not invalid_users.empty:
                logger.warning(f"Found {len(invalid_users)} jobs with posted_by_user_id not referencing a client user")
                
        # Check required_category_id if present
        if 'required_category_id' in jobs_df.columns and categories_df is not None:
            valid_categories = categories_df['category_id'].unique()
            # Filter out NaN values for nullable category_id
            invalid_categories = jobs_df[
                jobs_df['required_category_id'].notna() & 
                ~jobs_df['required_category_id'].isin(valid_categories)
            ]
            
            if not invalid_categories.empty:
                logger.warning(f"Found {len(invalid_categories)} jobs with required_category_id "
                              f"that doesn't exist in categories data")
                
        logger.info("Jobs data validation passed")
        return True
        
    def validate_job_applications(self, 
                                 job_apps_df: Optional[pd.DataFrame],
                                 jobs_df: Optional[pd.DataFrame],
                                 users_df: Optional[pd.DataFrame]) -> bool:
        """
        Validate the job applications DataFrame and its relationships.
        
        Args:
            job_apps_df: DataFrame containing job application data.
            jobs_df: DataFrame containing job data.
            users_df: DataFrame containing user data.
            
        Returns:
            bool: True if validation passes, False otherwise.
        """
        if job_apps_df is None:
            logger.info("Job applications DataFrame is None (optional data)")
            return True
            
        required_columns = ['application_id', 'job_id', 'professional_user_id', 'application_status']
        if not self._check_required_columns(job_apps_df, required_columns, 'job_applications'):
            return False
            
        # Check unique constraint for application_id
        if not self._check_unique_constraint(job_apps_df, 'application_id', 'job_applications'):
            return False
            
        # Check foreign key relationships if the related DataFrames are available
        if jobs_df is not None:
            valid_jobs = jobs_df['job_id'].unique()
            invalid_jobs = job_apps_df[~job_apps_df['job_id'].isin(valid_jobs)]
            
            if not invalid_jobs.empty:
                logger.warning(f"Found {len(invalid_jobs)} job applications with job_id that doesn't exist in jobs data")
                
        if users_df is not None:
            # Check that professional_user_id references professionals
            prof_users = users_df[users_df['user_type'] == 'professional']['user_id'].unique()
            invalid_users = job_apps_df[~job_apps_df['professional_user_id'].isin(prof_users)]
            
            if not invalid_users.empty:
                logger.warning(f"Found {len(invalid_users)} job applications with professional_user_id not "
                              f"referencing a professional user")
                
        # Validate timestamp format if present
        if 'timestamp' in job_apps_df.columns:
            try:
                pd.to_datetime(job_apps_df['timestamp'])
            except ValueError:
                logger.warning("Invalid timestamp format found in job_applications data")
                
        logger.info("Job applications data validation passed")
        return True
        
    def validate_cross_dataset_relationships(self, data_dict: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate relationships between different datasets.
        
        Args:
            data_dict: Dictionary of DataFrames to validate.
            
        Returns:
            bool: True if validations pass, False otherwise.
        """
        users_df = data_dict.get('users')
        locations_df = data_dict.get('locations')
        
        # Check that user location_ids exist in locations
        if users_df is not None and locations_df is not None:
            valid_locations = locations_df['location_id'].unique()
            invalid_locations = users_df[~users_df['location_id'].isin(valid_locations)]
            
            if not invalid_locations.empty:
                logger.warning(f"Found {len(invalid_locations)} users with location_id that doesn't exist in locations data")
                
        logger.info("Cross-dataset relationship validation passed")
        return True
        
    def _check_required_columns(self, df: pd.DataFrame, required_columns: List[str], dataset_name: str) -> bool:
        """
        Check if the DataFrame contains all required columns.
        
        Args:
            df: DataFrame to check.
            required_columns: List of column names that should be present.
            dataset_name: Name of the dataset for logging.
            
        Returns:
            bool: True if all required columns are present, False otherwise.
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in {dataset_name} data: {', '.join(missing_columns)}")
            return False
            
        return True
        
    def _check_unique_constraint(self, df: pd.DataFrame, column_name: str, dataset_name: str) -> bool:
        """
        Check if values in the specified column are unique.
        
        Args:
            df: DataFrame to check.
            column_name: Name of the column that should contain unique values.
            dataset_name: Name of the dataset for logging.
            
        Returns:
            bool: True if all values are unique, False otherwise.
        """
        duplicate_values = df[df.duplicated(subset=[column_name], keep=False)]
        
        if not duplicate_values.empty:
            logger.error(f"Duplicate {column_name} values found in {dataset_name} data: "
                        f"{duplicate_values[column_name].nunique()} duplicate values")
            return False
            
        return True
