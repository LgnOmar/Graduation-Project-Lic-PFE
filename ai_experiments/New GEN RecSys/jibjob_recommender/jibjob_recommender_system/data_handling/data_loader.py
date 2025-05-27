"""
Data loader module for JibJob recommendation system.
Loads data from various sources into appropriate structures.
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class responsible for loading data from various sources into appropriate structures.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the DataLoader with the given configuration.
        
        Args:
            config: Configuration dictionary containing data paths.
        """
        self.config = config
        self.data_config = config['data']
        
        # Initialize data containers
        self.locations = None
        self.categories = None
        self.users = None
        self.professional_categories = None
        self.jobs = None
        self.job_applications = None
        
    def load_all_data(self) -> Tuple[bool, Dict[str, pd.DataFrame]]:
        """
        Load all data sources and return the data dictionary.
        
        Returns:
            Tuple containing:
            - bool: Success indicator
            - Dict: Dictionary of loaded DataFrames
        """
        try:
            # Load locations
            success, self.locations = self.load_locations()
            if not success:
                return False, {}
                
            # Load categories
            success, self.categories = self.load_csv('categories_file', 'categories')
            if not success:
                return False, {}
                
            # Load users
            success, self.users = self.load_csv('users_file', 'users')
            if not success:
                return False, {}
                
            # Load professional categories
            success, self.professional_categories = self.load_csv('professional_categories_file', 'professional_categories')
            if not success:
                return False, {}
                
            # Load jobs
            success, self.jobs = self.load_csv('jobs_file', 'jobs')
            if not success:
                return False, {}
                
            # Try to load job applications (optional)
            success, self.job_applications = self.load_csv('job_applications_file', 'job_applications', optional=True)
            
            # Create data dictionary
            data_dict = {
                'locations': self.locations,
                'categories': self.categories,
                'users': self.users,
                'professional_categories': self.professional_categories,
                'jobs': self.jobs
            }
            
            if self.job_applications is not None:
                data_dict['job_applications'] = self.job_applications
                
            logger.info("All data loaded successfully")
            return True, data_dict
            
        except Exception as e:
            logger.error(f"Unexpected error during data loading: {str(e)}")
            return False, {}
    
    def load_locations(self) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Load locations data from JSON file.
        
        Returns:
            Tuple containing:
            - bool: Success indicator
            - Optional[pd.DataFrame]: DataFrame containing the location data if successful, None otherwise
        """
        file_path = self._get_file_path('locations_file')
        if not file_path:
            return False, None
            
        try:
            logger.info(f"Loading locations from {file_path}")
            with open(file_path, 'r') as file:
                locations_data = json.load(file)
                
            # Convert to DataFrame
            locations_df = pd.DataFrame(locations_data)
            logger.info(f"Loaded {len(locations_df)} locations")
            return True, locations_df
            
        except FileNotFoundError:
            logger.error(f"Locations file not found at {file_path}")
            return False, None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing locations JSON file: {str(e)}")
            return False, None
        except Exception as e:
            logger.error(f"Unexpected error loading locations: {str(e)}")
            return False, None
            
    def load_csv(self, config_key: str, display_name: str, optional: bool = False) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Load data from a CSV file.
        
        Args:
            config_key: Key in the data config for the file path.
            display_name: Human-readable name for logging.
            optional: Whether the file is optional (don't fail if missing).
            
        Returns:
            Tuple containing:
            - bool: Success indicator
            - Optional[pd.DataFrame]: DataFrame if successful, None otherwise
        """
        file_path = self._get_file_path(config_key)
        if not file_path:
            if optional:
                logger.warning(f"Optional {display_name} file not specified in config")
                return True, None
            else:
                logger.error(f"Required {display_name} file path not specified in config")
                return False, None
                
        try:
            logger.info(f"Loading {display_name} from {file_path}")
            
            if not os.path.exists(file_path) and optional:
                logger.warning(f"Optional {display_name} file not found at {file_path}")
                return True, None
                
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} {display_name} records")
            return True, df
            
        except FileNotFoundError:
            if optional:
                logger.warning(f"Optional {display_name} file not found at {file_path}")
                return True, None
            else:
                logger.error(f"Required {display_name} file not found at {file_path}")
                return False, None
        except pd.errors.EmptyDataError:
            logger.error(f"{display_name} file is empty: {file_path}")
            return False, None
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing {display_name} CSV file: {str(e)}")
            return False, None
        except Exception as e:
            logger.error(f"Unexpected error loading {display_name}: {str(e)}")
            return False, None
    
    def _get_file_path(self, config_key: str) -> Optional[str]:
        """
        Get the file path from the data config.
        
        Args:
            config_key: Key in the data config for the file path.
            
        Returns:
            Optional[str]: The file path if found, None otherwise.
        """
        if config_key not in self.data_config:
            logger.error(f"Missing config key: {config_key}")
            return None
            
        file_path = self.data_config[config_key]
        if not isinstance(file_path, str):
            logger.error(f"Invalid file path for {config_key}: {file_path}")
            return None
            
        # If the path is not absolute, make it relative to the base path
        if not os.path.isabs(file_path):
            base_path = self.data_config.get('base_path', '')
            file_path = os.path.join(base_path, file_path)
            
        return file_path
