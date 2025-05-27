"""
Tests for the data handling module of JibJob recommendation system.
"""

import unittest
import os
import json
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock
from jibjob_recommender_system.data_handling.data_loader import DataLoader
from jibjob_recommender_system.data_handling.data_validator import DataValidator
from jibjob_recommender_system.data_handling.preprocessor import DataPreprocessor

class TestDataLoader(unittest.TestCase):
    """Test cases for the DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'data': {
                'sample_data_path': './sample_data/',
                'output_path': './output/',
                'required_files': ['locations.json', 'users.csv', 'jobs.csv', 'categories.csv']
            }
        }
        self.data_loader = DataLoader(config=self.config)
        
        # Sample data
        self.sample_locations = [
            {
                "location_id": "loc_001",
                "post_code": "01001",
                "name": "Adrar",
                "wilaya_name": "Adrar",
                "longitude": -0.4841573,
                "latitude": 27.9763317
            },
            {
                "location_id": "loc_002",
                "post_code": "01002",
                "name": "Reggane",
                "wilaya_name": "Adrar",
                "longitude": -0.1712033,
                "latitude": 26.7208076
            }
        ]
        
        self.sample_users = pd.DataFrame({
            'user_id': ['prof_001', 'client_001'],
            'username': ['pro_user_A', 'client_user_X'],
            'user_type': ['professional', 'client'],
            'location_id': ['loc_001', 'loc_002'],
            'profile_bio': ['Experienced electrician', 'Homeowner looking for help']
        })
        
        self.sample_jobs = pd.DataFrame({
            'job_id': ['job_001', 'job_002'],
            'title': ['Urgent: Electrician', 'Math Tutor Needed'],
            'description': ['Need certified electrician', 'Need math tutor for grade 10'],
            'location_id': ['loc_001', 'loc_002'],
            'posted_by_user_id': ['client_001', 'client_001'],
            'required_category_id': ['cat_001', 'cat_002']
        })
        
        self.sample_categories = pd.DataFrame({
            'category_id': ['cat_001', 'cat_002'],
            'category_name': ['Electrician', 'Tutoring']
        })
        
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data=json.dumps([{"location_id": "loc_001"}]))
    def test_load_locations(self, mock_file, mock_exists):
        """Test loading locations from JSON file."""
        # Set up mock
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(self.sample_locations)
        
        # Call function
        locations = self.data_loader._load_locations('./sample_data/locations.json')
        
        # Verify result
        self.assertIsInstance(locations, pd.DataFrame)
        self.assertEqual(len(locations), len(self.sample_locations))
        self.assertListEqual(list(locations['location_id']), ['loc_001', 'loc_002'])
        self.assertEqual(locations['longitude'].dtype, np.float64)
        self.assertEqual(locations['latitude'].dtype, np.float64)
        
    @patch('os.path.exists', return_value=True)
    def test_load_csv_file(self, mock_exists):
        """Test loading data from CSV file."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            self.sample_users.to_csv(tmp.name, index=False)
            tmp_path = tmp.name
            
        try:
            # Call function
            users = self.data_loader._load_csv_file(tmp_path)
            
            # Verify result
            self.assertIsInstance(users, pd.DataFrame)
            self.assertEqual(len(users), len(self.sample_users))
            self.assertListEqual(list(users['user_id']), list(self.sample_users['user_id']))
            self.assertListEqual(list(users.columns), list(self.sample_users.columns))
        finally:
            # Clean up
            os.unlink(tmp_path)
            
    @patch.object(DataLoader, '_load_locations')
    @patch.object(DataLoader, '_load_csv_file')
    def test_load_data(self, mock_load_csv, mock_load_locations):
        """Test loading all data."""
        # Set up mocks
        mock_load_locations.return_value = pd.DataFrame(self.sample_locations)
        mock_load_csv.side_effect = [
            self.sample_users, 
            self.sample_jobs, 
            self.sample_categories,
            None  # For job_applications.csv (optional)
        ]
        
        # Call function
        data_dict = self.data_loader.load_data('./sample_data')
        
        # Verify result
        self.assertIsInstance(data_dict, dict)
        self.assertIn('locations', data_dict)
        self.assertIn('users', data_dict)
        self.assertIn('jobs', data_dict)
        self.assertIn('categories', data_dict)
        
        # Check call counts
        self.assertEqual(mock_load_locations.call_count, 1)
        self.assertEqual(mock_load_csv.call_count, 4)  # users, jobs, categories, job_applications


class TestDataValidator(unittest.TestCase):
    """Test cases for the DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'validation': {
                'required_columns': {
                    'locations': ['location_id', 'longitude', 'latitude'],
                    'users': ['user_id', 'username', 'user_type', 'location_id'],
                    'jobs': ['job_id', 'title', 'description', 'location_id', 'posted_by_user_id']
                }
            }
        }
        self.validator = DataValidator(config=self.config)
        
        # Sample valid data
        self.valid_data = {
            'locations': pd.DataFrame({
                'location_id': ['loc_001', 'loc_002'],
                'post_code': ['01001', '01002'],
                'name': ['Adrar', 'Reggane'],
                'wilaya_name': ['Adrar', 'Adrar'],
                'longitude': [-0.4841573, -0.1712033],
                'latitude': [27.9763317, 26.7208076]
            }),
            'users': pd.DataFrame({
                'user_id': ['prof_001', 'client_001'],
                'username': ['pro_user_A', 'client_user_X'],
                'user_type': ['professional', 'client'],
                'location_id': ['loc_001', 'loc_002'],
                'profile_bio': ['Experienced electrician', 'Homeowner looking for help']
            }),
            'jobs': pd.DataFrame({
                'job_id': ['job_001', 'job_002'],
                'title': ['Urgent: Electrician', 'Math Tutor Needed'],
                'description': ['Need certified electrician', 'Need math tutor for grade 10'],
                'location_id': ['loc_001', 'loc_002'],
                'posted_by_user_id': ['client_001', 'client_001'],
                'required_category_id': ['cat_001', 'cat_002']
            })
        }
        
    def test_validate_schema(self):
        """Test schema validation."""
        # Valid data should pass
        result = self.validator.validate_schema(self.valid_data)
        self.assertTrue(result)
        
        # Invalid data (missing column) should fail
        invalid_data = self.valid_data.copy()
        invalid_data['users'] = invalid_data['users'].drop('user_type', axis=1)
        with self.assertLogs(level='ERROR'):
            result = self.validator.validate_schema(invalid_data)
        self.assertFalse(result)
        
    def test_validate_foreign_keys(self):
        """Test foreign key validation."""
        # Valid data should pass
        result = self.validator.validate_foreign_keys(self.valid_data)
        self.assertTrue(result)
        
        # Invalid data (missing foreign key) should fail
        invalid_data = self.valid_data.copy()
        invalid_data['users'].loc[0, 'location_id'] = 'loc_999'  # Non-existent location
        with self.assertLogs(level='ERROR'):
            result = self.validator.validate_foreign_keys(invalid_data)
        self.assertFalse(result)
        
    def test_validate_data_types(self):
        """Test data type validation."""
        # Valid data should pass
        result = self.validator.validate_data_types(self.valid_data)
        self.assertTrue(result)
        
        # Invalid data (wrong type) should fail
        invalid_data = self.valid_data.copy()
        invalid_data['locations'].loc[0, 'longitude'] = 'not-a-number'
        with self.assertLogs(level='ERROR'):
            result = self.validator.validate_data_types(invalid_data)
        self.assertFalse(result)
        
    def test_validate_all(self):
        """Test comprehensive validation."""
        # Valid data should pass
        with patch.object(self.validator, 'validate_schema', return_value=True):
            with patch.object(self.validator, 'validate_foreign_keys', return_value=True):
                with patch.object(self.validator, 'validate_data_types', return_value=True):
                    result = self.validator.validate_all(self.valid_data)
                    self.assertTrue(result)
        
        # If any validation fails, validate_all should fail
        with patch.object(self.validator, 'validate_schema', return_value=True):
            with patch.object(self.validator, 'validate_foreign_keys', return_value=False):
                with patch.object(self.validator, 'validate_data_types', return_value=True):
                    with self.assertLogs(level='ERROR'):
                        result = self.validator.validate_all(self.valid_data)
                    self.assertFalse(result)


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for the DataPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'preprocessing': {
                'text_cleaning': {
                    'lowercase': True,
                    'remove_special_chars': True
                },
                'category_handling': 'one_hot'  # or 'multi_label'
            }
        }
        self.preprocessor = DataPreprocessor(config=self.config)
        
        # Sample data
        self.sample_data = {
            'users': pd.DataFrame({
                'user_id': ['prof_001', 'client_001'],
                'username': ['pro_user_A', 'client_user_X'],
                'user_type': ['professional', 'client'],
                'location_id': ['loc_001', 'loc_002'],
                'profile_bio': ['Experienced ELECTRICIAN with 10+ years!', 'Homeowner looking for HELP.']
            }),
            'jobs': pd.DataFrame({
                'job_id': ['job_001', 'job_002'],
                'title': ['Urgent: Electrician Needed!', 'Math Tutor Needed ASAP'],
                'description': ['Need certified electrician for wiring (residential)', 'Need math tutor for grade 10 student.'],
                'location_id': ['loc_001', 'loc_002'],
                'posted_by_user_id': ['client_001', 'client_001'],
                'required_category_id': ['cat_001', 'cat_002']
            }),
            'professional_categories': pd.DataFrame({
                'user_id': ['prof_001', 'prof_001'],
                'category_id': ['cat_001', 'cat_003']
            })
        }
        
    def test_clean_text(self):
        """Test text cleaning."""
        # Test with sample text
        sample_text = "This is a TEST with Special-Characters! 123"
        
        # With default settings
        cleaned = self.preprocessor._clean_text(sample_text)
        self.assertEqual(cleaned, "this is a test with specialcharacters 123")
        
        # Without lowercase
        with patch.dict(self.config['preprocessing']['text_cleaning'], {'lowercase': False}):
            preprocessor = DataPreprocessor(config=self.config)
            cleaned = preprocessor._clean_text(sample_text)
            self.assertEqual(cleaned, "This is a TEST with SpecialCharacters 123")
            
        # Without special char removal
        with patch.dict(self.config['preprocessing']['text_cleaning'], {'remove_special_chars': False}):
            preprocessor = DataPreprocessor(config=self.config)
            cleaned = preprocessor._clean_text(sample_text)
            self.assertEqual(cleaned, "this is a test with special-characters! 123")
            
    def test_preprocess_text_fields(self):
        """Test preprocessing of text fields."""
        # Create a copy to avoid modifying original
        data_dict = self.sample_data.copy()
        
        # Preprocess text fields
        processed = self.preprocessor._preprocess_text_fields(data_dict)
        
        # Check user bios
        self.assertEqual(processed['users'].loc[0, 'profile_bio'], 
                        "experienced electrician with 10 years")
        
        # Check job titles and descriptions
        self.assertEqual(processed['jobs'].loc[0, 'title'],
                        "urgent electrician needed")
        self.assertEqual(processed['jobs'].loc[0, 'description'],
                        "need certified electrician for wiring residential")
        
    def test_preprocess_categories(self):
        """Test preprocessing of categories."""
        # Create a mock implementation for one-hot encoding
        with patch.object(self.preprocessor, '_create_category_features', return_value=pd.DataFrame({
            'user_id': ['prof_001'],
            'cat_001': [1],
            'cat_002': [0],
            'cat_003': [1]
        })):
            # Process data
            data_dict = self.sample_data.copy()
            processed = self.preprocessor._preprocess_categories(data_dict)
            
            # Check that categories were processed
            self.assertIn('user_categories', processed)
            self.assertIn('cat_001', processed['user_categories'].columns)
            self.assertIn('cat_003', processed['user_categories'].columns)
    
    def test_preprocess(self):
        """Test the complete preprocessing pipeline."""
        # Mock the individual preprocessing methods
        with patch.object(self.preprocessor, '_preprocess_text_fields', return_value=self.sample_data):
            with patch.object(self.preprocessor, '_preprocess_categories', return_value=self.sample_data):
                with patch.object(self.preprocessor, '_preprocess_locations', return_value=self.sample_data):
                    # Call the preprocess method
                    result = self.preprocessor.preprocess(self.sample_data)
                    
                    # Verify that all methods were called
                    self.assertEqual(id(result), id(self.sample_data))  # Should return the same dict object


if __name__ == '__main__':
    unittest.main()
