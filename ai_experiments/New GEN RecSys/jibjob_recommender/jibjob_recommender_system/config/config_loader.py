"""
Configuration loader for JibJob recommendation system.
"""

import os
import yaml
import logging
from typing import Dict, Any


class ConfigLoader:
    """
    Configuration loader class for JibJob Recommendation System.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file. If None, uses default path.
        """
        self.config_path = config_path
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the specified YAML file.
        
        Returns:
            Dict containing the configuration.
            
        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is not properly formatted.
        """
        if self.config_path is None:
            # Get the directory of this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.config_path = os.path.join(current_dir, 'settings.yaml')
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Validate config
            validate_config(config)
            return config
        
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing configuration file: {e}")
            raise


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Backward compatibility function for loading configuration.
    
    Args:
        config_path: Path to the configuration file. If None, uses default path.
        
    Returns:
        Dict containing the configuration.
    """
    loader = ConfigLoader(config_path)
    return loader.load_config()

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the loaded configuration to ensure required fields are present.
    
    Args:
        config: Configuration dictionary to validate.
        
    Raises:
        ValueError: If required fields are missing.
    """
    # Required top-level sections
    required_sections = ['data', 'features', 'model', 'training', 'inference']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Check for essential data paths
    if 'base_path' not in config['data']:
        raise ValueError("Missing data.base_path in configuration")
    
    # Additional validations can be added here
    
    # Resolve paths relative to base path
    data_config = config['data']
    base_path = data_config['base_path']
    
    # Make data paths absolute if they're relative
    for key, path in data_config.items():
        if key != 'base_path' and isinstance(path, str):
            if not os.path.isabs(path):
                data_config[key] = os.path.join(base_path, path)

if __name__ == "__main__":
    # Simple test to validate that config can be loaded
    config = load_config()
    print("Config loaded successfully!")
    print(f"Data base path: {config['data']['base_path']}")
    print(f"Model hidden dimension: {config['model']['hidden_dim']}")
