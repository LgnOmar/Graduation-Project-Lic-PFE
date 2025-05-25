"""
Configuration utilities for JibJob recommendation system.
This module handles loading and managing configuration settings.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()

class Config:
    """
    Configuration manager for the recommendation system.
    
    This class handles loading and accessing configuration settings from
    environment variables, JSON, or YAML files.
    """
    
    def __init__(
        self,
        config_file: Optional[str] = None,
        env_prefix: str = "JIBJOB_"
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to a configuration file (JSON or YAML).
                         If None, only environment variables are used.
            env_prefix: Prefix for environment variables.
        """
        self._config = {}
        self._env_prefix = env_prefix
        
        # Load configuration from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Load configuration from environment variables
        self.load_from_env()
    
    def load_from_file(self, config_file: str):
        """
        Load configuration from a file.
        
        Args:
            config_file: Path to the configuration file.
        
        Raises:
            ValueError: If the file format is not supported or file not found.
        """
        path = Path(config_file)
        
        if not path.exists():
            raise ValueError(f"Configuration file not found: {config_file}")
        
        suffix = path.suffix.lower()
        
        try:
            if suffix == '.json':
                with open(path, 'r') as f:
                    file_config = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    file_config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {suffix}")
            
            # Update configuration
            self._config.update(file_config)
            logger.info(f"Loaded configuration from: {config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration from file: {e}")
            raise
    
    def load_from_env(self):
        """
        Load configuration from environment variables.
        
        Environment variables with the specified prefix are added to the configuration.
        The prefix is removed and the rest of the variable name is converted to lowercase.
        
        For example:
            JIBJOB_MODEL_PATH -> model_path
        """
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix):].lower()
                
                # Parse boolean and numeric values
                if value.lower() in ['true', 'yes']:
                    value = True
                elif value.lower() in ['false', 'no']:
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    value = float(value)
                
                self._config[config_key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key.
            default: Default value if key is not found.
            
        Returns:
            Any: The configuration value or default.
        """
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key.
            value: Configuration value.
        """
        self._config[key] = value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a section of the configuration.
        
        Args:
            section: Section name.
            
        Returns:
            Dict[str, Any]: Dictionary with the section configuration.
        """
        result = {}
        prefix = section + "_"
        
        # Find all keys that start with the section prefix
        for key, value in self._config.items():
            if key == section and isinstance(value, dict):
                # Direct section match
                result.update(value)
            elif key.startswith(prefix):
                # Section prefix match
                result[key[len(prefix):]] = value
        
        return result
    
    def save(self, file_path: str):
        """
        Save configuration to a file.
        
        Args:
            file_path: Path to save the configuration file.
            
        Raises:
            ValueError: If the file format is not supported.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        os.makedirs(path.parent, exist_ok=True)
        
        try:
            if suffix == '.json':
                with open(path, 'w') as f:
                    json.dump(self._config, f, indent=2)
            elif suffix in ['.yaml', '.yml']:
                with open(path, 'w') as f:
                    yaml.dump(self._config, f)
            else:
                raise ValueError(f"Unsupported configuration file format: {suffix}")
            
            logger.info(f"Saved configuration to: {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration to file: {e}")
            raise
    
    @property
    def as_dict(self) -> Dict[str, Any]:
        """Get the complete configuration as a dictionary."""
        return dict(self._config)


# Default configuration with sensible defaults
DEFAULT_CONFIG = {
    "model": {
        "bert_model_name": "bert-base-multilingual-cased",
        "sentiment_model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
        "embedding_dim": 64,
        "hidden_dim": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "conv_type": "gcn",
        "sentiment_weight": 0.5,
        "rating_weight": 0.5,
        "cache_dir": "cache"
    },
    "training": {
        "batch_size": 64,
        "epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "early_stop_patience": 10,
        "val_ratio": 0.2,
        "test_ratio": 0.1
    },
    "data": {
        "rating_col": "rating",
        "comment_col": "comment",
        "user_col": "user_id",
        "job_col": "job_id",
        "rating_max": 5.0,
        "clean_text": {
            "remove_numbers": False,
            "remove_punctuation": True,
            "lowercase": True,
            "remove_stopwords": True,
            "stemming": False
        }
    },
    "paths": {
        "data_dir": "data",
        "processed_dir": "data/processed",
        "model_dir": "models",
        "output_dir": "output",
        "log_dir": "logs"
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "workers": 1
    },
    "system": {
        "seed": 42,
        "device": "cuda",  # or "cpu"
        "num_workers": 4,
        "log_level": "INFO"
    }
}

# Create a global config instance with default values
config = Config()

# Initialize with default values
for section, values in DEFAULT_CONFIG.items():
    if isinstance(values, dict):
        for key, value in values.items():
            config.set(f"{section}_{key}", value)
    else:
        config.set(section, values)

def load_config(config_file: Optional[str] = None):
    """
    Load configuration from file and environment variables.
    
    Args:
        config_file: Path to configuration file.
        
    Returns:
        Config: Configuration object.
    """
    global config
    
    # Reset config
    config = Config()
    
    # Initialize with default values
    for section, values in DEFAULT_CONFIG.items():
        if isinstance(values, dict):
            for key, value in values.items():
                config.set(f"{section}_{key}", value)
        else:
            config.set(section, values)
    
    # Load from file if provided
    if config_file:
        config.load_from_file(config_file)
    
    # Load from environment
    config.load_from_env()
    
    return config

def get_config() -> Config:
    """
    Get the global configuration object.
    
    Returns:
        Config: Global configuration object.
    """
    global config
    return config
