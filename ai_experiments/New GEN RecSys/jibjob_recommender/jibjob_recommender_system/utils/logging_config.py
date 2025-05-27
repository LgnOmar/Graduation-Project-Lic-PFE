"""
Logging configuration for the JibJob recommendation system.
"""

import os
import logging
import logging.handlers
from typing import Dict, Any, Optional

def setup_logging(config: Optional[Dict[str, Any]] = None, log_level: int = None, log_file: str = None) -> None:
    """
    Configure logging for the application based on the provided config.
    
    Args:
        config: The logging configuration from settings.yaml. 
               If None, default configuration will be used.
        log_level: Explicit log level to override config.
        log_file: Explicit log file path to override config.
    """
    if config is None:
        default_log_level = logging.INFO
        default_log_file = "jibjob_recsys.log"
    else:
        log_level_str = config.get('level', 'INFO').upper()
        default_log_file = config.get('file', 'jibjob_recsys.log')
        
        # Map string level to logging constant
        log_level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        default_log_level = log_level_map.get(log_level_str, logging.INFO)

    # Use explicit parameters if provided, otherwise use defaults from config
    log_level = log_level if log_level is not None else default_log_level
    log_file = log_file if log_file is not None else default_log_file
    
    # Create the logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logging.info("Logging configured successfully")
    
if __name__ == "__main__":
    # Simple test of the logging configuration
    setup_logging()
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")
