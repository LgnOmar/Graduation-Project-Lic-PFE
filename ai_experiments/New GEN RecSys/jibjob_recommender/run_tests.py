#!/usr/bin/env python
"""
Run all tests for the JibJob recommendation system.

This script discovers and runs all test modules in the tests directory.
"""
import os
import sys
import unittest
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Get the directory containing this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the tests directory
    tests_dir = os.path.join(current_dir, 'tests')
    
    # Discover and run tests
    logger.info(f"Discovering tests in {tests_dir}")
    test_suite = unittest.defaultTestLoader.discover(tests_dir)
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return True if all tests passed, False otherwise
    return result.wasSuccessful()

def main():
    """Main entry point."""
    logger.info("Starting JibJob Recommender System Test Suite")
    
    success = run_all_tests()
    
    if success:
        logger.info("All tests passed successfully!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
