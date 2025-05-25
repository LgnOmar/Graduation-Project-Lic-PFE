"""
Pipeline script to run all components of the JibJob recommendation system.
"""
import os
import sys
import logging
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_step(step_name: str, script_path: str) -> bool:
    """Run a pipeline step and return True if successful."""
    logger.info(f"Running {step_name}...")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"{step_name} output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"{step_name} warnings:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{step_name} failed with error:\n{e.stderr}")
        return False

def main():
    # Get the project root directory (where pipeline.py is located)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Project root: {project_root}")
    
    # Add project root to Python path
    sys.path.insert(0, project_root)
    os.chdir(project_root)
    
    # Install dependencies
    logger.info("Installing dependencies...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies:\n{e.stderr}")
        sys.exit(1)
    
    # Pipeline steps
    steps = [
        ("Data Simulation", "src/data_simulation.py"),
        ("Feature Engineering", "src/feature_engineering.py"),
        ("Model Training", "src/train_gcn.py")
    ]
    
    # Run each step
    for step_name, script_path in steps:
        if not run_step(step_name, script_path):
            logger.error(f"Pipeline failed at {step_name}")
            sys.exit(1)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
