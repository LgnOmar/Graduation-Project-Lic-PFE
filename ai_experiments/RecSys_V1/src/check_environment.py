"""
Validate environment and system requirements.
"""
import sys
import importlib
import logging
import os
import platform
import torch
from typing import List, Tuple, Dict
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version() -> bool:
    """Check if Python version is compatible."""
    required_version = (3, 6)
    current_version = sys.version_info[:2]
    
    is_compatible = current_version >= required_version
    logger.info(f"Python version: {'.'.join(map(str, current_version))}")
    logger.info(f"Required: >= {'.'.join(map(str, required_version))}")
    
    if not is_compatible:
        logger.error(f"Python version incompatible! Please use Python {'.'.join(map(str, required_version))} or higher.")
        
    return is_compatible

def check_dependencies() -> bool:
    """Check if all required packages are installed."""
    required_packages = [
        'torch',
        'torch_geometric',
        'transformers',
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib'
    ]
    
    missing_packages = []
    all_installed = True
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"Package {package} is installed.")
        except ImportError:
            logger.error(f"Package {package} is not installed!")
            missing_packages.append(package)
            all_installed = False
    
    if missing_packages:
        logger.info("Run the following command to install missing packages:")
        logger.info(f"pip install {' '.join(missing_packages)}")
        
    return all_installed

def check_gpu() -> bool:
    """Check if GPU is available and CUDA version."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "N/A"
        cuda_version = torch.version.cuda
        
        logger.info(f"CUDA is available: {gpu_count} GPU(s)")
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"CUDA version: {cuda_version}")
        
        return True
    else:
        logger.warning("CUDA is not available. Using CPU only.")
        logger.warning("Training may be slow without GPU acceleration.")
        return False

def check_system_resources() -> Dict:
    """Check available system resources."""
    import psutil
    
    # Memory information
    mem = psutil.virtual_memory()
    total_memory_gb = mem.total / (1024**3)
    available_memory_gb = mem.available / (1024**3)
    
    # CPU information
    cpu_count = psutil.cpu_count(logical=True)
    
    # Disk information
    disk = psutil.disk_usage('/')
    total_disk_gb = disk.total / (1024**3)
    free_disk_gb = disk.free / (1024**3)
    
    logger.info(f"System Information:")
    logger.info(f"- OS: {platform.system()} {platform.version()}")
    logger.info(f"- CPU: {cpu_count} logical cores")
    logger.info(f"- RAM: {total_memory_gb:.1f} GB total, {available_memory_gb:.1f} GB available")
    logger.info(f"- Disk: {total_disk_gb:.1f} GB total, {free_disk_gb:.1f} GB free")
    
    # Check if resources are sufficient
    is_sufficient = True
    if available_memory_gb < 4.0:
        logger.warning("Available memory is low (< 4GB). This may impact performance.")
        is_sufficient = False
    if free_disk_gb < 2.0:
        logger.warning("Free disk space is low (< 2GB). This may cause issues during data processing.")
        is_sufficient = False
        
    return {"is_sufficient": is_sufficient, "cpu_count": cpu_count, "memory_gb": available_memory_gb}

def check_data_directories() -> bool:
    """Check if required data directories exist."""
    required_dirs = ['data', 'models']
    missing_dirs = []
    
    for dirname in required_dirs:
        if not os.path.isdir(dirname):
            logger.warning(f"Directory '{dirname}' does not exist. Creating it.")
            os.makedirs(dirname, exist_ok=True)
            
    return True

def main():
    """Run all checks."""
    logger.info("=== Validating Environment and System Requirements ===")
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("GPU Availability", check_gpu()),
        ("Data Directories", check_data_directories())
    ]
    
    # Check system resources
    resources = check_system_resources()
    checks.append(("System Resources", resources["is_sufficient"]))
    
    # Summarize results
    logger.info("\n=== Validation Summary ===")
    all_passed = True
    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{name}: {status}")
        all_passed = all_passed and result
        
    if all_passed:
        logger.info("\nAll checks passed! The system is ready to run.")
    else:
        logger.warning("\nSome checks failed. Please address the issues before running the system.")
        
    # Provide recommendations based on system
    if resources["memory_gb"] < 8.0:
        logger.info("\nRecommendation: Reduce batch size in training to accommodate lower memory.")
    if not checks[2][1]:  # GPU check
        logger.info("\nRecommendation: For faster training, consider using a machine with GPU support.")
        
    return all_passed

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
