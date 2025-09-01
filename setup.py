#!/usr/bin/env python3
"""
Setup script for Crypto PCA Project
Downloads top 200 crypto tokens by market cap and performs PCA analysis
"""

import os
import sys
import subprocess
import logging
from setuptools import setup, find_packages

def setup_logging():
    """Setup logging for setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")

def install_requirements():
    """Install requirements from requirements.txt"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists('requirements.txt'):
        logger.error("requirements.txt not found!")
        return False
    
    logger.info("Installing requirements from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        logger.info("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories for the project"""
    logger = logging.getLogger(__name__)
    
    directories = ['data', 'results', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def setup_environment():
    """Main setup function"""
    logger = setup_logging()
    
    logger.info("Starting Crypto PCA Project Setup...")
    
    try:
        # Check Python version
        check_python_version()
        logger.info(f"Python version check passed: {sys.version}")
        
        # Create necessary directories
        create_directories()
        
        # Install requirements
        if install_requirements():
            logger.info("Setup completed successfully!")
            logger.info("You can now run: python main.py")
            return True
        else:
            logger.error("Setup failed during requirements installation")
            return False
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

if __name__ == "__main__":
    # Run setup when script is executed directly
    success = setup_environment()
    if success:
        print("\n✅ Setup completed successfully!")
        print("Next step: Run 'python main.py' to start the analysis")
    else:
        print("\n❌ Setup failed. Please check the logs above.")
        sys.exit(1)
else:
    # Standard setuptools setup for package installation
    setup(
        name="crypto-pca-analysis",
        version="0.1.0",
        description="PCA analysis of top 200 crypto tokens by market cap",
        author="Crypto PCA Team",
        packages=find_packages(),
        install_requires=[
            # Will be populated as we add dependencies
        ],
        python_requires=">=3.8",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Financial and Insurance Industry",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
    ) 