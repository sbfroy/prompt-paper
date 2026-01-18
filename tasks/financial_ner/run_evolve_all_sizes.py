#!/usr/bin/env python3
"""
Run the evolution stage multiple times with different dataset sizes.

This script runs run_evolve.py three times with dataset sizes:
- 5000 (large)
- 500 (medium)  
- 50 (small)

Each run gets its own wandb run for tracking.
"""

import subprocess
import sys
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Dataset sizes to test (in order)
DATASET_SIZES = [5000, 500, 50]

# Path to config file (same directory as this script)
CONFIG_PATH = Path(__file__).parent / "config.yaml"
RUN_EVOLVE_SCRIPT = Path(__file__).parent / "run_evolve.py"


def load_config():
    """Load the current config."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(config):
    """Save modified config back to file."""
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def run_evolution_with_size(dataset_size: int, original_config: dict):
    """
    Run the evolution script with a specific dataset size.
    
    Args:
        dataset_size: Number of examples to use for evolution
        original_config: Original config to restore after run
    """
    logging.info("=" * 60)
    logging.info(f"Starting evolution run with dataset_size={dataset_size}")
    logging.info("=" * 60)
    
    # Load current config and modify dataset size
    config = load_config()
    config["dataset"]["size"] = dataset_size
    save_config(config)
    
    logging.info(f"Config updated: dataset.size = {dataset_size}")
    
    try:
        # Run the evolution script
        result = subprocess.run(
            [sys.executable, str(RUN_EVOLVE_SCRIPT)],
            cwd=str(RUN_EVOLVE_SCRIPT.parent.parent.parent),
            check=True,
        )
        logging.info(f"Evolution run with size={dataset_size} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Evolution run with size={dataset_size} failed with return code {e.returncode}")
        return False
        
    except Exception as e:
        logging.error(f"Evolution run with size={dataset_size} failed with error: {e}")
        return False


def main():
    """Run evolution for all dataset sizes."""
    logging.info("Starting multi-size evolution experiment")
    logging.info(f"Dataset sizes to test: {DATASET_SIZES}")
    
    # Store original config to restore later
    original_config = load_config()
    original_size = original_config["dataset"]["size"]
    logging.info(f"Original dataset size in config: {original_size}")
    
    results = {}
    
    try:
        for size in DATASET_SIZES:
            success = run_evolution_with_size(size, original_config)
            results[size] = "SUCCESS" if success else "FAILED"
            
            if not success:
                logging.warning(f"Run with size={size} failed, continuing with next size...")
    
    finally:
        # Restore original config
        logging.info(f"Restoring original config (dataset.size = {original_size})")
        save_config(original_config)
    
    # Print summary
    logging.info("=" * 60)
    logging.info("EXPERIMENT SUMMARY")
    logging.info("=" * 60)
    for size, status in results.items():
        logging.info(f"  Dataset size {size:>5}: {status}")
    logging.info("=" * 60)
    
    # Return non-zero if any run failed
    if "FAILED" in results.values():
        sys.exit(1)


if __name__ == "__main__":
    main()
