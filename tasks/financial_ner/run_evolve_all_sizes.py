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
from pathlib import Path

import yaml

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


def run_evolution_with_size(dataset_size: int):
    """
    Run the evolution script with a specific dataset size.
    
    Args:
        dataset_size: Number of examples to use for evolution
    """
    print(f"\n{'='*60}\nRUNNING WITH {dataset_size} EXAMPLES\n{'='*60}\n")
    
    # Load current config and modify dataset size
    config = load_config()
    config["dataset"]["size"] = dataset_size
    save_config(config)
    
    # Run the evolution script
    subprocess.run(
        [sys.executable, str(RUN_EVOLVE_SCRIPT)],
        cwd=str(RUN_EVOLVE_SCRIPT.parent.parent.parent),
        check=True,
    )


def main():
    """Run evolution for all dataset sizes."""  

    # Store original config to restore later
    original_config = load_config()
    
    try:
        for size in DATASET_SIZES:
            run_evolution_with_size(size)
    
    finally:
        # Restore original config
        save_config(original_config)


if __name__ == "__main__":
    main()
