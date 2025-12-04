"""
Financial NER - Clustering from Existing Embeddings

This script runs the clustering stage using pre-generated embeddings, which:
1. Loads existing embedded dataset from wandb artifacts
2. Performs dimensionality reduction with UMAP
3. Clusters examples using HDBSCAN
4. Saves cluster dataset to wandb

This is useful when you want to:
- Re-run clustering with different UMAP/HDBSCAN parameters
- Avoid regenerating embeddings (saves time and compute)
- Experiment with different clustering configurations
"""

import sys
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from grasp.wandb_utils import init_wandb, finish_wandb
from grasp.stages.cluster import run_cluster_from_embeddings_stage

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Suppress verbose logging from external libraries
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # step out to 'GRaSp'


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    """Run the clustering stage from existing embeddings for financial NER."""
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent

    # Initialize wandb for experiment tracking
    init_wandb(task_name=config["task"], config=config)

    # Run clustering stage (without embedding generation)
    run_cluster_from_embeddings_stage(
        task=config["task"],
        base_dir=str(base_dir),
        config_dict={
            **config["clustering"], 
            "dataset_size": config["dataset"]["size"],
            "use_real": config["dataset"]["use_real"]
        }
    )

    finish_wandb()

    logging.info("Clustering from embeddings stage completed successfully!")


if __name__ == "__main__":
    # Note: No need to start embedding server since we're using existing embeddings
    main()
