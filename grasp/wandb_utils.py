"""Utilities for Weights & Biases (wandb) integration and artifact management."""

import os
import wandb
import json
from datetime import datetime
import tempfile
from pathlib import Path


def init_wandb(task_name, config):
    """Initialize a wandb run with flattened configuration.
    
    Args:
        task_name: Name of the task (used as project name)
        config: Configuration dictionary (will be flattened)
        
    Returns:
        Wandb run object
    """
    config = _flatten_dict(config)
    run = wandb.init(
        project=f"{task_name}_task",
        entity="icl-research-team",
        config=config
    )
    return run


def log_metrics(step, **metrics):
    """Log metrics to wandb at a specific step.
    
    Args:
        step: Step number (e.g., generation number in evolution)
        **metrics: Metric key-value pairs to log
        
    Returns:
        Result of wandb.log()
    """
    return wandb.log(metrics, step=step)


def finish_wandb():
    """Finish the current wandb run if one is active."""
    if wandb.run is not None:
        wandb.finish()


def save_artifact(data, artifact_name, artifact_type):
    """Save data as wandb artifact.
    
    Args:
        data: JSON-serializable data to save
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (e.g., 'dataset', 'model')
        
    Returns:
        Wandb artifact object
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"{artifact_name}.json"
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(str(temp_path))

        wandb.log_artifact(artifact)
        artifact.wait()  # Wait for upload to complete

        return artifact


def save_file_artifact(file_path, artifact_name, artifact_type):
    """Save existing file as wandb artifact.
    
    Args:
        file_path: Path to the file to save
        artifact_name: Name for the artifact
        artifact_type: Type of artifact
        
    Returns:
        Wandb artifact object
    """
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(str(file_path))

    wandb.log_artifact(artifact)
    artifact.wait()

    return artifact


def load_artifact(artifact_name):
    """Load wandb artifact and return the downloaded file path.
    
    Args:
        artifact_name: Name of the artifact to load (loads latest version)
        
    Returns:
        Path to the first file in the artifact directory
    """
    artifact = wandb.use_artifact(f"{artifact_name}:latest")
    artifact_dir = artifact.download()

    # Find the first file in the artifact directory
    files = list(Path(artifact_dir).glob("*"))

    return files[0]


def _flatten_dict(d: dict, parent_key: str = "", sep: str = "."):
    """Recursively flatten a nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Key prefix for nested items
        sep: Separator between nested keys
        
    Returns:
        Flattened dictionary
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
