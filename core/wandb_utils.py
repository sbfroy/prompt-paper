import os
import wandb
import json
from datetime import datetime
import tempfile
from pathlib import Path

def init_wandb(task_name, config):
    """
    Flattens config and initializes a wandb run.
    Uses task_name as project name for separate projects per task.
    """
    config = _flatten_dict(config)
    run = wandb.init(
        project=f"{task_name}_task",
        entity="sbfroy_work", 
        config=config
        )
    return run

def log_metrics(step, **metrics):
    """
    Logs metrics to wandb at a specific step.
    Steps should be the generation number in evolution.
    """
    return wandb.log(metrics, step=step)

def finish_wandb():
    if wandb.run is not None:
        wandb.finish()

def save_artifact(data, artifact_name, artifact_type):
    """
    Save data as wandb artifact. Returns the artifact.

    Args:
        data: Data to be saved (should be JSON-serializable).
        artifact_name: Name of the artifact.
        artifact_type: Type of the artifact.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"{artifact_name}.json"
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(str(temp_path))

        wandb.log_artifact(artifact)

        artifact.wait()  # Wait for upload to complete

        return artifact

def save_file_artifact(file_path, artifact_name, artifact_type):
    """
    Save existing file as wandb artifact. Returns the artifact.

    Args:
        file_path: Path to the file to be saved as artifact.
        artifact_name: Name of the artifact.
        artifact_type: Type of the artifact.
    """
    artifact = wandb.Artifact(artifact_name, type=artifact_type)
    artifact.add_file(str(file_path))

    wandb.log_artifact(artifact)

    artifact.wait() 

    return artifact

def load_artifact(artifact_name):
    """
    Load wandb artifact and return the downloaded file path.
    
    """
    artifact = wandb.use_artifact(f"{artifact_name}:latest")
    artifact_dir = artifact.download()

    # Find the first file in the artifact directory
    files = list(Path(artifact_dir).glob("*"))

    return files[0]
    

def _flatten_dict(d: dict, parent_key: str = "", sep: str = "."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
