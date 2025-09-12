import os
import wandb
import json
from datetime import datetime

def init_wandb(task_name, config):
    """
    Flattens config and initializes a wandb run.

    """
    # TODO: task_name should be project name, so i get different projects for different tasks
    config = _flatten_dict(config)
    run = wandb.init(project="icl_project_2025", config=config)
    return run

def log_metrics(step, **metrics):
    """
    Logs metrics to wandb at a specific step.
    Steps should be the generation number in evolution.
    """
    return wandb.log(metrics, step=step)

def log_best_examples(selected_examples):
    """
    Logs the best individual as a wandb table.

    """
    table = wandb.Table(columns=["cluster_id", "example_id", "text"])
    for example in selected_examples:
        table.add_data(
            example["cluster_id"],
            example["example_id"], 
            example["text"]
        )

    wandb.log({"best_examples": table})

def finish_wandb():
    if wandb.run is not None:
        wandb.finish()

def _flatten_dict(d: dict, parent_key: str = "", sep: str = "."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
