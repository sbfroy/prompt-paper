import sys
import logging
from pathlib import Path
from openai import OpenAI
import yaml
import os

logging.basicConfig(level=logging.INFO)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.data_manager import DataManager
from core.wandb_utils import init_wandb, finish_wandb
from core.stages.cluster import run_cluster_stage
from core.stages.evolve import run_evolve_stage
from .evaluation import Evaluator

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_evaluator(base_dir, eval_config, client):
    """
    Function factory because GA requires a single-argument function.
    """
    evaluator = Evaluator(base_dir, eval_config, client)
    return evaluator.evaluate_individual

def run_pipeline():
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent  

    # Initialize DataManager
    data_manager = DataManager(config['task'], str(base_dir))
    
    # Initialize wandb
    run = init_wandb(task_name=config['task'], config=config)

    # ====== Run CLUSTERING STAGE ======

    cluster_output = run_cluster_stage(
        task=config['task'],
        base_dir=str(base_dir),
        config_dict=config['clustering']
    )

    # Load cluster dataset from wandb
    cluster_dataset = data_manager.load_cluster_dataset()

    # ====== RUN EVOLUTION STAGE ======

    # Initialize OpenAI-compatible client and sampling parameters for evaluation
    logging.info("Creating OpenAI client for evaluation...")

    client = OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    )
    
    eval_fn = create_evaluator(
        base_dir=str(base_dir), 
        eval_config=config['evaluation'], 
        client=client
    )

    evolution_output = run_evolve_stage(
        task=config['task'],
        base_dir=str(base_dir),
        config=config['evolution'],
        eval_fn=eval_fn
    )

    finish_wandb()

    logging.info(f"Pipeline completed successfully!")
 
if __name__ == "__main__":
    run_pipeline()