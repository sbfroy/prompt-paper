import sys
import logging
from pathlib import Path
from vllm import LLM, SamplingParams
import yaml

logging.basicConfig(level=logging.INFO)

# suppress the batch logging 
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root)) # step out to 'prompt-paper'

from core.data_manager import DataManager
from core.wandb_utils import init_wandb, finish_wandb
from core.stages.cluster import run_cluster_stage

def load_config(): # Loading the config file
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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

    finish_wandb()

    logging.info(f"Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()