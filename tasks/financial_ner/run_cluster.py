import sys
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from grasp.wandb_utils import init_wandb, finish_wandb
from grasp.stages.cluster import run_cluster_stage
from grasp.run_vllm import start_vllm_servers

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Some logging suppression
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # step out to 'GRaSp'


def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent

    # Initialize wandb
    run = init_wandb(task_name=config["task"], config=config)

    # Run clustering stage
    run_cluster_stage(
        task=config["task"], base_dir=str(base_dir), config_dict=config["clustering"]
    )

    finish_wandb()

    logging.info("Cluster stage completed successfully!")


if __name__ == "__main__":
    start_vllm_servers()
    main()
