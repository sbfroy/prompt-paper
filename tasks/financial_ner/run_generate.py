import sys
import logging
import os
from pathlib import Path


import yaml
from dotenv import load_dotenv
from openai import OpenAI

from proptimize.wandb_utils import init_wandb, finish_wandb
from proptimize.stages.generate import run_generate_stage
from proptimize.run_vllm import start_vllm_servers

load_dotenv()  

logging.basicConfig(level=logging.INFO)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))  # step out to 'prompt-paper'

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

    # Initialize OpenAI-compatible client 
    logging.info("Creating OpenAI client for generation...")

    client = OpenAI(
            base_url=f'http://localhost:{os.getenv("LLM_PORT", "8000")}/v1',
            api_key=os.getenv("LLM_API_KEY", "prompt-paper"),
        )

    run_generate_stage(
        task=config["task"],
        base_dir=str(base_dir),
        config=config["generation"],
        client=client
    )

    finish_wandb()

    logging.info("Generation stage completed successfully!")


if __name__ == "__main__":
    start_vllm_servers()
    main()