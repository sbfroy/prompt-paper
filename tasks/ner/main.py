import sys
import logging
from pathlib import Path
from vllm import LLM, SamplingParams
import yaml

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

def create_evaluation_function(base_dir, eval_config, cluster_dataset, llm_instance, sampling_params):
    evaluator = Evaluator(base_dir, eval_config, llm_instance, sampling_params)
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

    # Initialize LLM and sampling parameters for evaluation
    logging.info(f"Creating vLLM instance for evaluation...")

    eval_config = config['evaluation']
    llm_config = eval_config['llm']
    llm_instance = LLM(**llm_config)
    sampling_config = eval_config['sampling']
    sampling_params = SamplingParams(**sampling_config)

    # Create custom evaluation function
    evaluate_fn = create_evaluation_function(
        str(base_dir), 
        eval_config, 
        cluster_dataset, 
        llm_instance, 
        sampling_params
    )

    evolution_output = run_evolve_stage(
        task=config['task'],
        base_dir=str(base_dir),
        config_dict=config['evolution'],
        evaluate_fn=evaluate_fn
    )

    finish_wandb()

    logging.info(f"Pipeline completed successfully!")
 
if __name__ == "__main__":
    run_pipeline()