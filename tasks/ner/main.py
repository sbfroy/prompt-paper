import sys
import os
import logging
from pathlib import Path
from vllm import LLM, SamplingParams
import yaml

logging.basicConfig(level=logging.INFO)

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.schemas import TaskType
from core.stages.cluster import run_cluster_stage
from core.stages.evolve import run_evolve_stage
from core.data_manager import DataManager
from core.wandb_utils import init_wandb, finish_wandb, log_metrics
from .evaluation import NERTaskEvaluator

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_evaluation_function(base_dir, eval_config, cluster_dataset, llm_instance, sampling_params):
    evaluator = NERTaskEvaluator(base_dir, eval_config, llm_instance, sampling_params)
    return evaluator.evaluate_individual

def run_pipeline():
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent  
    
    # Init wandb (basic)
    run = init_wandb(task_name=config.get('task_name', 'ner'), config=config)

    # ====== Run CLUSTERING STAGE ======
    cluster_output = run_cluster_stage(
        task=TaskType.NER,
        base_dir=str(base_dir),
        config_dict=config.get('clustering', {})
    )

    data_manager = DataManager(TaskType.NER, str(base_dir))
    cluster_dataset = data_manager.load_cluster_dataset()

    # ====== RUN EVOLUTION STAGE ======

    eval_config = config.get('evaluation', {})

    # Initialize LLM and sampling parameters for evaluation
    logging.info(f"Creating vLLM instance for evaluation...")
    llm_config = eval_config.get("llm", {})
    llm_instance = LLM(**llm_config)
    sampling_config = eval_config.get("sampling", {})
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
        task=TaskType.NER,
        base_dir=str(base_dir),
        config_dict=config.get('evolution', {}),
        evaluate_fn=evaluate_fn
    )

    finish_wandb()

    logging.info(f"Pipeline completed successfully!")
 
if __name__ == "__main__":
    run_pipeline()