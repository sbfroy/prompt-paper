import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.schemas import TaskType
from core.stages.cluster import run_cluster_stage
from core.stages.evolve import run_evolve_stage
from core.data_manager import DataManager
from .evaluation import NERTaskEvaluator
import yaml
from core.wandb_utils import init_wandb, finish_wandb, log_metrics

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_evaluation_function(base_dir, config, cluster_dataset):
    eval_config = config.get('evaluation', {}) if config else {}
    evaluator = NERTaskEvaluator(base_dir, eval_config)

    return evaluator.evaluate_individual

def run_pipeline():
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent  
    
    # Init wandb (basic)
    run = init_wandb(task_name=config.get('task_name', 'ner'), config=config)

    # ====== Run clustering stage ======
    cluster_output = run_cluster_stage(
        task=TaskType.NER,
        base_dir=str(base_dir),
        config_dict=config.get('clustering', {})
    )

    data_manager = DataManager(TaskType.NER, str(base_dir))
    cluster_dataset = data_manager.load_cluster_dataset()
    
    # ====== Run evolution stage ======
    # with custom evaluation script
    evaluate_fn = create_evaluation_function(str(base_dir), config, cluster_dataset)
    evolution_output = run_evolve_stage(
        task=TaskType.NER,
        base_dir=str(base_dir),
        config_dict=config.get('evolution', {}),
        evaluate_fn=evaluate_fn
    )

    finish_wandb()

    print(f"Pipeline completed successfully!")
 
if __name__ == "__main__":
    run_pipeline()