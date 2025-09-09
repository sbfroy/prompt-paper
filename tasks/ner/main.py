import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.schemas import TaskType
from core.stages.cluster.main import run_cluster_stage
from core.stages.evolve.main import run_evolve_stage
from .evaluation import evaluate_individual
import yaml

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_evaluation_function(base_dir: str):
    """Create evaluation function that has the right signature for the evolution stage."""
    def evaluate_fn(individual):
        # Load cluster dataset 
        from core.data_manager import DataManager
        data_manager = DataManager(TaskType.NER, base_dir)
        cluster_dataset = data_manager.load_cluster_dataset("cluster_dataset.jsonl")
        
        return evaluate_individual(
            individual=individual,
            cluster_dataset=cluster_dataset, 
            base_dir=base_dir
        )
    return evaluate_fn

def run_pipeline():
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent  
    
    try:
        # Run clustering stage
        cluster_output = run_cluster_stage(
            task=TaskType.NER,
            base_dir=str(base_dir),
            config_dict=config.get('clustering', {}),
            skip_embedding=config.get('skip_embedding', False)
        )
        
        # Run evolution stage with task-specific evaluation
        evaluate_fn = create_evaluation_function(str(base_dir))
        evolution_output = run_evolve_stage(
            task=TaskType.NER,
            base_dir=str(base_dir),
            config_dict=config.get('evolution', {}),
            evaluate_fn=evaluate_fn
        )
        
        print(f"Pipeline completed successfully!")
        print(f"Cluster output: {cluster_output}")
        print(f"Evolution output: {evolution_output}")
 
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
