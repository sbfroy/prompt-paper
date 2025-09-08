import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.schemas import TaskType
from core.stages.cluster.main import run_clustering_stage
import yaml

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline():
    config = load_config()

    # Set up paths
    task_dir = Path(__file__).parent
    base_dir = task_dir.parent  
    
    try:
        cluster_output = run_clustering_stage(
            task=TaskType.NER,
            base_dir=str(base_dir),
            config_dict=config.get('clustering', {}),
            skip_embedding=config.get('skip_embedding', False)
        )
 
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    run_pipeline()
