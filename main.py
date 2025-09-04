from core.schemas import TaskType
from core.stages.cluster.main import run_clustering_stage

def test_clustering_stage():
    config = {
        "input_filename": "regplans-train.jsonl",
    }
    
    try:
        output_path = run_clustering_stage(
            task=TaskType.NER,
            base_dir="tasks",
            config_dict=config,
            skip_embedding=False 
        )

        
    except Exception as e:
        print(f"Error during clustering: {e}")

if __name__ == "__main__":
    test_clustering_stage()

# I don't need a main orchestration file, but for each task,
# there should be a file that runs its stuff with the core stuff.

# TODO: Make this file later today
# Also need to find out the best structure for the tasks folders