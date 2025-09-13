from pathlib import Path
import pandas as pd
import json
import tempfile
from .schemas import (
    TaskType,
    InputExample,
    InputDataset,
    EmbeddedExample,
    EmbeddedDataset,
    Cluster,
    ClusterDataset,
)
from .wandb_utils import save_artifact, save_file_artifact, load_artifact

class DataManager:
    def __init__(self, task, base_dir):
        self.task = TaskType(task)

        # Create task-specific directory structure
        self.task_data_dir = Path(base_dir) / self.task.value / "data"
        self.task_data_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset dir only (output goes to wandb)
        self.get_dataset_dir().mkdir(parents=True, exist_ok=True)

    def save_input_dataset(self, dataset, file_name="input_dataset.jsonl"):
        """
        Save InputDataset as JSONL in the dataset directory.
        This is the standardized format that enters the pipeline.
        """
        filepath = self.get_dataset_dir() / file_name

        with filepath.open("w", encoding="utf-8") as f:
            for example in dataset.examples:
                f.write(example.model_dump_json(by_alias=True) + "\n")

        return filepath
    
    def load_input_dataset(self, file_name="input_dataset.jsonl"):
        """Load InputDataset from dataset directory."""
        filepath = self.get_dataset_dir() / file_name
        examples: list[InputExample] = []
        
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(InputExample(**data))

        return InputDataset(examples=examples, task_type=self.task)

    def save_embedded_dataset(self, dataset):
        """Save embedded dataset as wandb artifact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "embedded_dataset.parquet"
            
            # Convert pydantic models to dict records for pandas
            records = [example.model_dump() for example in dataset.examples]
            df = pd.DataFrame(records) 
            df.to_parquet(temp_path, compression='snappy', index=False)
            
            artifact_name = f"{self.task.value}_embedded_dataset"
            return save_file_artifact(temp_path, artifact_name, "embedded_dataset")

    def load_embedded_dataset(self):
        """Load embedded dataset from wandb artifact."""
        artifact_name = f"{self.task.value}_embedded_dataset"
        file_path = load_artifact(artifact_name)

        # Read parquet back into df and convert to pydantic models
        df = pd.read_parquet(file_path)
        examples = [EmbeddedExample(**rec) for rec in df.to_dict(orient="records")]

        return EmbeddedDataset(examples=examples, task_type=self.task)

    def save_cluster_dataset(self, dataset):
        """Save cluster dataset as wandb artifact."""
        # Convert cluster objects to JSON-serializable format
        cluster_data = [cluster.model_dump(by_alias=True) for cluster in dataset.clusters]
        
        artifact_name = f"{self.task.value}_cluster_dataset"
        return save_artifact(cluster_data, artifact_name, "cluster_dataset")

    def load_cluster_dataset(self):
        """Load cluster dataset from wandb artifact."""
        artifact_name = f"{self.task.value}_cluster_dataset"
        file_path = load_artifact(artifact_name)
        
        # Load JSON data and reconstruct cluster objects
        with open(file_path, 'r') as f:
            cluster_data = json.load(f)
        
        clusters = [Cluster(**data) for data in cluster_data]
        
        return ClusterDataset(clusters=clusters, task_type=self.task)

    def save_results(self, data):
        """
        Save final pipeline results as wandb artifact.
        
        """
        artifact_name = f"{self.task.value}_results"
        return save_artifact(data, artifact_name, "GA_results")

    def get_dataset_dir(self) -> Path:
        return self.task_data_dir / "dataset"

    # TODO: Make the scope folder function