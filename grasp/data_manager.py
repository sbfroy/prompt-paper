"""Data management module for handling dataset I/O operations and artifact storage."""

from pathlib import Path
import pandas as pd
import json
import tempfile

from .schemas import (
    InputExample,
    InputDataset,
    EmbeddedExample,
    EmbeddedDataset,
    Cluster,
    ClusterDataset,
)
from .wandb_utils import save_artifact, save_file_artifact, load_artifact


class DataManager:
    """Manages dataset storage, loading, and artifact handling for pipeline stages."""

    def __init__(self, task, base_dir):
        """Initialize data manager with task-specific directory structure.
        
        Args:
            task: Name of the task (e.g., 'ner', 'financial_ner')
            base_dir: Base directory for data storage
        """
        self.task = task

        # Create task-specific directory structure
        self.task_data_dir = Path(base_dir) / self.task / "data"
        self.task_data_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset directory (outputs go to wandb)
        self.get_dataset_dir().mkdir(parents=True, exist_ok=True)

    def save_input_dataset(self, dataset, artifact_name_suffix):
        """Save InputDataset as wandb artifact.
        
        Args:
            dataset: InputDataset object to save
            artifact_name_suffix: Suffix for artifact name (e.g., 'train', 'val')
            
        Returns:
            Wandb artifact
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / f"{artifact_name_suffix}.jsonl"

            with temp_path.open("w", encoding="utf-8") as f:
                for example in dataset.examples:
                    f.write(example.model_dump_json(by_alias=True) + "\n")

            artifact_name = f"{self.task}_input_dataset_{artifact_name_suffix}"
            return save_file_artifact(temp_path, artifact_name, "input_dataset")

    def load_input_dataset(self, artifact_name_suffix):
        """Load InputDataset from wandb artifact.
        
        Args:
            artifact_name_suffix: Suffix for artifact name (e.g., 'train', 'val')
            
        Returns:
            InputDataset object
        """
        # Remove .jsonl extension if present to get the suffix
        artifact_suffix = artifact_name_suffix.replace('.jsonl', '')
        artifact_name = f"{self.task}_input_dataset_{artifact_suffix}"
        artifact_path = load_artifact(artifact_name)
        
        examples: list[InputExample] = []

        with open(artifact_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(InputExample(**data))

        return InputDataset(examples=examples, task_type=self.task)

    def save_embedded_dataset(self, dataset):
        """Save embedded dataset as wandb artifact.
        
        Args:
            dataset: EmbeddedDataset object to save
            
        Returns:
            Wandb artifact
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "embedded_dataset.parquet"

            # Convert pydantic models to dict records for pandas
            records = [example.model_dump() for example in dataset.examples]
            df = pd.DataFrame(records)
            df.to_parquet(temp_path, compression='snappy', index=False)

            artifact_name = f"{self.task}_embedded_dataset"
            return save_file_artifact(temp_path, artifact_name, "embedded_dataset")

    def load_embedded_dataset(self):
        """Load embedded dataset from wandb artifact.
        
        Returns:
            EmbeddedDataset object
        """
        artifact_name = f"{self.task}_embedded_dataset"
        file_path = load_artifact(artifact_name)

        # Read parquet and convert to pydantic models
        df = pd.read_parquet(file_path)
        examples = [EmbeddedExample(**rec) for rec in df.to_dict(orient="records")]

        return EmbeddedDataset(examples=examples, task_type=self.task)

    def save_cluster_dataset(self, dataset):
        """Save cluster dataset as wandb artifact.
        
        Args:
            dataset: ClusterDataset object to save
            
        Returns:
            Wandb artifact
        """
        # Convert cluster objects to JSON-serializable format
        cluster_data = [cluster.model_dump(by_alias=True) for cluster in dataset.clusters]

        artifact_name = f"{self.task}_cluster_dataset"
        return save_artifact(cluster_data, artifact_name, "cluster_dataset")

    def load_cluster_dataset(self):
        """Load cluster dataset from wandb artifact.
        
        Returns:
            ClusterDataset object
        """
        artifact_name = f"{self.task}_cluster_dataset"
        file_path = load_artifact(artifact_name)

        # Load JSON data and reconstruct cluster objects
        with open(file_path, 'r') as f:
            cluster_data = json.load(f)

        clusters = [Cluster(**data) for data in cluster_data]

        return ClusterDataset(clusters=clusters, task_type=self.task)

    def save_results(self, data):
        """Save final pipeline results as wandb artifact.
        
        Args:
            data: Results data to save
            
        Returns:
            Wandb artifact
        """
        artifact_name = f"{self.task}_results"
        return save_artifact(data, artifact_name, "GA_results")

    def save_artifact(self, data, artifact_name, artifact_type):
        """Generic method to save data as wandb artifact.
        
        Args:
            data: Data to save
            artifact_name: Name for the artifact
            artifact_type: Type of artifact
            
        Returns:
            Wandb artifact
        """
        full_artifact_name = f"{self.task}_{artifact_name}"
        return save_artifact(data, full_artifact_name, artifact_type)

    def get_dataset_dir(self) -> Path:
        """Get path to dataset directory."""
        return self.task_data_dir / "dataset"

    def get_scope_dir(self) -> Path:
        """Get path to scope directory."""
        return self.task_data_dir / "scope"