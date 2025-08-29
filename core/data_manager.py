from pathlib import Path
import pandas as pd
import json
from .schemas import (
    TaskType,
    InputExample,
    InputDataset,
    EmbeddedExample,
    EmbeddedDataset,
    Cluster,
    ClusterDataset,
)

class DataManager:
    def __init__(self, task: TaskType, base_dir: str):
        self.task = TaskType(task)

        # Creates the task data directory
        self.task_data_dir = Path(base_dir) / self.task.value / "data"
        self.task_data_dir.mkdir(parents=True, exist_ok=True)

        # Create all required subdirs
        for p in (
            self.get_scope_dir(),
            self.get_dataset_dir(),
            self.get_processed_dir(),
            self.get_output_dir(),
        ):
            p.mkdir(parents=True, exist_ok=True)

    def save_input_dataset(self, dataset: InputDataset, filename: str = "input_dataset.jsonl"):
        """
        Save InputDataset as JSONL in the processed directory.
        This is the standardized format that enters the pipeline.
        """
        filepath = self.get_processed_dir() / filename

        with filepath.open("w", encoding="utf-8") as f:
            for example in dataset.examples:
                f.write(example.json(ensure_ascii=False) + "\n")

        return filepath
    
    def load_input_dataset(self, filename: str = "input_dataset.jsonl"):
        """Load InputDataset from processed directory."""
        filepath = self.get_processed_dir() / filename
        examples: list[InputExample] = []
        
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(InputExample(**data))

        return InputDataset(examples=examples, task_type=self.task)

    def save_embedded_dataset(self, dataset: EmbeddedDataset, filename: str = "embedded_dataset.parquet"):
        """Save embedded dataset as Parquet in processed directory."""
        filepath = self.get_processed_dir() / filename
        
        records = [example.dict() for example in dataset.examples]
        df = pd.DataFrame(records) # To DF for efficient storage
        df.to_parquet(filepath, compression='snappy', index=False)

        return filepath
    

    def load_embedded_dataset(self, filename: str = "embedded_dataset.parquet"):
        """Load embedded dataset from processed directory."""
        filepath = self.get_processed_dir() / filename

        df = pd.read_parquet(filepath)
        examples = [EmbeddedExample(**rec) for rec in df.to_dict(orient="records")]

        return EmbeddedDataset(examples=examples, task_type=self.task)

    def save_cluster_dataset(self, dataset: ClusterDataset, filename: str = "cluster_dataset.jsonl"):
        """
        Save cluster dataset as JSONL in processed directory.
        Preserves cluster structure for analysis.
        """
        filepath = self.get_processed_dir() / filename

        with filepath.open("w", encoding="utf-8") as f:
            for cluster in dataset.clusters:
                f.write(cluster.json(ensure_ascii=False) + "\n")

        return filepath

    def load_cluster_dataset(self, filename: str = "cluster_dataset.jsonl"):
        """Load cluster dataset from processed directory."""
        filepath = self.get_processed_dir() / filename
        clusters: list[Cluster] = []

        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                clusters.append(Cluster(**data))

        return ClusterDataset(clusters=clusters, task_type=self.task)

    def save_final_output(self, data: dict, filename: str):
        """
        Save final pipeline results to output directory.
        Generic method for saving any final results as JSON.
        """
        filepath = self.get_output_dir() / filename
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def get_scope_dir(self) -> Path:
        return self.task_data_dir / "input" / "scope"
    
    def get_dataset_dir(self) -> Path:
        return self.task_data_dir / "input" / "dataset"
    
    def get_processed_dir(self) -> Path:
        return self.task_data_dir / "processed"
    
    def get_output_dir(self) -> Path:
        return self.task_data_dir / "output"