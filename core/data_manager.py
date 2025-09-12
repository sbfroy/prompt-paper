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

# TODO: Should probably make use of wandb artifacts in the future

class DataManager:
    def __init__(self, task, base_dir):
        self.task = TaskType(task)

        # Creates the task data directory
        self.task_data_dir = Path(base_dir) / self.task.value / "data"
        self.task_data_dir.mkdir(parents=True, exist_ok=True)

        # Create all required subdirs
        for p in (
            self.get_input_dir(),
            self.get_output_dir(),
        ):
            p.mkdir(parents=True, exist_ok=True)

    def save_input_dataset(self, dataset: InputDataset, filename="input_dataset.jsonl"):
        """
        Save InputDataset as JSONL in the input directory.
        This is the standardized format that enters the pipeline.
        """
        filepath = self.get_input_dir() / filename

        with filepath.open("w", encoding="utf-8") as f:
            for example in dataset.examples:
                f.write(example.model_dump_json(by_alias=True) + "\n")

        return filepath
    
    def load_input_dataset(self, filename="input_dataset.jsonl"):
        """Load InputDataset from input directory."""
        filepath = self.get_input_dir() / filename
        examples: list[InputExample] = []
        
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(InputExample(**data))

        return InputDataset(examples=examples, task_type=self.task)

    def save_embedded_dataset(self, dataset, filename="embedded_dataset.parquet"):
        """Save embedded dataset as Parquet in output directory."""
        filepath = self.get_output_dir() / filename
        
        records = [example.model_dump() for example in dataset.examples]
        df = pd.DataFrame(records) 
        df.to_parquet(filepath, compression='snappy', index=False)

        return filepath
    

    def load_embedded_dataset(self, filename="embedded_dataset.parquet"):
        """Load embedded dataset from output directory."""
        filepath = self.get_output_dir() / filename

        df = pd.read_parquet(filepath)
        examples = [EmbeddedExample(**rec) for rec in df.to_dict(orient="records")]

        return EmbeddedDataset(examples=examples, task_type=self.task)

    def save_cluster_dataset(self, dataset, filename="cluster_dataset.jsonl"):
        """
        Save cluster dataset as JSONL in output directory.
        Preserves cluster structure for analysis.
        """
        filepath = self.get_output_dir() / filename

        with filepath.open("w", encoding="utf-8") as f:
            for cluster in dataset.clusters:
                f.write(cluster.model_dump_json(by_alias=True) + "\n")

        return filepath

    def load_cluster_dataset(self, filename="cluster_dataset.jsonl"):
        """Load cluster dataset from output directory."""
        filepath = self.get_output_dir() / filename
        clusters: list[Cluster] = []

        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                clusters.append(Cluster(**data))

        return ClusterDataset(clusters=clusters, task_type=self.task)

    def save_final_output(self, data, filename):
        """
        Save final pipeline results to output directory.
        Generic method for saving any final results as JSON.
        """
        filepath = self.get_output_dir() / filename
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def get_input_dir(self) -> Path:
        return self.task_data_dir / "input"
    
    def get_output_dir(self) -> Path:
        return self.task_data_dir / "output"