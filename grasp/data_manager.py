"""Data management module for handling dataset I/O operations and artifact storage."""

from pathlib import Path
import pandas as pd
import json
import tempfile
import random
import logging

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

    def save_input_dataset(self, dataset, artifact_name_suffix, dataset_size):
        """Save InputDataset as wandb artifact.
        
        Args:
            dataset: InputDataset object to save
            artifact_name_suffix: Suffix for artifact name (e.g., 'train', 'val')
            dataset_size: Size to include in artifact name (e.g., 3000)
            
        Returns:
            Wandb artifact
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / f"{artifact_name_suffix}.jsonl"

            with temp_path.open("w", encoding="utf-8") as f:
                for example in dataset.examples:
                    f.write(example.model_dump_json(by_alias=True) + "\n")

            artifact_name = f"{self.task}_input_dataset_{dataset_size}_{artifact_name_suffix}"
            return save_file_artifact(temp_path, artifact_name, "input_dataset")

    def load_input_dataset(self, artifact_name_suffix, dataset_size):
        """Load InputDataset from wandb artifact.
        
        Args:
            artifact_name_suffix: Suffix for artifact name (e.g., 'train', 'val')
            dataset_size: Size included in artifact name (e.g., 3000)
            
        Returns:
            InputDataset object
        """
        # Remove .jsonl extension if present to get the suffix
        artifact_suffix = artifact_name_suffix.replace('.jsonl', '')
        
        artifact_name = f"{self.task}_input_dataset_{dataset_size}_{artifact_suffix}"
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

    def save_embedded_dataset(self, dataset, dataset_size):
        """Save embedded dataset as wandb artifact.
        
        Args:
            dataset: EmbeddedDataset object to save
            dataset_size: Size to include in artifact name (e.g., 3000)
            
        Returns:
            Wandb artifact
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "embedded_dataset.parquet"

            # Convert pydantic models to dict records for pandas
            records = [example.model_dump() for example in dataset.examples]
            df = pd.DataFrame(records)
            df.to_parquet(temp_path, compression='snappy', index=False)

            artifact_name = f"{self.task}_embedded_dataset_{dataset_size}"
            return save_file_artifact(temp_path, artifact_name, "embedded_dataset")

    def load_embedded_dataset(self, dataset_size):
        """Load embedded dataset from wandb artifact.
        
        Args:
            dataset_size: Size included in artifact name
        
        Returns:
            EmbeddedDataset object
        """
        artifact_name = f"{self.task}_embedded_dataset_{dataset_size}"
        file_path = load_artifact(artifact_name)

        # Read parquet and convert to pydantic models
        df = pd.read_parquet(file_path)
        examples = [EmbeddedExample(**rec) for rec in df.to_dict(orient="records")]

        return EmbeddedDataset(examples=examples, task_type=self.task)

    def save_cluster_dataset(self, dataset, dataset_size):
        """Save cluster dataset as wandb artifact.
        
        Args:
            dataset: ClusterDataset object to save
            dataset_size: Size to include in artifact name (e.g., 3000)
            
        Returns:
            Wandb artifact
        """
        # Convert cluster objects to JSON-serializable format
        cluster_data = [cluster.model_dump(by_alias=True) for cluster in dataset.clusters]

        artifact_name = f"{self.task}_cluster_dataset_{dataset_size}"
        return save_artifact(cluster_data, artifact_name, "cluster_dataset")

    def load_cluster_dataset(self, dataset_size):
        """Load cluster dataset from wandb artifact.
        
        If dataset_size differs from the original clustered dataset size, this will
        load the original full cluster dataset and sample examples using a round-robin
        strategy (starting from the largest cluster, moving to smallest).
        
        Args:
            dataset_size: Size included in artifact name (e.g., 3000)
        
        Returns:
            ClusterDataset object (sampled if dataset_size differs from original)
        """
        
        # Try to load the exact artifact first
        artifact_name = f"{self.task}_cluster_dataset_{dataset_size}"
        try:
            file_path = load_artifact(artifact_name)
            
            # Load JSON data and reconstruct cluster objects
            with open(file_path, 'r') as f:
                cluster_data = json.load(f)

            clusters = [Cluster(**data) for data in cluster_data]
            return ClusterDataset(clusters=clusters, task_type=self.task)
            
        except Exception as e:
            # If artifact doesn't exist, try to load the original (10000) and sample
            logging.info(f"Artifact '{artifact_name}' not found. Attempting to load original cluster dataset and sample...")

            # Try loading the original dataset (assuming 10000 is the default size)
            original_artifact_name = f"{self.task}_cluster_dataset_10000"
            try:
                file_path = load_artifact(original_artifact_name)
            except Exception as original_e:
                # If neither exists, raise the original error
                raise Exception(f"Could not find artifact '{artifact_name}' or original '{original_artifact_name}': {e}")
            
            # Load full dataset
            with open(file_path, 'r') as f:
                cluster_data = json.load(f)
            
            full_clusters = [Cluster(**data) for data in cluster_data]
            
            # Filter out noise cluster (-1)
            full_clusters = [c for c in full_clusters if c.cluster_id != -1]
            
            # Calculate total examples in full dataset
            total_examples = sum(len(c.examples) for c in full_clusters)
            logging.info(f"Original cluster dataset has {len(full_clusters)} clusters with {total_examples} total examples")
            
            # Validate that we have enough clusters and examples
            if dataset_size < len(full_clusters):
                raise ValueError(
                    f"Requested dataset_size ({dataset_size}) must be >= number of clusters ({len(full_clusters)}). "
                    "All clusters must be represented with at least one example each."
                )
            
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(full_clusters, key=lambda c: len(c.examples), reverse=True)
            
            # Create sampled clusters using round-robin
            logging.info(f"Sampling {dataset_size} examples using round-robin strategy...")
            
            sampled_clusters_map = {c.cluster_id: [] for c in sorted_clusters}
            examples_sampled = 0
            
            # Keep track of which examples we've already sampled from each cluster
            cluster_example_pools = {
                c.cluster_id: list(c.examples) for c in sorted_clusters
            }
            
            # Shuffle each cluster's examples for random selection
            for cluster_id in cluster_example_pools:
                random.shuffle(cluster_example_pools[cluster_id])
            
            # Track which clusters are still available
            available_cluster_indices = list(range(len(sorted_clusters)))
            current_position = 0  # Track position in the round-robin cycle
            
            # Round-robin sampling
            while examples_sampled < dataset_size and available_cluster_indices:
                # Get the actual cluster index from available clusters
                actual_cluster_index = available_cluster_indices[current_position]
                
                current_cluster = sorted_clusters[actual_cluster_index]
                cluster_id = current_cluster.cluster_id
                
                # Get one example from this cluster's pool
                if cluster_example_pools[cluster_id]:
                    example = cluster_example_pools[cluster_id].pop(0)
                    sampled_clusters_map[cluster_id].append(example)
                    examples_sampled += 1
                    
                    # If this cluster is now exhausted, remove it from available clusters
                    if not cluster_example_pools[cluster_id]:
                        logging.info(f"Cluster {cluster_id} exhausted after contributing {len(sampled_clusters_map[cluster_id])} examples")
                        available_cluster_indices.remove(actual_cluster_index)
                        # After removal, keep position at same index (which now points to next cluster)
                        # unless we're past the end of the list
                        if available_cluster_indices:
                            if current_position >= len(available_cluster_indices):
                                current_position = 0
                            # else: keep current_position as is, it now points to the next cluster
                    else:
                        # Move to next cluster in round-robin
                        current_position = (current_position + 1) % len(available_cluster_indices)
                else:
                    # This shouldn't happen, but handle it gracefully
                    logging.warning(f"Cluster {cluster_id} pool empty but still in available list, removing...")
                    available_cluster_indices.remove(actual_cluster_index)
                    if available_cluster_indices:
                        if current_position >= len(available_cluster_indices):
                            current_position = 0
            
            # Reconstruct clusters with sampled examples
            sampled_clusters = []
            for cluster in sorted_clusters:
                if sampled_clusters_map[cluster.cluster_id]:
                    sampled_clusters.append(
                        Cluster(
                            cluster_id=cluster.cluster_id,
                            examples=sampled_clusters_map[cluster.cluster_id]
                        )
                    )
            
            logging.info(f"Sampled {examples_sampled} examples across {len(sampled_clusters)} clusters")
            for cluster in sampled_clusters:
                logging.info(f"  Cluster {cluster.cluster_id}: {len(cluster.examples)} examples")
            
            return ClusterDataset(clusters=sampled_clusters, task_type=self.task)

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