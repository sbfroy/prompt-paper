import numpy as np
from pathlib import Path

from ...data_manager import DataManager
from ...schemas import TaskType, Cluster, ClusterDataset, ClusterExample
from .embedding_generator import EmbeddingGenerator
from .dimensionality_reducer import DimensionalityReducer
from .clusterer import HDBSCANClusterer
from .config import ClusterConfig

class ClusterStage:
    def __init__(self, data_manager: DataManager, config: ClusterConfig):
        self.data_manager = data_manager
        self.config = config

        self.embedding_generator = EmbeddingGenerator(
            model=config.embedding_model
        )
        self.reducer = DimensionalityReducer(
            random_state=config.random_seed
        )
        self.clusterer = HDBSCANClusterer(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_epsilon=config.cluster_selection_epsilon
        )

    def run(self, skip_embedding: bool = False):
        print("Starting clustering stage...")
        
        if skip_embedding:
            print("Loading existing embeddings...")
            embedded_dataset = self.data_manager.load_embedded_dataset(self.config.embedded_filename)
        else:
            input_dataset = self.data_manager.load_input_dataset(self.config.input_filename)
            embedded_dataset = self.embedding_generator.generate_embeddings(input_dataset)
            self.data_manager.save_embedded_dataset(embedded_dataset, self.config.embedded_filename)

        # Embeddings to numpy array
        embeddings = np.array([example.embedding for example in embedded_dataset.examples])

        print("Reducing dimensionality with UMAP...")
        reduced_embeddings = self.reducer.reduce(
            embeddings,
            n_components=self.config.umap_n_components
        )

        print("Clustering with HDBSCAN...")
        labels, probabilities = self.clusterer.cluster(reduced_embeddings)

        # Convert to schema format
        cluster_dataset = self._create_cluster_dataset(
            embedded_dataset,
            labels,
            probabilities
        )
        output_path = self.data_manager.save_cluster_dataset(
            cluster_dataset, self.config.output_filename
        )

        print(f"Clustering stage completed! Output saved to: {output_path}")
        return output_path

    def _create_cluster_dataset(
        self,
        embedded_dataset,
        labels: np.ndarray,
        probabilities: np.ndarray
    ):
        from collections import defaultdict

        # Group examples by cluster
        cluster_groups = defaultdict(list)

        for idx, (label, prob) in enumerate(zip(labels, probabilities)):
            example = embedded_dataset.examples[idx]
            
            cluster_example = ClusterExample(
                example_id=example.example_id,
                text=example.text,
                membership_probability=float(prob)
            )
            
            cluster_groups[int(label)].append(cluster_example)
        
        # Create Cluster objects
        clusters = []
        for cluster_id, examples in cluster_groups.items():
            cluster = Cluster(
                cluster_id=cluster_id,
                examples=examples,
            )
            clusters.append(cluster)
        
        return ClusterDataset(
            clusters=clusters,
            task_type=embedded_dataset.task_type
        )
    
def run_clustering_stage(
    task: TaskType,
    base_dir: str,
    config_dict: dict,
    skip_embedding: bool = False
):
    # Setup
    data_manager = DataManager(task, base_dir)
    config = ClusterConfig.from_dict(config_dict or {})

    # Run clustering
    stage = ClusterStage(data_manager, config)
    return stage.run(skip_embedding=skip_embedding)
