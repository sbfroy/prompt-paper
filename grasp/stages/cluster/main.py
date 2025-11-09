"""Main orchestration for the clustering stage pipeline."""

import logging
import warnings
from collections import defaultdict

import numpy as np

from grasp.data_manager import DataManager
from grasp.schemas import Cluster, ClusterDataset, ClusterExample
from grasp.stages.cluster.embedding_generator import EmbeddingGenerator
from grasp.stages.cluster.dimensionality_reducer import DimensionalityReducer
from grasp.stages.cluster.clusterer import HDBSCANClusterer
from grasp.run_vllm import shutdown_embedding_server

logging.basicConfig(level=logging.INFO)


class ClusterStage:
    """Orchestrates embedding generation, dimensionality reduction, and clustering."""

    def __init__(self, data_manager: DataManager, config: dict):
        """Initialize the clustering stage.
        
        Args:
            data_manager: Handles dataset loading and saving
            config: Configuration dictionary with parameters for all components
        """
        self.data_manager = data_manager
        self.config = config
        
        # Get dataset size from config for artifact naming
        self.dataset_size = config["dataset_size"]

        self.embedding_generator = EmbeddingGenerator()
        self.reducer = DimensionalityReducer(random_state=config["random_seed"])

        # Pass all HDBSCAN parameters directly from config
        hdbscan_params = config["hdbscan"]
        self.clusterer = HDBSCANClusterer(**hdbscan_params)

    def run(self):
        """Execute the complete clustering pipeline.
        
        Steps:
            1. Generate embeddings for input dataset
            2. Reduce dimensionality with UMAP
            3. Cluster with HDBSCAN
            4. Save results as ClusterDataset
        
        Returns:
            Artifact containing the clustered dataset
        """
        logging.info("Starting clustering stage...")

        # ====== EMBEDDING ======
        logging.info("Generating embeddings...")
        # Load training dataset from wandb artifact
        input_dataset = self.data_manager.load_input_dataset("train", dataset_size=self.dataset_size)
        embedded_dataset = self.embedding_generator.generate_embeddings(
            input_dataset, batch_size=self.config["batch_size"]
        )

        # Save the embedded dataset with size in artifact name
        self.data_manager.save_embedded_dataset(embedded_dataset, dataset_size=self.dataset_size)

        shutdown_embedding_server()
        logging.info("Embedding server shut down...")

        # Convert embeddings to numpy array
        embeddings = np.array(
            [example.embedding for example in embedded_dataset.examples]
        )

        # ====== UMAP ======
        logging.info("Reducing dimensionality with UMAP...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="umap")
            reduced_embeddings = self.reducer.reduce(
                embeddings, n_components=self.config["umap_n_components"]
            )

        # ====== HDBSCAN ======
        logging.info("Clustering with HDBSCAN...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            labels, probabilities = self.clusterer.cluster(reduced_embeddings)

        # Convert to schema format
        cluster_dataset = self._create_cluster_dataset(
            embedded_dataset, labels, probabilities
        )
        artifact = self.data_manager.save_cluster_dataset(cluster_dataset, dataset_size=self.dataset_size)

        logging.info(
            f"Clustering stage completed! Output saved as artifact: {artifact.name}"
        )
        return artifact

    def _create_cluster_dataset(
        self, embedded_dataset, labels: np.ndarray, probabilities: np.ndarray
    ):
        """Create a ClusterDataset from embeddings and clustering results.

        Args:
            embedded_dataset: The dataset with embeddings
            labels: Cluster labels for each example (-1 for noise)
            probabilities: Membership probabilities for each example

        Returns:
            ClusterDataset containing all clusters and their examples
        """
        # Group examples by cluster
        cluster_groups = defaultdict(list)

        for idx, (label, prob) in enumerate(zip(labels, probabilities)):
            example = embedded_dataset.examples[idx]

            cluster_example = ClusterExample(
                id=example.id,
                input=example.input,
                output=example.output,
                membership_probability=float(prob),
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

        return ClusterDataset(clusters=clusters, task_type=embedded_dataset.task_type)


def run_cluster_stage(task, base_dir, config_dict):
    """Run the clustering stage for a given task.
    
    Args:
        task: Task name/identifier
        base_dir: Base directory for data storage
        config_dict: Configuration parameters for the clustering pipeline
    
    Returns:
        Artifact containing the clustered dataset
    """
    # Setup data manager
    data_manager = DataManager(task, base_dir)

    # Run clustering pipeline
    stage = ClusterStage(data_manager, config_dict)
    return stage.run()

