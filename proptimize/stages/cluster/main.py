import numpy as np
import warnings
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

from proptimize.data_manager import DataManager
from proptimize.schemas import Cluster, ClusterDataset, ClusterExample
from proptimize.stages.cluster.embedding_generator import EmbeddingGenerator
from proptimize.stages.cluster.dimensionality_reducer import DimensionalityReducer
from proptimize.stages.cluster.clusterer import HDBSCANClusterer
from proptimize.stages.cluster.config import ClusterConfig


class ClusterStage:
    def __init__(self, data_manager: DataManager, config: ClusterConfig):
        self.data_manager = data_manager
        self.config = config

        self.embedding_generator = EmbeddingGenerator()
        self.reducer = DimensionalityReducer(random_state=config.random_seed)
        self.clusterer = HDBSCANClusterer(
            min_cluster_size=config.min_cluster_size,
            min_samples=config.min_samples,
            cluster_selection_epsilon=config.cluster_selection_epsilon,
        )

    def run(self):
        logging.info("Starting clustering stage...")

        # ====== EMBEDDING ======
        if self.config.skip_embedding:
            logging.info("Loading existing embeddings...")
            embedded_dataset = self.data_manager.load_embedded_dataset()
        else:
            input_dataset = self.data_manager.load_input_dataset(
                self.config.input_filename
            )
            embedded_dataset = self.embedding_generator.generate_embeddings(
                input_dataset, self.config.batch_size
            )
            self.data_manager.save_embedded_dataset(embedded_dataset)

        # Embeddings to numpy array
        embeddings = np.array(
            [example.embedding for example in embedded_dataset.examples]
        )

        # ====== UMAP ======
        logging.info("Reducing dimensionality with UMAP...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="umap")
            reduced_embeddings = self.reducer.reduce(
                embeddings, n_components=self.config.umap_n_components
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
        artifact = self.data_manager.save_cluster_dataset(cluster_dataset)

        # TODO: Implement some details logging (stats about clusters)

        logging.info(
            f"Clustering stage completed! Output saved as artifact: {artifact.name}"
        )
        return artifact

    def _create_cluster_dataset(
        self, embedded_dataset, labels: np.ndarray, probabilities: np.ndarray
    ):
        # Group examples by cluster
        cluster_groups = defaultdict(list)

        for idx, (label, prob) in enumerate(zip(labels, probabilities)):
            example = embedded_dataset.examples[idx]

            cluster_example = ClusterExample(
                example_id=example.example_id,
                text=example.text,
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


def run_cluster_stage(
    task,
    base_dir,
    config_dict,
):
    # Setup
    data_manager = DataManager(task, base_dir)
    config = ClusterConfig.from_dict(config_dict or {})

    # Run clustering
    stage = ClusterStage(data_manager, config)
    return stage.run()
