import numpy as np
import hdbscan

class HDBSCANClusterer:
    def __init__(
        self, 
        min_cluster_size: int = 15,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.3,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

    def cluster(self, embeddings: np.ndarray):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon
        )
        
        labels = clusterer.fit_predict(embeddings)
        probabilities = clusterer.probabilities_

        return labels, probabilities