import hdbscan

class HDBSCANClusterer:
    def __init__(
        self, 
        min_cluster_size: int = 15,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.3,
        **kwargs
    ):

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            **kwargs
        )

    def cluster(self, embeddings):
        """
        Cluster embeddings using HDBSCAN.

        Returns:
            labels: Cluster labels
            probabilities: Probability for belonging to assigned cluster
        """
        labels = self.clusterer.fit_predict(embeddings)
        probabilities = self.clusterer.probabilities_
        return labels, probabilities