import hdbscan

class HDBSCANClusterer:
    def __init__(
        self, 
        min_cluster_size,
        min_samples,
        cluster_selection_epsilon,
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