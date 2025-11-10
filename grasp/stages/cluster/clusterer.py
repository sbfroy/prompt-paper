"""HDBSCAN-based clustering for grouping similar embeddings."""

import hdbscan


class HDBSCANClusterer:
    """Wrapper for HDBSCAN clustering algorithm."""

    def __init__(self, **kwargs):
        """Initialize HDBSCAN clusterer with custom parameters.
        
        Args:
            **kwargs: HDBSCAN parameters (min_cluster_size, min_samples, etc.)
        """
        self.clusterer = hdbscan.HDBSCAN(**kwargs)

    def cluster(self, embeddings):
        """Cluster embeddings using HDBSCAN.
        
        Args:
            embeddings: Input embeddings to cluster (numpy array)
        
        Returns:
            tuple: (labels, probabilities)
                - labels: Cluster assignments for each embedding (-1 for noise)
                - probabilities: Membership confidence for assigned clusters
        """
        labels = self.clusterer.fit_predict(embeddings)
        probabilities = self.clusterer.probabilities_

        return labels, probabilities

