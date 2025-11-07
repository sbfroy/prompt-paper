import hdbscan


class HDBSCANClusterer:
    def __init__(self, **kwargs):
        self.clusterer = hdbscan.HDBSCAN(**kwargs)

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
