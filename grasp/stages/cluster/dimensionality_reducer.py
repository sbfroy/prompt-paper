"""UMAP-based dimensionality reduction for embedding visualization and clustering."""

import umap


class DimensionalityReducer:
    """Wrapper for UMAP dimensionality reduction."""

    def __init__(self, random_state, **umap_params):
        """Initialize the dimensionality reducer.
        
        Args:
            random_state: Random seed for reproducibility
            **umap_params: UMAP parameters 
        """
        self.random_state = random_state
        self.umap_params = umap_params

    def reduce(self, embeddings):
        """Reduce embedding dimensionality using UMAP.
        
        Args:
            embeddings: High-dimensional embeddings to reduce
        
        Returns:
            Reduced embeddings with shape (n_samples, n_components)
        """
        reducer = umap.UMAP(
            random_state=self.random_state,
            **self.umap_params
        )

        return reducer.fit_transform(embeddings)
