"""UMAP-based dimensionality reduction for embedding visualization and clustering."""

import numpy as np
import umap


class DimensionalityReducer:
    """Wrapper for UMAP dimensionality reduction."""

    def __init__(self, random_state):
        """Initialize the dimensionality reducer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state

    def reduce(self, embeddings, n_components, **kwargs):
        """Reduce embedding dimensionality using UMAP.
        
        Args:
            embeddings: High-dimensional embeddings to reduce
            n_components: Target number of dimensions
            **kwargs: Additional UMAP parameters
        
        Returns:
            Reduced embeddings with shape (n_samples, n_components)
        """
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=self.random_state,
            **kwargs
        )

        return reducer.fit_transform(embeddings)
