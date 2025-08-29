import numpy as np
import umap

class DimensionalityReducer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def reduce(
        self,
        embeddings: np.ndarray,
        n_components: int = 30
    ):

        reducer = umap.UMAP(
            n_components=n_components, 
            random_state=self.random_state
            )

        return reducer.fit_transform(embeddings)