import numpy as np
import umap

class DimensionalityReducer:
    def __init__(self, random_state):
        self.random_state = random_state
        
    def reduce(
        self,
        embeddings,
        n_components,
        **kwargs
    ):

        reducer = umap.UMAP(
            n_components=n_components, 
            random_state=self.random_state,
            **kwargs
        )

        return reducer.fit_transform(embeddings)