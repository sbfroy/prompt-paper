from dataclasses import dataclass

@dataclass
class ClusterConfig:
    random_seed: int = 42
    
    # Embedding generation params
    embedding_model: str = "NbAiLab/nb-sbert-base"
    batch_size: int = 64

    # UMAP params
    umap_n_components: int = 30

    # HDBSCAN params
    min_cluster_size: int = 15
    min_samples: int = 1
    cluster_selection_epsilon: float = 0.3
    
    input_filename: str = "input_dataset.jsonl"
    
    @classmethod
    def from_dict(cls, config_dict):
        # Creates config from dict, using defaults for missing keys.
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})