from dataclasses import dataclass

@dataclass
class ClusterConfig:
    """Configuration for clustering stage."""
    random_seed: int = 42
    
    # Embedding generation parameters
    embedding_model: str = "text-embedding-3-small"
    
    # UMAP parameters
    umap_n_components: int = 30
    
    # HDBSCAN parameters
    min_cluster_size: int = 15
    min_samples: int = 1
    cluster_selection_epsilon: float = 0.3
    
    # File names
    input_filename: str = "input_dataset.jsonl"
    embedded_filename: str = "embedded_dataset.parquet"
    output_filename: str = "cluster_dataset.jsonl"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ClusterConfig':
        """Create config from dictionary, using defaults for missing keys."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})