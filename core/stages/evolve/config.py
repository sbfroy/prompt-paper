from dataclasses import dataclass

@dataclass
class EvolveConfig:
    random_seed: int = 42
    
    # GA parameters
    subset_size: int = 5  # Number of examples to select
    pop_size: int = 50    # Population size
    generations: int = 20 # Number of generations
    cxpb: float = 0.7     # Crossover probability
    mutpb: float = 0.2    # Mutation probability
    tournsize: int = 3    # Tournament size for selection
    
    # Mutation parameters
    indpb: float = 0.1  # Independent probability for each example to be mutated
    inter_prob: float = 0.4  # Probability of inter-cluster mutation
    intra_prob: float = 0.4  # Probability of intra-cluster mutation

    # Client parameters
    openai_model: str = "gpt-4o-mini"

    # File names
    input_filename: str = "cluster_dataset.jsonl"
    output_filename: str = "evolved_dataset.jsonl"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'EvolveConfig':
        """Create config from dictionary, using defaults for missing keys."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
