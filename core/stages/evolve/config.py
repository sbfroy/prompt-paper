from dataclasses import dataclass

@dataclass
class EvolveConfig:
    random_seed: int = 42
    
    # GA params
    subset_size: int = 5  # Number of ICL examples
    pop_size: int = 50   
    generations: int = 20 
    cxpb: float = 0.6     
    mutpb: float = 0.2    
    tournsize: int = 3  
    
    # Mutation params
    indpb: float = 0.2  # Probability for each example to be mutated
    inter_prob: float = 0.4  # Probability of inter-cluster mutation
    intra_prob: float = 0.4  # Probability of intra-cluster mutation

    # Client params
    openai_model: str = "gpt-4o-mini"
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
