from deap import base, creator, tools, algorithms
import random

@dataclass
class GAConfig:
    subset_size: int
    pop_size: int
    generations : int
    cxpb: float
    mutpb: float
    tournsize: int
    seed: int

class GA:
    def __init__(self, example_pool, evaluate_fn, mate_fn, mutate_fn, select_fn, config: GAConfig):
        self.example_pool = example_pool
        self.evaluate_fn = evaluate_fn #TODO: Should take in a structured input (JSON)
        self.mate_fn = mate_fn
        self.mutate_fn = mutate_fn
        self.select_fn = select_fn
        self.config = config

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "attr_sample",
            lambda: random.sample(range(len(self.example_pool)), self.config.subset_size)
        )
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_sample)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        def _evaluate():
            pass

        self.toolbox.register("evaluate", _evaluate)
        self.toolbox.register("mate", self.mate_fn)
        self.toolbox.register("mutate", self.mutate_fn)
        self.toolbox.register("select", self.select_fn, tournsize=self.config.tournsize)

    def run(self):
        pass