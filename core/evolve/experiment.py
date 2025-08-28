from deap import base, creator, tools, algorithms
import numpy as np
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
        self.evaluate_fn = evaluate_fn #TODO: Should take in a structured input (JSON) aka the llm should return a json
        self.mate_fn = mate_fn
        self.mutate_fn = mutate_fn
        self.select_fn = select_fn
        self.config = config
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # set to maximize the fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "attr_sample", # 'attr_sample' randomly picks indices from the pool
            lambda: random.sample(range(len(self.example_pool)), self.config.subset_size)
        )
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_sample) # builds one individual
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) # repeats individual to make the whole population

        def _evaluate():
            pass

        self.toolbox.register("evaluate", _evaluate)
        self.toolbox.register("mate", self.mate_fn)
        self.toolbox.register("mutate", self.mutate_fn)
        self.toolbox.register("select", self.select_fn, tournsize=self.config.tournsize)

    def run(self):
        # sets up the statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", np.mean); stats.register("std", np.std)
        stats.register("min", np.min); stats.register("max", np.max)

        pop = self.toolbox.population(n=self.config.pop_size) # creates the init population
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox,
            cxpb=self.config.cxpb, 
            mutpb=self.config.mutpb,
            ngen=self.config.generations, 
            stats=stats
        )

        return pop, logbook # final population, stats over generations