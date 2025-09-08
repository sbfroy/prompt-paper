from deap import base, creator, tools, algorithms
import numpy as np
import random
from dataclasses import dataclass

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
    def __init__(self, cluster_dataset, evaluate_fn, mate_fn, mutate_fn, select_fn, config: GAConfig):
        self.cluster_dataset = cluster_dataset  
        self.evaluate_fn = evaluate_fn 
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
            "attr_sample", # selects one example from each cluster
            self._create_individual
        )
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_sample) # builds one individual
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual) # repeats individual to make the whole population

        self.toolbox.register("evaluate", self.evaluate_fn)
        self.toolbox.register("mate", self.mate_fn)
        self.toolbox.register("mutate", self.mutate_fn, cluster_dataset=self.cluster_dataset)
        self.toolbox.register("select", self.select_fn, tournsize=self.config.tournsize)

    def _create_individual(self):
        """
        Creates an individual by selecting one random example from different clusters.

        Returns:
            List of (cluster_id, example) pairs 
        """
        # Filter out noise cluster
        valid_clusters = [cluster for cluster in self.cluster_dataset.clusters if cluster.cluster_id != -1]
        random_clusters = random.sample(valid_clusters, self.config.subset_size)

        examples = []
        for cluster in random_clusters:
            # Select one random example from each randomly selected cluster
            random_example = random.choice(cluster.examples)
            examples.append((cluster.cluster_id, random_example))

        return examples

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