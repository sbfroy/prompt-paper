from deap import base, creator, tools, algorithms
import numpy as np
import random
from dataclasses import dataclass

from .client import get_total_cost, get_total_tokens, print_generation_cost_summary
from ...wandb_utils import log_metrics

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

        # Wrap evaluate since toolbox expects a function(pop_member) -> fitness tuple
        def _eval(individual):
            return self.evaluate_fn(individual)

        self.toolbox.register("evaluate", _eval)
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
        
        # sets up the statistics with simple cost printing
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean); stats.register("std", np.std)
        stats.register("min", np.min); stats.register("max", np.max)

        # Custom stats for cost tracking
        stats.register("gen_cost",   lambda _: 0.0)
        stats.register("total_cost", lambda _: 0.0)

        # Simple cost printing after each generation
        original_compile = stats.compile
        _last_total = [get_total_cost()]

        _gen = [-1] # generation counter

        def compile_with_cost(population):
            _gen[0] += 1
            generation = _gen[0]

            rec = original_compile(population)  # avg/std/min/max stay numeric

            total = float(get_total_cost())
            gen_cost = max(0.0, total - _last_total[0])
            _last_total[0] = total

            # Right-align numbers to fixed width; headers already padded
            rec["gen_cost"] = f"{gen_cost:.3f} NOK"
            rec["total_cost"] = f"{total:.3f} NOK"

            # Numeric best fitness in current population
            try:
                best_fitness = max(ind.fitness.values[0] for ind in population)
            except Exception:
                best_fitness = None

            # Log numeric metrics to wandb (skip if no run active)
            try:
                log_metrics(
                    step=generation,
                    avg=rec.get('avg'),
                    max=rec.get('max'),
                    min=rec.get('min'),
                    std=rec.get('std'),
                    nevals=rec.get('nevals'),
                    gen_cost_nok=gen_cost,
                    total_cost_nok=total,
                    best_fitness=best_fitness
                )
            except Exception:
                # Silent fail to avoid interfering with evolution if wandb not initialized
                pass

            return rec

        stats.compile = compile_with_cost

        pop = self.toolbox.population(n=self.config.pop_size) # creates the init population
        
        import sys, shutil
        width = shutil.get_terminal_size().columns
        print("-" * width)
     
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox,
            cxpb=self.config.cxpb, 
            mutpb=self.config.mutpb,
            ngen=self.config.generations, 
            stats=stats,
            verbose=True  
        )
        
        print("-" * width)

        return pop, logbook # final population, stats over generations