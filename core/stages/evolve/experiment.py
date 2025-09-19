from deap import base, creator, tools, algorithms
import numpy as np
import random
import sys, shutil
import logging

logging.basicConfig(level=logging.INFO)

from ...wandb_utils import log_metrics

class GA:
    def __init__(self, cluster_dataset, evaluate_fn, mate_fn, mutate_fn, select_fn, config):
        self.cluster_dataset = cluster_dataset  
        self.evaluate_fn = evaluate_fn 
        self.mate_fn = mate_fn
        self.mutate_fn = mutate_fn
        self.select_fn = select_fn
        self.config = config
        
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)

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
            score = self.evaluate_fn(individual)
            return (score,)  # Convert single score to tuple for DEAP

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
        random_clusters = random.sample(self.cluster_dataset.clusters, self.config.subset_size)

        examples = []
        for cluster in random_clusters:
            # Select one random example from each randomly selected cluster
            random_example = random.choice(cluster.examples)
            examples.append((cluster.cluster_id, random_example))

        return examples

    def run(self):
        
        # sets up the statistics
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean); stats.register("std", np.std)
        stats.register("min", np.min); stats.register("max", np.max)

        # Log metrics to wandb after each generation
        original_compile = stats.compile
        _gen = [-1] # gen counter

        def compile_with_logging(population):
            _gen[0] += 1
            generation = _gen[0]

            rec = original_compile(population)

            # Log numeric metrics to wandb
            log_metrics(
                step=generation,
                avg=rec.get('avg'),
                max=rec.get('max'),
                min=rec.get('min'),
                std=rec.get('std'),
                nevals=rec.get('nevals')
            )
            return rec

        stats.compile = compile_with_logging

        mu = self.config.mu # parents
        lambda_ = self.config.lambda_ # offspring

        pop = self.toolbox.population(n=mu) # creates the init population
        hof = tools.HallOfFame(self.config.hof_size)  # Keep the best individuals
        
        width = shutil.get_terminal_size().columns
        logging.info("-" * width)

        pop, logbook = algorithms.eaMuPlusLambda(
            pop, self.toolbox,
            mu=mu,
            lambda_=lambda_,
            cxpb=self.config.cxpb, 
            mutpb=self.config.mutpb,
            ngen=self.config.generations, 
            stats=stats,
            halloffame=hof
        )
        
        logging.info("-" * width)

        return pop, logbook, hof # final population, stats over generations, hall of fame