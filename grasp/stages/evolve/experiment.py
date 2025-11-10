"""Genetic algorithm implementation using DEAP framework."""

import logging
import random
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from deap import algorithms, base, creator, tools

from grasp.schemas import ClusterDataset
from grasp.wandb_utils import log_metrics

logging.basicConfig(level=logging.INFO)


class EarlyStoppingException(Exception):
    """Exception raised when early stopping is triggered."""

    pass


class GA:
    """
    Genetic Algorithm for optimizing example selection from clustered datasets.

    Uses DEAP library to evolve populations of individuals (example subsets)
    toward higher fitness scores. Supports adaptive mutation, early stopping,
    and parallel evaluation.
    """

    def __init__(
        self,
        cluster_dataset: ClusterDataset,
        evaluate_fn: callable,
        mate_fn: callable,
        mutate_fn: callable,
        select_fn: callable,
        config: dict,
    ):
        self.cluster_dataset = cluster_dataset
        self.evaluate_fn = evaluate_fn
        self.mate_fn = mate_fn
        self.mutate_fn = mutate_fn
        self.select_fn = select_fn
        self.config = config
        self._eval_calls_total = 0
        self._eval_lock = threading.Lock()
        self.evolution_trace_callback = None  # Callback to log evolution trace

        # Set random seeds for reproducibility
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

        # Define fitness and individual types
        creator.create(
            "FitnessMax", base.Fitness, weights=(1.0,)
        )  # Maximize fitness
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Register genetic operators and population initialization
        self.toolbox = base.Toolbox()
        self.toolbox.register(
            "attr_sample", self._create_individual  # Select one example per cluster
        )
        self.toolbox.register(
            "individual", tools.initIterate, creator.Individual, self.toolbox.attr_sample
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Wrap evaluate function to track calls and return DEAP-compatible tuple
        def _eval(individual):
            score = self.evaluate_fn(individual)
            with self._eval_lock:  # Thread-safe counter update
                self._eval_calls_total += 1
            return (score,)  # DEAP expects tuple

        self.toolbox.register("evaluate", _eval)
        self.toolbox.register("map", self._parallel_map)
        self.toolbox.register("mate", self.mate_fn)
        self.toolbox.register(
            "mutate", self.mutate_fn, cluster_dataset=self.cluster_dataset
        )
        self.toolbox.register(
            "select", self.select_fn, tournsize=self.config["tournsize"]
        )

        # Early stopping state
        self.early_stopping_counter = 0
        self.best_max_value = None

        # Store total number of clusters for diversity calculation
        self.total_clusters = len(cluster_dataset.clusters)

    def calculate_adaptive_inter_prob(self, population):
        """
        Calculate adaptive inter-cluster mutation probability based on population diversity.

        High diversity (exploration phase) -> higher inter_prob to explore clusters
        Low diversity (exploitation phase) -> lower inter_prob to refine within clusters

        Args:
            population: Current population of individuals.

        Returns:
            Tuple of (adaptive_inter_prob, diversity) both floats in [0, 1].
        """
        from .operators import calculate_cluster_diversity

        # Calculate current diversity (0 to 1)
        diversity = calculate_cluster_diversity(population, self.total_clusters)

        # Get bounds from config
        initial_inter_prob = self.config["max_inter_prob"]
        min_inter_prob = self.config["min_inter_prob"]

        # Scale inter_prob linearly with diversity
        # High diversity -> inter_prob close to max (explore)
        # Low diversity -> inter_prob close to min (exploit)
        adaptive_inter_prob = (
            min_inter_prob + (initial_inter_prob - min_inter_prob) * diversity
        )

        return adaptive_inter_prob, diversity

    def _create_individual(self):
        """
        Create an individual by selecting random examples from clusters.

        Samples with replacement to allow multiple examples from same cluster
        if beneficial for fitness.

        Returns:
            List of (cluster_id, example) pairs.
        """
        examples = []
        for _ in range(self.config["subset_size"]):
            # Select random cluster and example
            random_cluster = random.choice(self.cluster_dataset.clusters)
            random_example = random.choice(random_cluster.examples)
            examples.append((random_cluster.cluster_id, random_example))

        return examples

    def _parallel_map(self, func, iterable):
        """
        Parallel mapping function for population evaluation.

        Uses ThreadPoolExecutor to evaluate multiple individuals concurrently.
        """
        with ThreadPoolExecutor(max_workers=self.config["workers"]) as executor:
            results = list(executor.map(func, iterable))
        return results

    def _should_early_stop(self, current_max_value):
        """
        Check if early stopping should be triggered.

        Monitors improvement in best fitness and triggers stop if no significant
        improvement occurs for a specified number of generations.

        Args:
            current_max_value: Best fitness in current generation.

        Returns:
            True if early stopping should be triggered, False otherwise.
        """
        if self.best_max_value is None:
            self.best_max_value = current_max_value
            return False

        # Check if there's improvement
        improvement = current_max_value - self.best_max_value

        if improvement > self.config["early_stopping_min_delta"]:
            # Significant improvement found, reset counter
            self.early_stopping_counter = 0
            self.best_max_value = current_max_value
            return False
        else:
            # No improvement, increment counter
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= self.config["early_stopping_patience"]

    def run(self):
        """
        Execute the genetic algorithm.

        Returns:
            Tuple of (final_population, logbook, hall_of_fame).
        """
        # Set up statistics tracking
        stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Wrap stats compilation to add logging and adaptive behavior
        original_compile = stats.compile
        _gen = [-1]  # Generation counter
        prev_eval_calls = [0]  # Track evaluations per generation

        # Store state for early stopping recovery
        logbook_entries = []
        current_population = [None]

        def compile_with_logging(population):
            """Enhanced compile function with logging and adaptive mutation."""
            _gen[0] += 1
            generation = _gen[0]

            # Calculate evaluations since last generation
            nevals = self._eval_calls_total - prev_eval_calls[0]
            prev_eval_calls[0] = self._eval_calls_total

            # Compile statistics
            rec = original_compile(population)
            logbook_entries.append(rec)
            current_population[0] = population[:]

            # Call evolution trace callback if set
            if self.evolution_trace_callback:
                self.evolution_trace_callback(generation, population)

            # Calculate adaptive inter_prob and update mutation function
            adaptive_inter_prob, diversity = self.calculate_adaptive_inter_prob(
                population
            )
            self.toolbox.register(
                "mutate",
                self.mutate_fn,
                cluster_dataset=self.cluster_dataset,
                inter_prob=adaptive_inter_prob,
            )

            # Log metrics to wandb
            log_metrics(
                step=generation,
                avg=rec.get("avg"),
                max=rec.get("max"),
                min=rec.get("min"),
                std=rec.get("std"),
                nevals=nevals,
                inter_prob=adaptive_inter_prob,
                cluster_diversity=diversity,
            )

            # Update evaluator with stats for individual early stopping
            if hasattr(self.evaluate_fn, "__self__") and hasattr(
                self.evaluate_fn.__self__, "update_generation_stats"
            ):
                self.evaluate_fn.__self__.update_generation_stats(
                    avg=rec.get("avg"), std=rec.get("std")
                )

            # Check early stopping condition
            if self.config["early_stopping"]:
                current_max = rec.get("max")
                if self._should_early_stop(current_max):
                    logging.info("Early stopping triggered!")
                    raise EarlyStoppingException()

            return rec

        stats.compile = compile_with_logging

        # Initialize population and hall of fame
        mu = self.config["mu"]  # Number of parents
        lambda_ = self.config["lambda_"]  # Number of offspring

        pop = self.toolbox.population(n=mu)
        current_population[0] = pop[:]
        hof = tools.HallOfFame(self.config["hof_size"])

        # Log separator for clarity
        width = shutil.get_terminal_size().columns
        logging.info("-" * width)

        # Run evolutionary algorithm
        logbook = tools.Logbook()
        try:
            pop, logbook = algorithms.eaMuPlusLambda(
                pop,
                self.toolbox,
                mu=mu,
                lambda_=lambda_,
                cxpb=self.config["cxpb"],
                mutpb=self.config["mutpb"],
                ngen=self.config["generations"],
                stats=stats,
                halloffame=hof,
            )
        except EarlyStoppingException:
            # Recover state from before early stopping
            logbook = tools.Logbook()
            for entry in logbook_entries:
                logbook.record(**entry)
            pop = current_population[0]
            if hof is not None:
                hof.update(pop)

        logging.info("-" * width)

        return pop, logbook, hof
