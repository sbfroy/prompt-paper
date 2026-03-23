"""Genetic algorithm implementation using DEAP framework."""

import json
import logging
import os
import random
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
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
        test_evaluate_fn: callable = None,
        checkpoint_dir: str = None,
    ):
        self.cluster_dataset = cluster_dataset
        self.evaluate_fn = evaluate_fn
        self.mate_fn = mate_fn
        self.mutate_fn = mutate_fn
        self.select_fn = select_fn
        self.config = config
        self.test_evaluate_fn = test_evaluate_fn
        self.checkpoint_dir = checkpoint_dir
        self._eval_calls_total = 0
        self._eval_lock = threading.Lock()
        self._current_generation = 0 # Track current generation for progress bar
        self.evolution_trace_callback = None  # Callback to log evolution trace
        self.best_individuals_history = []  # Track best individual per generation

        # Reference to EvolveStage's evolution_trace list for checkpointing
        self.evolution_trace_ref = None

        # Store latest evaluation metrics for logging (thread-safe dict)
        self._latest_metrics = {}
        self._metrics_lock = threading.Lock()

        # Set random seeds for reproducibility
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])

        # Define fitness and individual types
        if not hasattr(creator, "FitnessMax"):
            creator.create(
                "FitnessMax", base.Fitness, weights=(1.0,)
            )  # Maximize fitness
        if not hasattr(creator, "Individual"):
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
        # Evaluate function now returns dict with micro_f1, micro_precision, micro_recall, macro_f1
        def _eval(individual):
            result = self.evaluate_fn(individual)
            with self._eval_lock:  # Thread-safe counter update
                self._eval_calls_total += 1
            # Extract micro_f1 as the fitness score (main evolution metric)
            # Store full metrics for later logging
            if isinstance(result, dict):
                fitness = result.get("micro_f1", 0.0)
                # Store metrics on the individual for later aggregation
                individual._eval_metrics = result
            else:
                # Backward compatibility: if result is a float, use it directly
                fitness = result
                individual._eval_metrics = {"micro_f1": fitness}
            return (fitness,)  # DEAP expects tuple

        self.toolbox.register("evaluate", _eval)
        self.toolbox.register("map", self._parallel_map)
        self.toolbox.register("mate", self.mate_fn)
        self.toolbox.register(
            "mutate", self.mutate_fn, cluster_dataset=self.cluster_dataset
        )
        self.toolbox.register(
            "select", self.select_fn, tournsize=self.config["tournsize"]
        )

        # Early stopping state - track both max and avg fitness
        self.max_stagnant_counter = 0
        self.avg_stagnant_counter = 0
        self.best_max_value = None
        self.best_avg_value = None

        # Store total number of clusters for diversity calculation
        self.total_clusters = len(cluster_dataset.clusters)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_example_index(cluster_dataset):
        """Build a lookup dict from (cluster_id, example_id) to (cluster_id, example)."""
        index = {}
        for cluster in cluster_dataset.clusters:
            for example in cluster.examples:
                index[(cluster.cluster_id, example.id)] = (cluster.cluster_id, example)
        return index

    @staticmethod
    def _serialize_individual(individual):
        """Convert a DEAP Individual to a JSON-serializable dict."""
        data = {
            "genes": [
                {"cluster_id": cluster_id, "example_id": example.id}
                for cluster_id, example in individual
            ],
            "fitness": individual.fitness.values[0] if individual.fitness.valid else None,
        }
        if hasattr(individual, "_eval_metrics"):
            data["eval_metrics"] = individual._eval_metrics
        return data

    @staticmethod
    def _reconstruct_individual(ind_data, example_index):
        """Rebuild a DEAP Individual from serialized data."""
        genes = []
        for gene in ind_data["genes"]:
            key = (gene["cluster_id"], gene["example_id"])
            if key not in example_index:
                raise ValueError(
                    f"Example (cluster={gene['cluster_id']}, id={gene['example_id']}) "
                    "not found in cluster dataset. Was the dataset changed since the checkpoint?"
                )
            genes.append(example_index[key])

        individual = creator.Individual(genes)
        if ind_data.get("fitness") is not None:
            individual.fitness.values = (ind_data["fitness"],)
        if "eval_metrics" in ind_data:
            individual._eval_metrics = ind_data["eval_metrics"]
        return individual

    @staticmethod
    def _get_rng_state():
        """Capture current RNG state in a JSON-serializable format."""
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        return {
            "random": [random_state[0], list(random_state[1]), random_state[2]],
            "numpy": [
                numpy_state[0],
                numpy_state[1].tolist(),
                int(numpy_state[2]),
                int(numpy_state[3]),
                float(numpy_state[4]),
            ],
        }

    @staticmethod
    def _set_rng_state(rng_state):
        """Restore RNG state from a checkpoint."""
        rs = rng_state["random"]
        random.setstate((rs[0], tuple(rs[1]), rs[2]))
        ns = rng_state["numpy"]
        np.random.set_state(
            (ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
        )

    def _save_checkpoint(self, generation, population, hof, logbook_entries):
        """Atomically save a checkpoint to disk after a completed generation."""
        if self.checkpoint_dir is None:
            return

        checkpoint = {
            "generation": generation,
            "population": [self._serialize_individual(ind) for ind in population],
            "hall_of_fame": [self._serialize_individual(ind) for ind in hof],
            "logbook_entries": logbook_entries,
            "best_individuals_history": self.best_individuals_history,
            "evolution_trace": self.evolution_trace_ref if self.evolution_trace_ref is not None else [],
            "early_stopping_state": {
                "max_stagnant_counter": self.max_stagnant_counter,
                "avg_stagnant_counter": self.avg_stagnant_counter,
                "best_max_value": self.best_max_value,
                "best_avg_value": self.best_avg_value,
            },
            "eval_calls_total": self._eval_calls_total,
            "rng_state": self._get_rng_state(),
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.json")

        # Write to a temp file first, then atomically replace
        fd, temp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(checkpoint, f)
            os.replace(temp_path, checkpoint_path)  # Atomic on POSIX
            logging.info(f"Checkpoint saved at generation {generation}")
        except BaseException:
            # Clean up temp file on any failure
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def load_checkpoint(checkpoint_path, cluster_dataset):
        """Load a checkpoint and reconstruct the population and hall of fame.

        Args:
            checkpoint_path: Path to checkpoint.json
            cluster_dataset: The same ClusterDataset used during the original run
                (after noise-cluster filtering).

        Returns:
            Dict with restored state ready to pass into GA.run().
        """

        if not hasattr(creator, "FitnessMax"):
              creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
              creator.create("Individual", list, fitness=creator.FitnessMax)

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        example_index = GA._build_example_index(cluster_dataset)

        population = [
            GA._reconstruct_individual(ind, example_index)
            for ind in data["population"]
        ]
        hall_of_fame_individuals = [
            GA._reconstruct_individual(ind, example_index)
            for ind in data["hall_of_fame"]
        ]

        return {
            "generation": data["generation"],
            "population": population,
            "hall_of_fame_individuals": hall_of_fame_individuals,
            "logbook_entries": data["logbook_entries"],
            "best_individuals_history": data["best_individuals_history"],
            "evolution_trace": data.get("evolution_trace", []),
            "early_stopping_state": data["early_stopping_state"],
            "eval_calls_total": data["eval_calls_total"],
            "rng_state": data["rng_state"],
        }

    # ------------------------------------------------------------------

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
      items = list(iterable)
      desc = f"Gen {self._current_generation}/{self.config['generations']}"

      with ThreadPoolExecutor(max_workers=self.config["workers"]) as executor:
          results = list(tqdm(
              executor.map(func, items),
              total=len(items),
              desc=desc,
              leave=False,
              ncols=80,
          ))

      self._current_generation += 1
      return results

    def _should_early_stop(self, current_max_value, current_avg_value, generation):
        """
        Check if early stopping should be triggered.

        Uses a combined criterion: stops only when BOTH max and avg fitness
        have stagnated for the patience period. This is more robust than
        tracking max alone, as it allows the population to continue improving
        even if the best individual has plateaued.

        Args:
            current_max_value: Best fitness in current generation.
            current_avg_value: Average fitness in current generation.
            generation: Current generation number.

        Returns:
            True if early stopping should be triggered, False otherwise.
        """
        # Don't early stop until minimum generations reached
        min_gens = self.config["early_stopping_min_generations"]
        if generation < min_gens:
            return False

        min_delta = self.config["early_stopping_min_delta"]
        patience = self.config["early_stopping_patience"]

        # Initialize best values on first call after min_gens
        if self.best_max_value is None:
            self.best_max_value = current_max_value
            self.best_avg_value = current_avg_value
            return False

        # Check max fitness improvement
        max_improvement = current_max_value - self.best_max_value
        if max_improvement > min_delta:
            self.best_max_value = current_max_value
            self.max_stagnant_counter = 0
        else:
            self.max_stagnant_counter += 1

        # Check avg fitness improvement
        avg_improvement = current_avg_value - self.best_avg_value
        if avg_improvement > min_delta:
            self.best_avg_value = current_avg_value
            self.avg_stagnant_counter = 0
        else:
            self.avg_stagnant_counter += 1

        # Stop only when BOTH have stagnated for patience generations
        return (
            self.max_stagnant_counter >= patience and
            self.avg_stagnant_counter >= patience
        )

    def _track_best_individual(self, generation, individual, fitness, test_score=None):
        """
        Track the best individual from a generation in memory.

        Args:
            generation: Generation number
            individual: The best individual to track
            fitness: Validation fitness score
            test_score: Optional test score if evaluation was performed
        """
        # Convert individual to serializable format
        individual_data = []
        for cluster_id, example in individual:
            individual_data.append({
                "cluster_id": cluster_id,
                "id": example.id,
                "input": example.input,
                "output": example.output,
            })

        save_data = {
            "generation": generation,
            "validation_fitness": fitness,
            "test_score": test_score,
            "individual": individual_data,
        }

        self.best_individuals_history.append(save_data)

    def run(self, resume_state=None):
        """
        Execute the genetic algorithm.

        Args:
            resume_state: Optional dict from GA.load_checkpoint() to resume
                a previously interrupted run.

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

        if resume_state is not None:
            _gen = [resume_state["generation"]]
            prev_eval_calls = [resume_state["eval_calls_total"]]
            logbook_entries = list(resume_state["logbook_entries"])
            _skip_initial_compile = [True]
            remaining_gens = self.config["generations"] - resume_state["generation"]
        else:
            _gen = [-1]  # Generation counter
            prev_eval_calls = [0]  # Track evaluations per generation
            logbook_entries = []
            _skip_initial_compile = [False]
            remaining_gens = self.config["generations"]

        # Store state for early stopping recovery
        current_population = [None]

        def compile_with_logging(population):
            """Enhanced compile function with logging and adaptive mutation."""
            rec = original_compile(population)

            # When resuming, skip the initial compile call (the restored
            # population is re-reported by eaMuPlusLambda as gen 0).
            if _skip_initial_compile[0]:
                _skip_initial_compile[0] = False
                current_population[0] = population[:]
                return rec

            _gen[0] += 1
            generation = _gen[0]

            # Calculate evaluations since last generation
            nevals = self._eval_calls_total - prev_eval_calls[0]
            prev_eval_calls[0] = self._eval_calls_total

            # Compile statistics
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

            # Find best individual in current population
            best_individual = max(population, key=lambda ind: ind.fitness.values[0])
            best_fitness = best_individual.fitness.values[0]

            # Log validation metrics from best individual if available
            if hasattr(best_individual, "_eval_metrics"):
                best_metrics = best_individual._eval_metrics
                log_metrics(
                    step=generation,
                    val_micro_precision=best_metrics.get("micro_precision"),
                    val_micro_recall=best_metrics.get("micro_recall"),
                    val_micro_f1=best_metrics.get("micro_f1"),
                    val_macro_f1=best_metrics.get("macro_f1"),
                )

            # Evaluate best individual on test set if test evaluator is provided
            test_metrics = None
            if self.test_evaluate_fn is not None:
                test_result = self.test_evaluate_fn(best_individual)
                # Handle both dict and float return types
                if isinstance(test_result, dict):
                    test_metrics = test_result
                    # Log test micro_f1 and macro_f1 to wandb
                    log_metrics(
                        step=generation,
                        test_micro_f1=test_result.get("micro_f1"),
                        test_macro_f1=test_result.get("macro_f1"),
                    )
                else:
                    # Backward compatibility: single score
                    test_metrics = {"micro_f1": test_result}
                    log_metrics(step=generation, test_score=test_result)

            # Track best individual in memory
            self._track_best_individual(generation, best_individual, best_fitness, test_metrics)

            # Update evaluator with stats for individual early stopping
            if hasattr(self.evaluate_fn, "__self__") and hasattr(
                self.evaluate_fn.__self__, "update_generation_stats"
            ):
                self.evaluate_fn.__self__.update_generation_stats(
                    avg=rec.get("avg"), std=rec.get("std")
                )

            # Save checkpoint after each generation
            self._save_checkpoint(generation, population, hof, logbook_entries)

            # Check early stopping condition (now uses both max and avg)
            if self.config["early_stopping"]:
                current_max = rec.get("max")
                current_avg = rec.get("avg")
                if self._should_early_stop(current_max, current_avg, generation):
                    logging.info(
                        f"Early stopping triggered! "
                        f"Max stagnant for {self.max_stagnant_counter} gens, "
                        f"Avg stagnant for {self.avg_stagnant_counter} gens."
                    )
                    raise EarlyStoppingException()

            return rec

        stats.compile = compile_with_logging

        # Initialize population and hall of fame
        mu = self.config["mu"]  # Number of parents
        lambda_ = self.config["lambda_"]  # Number of offspring

        hof = tools.HallOfFame(self.config["hof_size"])

        if resume_state is not None:
            pop = resume_state["population"]
            # Restore hall of fame from checkpoint
            hof.update(resume_state["hall_of_fame_individuals"])
            # Restore GA state
            self._eval_calls_total = resume_state["eval_calls_total"]
            self.best_individuals_history = resume_state["best_individuals_history"]
            es = resume_state["early_stopping_state"]
            self.max_stagnant_counter = es["max_stagnant_counter"]
            self.avg_stagnant_counter = es["avg_stagnant_counter"]
            self.best_max_value = es["best_max_value"]
            self.best_avg_value = es["best_avg_value"]
            # Restore RNG state
            self._set_rng_state(resume_state["rng_state"])
            # Set progress bar counter (initial empty eval will increment once)
            self._current_generation = resume_state["generation"]

            logging.info(
                f"Resuming from generation {resume_state['generation']}, "
                f"running {remaining_gens} more generations."
            )
        else:
            pop = self.toolbox.population(n=mu)

        current_population[0] = pop[:]

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
                ngen=remaining_gens,
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

        return pop, logbook, hof
