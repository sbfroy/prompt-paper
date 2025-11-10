"""Main evolution stage: manages GA execution and result tracking."""

import logging
from deap import tools

from grasp.data_manager import DataManager
from grasp.stages.evolve.experiment import GA
from grasp.stages.evolve.operators import mate, composite_mutate

logging.basicConfig(level=logging.INFO)


class EvolveStage:
    """
    Orchestrates the evolution process using a genetic algorithm.

    Loads clustered data, runs GA to find optimal example combinations,
    and saves results including hall of fame and evolution traces.
    """

    def __init__(self, data_manager, config, eval_fn):
        self.data_manager = data_manager
        self.config = config
        self.eval_fn = eval_fn
        self.evolution_trace = []  # Store example usage across generations
        
        # Get dataset size from config for artifact naming
        self.dataset_size = config["dataset_size"]

    def run(self):
        """Execute the evolution stage and return results."""
        logging.info("Starting evolution stage...")

        # Load the clustered dataset with size
        cluster_dataset = self.data_manager.load_cluster_dataset(dataset_size=self.dataset_size)

        # Filter out noise cluster (outliers don't generalize well)
        cluster_dataset.clusters = [
            cluster
            for cluster in cluster_dataset.clusters
            if cluster.cluster_id != -1
        ]

        # Validate configuration
        if len(cluster_dataset.clusters) < self.config["subset_size"]:
            raise ValueError(
                "subset_size is larger than the number of clusters. "
                "Should probably check if clustering stage is okay."
            )

        def _mutate(individual, cluster_dataset, inter_prob=None):
            """
            Wrapper for mutation with adaptive inter_prob support.

            If inter_prob is not provided, defaults to max_inter_prob from config.
            This allows the GA to adapt mutation strategy during evolution.
            """
            if inter_prob is None:
                inter_prob = self.config["max_inter_prob"]
            return composite_mutate(individual, cluster_dataset, inter_prob=inter_prob)

        # Initialize the genetic algorithm
        ga = GA(
            cluster_dataset=cluster_dataset,
            evaluate_fn=self.eval_fn,
            mate_fn=mate,
            mutate_fn=_mutate,
            select_fn=tools.selTournament,
            config=self.config,
        )

        # Set up evolution trace logging callback
        ga.evolution_trace_callback = self._log_examples

        logging.info("Running GA...")
        best_population, logbook, hof = ga.run()

        # Save results to artifacts
        artifact = self._save_results(logbook, hof)
        self._save_evolution_trace_artifact()

        logging.info(
            f"Evolution stage completed! Results saved as artifact: {artifact.name}"
        )
        return artifact, logbook, hof

    def _log_examples(self, generation, population):
        """Callback to track example usage for each generation."""
        generation_examples = []
        for individual in population:
            individual_examples = []
            for cluster_id, example in individual:
                individual_examples.append(
                    {"cluster_id": cluster_id, "id": example.id}
                )
            generation_examples.append(individual_examples)

        self.evolution_trace.append(
            {"generation": generation, "examples": generation_examples}
        )

    def _save_evolution_trace_artifact(self):
        """Save evolution trace as wandb artifact."""
        artifact = self.data_manager.save_artifact(
            data=self.evolution_trace,
            artifact_name="evolution_trace",
            artifact_type="evolution_data",
        )
        logging.info(f"Evolution trace saved as wandb artifact: {artifact.name}")

    def _save_results(self, logbook, hof):
        """Save hall of fame and evolution statistics."""
        hall_of_fame = []

        # Extract best individuals and their fitness
        for i, individual in enumerate(hof):
            hof_examples = []
            for cluster_id, example in individual:
                hof_examples.append(
                    {
                        "cluster_id": cluster_id,
                        "id": example.id,
                        "input": example.input,
                        "output": example.output,
                    }
                )
            hall_of_fame.append(
                {
                    "rank": i + 1,
                    "fitness": individual.fitness.values[0],
                    "selected_examples": hof_examples,
                }
            )

        # Compile results
        results = {
            "hall_of_fame": hall_of_fame,
            "evolution_stats": {
                "generations": len(logbook),
                "final_avg_fitness": logbook[-1]["avg"],
                "final_max_fitness": logbook[-1]["max"],
                "final_min_fitness": logbook[-1]["min"],
            },
        }

        return self.data_manager.save_results(results)


def run_evolve_stage(task, base_dir, config, eval_fn):
    """
    Entry point for running the evolution stage.

    Args:
        task: Task identifier for data management.
        base_dir: Base directory for artifacts.
        config: Configuration dictionary with GA parameters.
        eval_fn: Fitness evaluation function for individuals.

    Returns:
        Tuple of (artifact, logbook, hall_of_fame).
    """
    # Setup data manager
    data_manager = DataManager(task, base_dir)

    # Run evolution
    stage = EvolveStage(data_manager, config, eval_fn)
    return stage.run()
