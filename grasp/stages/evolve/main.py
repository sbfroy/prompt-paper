"""Main evolution stage: manages GA execution and result tracking."""

import json
import logging
import os
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

    def __init__(self, data_manager, config, eval_fn, test_eval_fn=None,
                 checkpoint_dir=None, resume_from=None):
        self.data_manager = data_manager
        self.config = config
        self.eval_fn = eval_fn
        self.test_eval_fn = test_eval_fn
        self.evolution_trace = []  # Store example usage across generations
        self.checkpoint_dir = checkpoint_dir
        self.resume_from = resume_from

        # Get sample size and use_real flag from config for round-robin sampling
        self.sample_size = config["dataset_size"]
        self.use_real = config["use_real"]

    def run(self):
        """Execute the evolution stage and return results."""
        logging.info("Starting evolution stage...")

        # When resuming a dataset_size=50 run, round-robin sampling is not
        # reproducible (it depends on the global RNG state at sampling time),
        # so we rebuild the cluster dataset directly from the (cluster_id,
        # example_id) pairs stored in the checkpoint.
        rebuild_from_checkpoint = (
            self.resume_from is not None and self.sample_size == 50
        )

        if rebuild_from_checkpoint:
            cluster_dataset = self.data_manager.load_cluster_dataset(
                use_real=self.use_real,
                sample_size=None,
            )
            cluster_dataset = self._filter_dataset_to_checkpoint_pairs(
                cluster_dataset, self.resume_from
            )
        else:
            # Load the clustered dataset with optional round-robin sampling
            cluster_dataset = self.data_manager.load_cluster_dataset(
                use_real=self.use_real,
                sample_size=self.sample_size
            )

        # Filter out noise cluster (outliers don't generalize well)
        cluster_dataset.clusters = [
            cluster
            for cluster in cluster_dataset.clusters
            if cluster.cluster_id != -1
        ]

        # Validate configuration
        if len(cluster_dataset.clusters) < self.config["subset_size"]:
            logging.warning(
                f"subset_size ({self.config['subset_size']}) is larger than the number of clusters ({len(cluster_dataset.clusters)}). "
                "The GA will sample clusters with replacement. Consider checking clustering stage parameters."
            )

        # Load checkpoint if resuming
        resume_state = None
        if self.resume_from is not None:
            logging.info(f"Loading checkpoint from {self.resume_from}")
            resume_state = GA.load_checkpoint(self.resume_from, cluster_dataset)
            # Restore evolution trace from checkpoint
            self.evolution_trace = resume_state.get("evolution_trace", [])

        def _mutate(individual, cluster_dataset, inter_prob=None):
            """
            Wrapper for mutation with adaptive inter_prob support.

            If inter_prob is not provided, defaults to max_inter_prob from config.
            This allows the GA to adapt mutation strategy during evolution.
            """
            if inter_prob is None:
                inter_prob = self.config["max_inter_prob"]
            return composite_mutate(individual, cluster_dataset, inter_prob=inter_prob)

        # Create checkpoint directory if needed
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize the genetic algorithm
        ga = GA(
            cluster_dataset=cluster_dataset,
            evaluate_fn=self.eval_fn,
            mate_fn=mate,
            mutate_fn=_mutate,
            select_fn=tools.selTournament,
            config=self.config,
            test_evaluate_fn=self.test_eval_fn,
            checkpoint_dir=self.checkpoint_dir,
        )

        # Set up evolution trace logging callback and reference for checkpointing
        ga.evolution_trace_callback = self._log_examples
        ga.evolution_trace_ref = self.evolution_trace

        # Restore best_individuals_history from checkpoint before run
        if resume_state is not None:
            ga.best_individuals_history = resume_state["best_individuals_history"]

        logging.info("Running GA...")
        best_population, logbook, hof = ga.run(resume_state=resume_state)

        # Save results to artifacts
        artifact = self._save_results(logbook, hof)
        self._save_evolution_trace_artifact()
        self._save_best_individuals_artifact(ga.best_individuals_history)

        logging.info(
            f"Evolution stage completed!"
        )
        return artifact, logbook, hof

    @staticmethod
    def _filter_dataset_to_checkpoint_pairs(cluster_dataset, checkpoint_path):
        """Rebuild a cluster dataset containing only examples referenced in a checkpoint.

        Round-robin sampling is not reproducible across runs, so on resume we
        reconstruct the sampled pool by keeping only the (cluster_id, example_id)
        pairs that appear in the checkpoint's population or hall of fame. Any
        example that was sampled originally but evolved out before the checkpoint
        was saved is unrecoverable and will be missing from the resumed pool.
        """
        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        pairs = set()
        for ind in data.get("population", []):
            for gene in ind["genes"]:
                pairs.add((gene["cluster_id"], gene["example_id"]))
        for ind in data.get("hall_of_fame", []):
            for gene in ind["genes"]:
                pairs.add((gene["cluster_id"], gene["example_id"]))

        wanted_ids_by_cluster = {}
        for cluster_id, example_id in pairs:
            wanted_ids_by_cluster.setdefault(cluster_id, set()).add(example_id)

        kept_clusters = []
        for cluster in cluster_dataset.clusters:
            wanted_ids = wanted_ids_by_cluster.get(cluster.cluster_id)
            if not wanted_ids:
                continue
            filtered_examples = [ex for ex in cluster.examples if ex.id in wanted_ids]
            if not filtered_examples:
                continue
            cluster.examples = filtered_examples
            kept_clusters.append(cluster)

        cluster_dataset.clusters = kept_clusters

        total_examples = sum(len(c.examples) for c in kept_clusters)
        logging.info(
            f"Rebuilt cluster dataset from checkpoint: {total_examples} examples "
            f"across {len(kept_clusters)} clusters (expected {len(pairs)} pairs from checkpoint)."
        )

        return cluster_dataset

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
        # logging.info(f"Evolution trace saved as wandb artifact: {artifact.name}")

    def _save_best_individuals_artifact(self, best_individuals_history):
        """Save best individuals history as wandb artifact."""
        if not best_individuals_history:
            logging.info("No best individuals to save")
            return

        artifact = self.data_manager.save_artifact(
            data=best_individuals_history,
            artifact_name="best_individuals_history",
            artifact_type="evolution_data",
        )
        # logging.info(f"Best individuals history saved as wandb artifact: {artifact.name}")

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


def run_evolve_stage(task, base_dir, config, eval_fn, test_eval_fn=None,
                     resume_from=None):
    """
    Entry point for running the evolution stage.

    Args:
        task: Task identifier for data management.
        base_dir: Base directory for artifacts.
        config: Configuration dictionary with GA parameters.
        eval_fn: Fitness evaluation function for individuals.
        test_eval_fn: Optional test evaluation function for best individuals.
        resume_from: Optional path to a checkpoint file to resume from.

    Returns:
        Tuple of (artifact, logbook, hall_of_fame).
    """
    # Setup data manager
    data_manager = DataManager(task, base_dir)

    # Default checkpoint dir alongside data
    checkpoint_dir = os.path.join(base_dir, task, "checkpoints")

    # Run evolution
    stage = EvolveStage(
        data_manager, config, eval_fn, test_eval_fn,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
    )
    return stage.run()
