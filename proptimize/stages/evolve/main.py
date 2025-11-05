import logging
from deap import tools

logging.basicConfig(level=logging.INFO)

from proptimize.data_manager import DataManager
from proptimize.stages.evolve.experiment import GA
from proptimize.stages.evolve.operators import mate, composite_mutate

class EvolveStage:
    def __init__(self, data_manager, config, eval_fn):
        self.data_manager = data_manager
        self.config = config
        self.eval_fn = eval_fn
        self.evolution_trace = []  # Store example usage across generations

    def run(self):
        logging.info("Starting evolution stage...")
        
        # Load the clustered dataset 
        cluster_dataset = self.data_manager.load_cluster_dataset()

        # Filter out noise cluster
        # My idea is that noise/outliers don't generalize
        cluster_dataset.clusters = [
            cluster for cluster in cluster_dataset.clusters 
            if cluster.cluster_id != -1
        ]

        if len(cluster_dataset.clusters) < self.config['subset_size']:
            raise ValueError(
                "subset_size is larger than the number of clusters. "
                "Should probably check if clustering stage is okay."
            )

        def _mutate(individual, cluster_dataset, inter_prob=None):
            # If inter_prob is not provided, use the config value
            # This allows for adaptive inter_prob to be passed in
            # aka defaults to max_inter_prob if inter_prob=None
            if inter_prob is None:
                inter_prob = self.config['max_inter_prob'] 
            return composite_mutate(
                individual, 
                cluster_dataset, 
                inter_prob=inter_prob
            )

        # Initialize the GA
        ga = GA(
            cluster_dataset=cluster_dataset,
            evaluate_fn=self.eval_fn,
            mate_fn=mate,
            mutate_fn=_mutate,
            select_fn=tools.selTournament,
            config=self.config
        )
        
        # Set up evolution trace logging
        ga.evolution_trace_callback = self._log_examples
        
        logging.info("Running GA...")
        best_population, logbook, hof = ga.run()
        
        # Save results
        artifact = self._save_results(logbook, hof)
        
        # Save evolution trace as wandb artifact
        self._save_evolution_trace_artifact()
        
        logging.info(f"Evolution stage completed! Results saved as artifact: {artifact.name}")
        return artifact, logbook, hof
    
    def _log_examples(self, generation, population):
        """Simple callback to log example usage for each generation"""
        generation_examples = []
        for individual in population:
            individual_examples = []
            for cluster_id, example in individual:
                individual_examples.append({
                    "cluster_id": cluster_id,
                    "id": example.id
                })
            generation_examples.append(individual_examples)
        
        self.evolution_trace.append({
            "generation": generation,
            "examples": generation_examples
        })

    def _save_evolution_trace_artifact(self):
        artifact = self.data_manager.save_artifact(
            data=self.evolution_trace,
            artifact_name="evolution_trace",
            artifact_type="evolution_data"
        )
        
        logging.info(f"Evolution trace saved as wandb artifact: {artifact.name}")

    def _save_results(self, logbook, hof):
      
        hall_of_fame = [] # hof data

        for i, individual in enumerate(hof):
            hof_examples = []
            for cluster_id, example in individual:
                hof_examples.append({
                    "cluster_id": cluster_id,
                    "id": example.id,
                    "input": example.input,
                    "output": example.output
                })
            hall_of_fame.append({
                "rank": i + 1,
                "fitness": individual.fitness.values[0],
                "selected_examples": hof_examples
            })
 
        results = {
            "hall_of_fame": hall_of_fame,
            "evolution_stats": {
                "generations": len(logbook),
                "final_avg_fitness": logbook[-1]["avg"],
                "final_max_fitness": logbook[-1]["max"],
                "final_min_fitness": logbook[-1]["min"]
            }
        }
        
        return self.data_manager.save_results(results)

def run_evolve_stage(
    task,
    base_dir,
    config,
    eval_fn
):
    # Setup
    data_manager = DataManager(task, base_dir)

    # Run evolution
    stage = EvolveStage(data_manager, config, eval_fn)
    return stage.run()
