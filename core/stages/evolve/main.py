from deap import tools

from .experiment import GA
from .operators import mate, composite_mutate
from .config import EvolveConfig
from ...data_manager import DataManager
from .client import get_total_cost, get_total_tokens, reset_cost_tracking

class EvolveStage:
    def __init__(self, data_manager, config, evaluate_fn):
        self.data_manager = data_manager
        self.config = config
        self.evaluate_fn = evaluate_fn

    def run(self):
        print("Starting evolution stage...")
        
        # Reset cost tracking for this run
        reset_cost_tracking()
        
        # Load the clustered dataset 
        cluster_dataset = self.data_manager.load_cluster_dataset()
        
        # Filter out noise cluster for subset size calculation
        valid_clusters = [cluster for cluster in cluster_dataset.clusters if cluster.cluster_id != -1]
        if len(valid_clusters) < self.config.subset_size:
            print(f"Warning: Subset size {self.config.subset_size} is more than num clusters {len(valid_clusters)}. Reducing subset size.")

        # Update subset size if needed
        self.config.subset_size = min(self.config.subset_size, len(valid_clusters))
        
        def _mutate(individual, cluster_dataset):
            return composite_mutate(
                individual, 
                cluster_dataset, 
                indpb=self.config.indpb,
                inter_prob=self.config.inter_prob,
                intra_prob=self.config.intra_prob
            )

        # Initialize the algorithm
        ga = GA(
            cluster_dataset=cluster_dataset,
            evaluate_fn=self.evaluate_fn,
            mate_fn=mate,
            mutate_fn=_mutate,
            select_fn=tools.selTournament,
            config=self.config
        )
        
        print("Running GA...")
        best_population, logbook, hof = ga.run()
        
        # Save results
        artifact = self._save_results(logbook, hof)
        
        print(f"Evolution stage completed! Output saved as artifact: {artifact.name}")
        return artifact, logbook, hof
    
    def _save_results(self, logbook, hof):
      
        hall_of_fame = [] # hof data
        
        for i, individual in enumerate(hof):
            hof_examples = []
            for cluster_id, example in individual:
                hof_examples.append({
                    "cluster_id": cluster_id,
                    "example_id": example.example_id,
                    "text": example.text
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
                "final_avg_fitness": logbook[-1]["avg"] if logbook else 0,
                "final_max_fitness": logbook[-1]["max"] if logbook else 0,
                "final_min_fitness": logbook[-1]["min"] if logbook else 0
            },
            "api_costs": {
                "total_cost_nok": get_total_cost(),
                "total_tokens": get_total_tokens()
            }
        }
        
        return self.data_manager.save_results(results)

def run_evolve_stage(
    task,
    base_dir,
    config_dict,
    evaluate_fn
):
    # Setup
    data_manager = DataManager(task, base_dir)
    config = EvolveConfig.from_dict(config_dict or {})

    # Run evolution
    stage = EvolveStage(data_manager, config, evaluate_fn)
    return stage.run()
