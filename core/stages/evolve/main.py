from pathlib import Path
from deap import tools

from .experiment import GAConfig, GA
from .operators import mate, composite_mutate
from .config import EvolveConfig
from ...data_manager import DataManager
from ...schemas import TaskType

# TODO: 4. implement wandb
# TODO: 5. set up something that tracks cost

class EvolveStage:
    def __init__(self, data_manager: DataManager, config: EvolveConfig, evaluate_fn):
        self.data_manager = data_manager
        self.config = config
        self.evaluate_fn = evaluate_fn

    def run(self):
        print("Starting evolution stage...")
        
        # Load the clustered dataset 
        cluster_dataset = self.data_manager.load_cluster_dataset(self.config.input_filename)
        
        # Filter out noise cluster for subset size calculation
        valid_clusters = [cluster for cluster in cluster_dataset.clusters if cluster.cluster_id != -1]
        if len(valid_clusters) < self.config.subset_size:
            print(f"Warning: Subset size {self.config.subset_size} is more than num clusters {len(valid_clusters)}. Reducing subset size.")

        ga_config = GAConfig(
            subset_size=min(self.config.subset_size, len(valid_clusters)), 
            pop_size=self.config.pop_size,
            generations=self.config.generations,
            cxpb=self.config.cxpb,
            mutpb=self.config.mutpb,
            tournsize=self.config.tournsize,
            seed=self.config.random_seed
        )
        
        def _mutate(individual):
            return composite_mutate(
                individual, 
                cluster_dataset, 
                indpb=self.config.mutation_indpb,
                inter_prob=self.config.inter_cluster_mutation_prob,
                intra_prob=self.config.intra_cluster_mutation_prob
            )

        # Initialize the algorithm
        ga = GA(
            cluster_dataset=cluster_dataset,
            evaluate_fn=self.evaluate_fn,
            mate_fn=mate,
            mutate_fn=_mutate,
            select_fn=tools.selTournament,
            config=ga_config
        )
        
        print("Running genetic algorithm...")
        best_population, logbook = ga.run()
        
        # Get the best individual
        best_individual = tools.selBest(best_population, 1)[0]
        
        print(f"Evolution completed!")
        print(f"Best individual fitness: {best_individual.fitness.values[0]}")
        print(f"Best individual contains {len(best_individual)} examples:")
        for cluster_id, example in best_individual:
            print(f"  - Example {example.example_id} from cluster {cluster_id} (text: {example.text[:50]}...)")
        
        # Save results
        output_path = self._save_results(best_individual, logbook)
        
        print(f"Evolution stage completed! Output saved to: {output_path}")
        return output_path, best_individual, logbook
    
    def _save_results(self, best_individual, logbook):
        """
        Save the evolution results to output directory.
        """
        # Extract examples from the best individual
        selected_examples = []
        for cluster_id, example in best_individual:
            selected_examples.append({
                "cluster_id": cluster_id,
                "example_id": example.example_id,
                "text": example.text,
                "membership_probability": example.membership_probability
            })
        
        # Prepare results data
        results = {
            "best_individual": {
                "fitness": best_individual.fitness.values[0],
                "selected_examples": selected_examples
            },
            "evolution_stats": {
                "generations": len(logbook),
                "final_avg_fitness": logbook[-1]["avg"] if logbook else 0,
                "final_max_fitness": logbook[-1]["max"] if logbook else 0,
                "final_min_fitness": logbook[-1]["min"] if logbook else 0
            }
        }
        
        # Save to output directory
        output_path = self.data_manager.save_final_output(results, self.config.output_filename)
        return output_path

def run_evolve_stage(
    task: TaskType,
    base_dir: str,
    config_dict: dict = None,
    evaluate_fn=None
):
    """
    Run the evolution stage to find optimal example subsets.
    
    Args:
        task: The task type
        base_dir: Base directory for task data  
        config_dict: Configuration dictionary for evolution
        evaluate_fn: Function to evaluate individuals. Should take (individual, cluster_dataset, base_dir) 
                    and return a fitness tuple. If None, uses default evaluation.
    """
    # Setup
    data_manager = DataManager(task, base_dir)
    config = EvolveConfig.from_dict(config_dict or {})

    # Run evolution
    stage = EvolveStage(data_manager, config, evaluate_fn)
    return stage.run()
