from deap import tools
import random

def composite_mutate(individual, cluster_dataset, indpb, inter_prob):
    """
    Per-gene mutation that applies inter-cluster and intra-cluster mutations
    based on specified probabilities.
    
    Args:
        individual: The individual to mutate, a list of (cluster_id, ClusterExample) pairs.
        cluster_dataset: The dataset containing clusters and their examples.
        indpb: Independent probability for each example to be mutated.
        inter_prob: Probability of applying inter-cluster mutation.
    
    Returns:
        A tuple containing the mutated individual.
    """
    cluster_map = {} # lookup table
    for cluster in cluster_dataset.clusters:
        if cluster.examples: # skip empty clusters
            cluster_map[cluster.cluster_id] = cluster

    for i in range(len(individual)):
        if random.random() >= indpb:
            continue  # Skip mutation for this example
        
        did_mutate = False
        if random.random() < inter_prob:
            # Inter-cluster mutation that replaces examples with examples from any cluster
            # inter means between
            new_cluster = random.choice(cluster_dataset.clusters)
            new_example = random.choice(new_cluster.examples)
            individual[i] = (new_cluster.cluster_id, new_example)
            did_mutate = True

        if not did_mutate:
            # Intra-cluster mutation that replaces examples with other examples from the same cluster
            # intra means within
            cluster_id, current_example = individual[i]
            current_cluster = cluster_map.get(cluster_id)
            if current_cluster and len(current_cluster.examples) > 1:
                other_examples = [ex for ex in current_cluster.examples if ex.example_id != current_example.example_id]
                if other_examples:
                    individual[i] = (cluster_id, random.choice(other_examples))

    return individual,

def mate(ind1, ind2):
    """
    Crosses two individuals by swapping the middle segment between two points.
    
    """
    return tools.cxTwoPoint(ind1, ind2)
