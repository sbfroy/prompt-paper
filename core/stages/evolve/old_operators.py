from deap import tools
import random

def inter_cluster_mutate(individual, cluster_dataset, indpb):
    """
    Cluster-aware mutation that replaces examples with others from different clusters.
    'inter' = 'between'
    Args:
        (described in composite_mutate)
    """
    # Clusters that are already represented
    used_cluster_ids = {cluster_id for cluster_id, example in individual}
    
    available_clusters = [
        cluster for cluster in cluster_dataset.clusters 
        if cluster.cluster_id not in used_cluster_ids
    ]
    
    for i in range(len(individual)):
        if random.random() < indpb and available_clusters:
            new_cluster = random.choice(available_clusters)
            new_example = random.choice(new_cluster.examples)

            # Get the old cluster and example id
            old_cluster_id, old_example = individual[i]
    
            old_cluster = next(c for c in cluster_dataset.clusters if c.cluster_id == old_cluster_id)
            if old_cluster not in available_clusters:
                # make the old cluster available again
                available_clusters.append(old_cluster)
            
            # Remove the new cluster from available clusters  
            available_clusters.remove(new_cluster)
            used_cluster_ids.add(new_cluster.cluster_id)
            used_cluster_ids.remove(old_cluster_id)
            
            # Replace the old example with the new one
            individual[i] = (new_cluster.cluster_id, new_example)
    
    return individual,

def intra_cluster_mutate(individual, cluster_dataset, indpb):
    """
    Mutation that replaces examples with other examples from the same cluster.
    'intra' = 'within'
    Args:
        (described in composite_mutate)
    """
    for i in range(len(individual)):
        if random.random() < indpb:
            cluster_id, current_example = individual[i]
            
            # Find the cluster this example belongs to
            current_cluster = next(c for c in cluster_dataset.clusters if c.cluster_id == cluster_id)
            
            if current_cluster and len(current_cluster.examples) > 1:
                # Select a different example from the same cluster
                other_examples = [ex for ex in current_cluster.examples if ex.example_id != current_example.example_id]
                if other_examples:
                    individual[i] = (cluster_id, random.choice(other_examples))
    
    return individual,

def composite_mutate(individual, cluster_dataset, indpb, inter_prob, intra_prob):

    """
    Composite mutation that randomly selects between inter-cluster, intra-cluster, and noise injection mutation.
    
    Args:
        individual: List of (cluster_id, ClusterExample) tuples
        cluster_dataset: The dataset containing all clusters
        indpb: Independent probability for each example to be mutated
        inter_prob: Probability of using inter-cluster mutation
        intra_prob: Probability of using intra-cluster mutation
    """
    rand_val = random.random()
    
    if rand_val < inter_prob:
        return inter_cluster_mutate(individual, cluster_dataset, indpb)
    else:
        return intra_cluster_mutate(individual, cluster_dataset, indpb)


def mate(ind1, ind2):
    """Crosses two individuals by swapping the middle segment between two random cut points."""
    return tools.cxTwoPoint(ind1, ind2)
