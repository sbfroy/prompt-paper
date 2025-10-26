import random

from deap import tools


def calculate_cluster_diversity(population, total_clusters):
    """Calculate the diversity of clusters present in the population.

    This measures how many unique clusters are represented across all individuals
    in the population, normalized by the total number of available clusters.

    Args:
        population: List of individuals, where each individual is a list of
            (cluster_id, ClusterExample) pairs.
        total_clusters: Total number of clusters available in the dataset.

    Returns:
        Float between 0 and 1, where:
            - 1.0 = maximum diversity (all clusters represented)
            - 0.0 = minimum diversity (only one cluster represented)
    """
    if total_clusters == 0:
        return 0.0

    # Collect all unique cluster IDs from the population
    unique_clusters = set()
    for individual in population:
        for cluster_id, example in individual:
            unique_clusters.add(cluster_id)

    # Normalize by total available clusters
    diversity = len(unique_clusters) / total_clusters
    return diversity


def composite_mutate(individual, cluster_dataset, indpb, inter_prob):
    """Apply per-gene mutation with inter-cluster and intra-cluster mutations.

    Args:
        individual: The individual to mutate, a list of (cluster_id, ClusterExample) pairs.
        cluster_dataset: The dataset containing clusters and their examples.
        indpb: Independent probability for each example to be mutated.
        inter_prob: Probability of applying inter-cluster mutation.

    Returns:
        A tuple containing the mutated individual.
    """
    cluster_map = {}  # Lookup table
    for cluster in cluster_dataset.clusters:
        if cluster.examples:  # Skip empty clusters
            cluster_map[cluster.cluster_id] = cluster

    for i in range(len(individual)):
        if random.random() >= indpb:
            continue  # Skip mutation for this example

        did_mutate = False
        if random.random() < inter_prob:
            # Inter-cluster mutation: replaces examples with examples from any cluster
            new_cluster = random.choice(cluster_dataset.clusters)
            new_example = random.choice(new_cluster.examples)
            individual[i] = (new_cluster.cluster_id, new_example)
            did_mutate = True

        if not did_mutate:
            # Intra-cluster mutation: replaces examples with other examples from same cluster
            cluster_id, current_example = individual[i]
            current_cluster = cluster_map.get(cluster_id)
            if current_cluster and len(current_cluster.examples) > 1:
                other_examples = [ex for ex in current_cluster.examples if ex.id != current_example.id]
                if other_examples:
                    individual[i] = (cluster_id, random.choice(other_examples))

    return individual,

def mate(ind1, ind2):
    """Cross two individuals by swapping the middle segment between two points.

    Args:
        ind1: First parent individual.
        ind2: Second parent individual.

    Returns:
        A tuple containing the two crossed offspring.
    """
    return tools.cxTwoPoint(ind1, ind2)
