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


def composite_mutate(individual, cluster_dataset, inter_prob):
    """
    Mutate exactly one gene in the individual.

    Behavior:
    - When this operator is called it will always perform exactly one gene
      mutation (selecting a gene uniformly at random).
    - For that mutation event the operator chooses the mutation mode once:
      inter-cluster mutation with probability ``inter_prob`` (replace the
      example with one from any cluster), otherwise intra-cluster mutation
      (replace the example with another example from the same cluster).

    Args:
        individual: The individual to mutate, a list of (cluster_id, ClusterExample) pairs.
        cluster_dataset: The dataset containing clusters and their examples.
        inter_prob: Probability of applying inter-cluster mutation.

    Returns:
        A tuple containing the mutated individual.
    """
    cluster_map = {}  # Lookup table
    for cluster in cluster_dataset.clusters:
        if cluster.examples:  # Skip empty clusters
            cluster_map[cluster.cluster_id] = cluster

    i = random.randrange(len(individual))

    if random.random() < inter_prob:
        # Inter-cluster mutation: replace example with an example from any cluster
        new_cluster = random.choice(cluster_dataset.clusters)
        new_example = random.choice(new_cluster.examples)
        individual[i] = (new_cluster.cluster_id, new_example)
    else:
        # Intra-cluster mutation: replace example with another example from same cluster
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
