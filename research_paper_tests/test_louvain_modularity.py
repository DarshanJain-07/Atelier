import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from schema import SimConfig, DIMENSIONS
from generate_society import generate_society
from society_evolution import SocietyEvolution
import networkx as nx
import community as community_louvain  # python-louvain


def build_threshold_graph(exposures, influence, threshold=0.85):
    """
    Builds a social network using Cosine Similarity + Influence scaling.
    Uses a Threshold approach instead of K-NN to allow isolated bubbles to form properly.
    """
    print(f"Building Threshold Graph (t={threshold}) for {len(exposures)} agents...")
    similarity_matrix = cosine_similarity(exposures.numpy())

    # We want true homophily: people only connect if their beliefs are highly aligned.
    # similarity_matrix goes from -1 to 1.
    # A threshold of 0.85 means they must be very ideologically similar to form a connection.

    # We don't scale by influence for edge *existence* in a pure ideological graph,
    # because that artificially forces weak-aligned people to connect to elites,
    # which flattens the Louvain community detection into a single "Elite-worshipping" block.

    np.fill_diagonal(similarity_matrix, -np.inf)

    G = nx.Graph()
    G.add_nodes_from(range(len(exposures)))

    edges = []
    # Only keep edges where similarity > threshold
    for i in range(len(exposures)):
        # Find all j where similarity is high
        for j in np.where(similarity_matrix[i] > threshold)[0]:
            edges.append((i, j))

    G.add_edges_from(edges)
    return G


def run_louvain_modularity_test():
    print("--- Running Network Modularity (Louvain) Analysis ---")
    config = SimConfig(
        num_agents=2000,
        seed=42,
        enable_evolution=True,
        evolution_generations=50,
        use_ideological_drift=True,
        elite_influence_drift_chance=0.20,  # Higher hegemony for more distinct tribe formation
        record_history=False,
    )
    config.wealth_dim_idx = DIMENSIONS.index("Wealth")

    print("\n[ Generation 0: Initial Society ]")
    df_meta, exposures, personalities, affinities, _ = generate_society(config)
    influence = torch.tensor(df_meta["Influence"].values)

    # Use Threshold Graph!
    G_initial = build_threshold_graph(exposures, influence, threshold=0.90)

    # Run Louvain Algorithm
    partition_init = community_louvain.best_partition(G_initial)
    modularity_init = community_louvain.modularity(partition_init, G_initial)

    # Filter out tiny communities (e.g., pairs of 2 people) to count real tribes
    counts_init = np.bincount(list(partition_init.values()))
    num_communities_init = sum(counts_init > 10)

    print(f"Initial Network Modularity (Q): {modularity_init:.4f}")
    print(f"Initial Number of Major Communities (>10 agents): {num_communities_init}")

    print(f"\n[ Evolving Society ({config.evolution_generations} Generations) ]")
    evolver = SocietyEvolution(config, df_meta, exposures, personalities)
    df_meta, exposures_final, personalities_final = evolver.evolve()
    influence_final = torch.tensor(df_meta["Influence"].values)

    print("\n[ Generation 50: Final Society ]")
    G_final = build_threshold_graph(exposures_final, influence_final, threshold=0.90)

    partition_final = community_louvain.best_partition(G_final)
    modularity_final = community_louvain.modularity(partition_final, G_final)

    counts_final = np.bincount(list(partition_final.values()))
    num_communities_final = sum(counts_final > 10)

    print(f"Final Network Modularity (Q): {modularity_final:.4f}")
    print(f"Final Number of Major Communities (>10 agents): {num_communities_final}")

    diff = modularity_final - modularity_init

    print("\n[ Analysis ]")
    print(
        "Note: Modularity (Q) > 0.3 typically indicates strong, dense community structure (Echo Chambers)."
    )
    print(f"Change in Modularity: {diff:+.4f}")
    if modularity_final > 0.3:
        print("-> The society has fractured into strong, isolated ideological bubbles.")
    else:
        print("-> The society remains a relatively fluid, interconnected web.")


if __name__ == "__main__":
    run_louvain_modularity_test()
