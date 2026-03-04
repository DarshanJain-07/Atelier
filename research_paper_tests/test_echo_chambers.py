import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from schema import SimConfig, DIMENSIONS
from generate_society import generate_society
from society_evolution import SocietyEvolution
import networkx as nx

def build_knn_graph(exposures, influence, k=10):
    """
    Builds a social network where each agent connects to 'k' other agents.
    Connection probability is based on a mix of Homophily (Cosine Similarity of ideology)
    and Preferential Attachment (Influence of the target node).
    """
    print(f"Building {k}-NN Social Graph for {len(exposures)} agents...")
    
    # Calculate cosine similarity between all agents (Ideological Homophily)
    # exposures is (N, 12).
    similarity_matrix = cosine_similarity(exposures.numpy())
    
    # Scale similarity by target agent's influence
    # Broadcasting influence across rows (each agent looks at the same target influences)
    influence_np = influence.numpy()
    weighted_matrix = similarity_matrix * influence_np
    
    # Prevent self-connections
    np.fill_diagonal(weighted_matrix, -np.inf)
    
    # Find top K indices for each agent
    top_k_indices = np.argsort(weighted_matrix, axis=1)[:, -k:]
    
    # Build NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(len(exposures)))
    
    edges = []
    for i in range(len(exposures)):
        for j in top_k_indices[i]:
            edges.append((i, j))
            
    G.add_edges_from(edges)
    return G

def run_echo_chamber_analysis():
    print("--- Running Echo Chamber (Assortativity) Analysis ---")
    config = SimConfig(
        num_agents=10000, # Lower agent count for faster N^2 matrix calculation
        seed=69, 
        enable_evolution=True, 
        evolution_generations=30,
        use_ideological_drift=True,
        elite_influence_drift_chance=0.05,
        record_history=False
    )
    config.wealth_dim_idx = DIMENSIONS.index("Wealth")
    dim_idx_to_test = DIMENSIONS.index("Fairness")
    
    print("\n[ Generation 0: Initial Society ]")
    df_meta, exposures, personalities, affinities = generate_society(config)
    influence = torch.tensor(df_meta["Influence"].values)
    
    # 1. Build initial graph
    G_initial = build_knn_graph(exposures, influence, k=5)
    
    # 2. Assign ideology as a node attribute
    fairness_values_init = exposures[:, dim_idx_to_test].numpy()
    dict_fairness_init = {i: float(fairness_values_init[i]) for i in range(config.num_agents)}
    nx.set_node_attributes(G_initial, dict_fairness_init, "fairness")
    
    # 3. Calculate Assortativity
    assort_init = nx.numeric_assortativity_coefficient(G_initial, "fairness")
    print(f"Initial Network Assortativity on Fairness: {assort_init:.3f}")
    
    print(f"\n[ Evolving Society ({config.evolution_generations} Generations) ]")
    evolver = SocietyEvolution(config, df_meta, exposures, personalities)
    df_meta, exposures_final, personalities_final = evolver.evolve()
    influence_final = torch.tensor(df_meta["Influence"].values)
    
    print("\n[ Generation 30: Final Society ]")
    # 1. Build final graph
    G_final = build_knn_graph(exposures_final, influence_final, k=5)
    
    # 2. Assign final ideology as a node attribute
    fairness_values_final = exposures_final[:, dim_idx_to_test].numpy()
    dict_fairness_final = {i: float(fairness_values_final[i]) for i in range(config.num_agents)}
    nx.set_node_attributes(G_final, dict_fairness_final, "fairness")
    
    # 3. Calculate final Assortativity
    assort_final = nx.numeric_assortativity_coefficient(G_final, "fairness")
    print(f"Final Network Assortativity on Fairness: {assort_final:.3f}")
    
    diff = assort_final - assort_init
    trend = "FORMING ECHO CHAMBERS" if diff > 0 else "DISSOLVING ECHO CHAMBERS"
    print(f"Change in Assortativity: {diff:+.3f} -> {trend}")

if __name__ == "__main__":
    run_echo_chamber_analysis()
