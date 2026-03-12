import numpy as np
import torch
import scipy.stats as st
from sklearn.metrics.pairwise import cosine_similarity
from schema import SimConfig, DIMENSIONS
from generate_society import generate_society
from society_evolution import SocietyEvolution
import networkx as nx
import community as community_louvain  # python-louvain


def get_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    if n < 2: return np.mean(a), 0.0
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def build_threshold_graph(exposures, influence, threshold=0.85):
    similarity_matrix = cosine_similarity(exposures.numpy())
    np.fill_diagonal(similarity_matrix, -np.inf)

    G = nx.Graph()
    G.add_nodes_from(range(len(exposures)))

    edges = []
    for i in range(len(exposures)):
        for j in np.where(similarity_matrix[i] > threshold)[0]:
            edges.append((i, j))

    G.add_edges_from(edges)
    return G

def run_louvain_modularity_test():
    print("--- Running Network Modularity (Louvain) Analysis (Monte Carlo) ---")
    
    num_runs = 5
    modularity_inits = []
    modularity_finals = []
    diffs = []
    
    for run in range(num_runs):
        seed = 42 + run
        config = SimConfig(
            num_agents=2000,
            seed=seed,
            enable_evolution=True,
            evolution_generations=50,
            use_ideological_drift=True,
            elite_influence_drift_chance=0.20,
            record_history=False,
        )
        config.wealth_dim_idx = DIMENSIONS.index("Wealth")

        df_meta, exposures, personalities, affinities, _ = generate_society(config)
        influence = torch.tensor(df_meta["Influence"].values)

        G_initial = build_threshold_graph(exposures, influence, threshold=0.90)
        partition_init = community_louvain.best_partition(G_initial)
        modularity_init = community_louvain.modularity(partition_init, G_initial)
        modularity_inits.append(modularity_init)

        evolver = SocietyEvolution(config, df_meta, exposures, personalities)
        df_meta, exposures_final, personalities_final = evolver.evolve()
        influence_final = torch.tensor(df_meta["Influence"].values)

        G_final = build_threshold_graph(exposures_final, influence_final, threshold=0.90)
        partition_final = community_louvain.best_partition(G_final)
        modularity_final = community_louvain.modularity(partition_final, G_final)
        modularity_finals.append(modularity_final)
        
        diffs.append(modularity_final - modularity_init)

    m_init, h_init = get_confidence_interval(modularity_inits)
    m_final, h_final = get_confidence_interval(modularity_finals)
    m_diff, h_diff = get_confidence_interval(diffs)

    print("\n[ Analysis ]")
    print("Note: Modularity (Q) > 0.3 typically indicates strong, dense community structure (Echo Chambers).")
    print(f"Initial Network Modularity (Q): {m_init:.4f} ± {h_init:.4f} (95% CI)")
    print(f"Final Network Modularity (Q):   {m_final:.4f} ± {h_final:.4f} (95% CI)")
    print(f"Change in Modularity:           {m_diff:+.4f} ± {h_diff:.4f} (95% CI)")
    
    if m_final > 0.3:
        print("-> The society has fractured into strong, isolated ideological bubbles.")
    else:
        print("-> The society remains a relatively fluid, interconnected web.")

if __name__ == "__main__":
    run_louvain_modularity_test()
