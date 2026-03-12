import numpy as np
import torch
import random
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig, DIMENSIONS
from generate_society import generate_society
from society_evolution import SocietyEvolution
import networkx as nx

def build_probabilistic_graph(exposures, influence, k=10):
    """
    Builds a social network probabilistically based on Preferential Attachment and Homophily.
    """
    print(f"Building Probabilistic Social Graph for {len(exposures)} agents...")
    similarity_matrix = cosine_similarity(exposures.numpy())
    influence_np = influence.numpy()

    # Scale similarity by target agent's influence
    # Make similarity strictly positive for probabilities (shift from [-1, 1] to [0, 1])
    similarity_positive = (similarity_matrix + 1.0) / 2.0
    weighted_matrix = similarity_positive * influence_np
    np.fill_diagonal(weighted_matrix, 0)
    
    # Normalize probabilities per row
    row_sums = weighted_matrix.sum(axis=1, keepdims=True) + 1e-8
    prob_matrix = weighted_matrix / row_sums

    G = nx.Graph()
    G.add_nodes_from(range(len(exposures)))

    edges = []
    for i in range(len(exposures)):
        # Probabilistically sample k connections
        probs = prob_matrix[i]
        sampled_indices = np.random.choice(len(exposures), size=k, replace=False, p=probs)
        for j in sampled_indices:
            edges.append((i, j))

    G.add_edges_from(edges)
    return G

def rewire_graph(G, exposures, influence, rewire_fraction=0.1):
    """
    Agents drop connections with people who have diverged from their beliefs 
    and form new connections with high-influence aligned agents.
    """
    exposures_np = exposures.numpy()
    influence_np = influence.numpy()
    nodes = list(G.nodes())
    num_to_rewire = int(len(nodes) * rewire_fraction)
    
    nodes_to_rewire = random.sample(nodes, num_to_rewire)
    
    for u in nodes_to_rewire:
        neighbors = list(G.neighbors(u))
        if not neighbors: continue
        
        # Find neighbor with lowest similarity
        u_exp = exposures_np[u]
        
        # Fast cosine similarity for neighbors
        neighbor_exps = exposures_np[neighbors]
        sims = np.dot(neighbor_exps, u_exp) / (np.linalg.norm(neighbor_exps, axis=1) * np.linalg.norm(u_exp) + 1e-8)
        worst_idx = np.argmin(sims)
        worst_neighbor = neighbors[worst_idx]
        
        # Drop the edge
        G.remove_edge(u, worst_neighbor)
        
        # Find a new candidate to connect to:
        candidates = random.sample(nodes, min(50, len(nodes)))
        cand_exps = exposures_np[candidates]
        cand_sims = np.dot(cand_exps, u_exp) / (np.linalg.norm(cand_exps, axis=1) * np.linalg.norm(u_exp) + 1e-8)
        cand_scores = cand_sims * influence_np[candidates]
        
        # Avoid self and existing neighbors
        for i, c in enumerate(candidates):
            if c == u or G.has_edge(u, c):
                cand_scores[i] = -float('inf')
                
        best_candidate = candidates[np.argmax(cand_scores)]
        G.add_edge(u, best_candidate)

    return G

def run_echo_chamber_analysis():
    print("--- Running Echo Chamber (Assortativity) & Dynamic Rewiring Analysis ---")
    config = SimConfig(
        num_agents=5000,
        seed=69,
        enable_evolution=True,
        evolution_generations=30,
        use_ideological_drift=True,
        elite_influence_drift_chance=0.05,
        record_history=True,
    )
    config.wealth_dim_idx = DIMENSIONS.index("Wealth")
    dim_idx_to_test = DIMENSIONS.index("Fairness")

    print("\n[ Generation 0: Initial Society ]")
    df_meta, exposures, personalities, affinities, _ = generate_society(config)
    influence = torch.tensor(df_meta["Influence"].values)

    # 1. Build probabilistic initial graph
    G = build_probabilistic_graph(exposures, influence, k=5)

    # 2. Assign ideology as a node attribute
    fairness_values_init = exposures[:, dim_idx_to_test].numpy()
    dict_fairness_init = {i: float(fairness_values_init[i]) for i in range(config.num_agents)}
    nx.set_node_attributes(G, dict_fairness_init, "fairness")

    # 3. Calculate Assortativity
    assort_init = nx.numeric_assortativity_coefficient(G, "fairness")
    print(f"Initial Network Assortativity on Fairness: {assort_init:.3f}")

    print(f"\n[ Evolving Society ({config.evolution_generations} Generations with Dynamic Rewiring) ]")
    evolver = SocietyEvolution(config, df_meta, exposures, personalities)
    
    # Store history for UMAP
    history_exposures = [exposures.clone().numpy()]

    for gen in range(1, config.evolution_generations + 1):
        evolver.apply_inheritance()
        evolver.apply_reinvestment()
        evolver.apply_economic_shocks(gen)
        evolver.apply_mobility()
        evolver.apply_ideological_drift()

        if getattr(config, "use_dynamic_classes", False):
            evolver.reassign_classes()

        evolver.exposures[:, evolver.wealth_idx] = torch.clamp(
            evolver.exposures[:, evolver.wealth_idx], min=0.0, max=1e6
        )
        
        # Dynamic graph rewiring
        G = rewire_graph(G, evolver.exposures, torch.tensor(evolver.metadata["Influence"].values), rewire_fraction=0.1)

        if gen % 10 == 0:
            history_exposures.append(evolver.exposures.clone().numpy())

    df_meta, exposures_final, personalities_final = evolver.metadata, evolver.exposures, evolver.personalities
    influence_final = torch.tensor(df_meta["Influence"].values)

    print("\n[ Generation 30: Final Society ]")
    # 2. Assign final ideology as a node attribute
    fairness_values_final = exposures_final[:, dim_idx_to_test].numpy()
    dict_fairness_final = {i: float(fairness_values_final[i]) for i in range(config.num_agents)}
    nx.set_node_attributes(G, dict_fairness_final, "fairness")

    # 3. Calculate final Assortativity
    assort_final = nx.numeric_assortativity_coefficient(G, "fairness")
    print(f"Final Network Assortativity on Fairness: {assort_final:.3f}")

    diff = assort_final - assort_init
    trend = "FORMING ECHO CHAMBERS" if diff > 0 else "DISSOLVING ECHO CHAMBERS"
    print(f"Change in Assortativity: {diff:+.3f} -> {trend}")

    # Generate UMAP Trajectory
    try:
        import umap
        import matplotlib.pyplot as plt
        import os
        
        print("\n[ Generating UMAP Trajectory Plot ]")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        
        # Fit UMAP on Generation 0
        reducer.fit(history_exposures[0])
        
        fig, axes = plt.subplots(1, len(history_exposures), figsize=(15, 5))
        
        for i, (ax, data) in enumerate(zip(axes, history_exposures)):
            embedding = reducer.transform(data)
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=data[:, dim_idx_to_test], cmap='coolwarm', s=2, alpha=0.6)
            ax.set_title(f"Generation {i * 10}")
            ax.axis('off')
            
        fig.suptitle("UMAP Projection of Ideological Drift (Colored by Fairness)", fontsize=14)
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.05, pad=0.04)
        cbar.set_label('Fairness Score')
        
        output_path = os.path.join(os.path.dirname(__file__), "umap_echo_chambers.png")
        plt.savefig(output_path, bbox_inches='tight')
        print(f"[!] Saved UMAP Plot to: {output_path}")

    except ImportError:
        print("\n[!] umap-learn or matplotlib not installed. Skipping UMAP generation.")


if __name__ == "__main__":
    run_echo_chamber_analysis()
