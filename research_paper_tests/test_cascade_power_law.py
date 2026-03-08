import numpy as np
import torch
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import linregress

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig
from generate_society import generate_society
from cognitive_engine import CognitiveEngine


def build_directed_knn_graph(exposures, influence, k=30):
    # Directed graph: edge from j to i means information flows from j to i
    # (i.e., i follows j)
    print(f"Building Directed {k}-NN Social Graph for {len(exposures)} agents...")
    similarity_matrix = cosine_similarity(exposures.numpy())
    influence_np = influence.numpy()

    # Probability i follows j:
    weighted_matrix = similarity_matrix * influence_np  # j's influence broadcasted to i
    np.fill_diagonal(weighted_matrix, -np.inf)
    top_k_indices = np.argsort(weighted_matrix, axis=1)[:, -k:]

    G = nx.DiGraph()
    G.add_nodes_from(range(len(exposures)))
    edges = []
    for i in range(len(exposures)):
        for j in top_k_indices[i]:
            # i follows j. So information flows from j to i.
            edges.append((j, i))

    G.add_edges_from(edges)
    return G


def test_cascade_power_law():
    print("--- Running Cascade Size Power-Law Analysis ---")
    config = SimConfig(num_agents=10000, seed=78)

    df_meta, exposures, personalities, affinities, _ = generate_society(config)
    influence = torch.tensor(df_meta["Influence"].values)

    G = build_directed_knn_graph(exposures, influence, k=config.cascade_knn_k)

    cog_engine = CognitiveEngine(config)

    num_seeds = 4000
    np.random.seed(42)
    seed_indices = np.random.choice(config.num_agents, num_seeds, replace=False)

    threshold = config.cascade_threshold
    cascade_sizes = []

    print(f"\n[ Simulating Cascades for {num_seeds} seeds ]")
    for idx, seed in enumerate(seed_indices):
        thought_vector = exposures[seed]

        # Calculate how everyone in the network inherently reacts to this thought
        # We increase urgency slightly to stimulate engagement
        _, _, engagement_scores, _ = cog_engine.run(
            world_tensor_raw=thought_vector.unsqueeze(0),
            urgency=0.6,
            is_personal=False,
            exposures=exposures,
            personalities=personalities,
            agent_affinities=affinities,
        )

        will_infect = (engagement_scores > threshold).numpy()

        # BFS Cascade Propagation
        infected = {seed}
        frontier = {seed}

        while frontier:
            new_frontier = set()
            for u in frontier:
                for v in G.successors(u):
                    if v not in infected and will_infect[v]:
                        new_frontier.add(v)
            infected.update(new_frontier)
            frontier = new_frontier

        cascade_sizes.append(len(infected))

        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{num_seeds} seeds...")

    sizes = np.array(cascade_sizes)
    # Filter out cascades of size 1 (where the seed couldn't influence a single person)
    active_cascades = sizes[sizes > 1]

    print("\n[ Results ]")
    print(f"Total Seeds: {num_seeds}")
    print(f"Cascades > 1 hop: {len(active_cascades)}")
    print(f"Max Cascade Size: {np.max(sizes)} agents")

    if len(active_cascades) < 10:
        print(
            "Not enough active cascades to plot. The network is highly resistant to virality."
        )
        return

    print("\n[ Plotting Log-Log Distribution ]")
    # Calculate frequency of each size
    unique_sizes, counts = np.unique(active_cascades, return_counts=True)

    # Sort for plotting
    sort_idx = np.argsort(unique_sizes)
    x = unique_sizes[sort_idx]
    y = counts[sort_idx]

    log_x = np.log10(x)
    log_y = np.log10(y)

    # Linear regression on log-log to find the power-law exponent (alpha)
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)

    plt.figure(figsize=(8, 6))
    plt.scatter(log_x, log_y, color="blue", alpha=0.7, label="Simulated Cascades")
    plt.plot(
        log_x,
        intercept + slope * log_x,
        color="red",
        linestyle="--",
        label=f"Fit: $\\alpha$ = {slope:.2f}\n$R^2$ = {r_value**2:.2f}",
    )

    plt.title("Log-Log Plot of Viral Cascade Sizes")
    plt.xlabel("Log10(Cascade Size)")
    plt.ylabel("Log10(Frequency)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(os.path.dirname(__file__), "cascade_power_law.png")
    plt.savefig(output_path)
    print(f"Plot saved to: {output_path}")

    print(f"Power-Law Exponent (Slope): {slope:.2f}")
    if slope < -0.5 and r_value**2 > 0.5:
        print(
            "-> SUCCESS: The simulation exhibits power-law properties typical of real human social networks!"
        )
    else:
        print("-> WARNING: Distribution may not strongly follow a power-law.")


if __name__ == "__main__":
    test_cascade_power_law()
