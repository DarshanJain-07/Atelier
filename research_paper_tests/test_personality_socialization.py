import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_society import generate_society
from schema import SimConfig

def test_personality_socialization():
    print("--- Testing 2-Stage Personality: Socialization (Nurture) ---")
    
    # 1. Configuration
    config = SimConfig(
        num_agents=1000,
        seed=42,
        use_network_topology=True,
        base_connections=15,
        personality_socialization_gain=0.4 # Strong socialization (40% drift)
    )
    
    # 2. RUN GENERATION
    # We want to compare the original (Innate) and socialized results
    print(f"Generating Society with Socialization Gain = {config.personality_socialization_gain}...")
    
    # To get the Stage 1 (Innate) baseline, we temporarily disable socialization
    config.personality_socialization_gain = 0.0
    _, _, personalities_innate, _, adj = generate_society(config)
    
    # Now run Stage 2 manually on the same data
    config.personality_socialization_gain = 0.4
    gain = config.personality_socialization_gain
    
    local_mean = torch.sparse.mm(adj, personalities_innate)
    personalities_socialized = (1.0 - gain) * personalities_innate + gain * local_mean
    
    # 3. ANALYSIS: Intra-Cluster Similarity
    def calculate_neighbor_similarity(p_tensor, adj_matrix):
        # Euclidean distance to neighbors' mean personality
        local_mean = torch.sparse.mm(adj_matrix, p_tensor)
        distances = torch.norm(p_tensor - local_mean, dim=1)
        return distances.numpy()

    print("Calculating Neighbor Alignment...")
    base_distances = calculate_neighbor_similarity(personalities_innate, adj)
    social_distances = calculate_neighbor_similarity(personalities_socialized, adj)
    
    avg_base_dist = np.mean(base_distances)
    avg_social_dist = np.mean(social_distances)
    reduction = (1 - (avg_social_dist / avg_base_dist)) * 100
    
    print(f"\nResults (Euclidean Distance in 5D Big-Five Space):")
    print(f"Average Neighbor Distance (Innate):    {avg_base_dist:.4f}")
    print(f"Average Neighbor Distance (Socialized): {avg_social_dist:.4f}")
    print(f"Socialization Homogeneity Increase: {reduction:.2f}%")

    # 4. Global Variance Check
    # Does socialization cause global trait collapse?
    global_std_base = personalities_innate.std().item()
    global_std_social = personalities_socialized.std().item()
    print(f"Global Trait Std Dev (Innate):    {global_std_base:.4f}")
    print(f"Global Trait Std Dev (Socialized): {global_std_social:.4f}")

    # 5. Visualization: Distribution of Distances
    plt.figure(figsize=(10, 6))
    sns.histplot(base_distances, color="blue", label="Stage 1 (Innate Baseline)", kde=True, alpha=0.5)
    sns.histplot(social_distances, color="purple", label="Stage 2 (Socialized/Nurture)", kde=True, alpha=0.5)
    
    plt.title("Personality Socialization: Reduction in Personality Friction", fontsize=14)
    plt.xlabel("Euclidean Distance to Local Neighborhood Mean (Personality)")
    plt.ylabel("Agent Count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    output_path = os.path.join(os.path.dirname(__file__), "personality_socialization_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    test_personality_socialization()
