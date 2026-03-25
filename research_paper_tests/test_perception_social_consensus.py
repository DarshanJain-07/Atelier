import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from scipy.stats import pearsonr

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_society import generate_society
from cognitive_engine import CognitiveEngine
from schema import SimConfig, DIMENSION_INDICES

def test_perception_social_consensus():
    print("--- Testing 2-Stage Perception: Socially Constructed Reality ---")
    
    # 1. Configuration
    # We want a strong network and clear distortion to see the consensus effect
    config = SimConfig(
        num_agents=2000,
        seed=42,
        use_signal_distortion=True,
        distortion_max_noise=0.6,
        distortion_neurotic_gain=1.0,
        use_network_topology=True,
        base_connections=15,
        perception_social_consensus_gain=0.3  # 30% consensus blend
    )
    
    print(f"Generating Society with {config.num_agents} agents...")
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    
    if adjacency_matrix is None:
        print("Error: Adjacency matrix is None. Network topology is required for this test.")
        return

    # 2. Define the Objective Event
    # A negative event on Physical Safety (-0.5)
    world_tensor_raw = torch.zeros(1, 12)
    safety_idx = DIMENSION_INDICES["Physical_Safety"]
    world_tensor_raw[0, safety_idx] = -0.5
    
    engine = CognitiveEngine(config)
    
    # 3. RUN PERCEPTION - BASELINE (No Social Consensus)
    # We temporarily set the gain to 0 to get the individual-only baseline
    original_gain = config.perception_social_consensus_gain
    config.perception_social_consensus_gain = 0.0
    
    # We only need the distorted_world, but run() returns context_vector etc.
    # To get the raw distorted world, we can intercept the logic or just look at the output
    # since context_vector = perceived_world * attention_weights.
    # For this test, let's call the internal logic to get the raw perceived values.
    
    print("Running Perception: Stage 1 (Individual Baseline)...")
    with torch.no_grad():
        baseline_perceived = engine.distort_signal(
            world_tensor_raw.squeeze(),
            personalities
        )
    
    # 4. RUN PERCEPTION - 2-STAGE (With Social Consensus)
    config.perception_social_consensus_gain = original_gain
    print(f"Running Perception: Stage 2 (Social Consensus, Gain={original_gain})...")
    
    with torch.no_grad():
        # Re-run Stage 1 to ensure same random seed/noise if possible, 
        # but the engine.distort_signal uses torch.randn inside.
        # To be perfectly fair, we manually apply the consensus to the same baseline_perceived.
        local_consensus = torch.sparse.mm(adjacency_matrix, baseline_perceived)
        consensus_perceived = (1.0 - original_gain) * baseline_perceived + original_gain * local_consensus

    # 5. ANALYSIS: Local Variance & Alignment
    # We want to see if neighbors are MORE similar in the Consensus stage than in the Baseline stage.
    
    def calculate_neighbor_similarity(perceived_tensor, adj_matrix):
        # For each agent, calculate the average distance to their neighbors
        # We'll use Euclidean distance in the 12D space
        
        # This is expensive for dense, but adjacency is sparse.
        # Let's calculate the "Local Mean" first
        local_mean = torch.sparse.mm(adj_matrix, perceived_tensor)
        
        # Distance from individual perception to their local neighborhood mean
        distances = torch.norm(perceived_tensor - local_mean, dim=1)
        return distances.numpy()

    print("Calculating Neighbor Alignment...")
    baseline_distances = calculate_neighbor_similarity(baseline_perceived, adjacency_matrix)
    consensus_distances = calculate_neighbor_similarity(consensus_perceived, adjacency_matrix)
    
    avg_base_dist = np.mean(baseline_distances)
    avg_cons_dist = np.mean(consensus_distances)
    reduction = (1 - (avg_cons_dist / avg_base_dist)) * 100
    
    print(f"\nResults:")
    print(f"Average Distance to Neighbors (Baseline): {avg_base_dist:.4f}")
    print(f"Average Distance to Neighbors (Consensus): {avg_cons_dist:.4f}")
    print(f"Perception Alignment Increase: {reduction:.2f}%")

    # 6. Global Polarization Check (Standard Deviation of Safety Perception)
    # Does consensus increase or decrease global variance? 
    # Usually, it reduces global variance unless the network is highly clustered/polarized.
    base_safety = baseline_perceived[:, safety_idx].numpy()
    cons_safety = consensus_perceived[:, safety_idx].numpy()
    
    base_std = np.std(base_safety)
    cons_std = np.std(cons_safety)
    
    print(f"Global Std Dev of Safety Perception (Baseline): {base_std:.4f}")
    print(f"Global Std Dev of Safety Perception (Consensus): {cons_std:.4f}")

    # 7. Visualization: Distribution of Distances
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.histplot(baseline_distances, color="blue", label="Baseline (Individual)", kde=True, alpha=0.5)
    sns.histplot(consensus_distances, color="red", label="2-Stage (Social Consensus)", kde=True, alpha=0.5)
    plt.title("Local Perception Dissimilarity\n(Distance to Neighbors)")
    plt.xlabel("Euclidean Distance to Local Neighborhood Mean")
    plt.ylabel("Agent Count")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(base_safety, color="blue", label="Baseline", shade=True)
    sns.kdeplot(cons_safety, color="red", label="Consensus", shade=True)
    plt.axvline(world_tensor_raw[0, safety_idx].item(), color='black', linestyle='--', label='Objective')
    plt.title("Global Distribution of Perceived Safety")
    plt.xlabel("Perceived Magnitude (Safety Dimension)")
    plt.ylabel("Density")
    plt.legend()
    
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "perception_consensus_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    test_perception_social_consensus()
