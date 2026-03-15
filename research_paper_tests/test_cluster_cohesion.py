import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from schema import PSYCH_PROJECTION, SimConfig

def test_cluster_cohesion():
    print("--- Testing Cluster Cohesion (Silhouette & Davies-Bouldin) ---")

    # 1. Configuration
    config = SimConfig(num_agents=2000, emotion_temperature=0.1)
    
    # 2. Generate Society
    print("Generating Society...")
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    
    # 3. Simulate a highly polarizing event to create distinct emotional sub-factions
    # e.g., A controversial policy that negatively impacts wealth and fairness but appeals to in-group identity
    world_tensor_raw = torch.zeros(1, 12)
    world_tensor_raw[0, 0] = -0.5 # Wealth
    world_tensor_raw[0, 4] = -0.6 # Fairness
    world_tensor_raw[0, 5] = 0.5  # In_Group
    
    urgency = 0.8
    is_personal = True

    print("\nRunning Cognitive Engine...")
    agent_memory = torch.zeros_like(exposures)
    cog_engine = CognitiveEngine(config)
    ctx, att, eng, agent_memory = cog_engine.run(
        world_tensor_raw=world_tensor_raw,
        urgency=urgency,
        is_personal=is_personal,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=agent_memory,
    )
    
    device = ctx.device
    projection_matrix = PSYCH_PROJECTION.to(device)
    final_emotions = torch.matmul(ctx, projection_matrix)
    final_emotions = F.softmax(final_emotions / max(0.01, config.emotion_temperature), dim=1)
    
    emotions_np = final_emotions.cpu().numpy()

    # 4. Determine optimal number of clusters using Silhouette Score
    k_values = range(2, 9)
    silhouette_scores = []
    db_scores = []

    print("\nRunning K-Means Clustering on 8D Emotion Vectors...")
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(emotions_np)
        
        sil = silhouette_score(emotions_np, labels)
        db = davies_bouldin_score(emotions_np, labels)
        
        silhouette_scores.append(sil)
        db_scores.append(db)
        
        print(f"  k={k} -> Silhouette Score: {sil:.4f}, Davies-Bouldin Index: {db:.4f}")

    # Optimal K based on max Silhouette
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"\nOptimal number of emotional sub-factions (clusters): {optimal_k}")
    print(f"Max Silhouette Score: {max(silhouette_scores):.4f} (Higher is better, > 0.5 is good)")
    print(f"Davies-Bouldin at Optimal K: {db_scores[np.argmax(silhouette_scores)]:.4f} (Lower is better)")

    # Run with optimal K for plotting
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(emotions_np)

    # 5. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Metrics over K
    ax1.plot(k_values, silhouette_scores, marker='o', label='Silhouette Score (Higher = Better)', color='b')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Silhouette Score', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(k_values, db_scores, marker='s', label='Davies-Bouldin (Lower = Better)', color='r')
    ax1_twin.set_ylabel('Davies-Bouldin Index', fontsize=12, color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    ax1.set_title('Cluster Cohesion Metrics vs. k', fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: PCA Visualization of Clusters
    pca = PCA(n_components=2)
    emotions_pca = pca.fit_transform(emotions_np)
    
    scatter = ax2.scatter(emotions_pca[:, 0], emotions_pca[:, 1], c=labels, cmap='tab10', alpha=0.6, s=15)
    ax2.set_title(f'PCA Projection of {optimal_k} Emotional Sub-factions', fontsize=14, fontweight="bold")
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    plt.colorbar(scatter, ax=ax2, label='Cluster ID')

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "cluster_cohesion_metrics.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    test_cluster_cohesion()
