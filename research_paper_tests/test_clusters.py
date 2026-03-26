import torch
import numpy as np
from generate_society import generate_society
from schema import SimConfig
from sklearn.cluster import KMeans

def analyze_cluster_divergence():
    config = SimConfig(num_agents=5000, seed=42)
    df_meta, exposures, personalities, affinities, adj = generate_society(config)
    
    pers_np = personalities.numpy()
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    # Simulate 10 clusters/communities using KMeans in personality space
    # (In reality clusters are topological, but topology follows personality homophily)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pers_np)
    
    print("\n--- Cluster Aggregate Analysis (10 Clusters) ---")
    cluster_neuro_means = []
    for c in range(10):
        mask = (clusters == c)
        c_mean = pers_np[mask, 4].mean()
        cluster_neuro_means.append(c_mean)
        print(f"Cluster {c} (N={mask.sum()}): Mean Neuroticism = {c_mean:.3f}")
        
    print(f"\nCluster Mean Spread: {min(cluster_neuro_means):.3f} to {max(cluster_neuro_means):.3f}")
    print(f"Standard Deviation of Cluster Means: {np.std(cluster_neuro_means):.3f}")

if __name__ == "__main__":
    analyze_cluster_divergence()
