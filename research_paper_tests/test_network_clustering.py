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

from generate_society import create_topology, apply_triadic_closure
from schema import SimConfig

def calculate_average_clustering(adj):
    """
    Calculates average clustering coefficient for a sparse binary adjacency matrix.
    C_i = (triangles) / (triples)
    """
    N = adj.shape[0]
    adj = adj.coalesce()
    
    # Binary version
    indices = adj.indices()
    vals = torch.ones_like(adj.values())
    A = torch.sparse_coo_tensor(indices, vals, size=(N, N)).coalesce()
    
    # Degrees
    k = torch.sparse.sum(A, dim=1).to_dense()
    
    # Triples centered on i: k_i * (k_i - 1)
    # For undirected: k_i * (k_i - 1) / 2. 
    # Our graph is directed, but clustering usually assumes undirected or cycles.
    # Let's use the undirected formula on the symmetric version for a standard 'cohesion' metric.
    
    # Symmetric A
    A_sym = (A + A.t()).coalesce()
    A_sym = torch.sparse_coo_tensor(A_sym.indices(), torch.ones_like(A_sym.values()), size=(N, N)).coalesce()
    
    k_sym = torch.sparse.sum(A_sym, dim=1).to_dense()
    triples = k_sym * (k_sym - 1) / 2.0
    
    # Triangles: diag(A^3) / 2
    # This is expensive for sparse mm. Let's do a sample or a more efficient sparse way.
    # For N=1000, A^2 is fine.
    A2 = torch.sparse.mm(A_sym, A_sym).coalesce()
    
    # Triangle count for node i: (A2 * A_sym).sum(dim=1) / 2
    # Element-wise multiply sparse matrices is tricky in PyTorch. 
    # We can do it on the indices.
    
    # Find intersection of A2 indices and A_sym indices
    # Or just loop over the non-zero indices of A_sym (the edges)
    # A_sym[i,j] is 1. We want to know how many k exist s.t. A_sym[i,k]=1 and A_sym[j,k]=1.
    # This is exactly (A_sym @ A_sym)[i,j].
    
    A2_indices = A2.indices()
    A2_values = A2.values()
    
    # We only care about (A2)_{ij} where (A_sym)_{ij} is also 1.
    # Let's convert A_sym to a set of tuples for fast lookup
    edges = set(zip(A_sym.indices()[0].tolist(), A_sym.indices()[1].tolist()))
    
    triangles_per_node = torch.zeros(N)
    for idx in range(len(A2_values)):
        i, j = A2_indices[0, idx].item(), A2_indices[1, idx].item()
        if (i, j) in edges:
            val = A2_values[idx]
            triangles_per_node[i] += val
            
    # Each triangle (i,j,k) is counted twice for each node (as ij and ik)
    # So we divide by 2
    triangles_per_node = triangles_per_node / 2.0
    
    # C_i = triangles / triples
    clustering_coeffs = torch.where(triples > 0, triangles_per_node / triples, torch.zeros_like(triples))
    
    return clustering_coeffs.mean().item(), clustering_coeffs.numpy()

def test_network_clustering():
    print("--- Testing 2-Stage Topology: Triadic Closure & Community Cohesion ---")
    
    # 1. Configuration
    config = SimConfig(
        num_agents=1000,
        seed=42,
        base_connections=10,
        triadic_closure_prob=0.3, # 30% prob of closure
        triadic_closure_iterations=1
    )
    
    # 2. Setup Data for Topology
    exposures = torch.randn(config.num_agents, 12)
    personalities = torch.sigmoid(torch.randn(config.num_agents, 5))
    influence_scores = np.random.lognormal(mean=1.0, sigma=0.5, size=config.num_agents)
    
    # 3. STAGE 1: Backbone
    print("Generating Stage 1 Backbone...")
    # We'll manually call the internal parts to get the backbone
    N = config.num_agents
    inf_mean = np.mean(influence_scores)
    k_array = np.clip((influence_scores / inf_mean) * config.base_connections, 1, 100).astype(int)
    features = torch.cat([exposures, personalities], dim=1)
    features_norm = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)
    
    # Small scale backbone generation
    sim = torch.mm(features_norm, features_norm.T)
    sim = torch.pow((sim + 1.0) / 2.0, 2.0)
    inf_tensor = torch.tensor(influence_scores, dtype=torch.float32)
    prob_matrix = sim * (inf_tensor / inf_tensor.mean()).unsqueeze(0)
    
    indices_list = []
    for i in range(N):
        prob_matrix[i, i] = 0.0
        k = k_array[i]
        sampled = torch.multinomial(prob_matrix[i] + 1e-9, k, replacement=False)
        for s in sampled:
            indices_list.append([i, s.item()])
            
    indices_tensor = torch.tensor(indices_list).t()
    backbone_adj = torch.sparse_coo_tensor(indices_tensor, torch.ones(indices_tensor.shape[1]), size=(N, N)).coalesce()
    
    # 4. STAGE 2: Triadic Closure
    print("Applying Stage 2 Triadic Closure...")
    refined_adj = apply_triadic_closure(config, backbone_adj)
    
    # 5. Calculate Metrics
    print("Calculating Clustering Coefficients...")
    avg_c_base, c_list_base = calculate_average_clustering(backbone_adj)
    avg_c_refined, c_list_refined = calculate_average_clustering(refined_adj)
    
    increase = (avg_c_refined / avg_c_base - 1) * 100
    
    print(f"\nResults:")
    print(f"Average Clustering (Stage 1 Backbone): {avg_c_base:.4f}")
    print(f"Average Clustering (Stage 2 Refined):  {avg_c_refined:.4f}")
    print(f"Community Cohesion Increase: {increase:.2f}%")
    
    # 6. Density check
    dens_base = backbone_adj._nnz() / (N * (N-1))
    dens_refined = refined_adj._nnz() / (N * (N-1))
    print(f"Network Density (Stage 1): {dens_base:.4f}")
    print(f"Network Density (Stage 2): {dens_refined:.4f}")

    # 7. Visualization
    plt.figure(figsize=(10, 6))
    sns.kdeplot(c_list_base, color="blue", label=f"Stage 1 (Backbone, C={avg_c_base:.3f})", fill=True)
    sns.kdeplot(c_list_refined, color="green", label=f"Stage 2 (Refined, C={avg_c_refined:.3f})", fill=True)
    plt.title("Distribution of Local Clustering Coefficients\n(Community Cohesion)", fontsize=14)
    plt.xlabel("Clustering Coefficient [0.0 - 1.0]")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    
    output_path = os.path.join(os.path.dirname(__file__), "network_clustering_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    test_network_clustering()
