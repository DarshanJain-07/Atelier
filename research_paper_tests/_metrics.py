import networkx as nx
import numpy as np
import torch

from schema import DIMENSION_INDICES


def gini(values) -> float:
    array = np.asarray(values, dtype=np.float64).flatten()
    if array.size == 0:
        return 0.0
    if np.min(array) < 0:
        array = array - np.min(array)
    array = np.sort(array + 1e-9)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return float(np.sum((2 * index - n - 1) * array) / (n * np.sum(array)))


def bimodality_coefficient(values) -> float:
    array = np.asarray(values, dtype=np.float64).flatten()
    std = array.std()
    if std == 0.0:
        return 0.0
    centered = (array - array.mean()) / std
    skew = np.mean(centered**3)
    kurtosis = np.mean(centered**4)
    if kurtosis == 0.0:
        return 0.0
    return float((skew**2 + 1.0) / kurtosis)


def average_neighbor_distance(features: torch.Tensor, adjacency_matrix: torch.Tensor) -> float:
    local_mean = torch.sparse.mm(adjacency_matrix.coalesce(), features)
    distances = torch.norm(features - local_mean, dim=1)
    return float(distances.mean().item())


def adjacency_to_graph(adjacency_matrix: torch.Tensor) -> nx.Graph:
    adjacency = adjacency_matrix.coalesce()
    rows = adjacency.indices()[0].tolist()
    cols = adjacency.indices()[1].tolist()
    graph = nx.Graph()
    graph.add_edges_from((u, v) for u, v in zip(rows, cols) if u != v)
    return graph


def average_clustering(adjacency_matrix: torch.Tensor) -> float:
    graph = adjacency_to_graph(adjacency_matrix)
    if graph.number_of_edges() == 0:
        return 0.0
    return float(nx.average_clustering(graph))


def mean_edge_cosine_similarity(features: torch.Tensor, adjacency_matrix: torch.Tensor) -> float:
    adjacency = adjacency_matrix.coalesce()
    rows = adjacency.indices()[0]
    cols = adjacency.indices()[1]
    lhs = features[rows]
    rhs = features[cols]
    sim = torch.nn.functional.cosine_similarity(lhs, rhs, dim=1)
    return float(sim.mean().item())


def mean_edge_topology_similarity(
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    adjacency_matrix: torch.Tensor,
) -> float:
    wealth_idx = DIMENSION_INDICES["Wealth"]
    topology_exposures = exposures.clone()
    topology_exposures[:, wealth_idx] = 0.0
    topology_features = torch.cat([topology_exposures, personalities], dim=1)
    return mean_edge_cosine_similarity(topology_features, adjacency_matrix)


def mad_metrics(features: torch.Tensor, partition: dict[int, int]) -> dict[str, float]:
    n = features.shape[0]
    if n <= 1:
        return {"mad": 0.0, "mad_within": 0.0, "mad_between": 0.0, "madgap": 0.0, "gdr": 0.0}

    distances = torch.cdist(features, features, p=2)

    communities = torch.tensor([partition.get(i, -1) for i in range(n)], device=features.device)
    same_community = communities.unsqueeze(0) == communities.unsqueeze(1)

    eye_mask = torch.eye(n, dtype=torch.bool, device=features.device)

    mad = float(distances[~eye_mask].mean().item())

    within_mask = same_community & ~eye_mask
    mad_within = float(distances[within_mask].mean().item()) if within_mask.any() else 0.0

    between_mask = ~same_community & ~eye_mask
    mad_between = float(distances[between_mask].mean().item()) if between_mask.any() else 0.0

    madgap = mad_between - mad_within
    gdr = mad_between / mad_within if mad_within > 0 else float("inf")

    return {
        "mad": mad,
        "mad_within": mad_within,
        "mad_between": mad_between,
        "madgap": madgap,
        "gdr": gdr,
    }


def wl_graph_hash(adjacency_matrix: torch.Tensor, iterations: int = 3) -> str:
    """Computes a Weisfeiler-Lehman graph hash for structural comparison."""
    graph = adjacency_to_graph(adjacency_matrix)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return nx.weisfeiler_lehman_graph_hash(graph, iterations=iterations)


def wl_node_hashes(adjacency_matrix: torch.Tensor, iterations: int = 3) -> dict[int, list[str]]:
    """Computes Weisfeiler-Lehman subgraph hashes for node-level structural roles."""
    graph = adjacency_to_graph(adjacency_matrix)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return nx.weisfeiler_lehman_subgraph_hashes(graph, iterations=iterations)


def wl_echo_chamber_structural_similarity(
    adjacency_matrix: torch.Tensor,
    partition: dict[int, int],
    iterations: int = 3,
) -> dict[tuple[int, int], float]:
    """Computes pairwise structural similarity of echo chambers (communities)
    using WL subgraph hashes.
    """
    import itertools
    from collections import Counter

    hashes = wl_node_hashes(adjacency_matrix, iterations=iterations)

    # group hashes by community (use the final iteration hash as representation)
    community_hashes = {}
    for node, comm in partition.items():
        if comm not in community_hashes:
            community_hashes[comm] = []
        # Use the hash from the final iteration to represent the node's structural role
        community_hashes[comm].append(hashes[node][-1])

    similarities = {}
    communities = sorted(list(community_hashes.keys()))

    for c1, c2 in itertools.combinations(communities, 2):
        counter_a = Counter(community_hashes[c1])
        counter_b = Counter(community_hashes[c2])

        all_keys = set(counter_a.keys()).union(counter_b.keys())
        vec_a = np.array([counter_a[k] for k in all_keys], dtype=np.float64)
        vec_b = np.array([counter_b[k] for k in all_keys], dtype=np.float64)

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        sim = 0.0
        if norm_a > 0 and norm_b > 0:
            sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        similarities[(c1, c2)] = sim

    return similarities


def wl_kernel_similarity(adj_a: torch.Tensor, adj_b: torch.Tensor, iterations: int = 3) -> float:
    """Computes a continuous structural similarity score between two graphs using
    a WL Subtree Kernel (dot product of normalized hash frequency vectors).
    """
    from collections import Counter

    hashes_a = wl_node_hashes(adj_a, iterations=iterations)
    hashes_b = wl_node_hashes(adj_b, iterations=iterations)

    # Flatten the hashes for all iterations
    # In WL Subtree Kernel, features are counts of all subtrees across all iterations
    flat_a = [h for node_hashes in hashes_a.values() for h in node_hashes]
    flat_b = [h for node_hashes in hashes_b.values() for h in node_hashes]

    counter_a = Counter(flat_a)
    counter_b = Counter(flat_b)

    all_keys = set(counter_a.keys()).union(counter_b.keys())
    vec_a = np.array([counter_a[k] for k in all_keys], dtype=np.float64)
    vec_b = np.array([counter_b[k] for k in all_keys], dtype=np.float64)

    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

