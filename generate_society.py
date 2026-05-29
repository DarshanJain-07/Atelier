import math
import os

import numpy as np
import pandas as pd
import torch

from schema import (
    DIMENSION_INDICES,
    DIMENSIONS,
    PERSONALITY_CORRELATIONS,
    SimConfig,
)
from society_evolution import SocietyEvolution


# ================================
# Mutation Logic
# ================================
def apply_random_mutations(exposures, personality_logits, temperature, seed):
    if temperature <= 0.0:
        return exposures, personality_logits

    rng = torch.Generator()
    rng.manual_seed(seed + 999)

    n_agents = exposures.shape[0]
    num_dims = len(DIMENSIONS)

    mutation_prob = temperature
    mutant_mask = torch.rand(n_agents, generator=rng) < mutation_prob
    mutant_indices = torch.where(mutant_mask)[0]

    if len(mutant_indices) == 0:
        return exposures, personality_logits

    num_changes = math.ceil(3 * temperature)

    for _ in range(num_changes):
        col_indices = torch.randint(0, num_dims, (len(mutant_indices),), generator=rng)
        # Use normal distribution for mutations to preserve bell curve
        random_values = torch.clamp(
            torch.randn(len(mutant_indices), generator=rng) * 0.4, -1.0, 1.0,
        )
        exposures[mutant_indices, col_indices] = random_values

    # Apply additive noise to personality logits
    for _ in range(num_changes):
        col_indices = torch.randint(0, 5, (len(mutant_indices),), generator=rng)
        noise = torch.randn(len(mutant_indices), generator=rng) * 0.5
        personality_logits[mutant_indices, col_indices] += noise

    # ---- Radical Outliers ----
    radical_prob = 0.05 * temperature
    radical_mask = torch.rand(n_agents, generator=rng) < radical_prob
    radical_indices = torch.where(radical_mask)[0]

    if len(radical_indices) > 0:
        for _ in range(2):
            col_indices = torch.randint(0, 5, (len(radical_indices),), generator=rng)
            radical_values = torch.randn(len(radical_indices), generator=rng) * 2.5
            personality_logits[radical_indices, col_indices] = radical_values

    return exposures, personality_logits


# ================================
# Wealth + Structure Helpers
# ================================
def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32)

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=values.size, dtype=np.float32)
    return ranks


def normalize_wealth_exposure(raw_wealth: np.ndarray) -> np.ndarray:
    raw_wealth = np.asarray(raw_wealth, dtype=np.float32)
    if raw_wealth.size == 0:
        return raw_wealth

    log_wealth = np.log1p(np.clip(raw_wealth, a_min=0.0, a_max=None))
    min_w = float(np.min(log_wealth))
    max_w = float(np.max(log_wealth))
    if max_w <= min_w:
        return np.zeros_like(log_wealth, dtype=np.float32)

    normalized = 2.0 * ((log_wealth - min_w) / (max_w - min_w)) - 1.0
    return normalized.astype(np.float32)


def generate_structural_wealth(
    influence_scores,
    personality_logits,
    temperature,
    seed,
):
    """Generate raw wealth before topology exists.

    Wealth is driven by influence, productive traits, and a gated elite tail.
    """
    from scipy.stats import pareto

    np.random.seed(seed + 123)
    n = len(influence_scores)
    influence_scores = np.asarray(influence_scores, dtype=np.float32)
    influence_percentile = percentile_ranks(influence_scores)

    openness = torch.sigmoid(personality_logits[:, 0]).numpy()
    consc = torch.sigmoid(personality_logits[:, 1]).numpy()
    extraversion = torch.sigmoid(personality_logits[:, 2]).numpy()
    stability = 1.0 - torch.sigmoid(personality_logits[:, 4]).numpy()

    merit = (
        0.45 * consc
        + 0.20 * stability
        + 0.20 * extraversion
        + 0.15 * openness
    )

    seed_potential = (
        800.0
        + 2800.0 * merit
        + 1800.0 * np.power(np.maximum(influence_scores, 1e-6), 0.70)
    )
    realization_noise = np.random.lognormal(
        mean=0.0,
        sigma=0.18 + 0.22 * temperature,
        size=n,
    )
    wealth = seed_potential * realization_noise

    alpha = max(1.2, 2.3 - (temperature * 0.5))
    elite_gate = np.clip(0.55 * influence_percentile + 0.45 * merit, 0.0, 1.0)
    legacy_injection = (
        (pareto.rvs(alpha, size=n) + 1.0)
        * 4500.0
        * elite_gate
        * (0.4 + 0.6 * temperature)
    )
    wealth += legacy_injection

    wealth = np.maximum(wealth, 500.0)
    wealth = np.nan_to_num(wealth, nan=500.0, posinf=1e8)
    return wealth.astype(np.float32)


def _select_bridge_agents(
    wealth_percentile: np.ndarray,
    influence_percentile: np.ndarray,
    openness: np.ndarray,
) -> np.ndarray:
    n = len(wealth_percentile)
    bridge_mask = np.zeros(n, dtype=np.float32)
    if n == 0:
        return bridge_mask

    wealthy_slots = min(max(1, n // 500), 2)
    influence_slots = min(max(1, n // 500), 2)

    wealthy_candidates = np.argsort(wealth_percentile)[-wealthy_slots:]
    influence_candidates = np.argsort(influence_percentile)[-influence_slots:]
    candidate_indices = np.unique(
        np.concatenate([wealthy_candidates, influence_candidates], axis=0),
    )

    open_candidates = candidate_indices[openness[candidate_indices] >= 0.55]
    if open_candidates.size == 0 and candidate_indices.size > 0:
        open_candidates = np.array(
            [candidate_indices[np.argmax(openness[candidate_indices])]],
            dtype=np.int64,
        )

    bridge_mask[open_candidates] = 1.0
    return bridge_mask


# ================================
# Structure Builder (Wealth/Influence First)
# ================================
def apply_triadic_closure(config: SimConfig, adj: torch.Tensor, device: torch.device):
    """Stage 2: Community Cohesion via Iterative Triadic Closure.
    Forms new edges between neighbors-of-neighbors (A->B, B->C => A->C).
    """
    if not getattr(config, "triadic_closure_prob", 0.0) > 0:
        return adj

    N = config.num_agents
    prob = config.triadic_closure_prob
    iterations = getattr(config, "triadic_closure_iterations", 1)

    current_adj = adj.coalesce()

    # Get features for similarity filtering
    # We need to reach into the exposures/personalities
    # This is a bit hacky but efficient
    features_norm = getattr(config, "_features_norm_cache", None)

    for _ in range(iterations):
        # 1. Find neighbors-of-neighbors using sparse matrix multiplication
        # A_2[i, j] > 0 if there is a path i -> k -> j
        # We use a simplified version: just the structure, not the weights
        indices = current_adj.indices()
        vals = torch.ones_like(current_adj.values())
        binary_adj = torch.sparse_coo_tensor(indices, vals, size=(N, N)).coalesce()

        # Paths of length 2
        paths_2 = torch.sparse.mm(binary_adj, binary_adj).coalesce()

        p2_indices = paths_2.indices()
        p2_values = paths_2.values()

        # 2. Filter: Remove self-loops and existing edges
        # We only want NEW edges
        mask_self = p2_indices[0] != p2_indices[1]

        # Sample based on triadic_closure_prob
        sample_mask = torch.rand(len(p2_values), device=device) < prob
        valid_mask = mask_self & sample_mask

        if not valid_mask.any():
            break

        candidate_indices = p2_indices[:, valid_mask]

        if features_norm is not None:
            # Assign weights based on actual similarity
            src_features = features_norm[candidate_indices[0]]
            dst_features = features_norm[candidate_indices[1]]
            new_similarities = (src_features * dst_features).sum(dim=1)

            # Filter: Only keep new edges that meet a minimum homophily threshold
            homophily_threshold = getattr(config, "triadic_closure_homophily_threshold", 0.45)
            homophily_filter = new_similarities > homophily_threshold

            final_valid_mask = homophily_filter
            if not final_valid_mask.any():
                continue

            final_new_indices = candidate_indices[:, final_valid_mask]
            final_new_values = new_similarities[final_valid_mask]
        else:
            # Fallback if no features available
            final_new_indices = candidate_indices
            final_new_values = torch.full((final_new_indices.shape[1],), current_adj.values().mean(), device=device)

        # 3. Merge with original backbone
        combined_indices = torch.cat([current_adj.indices(), final_new_indices], dim=1)
        combined_values = torch.cat([current_adj.values(), final_new_values])

        current_adj = torch.sparse_coo_tensor(combined_indices, combined_values, size=(N, N), device=device).coalesce()

        # Cap connections to prevent exploding density
        if current_adj._nnz() > N * getattr(config, "max_connections", 100):
            break

    return current_adj

def create_topology(
    config: SimConfig,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    influence_scores: np.ndarray,
    raw_wealth: np.ndarray | None = None,
):
    """Wealth/influence-first echo chamber construction.
    """
    N = config.num_agents
    if N <= 1:
        return None

    max_available_neighbors = N - 1
    max_connections = min(config.max_connections, max_available_neighbors)
    wealth_idx = DIMENSION_INDICES["Wealth"]

    if raw_wealth is None:
        wealth_source = exposures[:, wealth_idx].detach().cpu().numpy()
    else:
        wealth_source = np.asarray(raw_wealth, dtype=np.float32)
    sorted_wealth = np.sort(np.asarray(wealth_source, dtype=np.float64) + 1e-9)
    wealth_gini = 0.0
    if sorted_wealth.size > 0 and sorted_wealth.sum() > 0:
        rank = np.arange(1, sorted_wealth.size + 1, dtype=np.float64)
        wealth_gini = float(
            np.sum((2.0 * rank - sorted_wealth.size - 1.0) * sorted_wealth)
            / (sorted_wealth.size * sorted_wealth.sum())
        )
    balkanization_gini = (
        wealth_gini
        if raw_wealth is not None and int(getattr(config, "evolution_generations", 10)) <= 3
        else 0.0
    )

    influence_scores = np.asarray(influence_scores, dtype=np.float32)
    influence_percentile = percentile_ranks(influence_scores)
    wealth_percentile = percentile_ranks(wealth_source)
    openness_np = personalities[:, 0].detach().cpu().numpy().astype(np.float32)
    bridge_mask_np = _select_bridge_agents(
        wealth_percentile, influence_percentile, openness_np,
    )

    elite_strength = np.maximum(wealth_percentile, influence_percentile)
    k_array = np.clip(
        np.rint(
            config.base_connections
            * (
                0.45
                + 1.15 * openness_np
                + 0.35 * elite_strength
                + 0.25 * bridge_mask_np
            ),
        ),
        1,
        max_connections,
    ).astype(int)
    h_strength_config = max(1.0, float(getattr(config, "homophily_strength", 6.0)))
    socialization_gain = float(getattr(config, "personality_socialization_gain", 0.08))
    degree_regularity = 0.0
    if socialization_gain <= 0.0:
        degree_regularity = min(0.65, max(0.0, h_strength_config - 1.0) / 10.0)
    if degree_regularity > 0:
        regular_degree = np.clip(config.base_connections, 1, max_connections)
        k_array = np.clip(
            np.rint((1.0 - degree_regularity) * k_array + degree_regularity * regular_degree),
            1,
            max_connections,
        ).astype(int)

    topology_exposures = exposures.clone()
    topology_exposures[:, wealth_idx] = 0.0
    features = torch.cat([topology_exposures, personalities], dim=1)
    features_norm = features / (torch.norm(features, dim=1, keepdim=True) + 1e-8)

    batch_size = 512
    indices_list = []
    values_list = []
    device = exposures.device
    features_norm = features_norm.to(device)

    influence_scale = influence_scores / max(float(np.mean(influence_scores)), 1e-6)
    influence_scale_tensor = torch.tensor(
        influence_scale, dtype=torch.float32, device=device,
    )
    influence_tensor = torch.tensor(
        influence_percentile, dtype=torch.float32, device=device,
    )
    wealth_tensor = torch.tensor(wealth_percentile, dtype=torch.float32, device=device)
    openness_tensor = torch.tensor(openness_np, dtype=torch.float32, device=device)
    bridge_tensor = torch.tensor(bridge_mask_np, dtype=torch.float32, device=device)
    target_influence_bias = torch.pow(
        torch.clamp(influence_scale_tensor, min=0.25),
        getattr(config, "influence_bias_exp", 0.4),
    )

    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_size_actual = end - i
        batch_features = features_norm[i:end]

        sim = torch.mm(batch_features, features_norm.T)
        h_strength = h_strength_config
        positive_similarity = torch.clamp(sim, min=0.0)
        homophily_sharpness = max(0.0, h_strength - 1.0)
        trait_similarity = positive_similarity * torch.exp(
            positive_similarity * homophily_sharpness,
        )

        batch_influence = influence_tensor[i:end].unsqueeze(1)
        batch_wealth = wealth_tensor[i:end].unsqueeze(1)
        batch_openness = openness_tensor[i:end].unsqueeze(1)
        batch_bridge = bridge_tensor[i:end].unsqueeze(1)

        influence_gap = torch.abs(batch_influence - influence_tensor.unsqueeze(0))
        wealth_gap = torch.abs(batch_wealth - wealth_tensor.unsqueeze(0))

        socioeconomic_sharpness = 2.0 + (0.35 * h_strength) + (24.0 * balkanization_gini)
        influence_homophily = torch.exp(-socioeconomic_sharpness * influence_gap)
        wealth_homophily = torch.exp(-socioeconomic_sharpness * wealth_gap)

        influence_elite = torch.sqrt(
            torch.clamp(batch_influence * influence_tensor.unsqueeze(0), min=0.0),
        )
        wealth_elite = torch.sqrt(
            torch.clamp(batch_wealth * wealth_tensor.unsqueeze(0), min=0.0),
        )
        cross_elite = 0.5 * (
            torch.sqrt(
                torch.clamp(batch_wealth * influence_tensor.unsqueeze(0), min=0.0),
            )
            + torch.sqrt(
                torch.clamp(batch_influence * wealth_tensor.unsqueeze(0), min=0.0),
            )
        )
        pair_openness = 0.5 * (batch_openness + openness_tensor.unsqueeze(0))

        homophily_mix = min(0.90, 0.24 + 0.035 * h_strength + 0.90 * balkanization_gini)
        bridge_mix = 1.0 - (0.45 * homophily_mix)

        prob_matrix = homophily_mix * trait_similarity
        prob_matrix += (
            bridge_mix
            * (0.18 + 0.24 * batch_influence)
            * influence_homophily
            * (0.25 + 0.75 * influence_elite)
        )
        prob_matrix += (
            bridge_mix
            * (1.0 + 4.0 * balkanization_gini)
            * (0.18 + 0.24 * batch_wealth)
            * wealth_homophily
            * (0.25 + 0.75 * wealth_elite)
        )
        inequality_bridge_damping = max(0.20, 1.0 - 1.5 * balkanization_gini)
        prob_matrix += (
            bridge_mix
            * inequality_bridge_damping
            * (0.05 + 0.18 * batch_openness)
            * cross_elite
        )
        prob_matrix += (0.04 + 0.10 * pair_openness) * trait_similarity
        prob_matrix += (
            bridge_mix
            * batch_bridge
            * (0.08 + 0.18 * batch_openness)
            * (0.35 + 0.65 * pair_openness)
        )
        prob_matrix = prob_matrix * target_influence_bias.unsqueeze(0)

        batch_indices = torch.arange(batch_size_actual, device=device)
        global_indices = batch_indices + i
        prob_matrix[batch_indices, global_indices] = 0.0

        batch_k = np.minimum(k_array[i:end], max_available_neighbors)
        max_k = int(np.max(batch_k))
        if max_k == 0:
            continue

        prob_matrix = prob_matrix + 1e-9
        sampled_indices = torch.multinomial(prob_matrix, max_k, replacement=False)

        sampled_vals = torch.gather(prob_matrix, 1, sampled_indices)

        range_tensor = torch.arange(max_k, device=device).unsqueeze(0)
        k_tensor = torch.tensor(batch_k, device=device).unsqueeze(1)
        mask = range_tensor < k_tensor

        valid_src = torch.arange(i, end, device=device).unsqueeze(1).expand(-1, max_k)[mask]
        valid_dst = sampled_indices[mask]
        valid_val = sampled_vals[mask]

        indices_list.append(torch.stack([valid_src, valid_dst]))
        values_list.append(valid_val)

    if not indices_list: return None

    indices_tensor = torch.cat(indices_list, dim=1)
    values_tensor = torch.cat(values_list)
    backbone_adj = torch.sparse_coo_tensor(indices_tensor, values_tensor, size=(N, N), device=device)

    config._features_norm_cache = features_norm

    print("Applying Triadic Closure (Stage 2)...")
    final_adj = apply_triadic_closure(config, backbone_adj, device)

    # Cleanup cache
    if hasattr(config, "_features_norm_cache"):
        delattr(config, "_features_norm_cache")

    # --- Final Normalization ---
    final_adj = final_adj.coalesce()
    if balkanization_gini > 0.0:
        edge_indices = final_adj.indices()
        edge_values = final_adj.values()
        edge_wealth_gap = torch.abs(
            wealth_tensor.index_select(0, edge_indices[0])
            - wealth_tensor.index_select(0, edge_indices[1])
        )
        max_cross_class_gap = max(0.12, 0.78 - (1.70 * balkanization_gini))
        keep_mask = edge_wealth_gap <= max_cross_class_gap
        row_keep_counts = torch.zeros(N, dtype=torch.long, device=device)
        row_keep_counts.index_add_(0, edge_indices[0], keep_mask.long())
        keep_mask = keep_mask | (row_keep_counts.index_select(0, edge_indices[0]) == 0)
        final_adj = torch.sparse_coo_tensor(
            edge_indices[:, keep_mask],
            edge_values[keep_mask],
            size=(N, N),
            device=device,
        ).coalesce()

    row_indices = final_adj.indices()[0]
    dense_sums = torch.sparse.sum(final_adj, dim=1).to_dense()
    dense_sums = torch.clamp(dense_sums, min=1e-8)

    normalized_values = final_adj.values() / dense_sums[row_indices]

    normalized_sparse_adj = torch.sparse_coo_tensor(
        final_adj.indices(), normalized_values, size=(N, N),
    ).coalesce()

    return normalized_sparse_adj


def assign_classes_from_topology(
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    influence_scores: np.ndarray,
    raw_wealth: np.ndarray,
    adjacency_matrix: torch.Tensor | None,
):
    n = len(influence_scores)
    if n == 0:
        return [], {}

    device = personalities.device
    wealth_percentile = torch.tensor(
        percentile_ranks(raw_wealth), dtype=torch.float32, device=device,
    )
    influence_percentile = torch.tensor(
        percentile_ranks(influence_scores), dtype=torch.float32, device=device,
    )
    openness = personalities[:, 0]

    if adjacency_matrix is not None:
        topology = adjacency_matrix.coalesce().to(device)
        local_wealth = torch.sparse.mm(topology, wealth_percentile.unsqueeze(1)).squeeze(1)
        local_influence = torch.sparse.mm(
            topology, influence_percentile.unsqueeze(1),
        ).squeeze(1)
        out_degree = torch.bincount(topology.indices()[0], minlength=n)
        in_degree = torch.bincount(topology.indices()[1], minlength=n)
        degree = (out_degree + in_degree).float()
    else:
        local_wealth = wealth_percentile
        local_influence = influence_percentile
        degree = torch.zeros(n, dtype=torch.float32, device=device)

    degree_percentile = torch.tensor(
        percentile_ranks(degree.detach().cpu().numpy()),
        dtype=torch.float32,
        device=device,
    )

    chamber_score = 0.5 * (local_wealth + local_influence)
    structural_score = (
        0.24 * wealth_percentile
        + 0.16 * influence_percentile
        + 0.22 * degree_percentile
        + 0.20 * local_wealth
        + 0.14 * local_influence
        + 0.04 * openness
    )

    class_labels = np.empty(n, dtype=object)
    elite_mask = (structural_score >= 0.86) | (
        (wealth_percentile >= 0.92) & (degree_percentile >= 0.65)
    )
    upper_middle_mask = (structural_score >= 0.67) & ~elite_mask
    middle_mask = (structural_score >= 0.44) & ~(elite_mask | upper_middle_mask)
    working_mask = (structural_score >= 0.22) & ~(
        elite_mask | upper_middle_mask | middle_mask
    )

    class_labels[:] = "Underclass"
    class_labels[working_mask.detach().cpu().numpy()] = "Working Class"
    class_labels[middle_mask.detach().cpu().numpy()] = "Middle Class"
    class_labels[upper_middle_mask.detach().cpu().numpy()] = "Upper Middle"
    class_labels[elite_mask.detach().cpu().numpy()] = "Elite"

    metrics = {
        "Topology_Degree": degree.detach().cpu().numpy(),
        "Chamber_Wealth": local_wealth.detach().cpu().numpy(),
        "Chamber_Influence": local_influence.detach().cpu().numpy(),
        "Structural_Class_Score": structural_score.detach().cpu().numpy(),
        "Chamber_Score": chamber_score.detach().cpu().numpy(),
    }
    return class_labels.tolist(), metrics


def finalize_social_structure(
    config: SimConfig,
    metadata: pd.DataFrame,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
):
    metadata = metadata.copy()
    influence_scores = metadata["Influence"].to_numpy(dtype=np.float32)
    raw_wealth = metadata["Raw_Wealth"].to_numpy(dtype=np.float32)

    adjacency_matrix = None
    if getattr(config, "use_network_topology", True):
        print("Generating Network Topology...")
        adjacency_matrix = create_topology(
            config,
            exposures,
            personalities,
            influence_scores,
            raw_wealth=raw_wealth,
        )

    if adjacency_matrix is not None and getattr(
        config, "personality_socialization_gain", 0.0,
    ) > 0:
        gain = getattr(config, "personality_socialization_gain", 0.05)
        print(f"Applying Personality Socialization (Stage 2, Gain={gain})...")
        local_personality_mean = torch.sparse.mm(adjacency_matrix, personalities)
        personalities = (1.0 - gain) * personalities + gain * local_personality_mean
        personalities = torch.clamp(personalities, 0.001, 0.999)

    classes, structure_metrics = assign_classes_from_topology(
        exposures,
        personalities,
        influence_scores,
        raw_wealth,
        adjacency_matrix,
    )
    metadata["Class"] = classes
    for metric_name, metric_values in structure_metrics.items():
        metadata[metric_name] = np.round(metric_values, 3)

    return metadata, personalities, adjacency_matrix


def generate_society(config: SimConfig, defer_structure: bool = False):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    print(f"Generating {config.num_agents} Agents via Wealth/Influence-First Model...")

    num_dims = len(DIMENSIONS)
    wealth_idx = DIMENSION_INDICES["Wealth"]
    num_personalities = 5
    total_dims = num_dims * 2 + num_personalities

    # 1. Initialize Continuous Traits (Wealth=0 for now)
    traits = torch.randn(config.num_agents, total_dims) * config.initial_trait_std_dev
    exposures = traits[:, :num_dims]
    exposures[:, wealth_idx] = 0.0 # Placeholder

    # 2. Correlated Personalities
    # Use a higher multiplier (1.5 vs 1.0) for personalities to ensure we cover sigmoid tails
    raw_personalities = (traits[:, num_dims : num_dims + num_personalities] / config.initial_trait_std_dev) * 1.5

    try:
        jitter = torch.eye(5) * 1e-4
        L = torch.linalg.cholesky(PERSONALITY_CORRELATIONS + jitter)
        raw_personalities = torch.matmul(raw_personalities, L.T)
    except Exception as e:
        print(f"Warning: Cholesky failed ({e}).")

    raw_affinities = traits[:, num_dims + num_personalities :]
    exposures, raw_personalities = apply_random_mutations(
        exposures, raw_personalities, config.mutation_temperature, config.seed,
    )
    personalities = torch.sigmoid(raw_personalities)

    # 3. Influence
    influence_scores = np.random.lognormal(
        mean=1.0, sigma=0.5 + config.mutation_temperature, size=config.num_agents,
    )
    if getattr(config, "use_power_law_influence", False):
        alpha = 1.16
        pareto_multiplier = (np.random.pareto(alpha, config.num_agents) + 1) * 2.0
        influence_scores *= pareto_multiplier
        influence_scores = np.minimum(influence_scores, 1000.0)

    # 4. Raw Wealth (before topology/evolution)
    wealth_values = generate_structural_wealth(
        influence_scores,
        raw_personalities,
        config.mutation_temperature,
        config.seed,
    )

    # 5. Normalize wealth only for exposure-space cognition
    wealth_normalized = normalize_wealth_exposure(wealth_values)
    exposures[:, wealth_idx] = torch.tensor(wealth_normalized, dtype=torch.float32)

    # Final Touches
    non_wealth_mask = torch.ones(num_dims, dtype=torch.bool)
    non_wealth_mask[wealth_idx] = False
    exposures[:, non_wealth_mask] = torch.tanh(exposures[:, non_wealth_mask])

    cognitive_bandwidth = torch.clamp(
        torch.randn(config.num_agents, 1) * 0.2 + 0.55, min=0.1, max=1.0,
    )
    positive_affinities = torch.clamp(
        torch.abs(raw_affinities), min=config.affinity_min_strength,
    )
    if getattr(config, "normalize_affinities_by_mean", True):
        mean_affinity = positive_affinities.mean(dim=1, keepdim=True)
        normalized_affinities = positive_affinities / torch.clamp(
            mean_affinity, min=1e-6,
        )
    else:
        normalized_affinities = positive_affinities
    affinities = normalized_affinities * cognitive_bandwidth

    df_metadata = pd.DataFrame(
        {
            "Agent_ID": range(config.num_agents),
            "Class": ["Agent"] * config.num_agents,
            "Region": ["Global"] * config.num_agents,
            "Influence": np.round(influence_scores, 3),
            "Raw_Wealth": np.round(wealth_values, 3),
            "Cognitive_Bandwidth": np.round(cognitive_bandwidth.squeeze().numpy(), 3),
        },
    )

    adjacency_matrix = None
    if not defer_structure:
        df_metadata, personalities, adjacency_matrix = finalize_social_structure(
            config,
            df_metadata,
            exposures,
            personalities,
        )

    df_metadata.to_parquet(f"{config.output_dir}/metadata.parquet")
    torch.save(exposures, f"{config.output_dir}/exposures.pt")
    torch.save(personalities, f"{config.output_dir}/personalities.pt")
    torch.save(affinities, f"{config.output_dir}/affinities.pt")
    if adjacency_matrix is not None:
        torch.save(adjacency_matrix, f"{config.output_dir}/adjacency.pt")

    print(f"Society Generated in '{config.output_dir}' (Wealth/Influence-First Model)")
    return df_metadata, exposures, personalities, affinities, adjacency_matrix


def main():
    conf = SimConfig(num_agents=10000, seed=69)
    conf.wealth_dim_idx = DIMENSION_INDICES["Wealth"]
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(
        conf,
        defer_structure=conf.enable_evolution,
    )
    if conf.enable_evolution:
        evolver = SocietyEvolution(conf, df_meta, exposures, personalities)
        df_meta, exposures, personalities = evolver.evolve()
        df_meta, personalities, adjacency_matrix = finalize_social_structure(
            conf,
            df_meta,
            exposures,
            personalities,
        )


if __name__ == "__main__":
    main()
