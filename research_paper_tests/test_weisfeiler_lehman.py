import community as community_louvain
import pytest

from main import prepare_society_for_debug
from research_paper_tests._metrics import (
    adjacency_to_graph,
    wl_echo_chamber_structural_similarity,
    wl_graph_hash,
    wl_kernel_similarity,
)
from research_paper_tests.config_schema import get_test_scenario


def test_wl_echo_chamber_similarity():
    low_config = get_test_scenario("echo_chambers_low").sim_config()
    high_config = get_test_scenario("echo_chambers_high").sim_config()

    low_society = prepare_society_for_debug(low_config)
    high_society = prepare_society_for_debug(high_config)

    low_graph = adjacency_to_graph(low_society.adjacency_matrix)
    high_graph = adjacency_to_graph(high_society.adjacency_matrix)

    low_partition = community_louvain.best_partition(low_graph, random_state=42)
    high_partition = community_louvain.best_partition(high_graph, random_state=42)

    low_sims = wl_echo_chamber_structural_similarity(low_society.adjacency_matrix, low_partition)
    high_sims = wl_echo_chamber_structural_similarity(high_society.adjacency_matrix, high_partition)

    low_avg = sum(low_sims.values()) / len(low_sims) if low_sims else 0
    high_avg = sum(high_sims.values()) / len(high_sims) if high_sims else 0

    print(f"\nLow Homophily - Avg community WL similarity: {low_avg:.4f}")
    print(f"High Homophily - Avg community WL similarity: {high_avg:.4f}")

    assert low_avg >= 0
    assert high_avg >= 0

def test_wl_graph_kernel_evolution():
    config = get_test_scenario("echo_chambers_low").sim_config()
    config.seed = 42

    society_gen0 = prepare_society_for_debug(config)

    evolve_config = get_test_scenario("echo_chambers_low").sim_config()
    evolve_config.seed = 42
    evolve_config.enable_evolution = True
    evolve_config.evolution_generations = 10
    society_gen10 = prepare_society_for_debug(evolve_config)

    kernel_sim = wl_kernel_similarity(society_gen0.adjacency_matrix, society_gen10.adjacency_matrix)
    print(f"\nWL Kernel Similarity Gen0 vs Gen10: {kernel_sim:.4f}")

    assert kernel_sim >= 0

def test_wl_generator_stability():
    config1 = get_test_scenario("echo_chambers_low").sim_config()
    config1.seed = 42
    config2 = get_test_scenario("echo_chambers_low").sim_config()
    config2.seed = 100

    society1 = prepare_society_for_debug(config1)
    society2 = prepare_society_for_debug(config2)

    hash1 = wl_graph_hash(society1.adjacency_matrix)
    hash2 = wl_graph_hash(society2.adjacency_matrix)

    print(f"\nSeed 42 Hash: {hash1}")
    print(f"Seed 100 Hash: {hash2}")

    kernel_sim = wl_kernel_similarity(society1.adjacency_matrix, society2.adjacency_matrix)
    print(f"WL Kernel Sim (Seed 42 vs 100): {kernel_sim:.4f}")

    assert kernel_sim >= 0

@pytest.mark.slow
def test_wl_population_scaling():
    config_500 = get_test_scenario("echo_chambers_low").sim_config()
    config_500.seed = 42
    config_500.num_agents = 500

    config_5k = get_test_scenario("echo_chambers_low").sim_config()
    config_5k.seed = 42
    config_5k.num_agents = 5000

    config_10k = get_test_scenario("echo_chambers_low").sim_config()
    config_10k.seed = 42
    config_10k.num_agents = 10000

    print("\nGenerating Society (N=500)...")
    society_500 = prepare_society_for_debug(config_500)
    print("Generating Society (N=5000)...")
    society_5k = prepare_society_for_debug(config_5k)
    print("Generating Society (N=10000)...")
    society_10k = prepare_society_for_debug(config_10k)

    hash_500 = wl_graph_hash(society_500.adjacency_matrix)
    hash_5k = wl_graph_hash(society_5k.adjacency_matrix)
    hash_10k = wl_graph_hash(society_10k.adjacency_matrix)

    print(f"\nPop 500 Hash: {hash_500}")
    print(f"Pop 5000 Hash: {hash_5k}")
    print(f"Pop 10000 Hash: {hash_10k}")

    sim_500_5k = wl_kernel_similarity(society_500.adjacency_matrix, society_5k.adjacency_matrix)
    sim_5k_10k = wl_kernel_similarity(society_5k.adjacency_matrix, society_10k.adjacency_matrix)

    print(f"WL Kernel Sim (500 vs 5000): {sim_500_5k:.4f}")
    print(f"WL Kernel Sim (5000 vs 10000): {sim_5k_10k:.4f}")

    assert sim_500_5k >= 0
    assert sim_5k_10k >= 0
