import torch
import numpy as np
from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from research_paper_tests.config_schema import SimConfig
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_statistically_greater,
    assert_monotonic_relationship
)

def test_institutional_distrust_backlash_gradient(n_seeds):
    """
    Validation: Historical distrust (low Reputation/Fairness in memory) must 
    monotonically increase the cognitive 'energy' of skeptical frames.
    This proves that agents prioritize historical narrative over objective current benefits.
    """
    # Sweep Reputation in memory from High Trust (0.8) to High Distrust (-0.8)
    reputation_sweep = [0.8, 0.4, 0.0, -0.4, -0.8]
    mean_skeptical_energies = []

    def get_sim_runner(rep_val):
        def runner():
            config = SimConfig(
                num_agents=200, # Focused sample
                use_backlash_ab_testing=True,
                backlash_sample_size=0.1,         
                use_agent_memory=True,
                memory_trigger_stacking_gain=3.0, # Stronger history effect
            )
            df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
            cognitive = CognitiveEngine(config)
            
            # Setup Beneficial 'Official' Signal (e.g., Safety Policy)
            official_signal = torch.zeros(12)
            official_signal[1] = 0.8 # Safety
            
            # Setup 'Skeptical' Frame (e.g., Loss of Freedom)
            skeptical_frame = torch.zeros(12)
            skeptical_frame[7] = -0.8 # Freedom
            
            # Seed memory with the specific institutional reputation level
            memory = torch.zeros(config.num_agents, 12)
            memory[:, 3] = rep_val # Institutional Reputation
            
            decision = cognitive.run_backlash_ab_test(
                world_tensor_off=official_signal.unsqueeze(0),
                world_tensor_skp=skeptical_frame.unsqueeze(0),
                backlash_potential=0.7, 
                urgency=0.3,
                is_personal=False,
                exposures=exposures,
                personalities=personalities,
                agent_affinities=affinities,
                agent_memory=memory,
                adjacency_matrix=adjacency_matrix
            )
            # Skeptical energy represents the 'attraction' of the skeptical narrative
            return decision.skeptical_energy
        return runner

    # 1. Execute Sweep
    for rep in reputation_sweep:
        results = run_monte_carlo(get_sim_runner(rep), n_seeds=n_seeds)
        mean_skeptical_energies.append(np.mean(results))

    # Assertion: As Reputation decreases, Skeptical Energy must increase (Negative Monotonicity)
    assert_monotonic_relationship(reputation_sweep, mean_skeptical_energies, "negative")

    # 2. Statistical Significance between Trust and Distrust extremes
    trust_results = run_monte_carlo(get_sim_runner(0.8), n_seeds=n_seeds)
    distrust_results = run_monte_carlo(get_sim_runner(-0.8), n_seeds=n_seeds)
    assert_statistically_greater(distrust_results, trust_results)
