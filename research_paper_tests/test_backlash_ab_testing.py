import torch
import numpy as np

from main import build_debug_society, run_debug_simulation
from research_paper_tests.config_schema import SimConfig
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def test_backlash_ab_testing_flips_to_skeptical_frame(n_seeds):
    def runner():
        config = SimConfig(
            seed=np.random.randint(0, 1000000),
            num_agents=48,
            enable_evolution=False,
            use_signal_distortion=False,
            use_backlash_ab_testing=True,
            use_agent_memory=True,
            backlash_sample_size=0.5,
            backlash_skepticism_threshold=0.45,
            backlash_decision_threshold=1.05,
        )

        exposures = torch.zeros((config.num_agents, 12), dtype=torch.float32)
        personalities = torch.zeros((config.num_agents, 5), dtype=torch.float32)
        personalities[:, 0] = 1.0
        personalities[:, 4] = 1.0
        affinities = torch.ones((config.num_agents, 12), dtype=torch.float32)
        society = build_debug_society(config, exposures, personalities, affinities=affinities)

        official_frame = torch.zeros((1, 12), dtype=torch.float32)
        official_frame[0, 6] = 0.2
        official_frame[0, 3] = 0.1

        skeptical_frame = torch.zeros((1, 12), dtype=torch.float32)
        skeptical_frame[0, 3] = -0.8
        skeptical_frame[0, 4] = -0.7
        skeptical_frame[0, 6] = -0.2

        result = run_debug_simulation(
            config,
            official_frame,
            society=society,
            urgency=0.45,
            world_tensor_skp=skeptical_frame,
            backlash_potential=0.95,
        )
        
        return {
            "narrative_frame": result.narrative_frame,
            "skeptical_energy": result.backlash_diagnostics["skeptical_energy"],
            "official_energy": result.backlash_diagnostics["official_energy"]
        }

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    
    for r in results:
        assert r["narrative_frame"] == "skeptical"
    
    skeptical_energies = [r["skeptical_energy"] for r in results]
    official_energies = [r["official_energy"] for r in results]
    
    assert_statistically_greater(skeptical_energies, official_energies)

def test_backlash_ab_testing_preserves_official_frame_for_conformists(n_seeds):
    def runner():
        config = SimConfig(
            seed=np.random.randint(0, 1000000),
            num_agents=48,
            enable_evolution=False,
            use_signal_distortion=False,
            use_backlash_ab_testing=True,
            backlash_sample_size=0.5,
            backlash_skepticism_threshold=0.6,
            backlash_decision_threshold=1.2,
        )

        exposures = torch.zeros((config.num_agents, 12), dtype=torch.float32)
        personalities = torch.zeros((config.num_agents, 5), dtype=torch.float32)
        personalities[:, 1] = 0.8
        personalities[:, 3] = 1.0
        affinities = torch.ones((config.num_agents, 12), dtype=torch.float32)
        society = build_debug_society(config, exposures, personalities, affinities=affinities)

        official_frame = torch.zeros((1, 12), dtype=torch.float32)
        official_frame[0, 6] = 0.45
        official_frame[0, 3] = 0.25

        skeptical_frame = torch.zeros((1, 12), dtype=torch.float32)
        skeptical_frame[0, 3] = -0.1

        result = run_debug_simulation(
            config,
            official_frame,
            society=society,
            urgency=0.3,
            world_tensor_skp=skeptical_frame,
            backlash_potential=0.15,
        )
        return result.narrative_frame

    results = run_monte_carlo(runner, n_seeds=n_seeds)
    for frame in results:
        assert frame == "official"
