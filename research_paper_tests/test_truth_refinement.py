import torch

from main import build_debug_society, create_sim_config, run_cognitive_cycle


def test_truth_refinement_prioritizes_long_term_for_skeptical_agents():
    config = create_sim_config(
        num_agents=2,
        use_signal_distortion=False,
        use_time_pressure=False,
        use_network_topology=False,
        enable_evolution=False,
    )
    config.skepticism_gain = 4.0
    config.logic_gap_threshold = 0.4

    world = torch.zeros(1, 12)
    world[0, 0] = 0.8
    world[0, 2] = -0.9
    world[0, 10] = 0.9
    world[0, 11] = -0.9

    personalities = torch.tensor(
        [
            [0.1, 0.1, 0.5, 0.5, 0.5],
            [0.9, 0.9, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    society = build_debug_society(config, torch.zeros(2, 12), personalities)

    _, attention, _ = run_cognitive_cycle(
        config,
        world,
        urgency=0.0,
        is_personal=False,
        exposures=society.exposures,
        personalities=society.personalities,
        affinities=society.affinities,
    )

    skeptical_long_term = attention[1, 11].item()
    skeptical_short_term = attention[1, 10].item()
    populist_long_term = attention[0, 11].item()
    populist_short_term = attention[0, 10].item()

    assert skeptical_long_term > populist_long_term
    assert skeptical_long_term > skeptical_short_term
    assert (skeptical_long_term - skeptical_short_term) > (
        populist_long_term - populist_short_term
    )
