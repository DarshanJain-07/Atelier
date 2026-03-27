import torch

from main import build_debug_society, create_sim_config, project_emotions, run_cognitive_cycle


def test_emotional_divergence_tracks_neuroticism():
    config = create_sim_config(
        num_agents=2,
        use_signal_distortion=False,
        use_network_topology=False,
        enable_evolution=False,
    )
    personalities = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.95],
        ],
        dtype=torch.float32,
    )
    exposures = torch.zeros((2, 12))
    society = build_debug_society(config, exposures, personalities)

    world = torch.zeros(1, 12)
    world[0, 1] = -0.4

    context, _, _ = run_cognitive_cycle(
        config,
        world,
        urgency=0.5,
        is_personal=False,
        exposures=society.exposures,
        personalities=society.personalities,
        affinities=society.affinities,
    )
    emotions = project_emotions(config, context)

    assert emotions[1, 2].item() > emotions[0, 2].item()
