import torch

from main import DIMENSION_INDICES, create_sim_config, prepare_society_for_debug, run_debug_simulation


def test_personal_events_stay_more_localized_than_general_events(tmp_path):
    config = create_sim_config(
        num_agents=200,
        use_signal_distortion=False,
        use_time_pressure=False,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "personal"), evolve=False
    )

    world = torch.zeros(1, 12)
    world[0, DIMENSION_INDICES["Care"]] = -0.8

    general = run_debug_simulation(config, world, society=society, urgency=0.2, is_personal=False)
    personal = run_debug_simulation(config, world, society=society, urgency=0.2, is_personal=True)

    assert personal.engagement_scores.mean().item() < general.engagement_scores.mean().item()
