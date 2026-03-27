import torch

from main import DIMENSION_INDICES, build_debug_society, create_sim_config, run_debug_simulation


def test_cognitive_gate_blocks_misaligned_low_openness_agents():
    config = create_sim_config(
        num_agents=400,
        use_signal_distortion=False,
        use_network_topology=False,
        enable_evolution=False,
    )
    setattr(config, "use_selective_exposure", True)
    setattr(config, "selective_exposure_base_tolerance", -0.3)
    setattr(config, "selective_exposure_openness_factor", 0.4)

    half = config.num_agents // 2
    exposures = torch.zeros(config.num_agents, 12)
    personalities = torch.ones(config.num_agents, 5) * 0.5
    personalities[:half, 0] = 0.1
    personalities[half:, 0] = 0.9

    for idx, value in [
        (DIMENSION_INDICES["Innovation"], 1.0),
        (DIMENSION_INDICES["Fairness"], 1.0),
        (DIMENSION_INDICES["Sanctity"], -1.0),
        (DIMENSION_INDICES["In_Group"], -1.0),
    ]:
        exposures[half:, idx] = value
        exposures[:half, idx] = -value

    society = build_debug_society(config, exposures, personalities)

    world = torch.zeros(1, 12)
    world[0, DIMENSION_INDICES["Innovation"]] = 0.8
    world[0, DIMENSION_INDICES["Fairness"]] = 0.7
    world[0, DIMENSION_INDICES["Sanctity"]] = -0.9
    world[0, DIMENSION_INDICES["In_Group"]] = -0.5

    result = run_debug_simulation(config, world, society=society, urgency=0.2)
    engagement = result.engagement_scores.numpy()

    assert engagement[:half].mean() < engagement[half:].mean()
    assert float((engagement[:half] == 0.0).mean()) > float((engagement[half:] == 0.0).mean())
