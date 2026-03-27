import numpy as np
import torch

from main import create_sim_config, distort_world_signal, prepare_society_for_debug


def test_signal_distortion_scales_with_neuroticism(tmp_path):
    config = create_sim_config(
        num_agents=500,
        use_signal_distortion=True,
        distortion_max_noise=0.8,
        distortion_neurotic_gain=1.5,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "distortion"), evolve=False
    )

    world = torch.zeros(1, 12)
    world[0, 1] = -0.4
    perceived = distort_world_signal(config, world, society.personalities)

    neuroticism = society.personalities[:, 4].numpy()
    distortion = np.abs(perceived[:, 1].numpy() - world[0, 1].item())
    correlation = float(np.corrcoef(neuroticism, distortion)[0, 1])

    assert correlation > 0.1
