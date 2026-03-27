import numpy as np
import pandas as pd

from main import PERSONALITY_CORRELATIONS, create_sim_config, prepare_society_for_debug


def test_generated_personalities_follow_target_correlations(tmp_path):
    config = create_sim_config(
        num_agents=4000,
        mutation_temperature=0.0,
        use_network_topology=False,
        enable_evolution=False,
    )
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "correlations"), evolve=False
    )

    observed = pd.DataFrame(
        society.personalities.numpy(),
        columns=["O", "C", "E", "A", "N"],
    ).corr()
    target = pd.DataFrame(
        PERSONALITY_CORRELATIONS.numpy(),
        index=["O", "C", "E", "A", "N"],
        columns=["O", "C", "E", "A", "N"],
    )

    rmse = np.sqrt(((observed - target) ** 2).mean().mean())
    assert rmse < 0.2
