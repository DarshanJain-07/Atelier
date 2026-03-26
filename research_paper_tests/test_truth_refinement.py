import torch

from cognitive_engine import CognitiveEngine
from schema import SimConfig


def test_truth_refinement_prioritizes_long_term_for_skeptical_agents():
    config = SimConfig(
        num_agents=2,
        use_signal_distortion=False,
        use_time_pressure=False,
    )
    config.skepticism_gain = 4.0
    config.logic_gap_threshold = 0.4

    engine = CognitiveEngine(config)

    world_tensor = torch.zeros(1, 12)
    world_tensor[0, 0] = 0.8
    world_tensor[0, 2] = -0.9
    world_tensor[0, 10] = 0.9
    world_tensor[0, 11] = -0.9

    personalities = torch.tensor(
        [
            [0.1, 0.1, 0.5, 0.5, 0.5],
            [0.9, 0.9, 0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    exposures = torch.zeros(2, 12)
    affinities = torch.ones(2, 12)

    _, attention, _ = engine.run(
        world_tensor_raw=world_tensor,
        urgency=0.0,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
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


if __name__ == "__main__":
    test_truth_refinement_prioritizes_long_term_for_skeptical_agents()
