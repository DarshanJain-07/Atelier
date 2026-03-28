import numpy as np

from main import DIMENSION_INDICES, distort_world_signal, prepare_society_for_debug
from research_paper_tests.config_schema import (
    PERSONALITY_INDICES,
    build_world,
    get_test_scenario,
)


def test_signal_distortion_scales_with_neuroticism(tmp_path):
    scenario = get_test_scenario("signal_distortion")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_society_for_debug(
        config, output_dir=str(tmp_path / "distortion"), evolve=False
    )

    world = build_world(settings["world"])
    perceived = distort_world_signal(config, world, society.personalities)

    safety_index = DIMENSION_INDICES["Physical_Safety"]
    neuroticism = society.personalities[:, PERSONALITY_INDICES["Neuroticism"]].numpy()
    distortion = np.abs(perceived[:, safety_index].numpy() - world[0, safety_index].item())
    correlation = float(np.corrcoef(neuroticism, distortion)[0, 1])

    assert correlation > settings["min_correlation"]
