import numpy as np

from main import DIMENSION_INDICES, distort_world_signal
from research_paper_tests.config_schema import (
    PERSONALITY_INDICES,
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)


def test_signal_distortion_scales_with_neuroticism(tmp_path):
    scenario = get_test_scenario("signal_distortion")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "signal_distortion",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="distortion",
    )

    world = build_world(settings["world"])
    perceived = distort_world_signal(config, world, society.personalities)

    safety_index = DIMENSION_INDICES["Physical_Safety"]
    neuroticism = society.personalities[:, PERSONALITY_INDICES["Neuroticism"]].numpy()
    distortion = np.abs(perceived[:, safety_index].numpy() - world[0, safety_index].item())
    correlation = float(np.corrcoef(neuroticism, distortion)[0, 1])
    low_cutoff = float(np.quantile(neuroticism, 0.25))
    high_cutoff = float(np.quantile(neuroticism, 0.75))
    low_neuroticism_distortion = distortion[neuroticism <= low_cutoff]
    high_neuroticism_distortion = distortion[neuroticism >= high_cutoff]

    assert correlation > 0.0
    assert high_neuroticism_distortion.mean() > low_neuroticism_distortion.mean()
