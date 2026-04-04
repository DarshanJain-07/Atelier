from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import aggregate_social_state
from research_paper_tests.config_schema import EMOTION_INDICES, get_test_scenario, zero_emotions

matplotlib.use("Agg")


def _viral_scaling_curve(config, settings):
    amplitudes = settings["amplitudes"]
    emotion_idx = EMOTION_INDICES[settings["emotion_name"]]
    influence = torch.ones(config.num_agents)
    mean_multiplier = []
    max_multiplier = []

    for amplitude in amplitudes:
        emotions = zero_emotions(config.num_agents)
        emotions[:, emotion_idx] = float(amplitude)
        state = aggregate_social_state(
            config,
            emotions,
            influence,
            engagement_scores=torch.ones(config.num_agents),
        )
        mean_multiplier.append(float(state["mean_outrage_multiplier"]))
        max_multiplier.append(float(state["max_outrage_multiplier"]))

    return np.asarray(amplitudes, dtype=np.float64), np.asarray(mean_multiplier), np.asarray(max_multiplier)


def test_viral_scaling_has_sigmoid_regime_and_cap():
    scenario = get_test_scenario("viral_scaling")
    config = scenario.sim_config()
    settings = scenario.settings()
    amplitudes, mean_multiplier, max_multiplier = _viral_scaling_curve(config, settings)

    diffs = np.diff(mean_multiplier)
    peak_idx = int(np.argmax(diffs))
    max_allowed = 1.0 + config.max_viral_multiplier

    assert np.all(np.diff(mean_multiplier) >= -1e-6)
    assert np.all(np.diff(max_multiplier) >= -1e-6)
    assert max_allowed - max_multiplier[-1] <= settings["near_cap_tolerance"]
    assert 0 < peak_idx < len(diffs) - 1
    assert diffs[peak_idx] > diffs[0]
    assert diffs[peak_idx] > diffs[-1]


def test_generate_viral_scaling_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "viral_scaling.png"

    scenario = get_test_scenario("viral_scaling")
    config = scenario.sim_config()
    settings = scenario.settings()
    amplitudes, mean_multiplier, max_multiplier = _viral_scaling_curve(config, settings)
    max_allowed = 1.0 + config.max_viral_multiplier
    slopes = np.diff(mean_multiplier) / np.diff(amplitudes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # x-axis: emotional amplitude injected into the same aligned population.
    # y-axis: outrage multiplier. The curve should accelerate through the sigmoid
    # middle region and then flatten near the configured cap.
    axes[0].plot(amplitudes, mean_multiplier, marker="o", linewidth=2, label="Mean")
    axes[0].plot(amplitudes, max_multiplier, marker="s", linewidth=2, label="Max")
    axes[0].axhline(
        max_allowed,
        color="#e63946",
        linestyle="--",
        linewidth=2,
        label="Configured cap",
    )
    axes[0].set_title("Viral Scaling Curve")
    axes[0].set_xlabel("Emotion Amplitude")
    axes[0].set_ylabel("Outrage Multiplier")
    axes[0].legend(fontsize=8)

    slope_x = 0.5 * (amplitudes[:-1] + amplitudes[1:])

    # x-axis: midpoint of each adjacent amplitude segment.
    # y-axis: local slope of the multiplier curve. The peak marks the regime where
    # small changes in arousal produce the fastest growth in virality.
    axes[1].plot(slope_x, slopes, marker="o", linewidth=2, color="#457b9d")
    axes[1].set_title("Steepest Viral Growth Region")
    axes[1].set_xlabel("Emotion Amplitude")
    axes[1].set_ylabel("d(Multiplier)/d(Amplitude)")

    for axis in axes:
        axis.grid(True, alpha=0.2)

    fig.suptitle("Viral Scaling", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
