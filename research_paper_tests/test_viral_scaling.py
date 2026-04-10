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
    output_dir = Path(__file__).resolve().parent / "generated" / "viral_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = get_test_scenario("viral_scaling")
    config = scenario.sim_config()
    settings = scenario.settings()
    amplitudes, mean_multiplier, max_multiplier = _viral_scaling_curve(config, settings)
    max_allowed = 1.0 + config.max_viral_multiplier
    slopes = np.diff(mean_multiplier) / np.diff(amplitudes)

    # Figure 1: Viral Scaling Curve
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(amplitudes, mean_multiplier, marker="o", linewidth=2, label="Mean")
    ax1.plot(amplitudes, max_multiplier, marker="s", linewidth=2, label="Max")
    ax1.axhline(
        max_allowed,
        color="#e63946",
        linestyle="--",
        linewidth=2,
        label="Configured cap",
    )
    ax1.set_title("Viral Scaling Curve", fontsize=16)
    ax1.set_xlabel("Emotion Amplitude", fontsize=12)
    ax1.set_ylabel("Outrage Multiplier", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)
    
    path1 = output_dir / "viral_scaling_curve.png"
    fig1.savefig(path1, dpi=220, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Steepest Viral Growth Region
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    slope_x = 0.5 * (amplitudes[:-1] + amplitudes[1:])
    ax2.plot(slope_x, slopes, marker="o", linewidth=2, color="#457b9d")
    ax2.set_title("Steepest Viral Growth Region", fontsize=16)
    ax2.set_xlabel("Emotion Amplitude", fontsize=12)
    ax2.set_ylabel("d(Multiplier)/d(Amplitude)", fontsize=12)
    ax2.grid(True, alpha=0.2)

    path2 = output_dir / "steepest_growth_region.png"
    fig2.savefig(path2, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    assert path1.exists()
    assert path2.exists()
    assert path1.stat().st_size > 0
    assert path2.stat().st_size > 0
