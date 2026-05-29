import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

from main import aggregate_social_state
from research_paper_tests.config_schema import (
    EMOTION_INDICES,
    get_test_scenario,
    zero_emotions,
)
from research_paper_tests.plotting_utils import (
    PAPER_PALETTE,
    apply_paper_style,
    save_paper_figure,
    setup_plot,
)
from research_paper_tests.stats_utils import (
    run_monte_carlo,
    assert_monotonic_relationship
)

matplotlib.use("Agg")
apply_paper_style()

def test_viral_scaling_statistical():
    """
    Statistically validates that viral scaling follows a monotonic sigmoid increase
    and respects configured caps using Monte Carlo simulations.
    """
    scenario = get_test_scenario("viral_scaling")
    config = scenario.sim_config()
    settings = scenario.settings()
    amplitudes = settings["amplitudes"]
    emotion_idx = EMOTION_INDICES[settings["emotion_name"]]
    influence = torch.ones(config.num_agents)
    
    def get_scaling_runner(amplitude):
        def runner():
            emotions = zero_emotions(config.num_agents)
            emotions[:, emotion_idx] = float(amplitude)
            state = aggregate_social_state(
                config,
                emotions,
                influence,
                engagement_scores=torch.ones(config.num_agents),
            )
            return {
                "mean": float(state["mean_outrage_multiplier"]),
                "max": float(state["max_outrage_multiplier"])
            }
        return runner

    print(f"\nRunning Viral Scaling Statistical Sweep ({len(amplitudes)} levels)...")
    
    all_amplitudes = []
    all_mean_results = []
    last_max_results = []
    
    for amp in amplitudes:
        # Using n_seeds=3 for the sweep to balance coverage and performance
        results = run_monte_carlo(get_scaling_runner(amp), n_seeds=3)
        all_amplitudes.extend([amp] * len(results))
        all_mean_results.extend([r["mean"] for r in results])
        if amp == amplitudes[-1]:
            last_max_results.extend([r["max"] for r in results])

    # 1. Monotonicity: Outrage multiplier must increase with amplitude (p < 0.05)
    assert_monotonic_relationship(all_amplitudes, all_mean_results, "positive")
    
    # 2. Cap Validation: Final multipliers must be near the configured cap
    max_allowed = 1.0 + config.max_viral_multiplier
    mean_final_max = np.mean(last_max_results)
    assert max_allowed - mean_final_max <= settings["near_cap_tolerance"]
    
    # 3. Shape Validation: Ensure sigmoid regime (growth peak in the middle)
    unique_means = [
        np.mean([all_mean_results[i] for i, a in enumerate(all_amplitudes) if a == amp]) 
        for amp in amplitudes
    ]
    diffs = np.diff(unique_means)
    peak_idx = int(np.argmax(diffs))
    assert 0 < peak_idx < len(diffs) - 1, f"Sigmoid peak should be interior, found at index {peak_idx}"

def _viral_scaling_curve(config, settings):
    amplitudes = settings["amplitudes"]
    emotion_idx = EMOTION_INDICES[settings["emotion_name"]]
    
    mean_multipliers = []
    max_multipliers = []
    
    for amp in amplitudes:
        emotions = zero_emotions(config.num_agents)
        emotions[:, emotion_idx] = float(amp)
        state = aggregate_social_state(
            config,
            emotions,
            torch.ones(config.num_agents),
            engagement_scores=torch.ones(config.num_agents),
        )
        mean_multipliers.append(float(state["mean_outrage_multiplier"]))
        max_multipliers.append(float(state["max_outrage_multiplier"]))
    return amplitudes, mean_multipliers, max_multipliers

def test_generate_viral_scaling_figure():
    """Generates the viral scaling research figures using the statistical mean."""
    output_dir = Path(__file__).resolve().parent / "generated" / "viral_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = get_test_scenario("viral_scaling")
    config = scenario.sim_config()
    settings = scenario.settings()
    
    amplitudes, mean_multipliers, max_multipliers = _viral_scaling_curve(config, settings)
    max_allowed = 1.0 + config.max_viral_multiplier
    
    # Figure: Viral Scaling Curve
    fig, ax = setup_plot(
        title="Viral Scaling Curve",
        xlabel="Emotion Amplitude",
        ylabel="Outrage Multiplier",
    )
    ax.plot(amplitudes, mean_multipliers, marker="o", label="Mean", color=PAPER_PALETTE["primary"])
    ax.plot(amplitudes, max_multipliers, marker="s", label="Max", color=PAPER_PALETTE["secondary"])
    ax.axhline(max_allowed, color=PAPER_PALETTE["neutral"], linestyle="--", label="Configured cap")
    ax.legend()

    save_paper_figure(fig, output_dir / "viral_scaling_curve.png")
    plt.close(fig)

    assert (output_dir / "viral_scaling_curve.png").exists()
