from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import build_debug_society, run_debug_simulation
from research_paper_tests.config_schema import (
    PERSONALITY_INDICES,
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
    set_dimensions,
    zero_personalities,
)

matplotlib.use("Agg")


def _sweep_society(config, trait_name: str, trait_values: list[float], exposure_values: dict[str, float], fill: float):
    exposures = torch.zeros(len(trait_values), WORLD_DIMENSION_COUNT)
    set_dimensions(exposures, exposure_values)
    personalities = zero_personalities(len(trait_values), fill=fill)
    personalities[:, PERSONALITY_INDICES[trait_name]] = torch.tensor(
        trait_values,
        dtype=torch.float32,
    )
    return build_debug_society(config, exposures, personalities)


def _action_cost_curve(personalities: torch.Tensor, base_action_cost: float):
    extraversion = personalities[:, PERSONALITY_INDICES["Extraversion"]]
    neuroticism = personalities[:, PERSONALITY_INDICES["Neuroticism"]]
    influence = torch.ones(personalities.shape[0], dtype=torch.float32)
    return torch.clamp(
        base_action_cost
        - 0.1 * extraversion
        - 0.1 * neuroticism
        - 0.05 * torch.log1p(influence),
        min=0.05,
    ).numpy()


def _trait_sweep_metrics(config, settings):
    trait_values = settings["trait_values"]
    fill = settings["baseline_fill"]
    urgency = settings["urgency"]
    metrics = {}

    sweep_specs = {
        "Openness": {
            "world": build_world(settings["openness_world"]),
            "exposures": settings["openness_exposure"],
        },
        "Extraversion": {
            "world": build_world(settings["threat_world"]),
            "exposures": settings["threat_world"],
        },
        "Neuroticism": {
            "world": build_world(settings["threat_world"]),
            "exposures": settings["threat_world"],
        },
        "Conscientiousness": {
            "world": build_world(settings["threat_world"]),
            "exposures": settings["threat_world"],
        },
    }

    for trait_name, spec in sweep_specs.items():
        society = _sweep_society(
            config,
            trait_name,
            trait_values,
            spec["exposures"],
            fill,
        )
        result = run_debug_simulation(
            config,
            spec["world"],
            society=society,
            urgency=urgency,
        )
        metrics[trait_name] = {
            "engagement": result.engagement_scores.detach().cpu().numpy(),
            "attention": result.attention_weights.detach().cpu().numpy(),
            "action_cost": _action_cost_curve(
                society.personalities,
                getattr(config, "base_action_cost", 0.5),
            ),
        }

    return np.asarray(trait_values, dtype=np.float64), metrics


def test_trait_sweeps_reveal_monotonic_behavioral_gradients():
    scenario = get_test_scenario("trait_sweeps")
    config = scenario.sim_config()
    settings = scenario.settings()
    trait_values, metrics = _trait_sweep_metrics(config, settings)
    tol = settings["monotonic_tolerance"]

    openness_engagement = metrics["Openness"]["engagement"]
    extraversion_short_term_attention = metrics["Extraversion"]["attention"][:, 10]
    conscientiousness_engagement = metrics["Conscientiousness"]["engagement"]
    extraversion_action_cost = metrics["Extraversion"]["action_cost"]
    neuroticism_action_cost = metrics["Neuroticism"]["action_cost"]

    assert np.all(np.diff(openness_engagement) >= -tol)
    assert np.all(np.diff(extraversion_short_term_attention) >= -tol)
    assert np.all(np.diff(conscientiousness_engagement) <= tol)
    assert np.all(np.diff(extraversion_action_cost) <= tol)
    assert np.all(np.diff(neuroticism_action_cost) <= tol)

    assert openness_engagement[-1] - openness_engagement[0] >= settings["min_openness_gain"]
    assert (
        extraversion_short_term_attention[-1] - extraversion_short_term_attention[0]
        >= settings["min_extraversion_attention_gain"]
    )
    assert (
        extraversion_action_cost[0] - extraversion_action_cost[-1]
        >= settings["min_extraversion_cost_drop"]
    )
    assert (
        neuroticism_action_cost[0] - neuroticism_action_cost[-1]
        >= settings["min_neuroticism_cost_drop"]
    )
    assert (
        conscientiousness_engagement[0] - conscientiousness_engagement[-1]
        >= settings["min_conscientiousness_engagement_drop"]
    )


def test_generate_trait_sweeps_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated" / "trait_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario = get_test_scenario("trait_sweeps")
    config = scenario.sim_config()
    settings = scenario.settings()
    trait_values, metrics = _trait_sweep_metrics(config, settings)

    # Figure 1: Openness vs Engagement
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(
        trait_values,
        metrics["Openness"]["engagement"],
        marker="o",
        linewidth=2,
        color="#457b9d",
    )
    ax1.set_title("Openness vs Engagement", fontsize=16)
    ax1.set_xlabel("Openness", fontsize=12)
    ax1.set_ylabel("Engagement", fontsize=12)
    ax1.grid(True, alpha=0.2)
    path1 = output_dir / "openness_vs_engagement.png"
    fig1.savefig(path1, dpi=220, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Threat Engagement by Trait
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(
        trait_values,
        metrics["Extraversion"]["engagement"],
        marker="o",
        linewidth=2,
        label="Extraversion",
        color="#e76f51",
    )
    ax2.plot(
        trait_values,
        metrics["Conscientiousness"]["engagement"],
        marker="s",
        linewidth=2,
        label="Conscientiousness",
        color="#6d597a",
    )
    ax2.set_title("Threat Engagement by Trait", fontsize=16)
    ax2.set_xlabel("Trait Value", fontsize=12)
    ax2.set_ylabel("Engagement", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)
    path2 = output_dir / "threat_engagement_by_trait.png"
    fig2.savefig(path2, dpi=220, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: Extraversion vs Short-Term Attention
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.plot(
        trait_values,
        metrics["Extraversion"]["attention"][:, 10],
        marker="o",
        linewidth=2,
        color="#264653",
    )
    ax3.set_title("Extraversion vs Short-Term Attention", fontsize=16)
    ax3.set_xlabel("Extraversion", fontsize=12)
    ax3.set_ylabel("Attention Weight", fontsize=12)
    ax3.grid(True, alpha=0.2)
    path3 = output_dir / "extraversion_vs_attention.png"
    fig3.savefig(path3, dpi=220, bbox_inches="tight")
    plt.close(fig3)

    # Figure 4: Trait vs Action Cost
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.plot(
        trait_values,
        metrics["Extraversion"]["action_cost"],
        marker="o",
        linewidth=2,
        label="Extraversion",
        color="#f4a261",
    )
    ax4.plot(
        trait_values,
        metrics["Neuroticism"]["action_cost"],
        marker="s",
        linewidth=2,
        label="Neuroticism",
        color="#2a9d8f",
    )
    ax4.set_title("Trait vs Action Cost", fontsize=16)
    ax4.set_xlabel("Trait Value", fontsize=12)
    ax4.set_ylabel("Action Cost", fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.2)
    path4 = output_dir / "trait_vs_action_cost.png"
    fig4.savefig(path4, dpi=220, bbox_inches="tight")
    plt.close(fig4)

    assert path1.exists()
    assert path2.exists()
    assert path3.exists()
    assert path4.exists()
    assert path1.stat().st_size > 0
    assert path2.stat().st_size > 0
    assert path3.stat().st_size > 0
    assert path4.stat().st_size > 0
