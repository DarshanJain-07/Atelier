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
from research_paper_tests.plotting_utils import (
    PAPER_PALETTE,
    apply_paper_style,
    compose_panel_grid,
    save_paper_figure,
    setup_plot,
)

matplotlib.use("Agg")
apply_paper_style()


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
    fig1, ax1 = setup_plot(
        title="Openness vs Engagement",
        xlabel="Openness",
        ylabel="Engagement",
    )
    ax1.plot(
        trait_values,
        metrics["Openness"]["engagement"],
        marker="o",
        color=PAPER_PALETTE["secondary"],
    )
    path1 = output_dir / "openness_vs_engagement.png"
    save_paper_figure(fig1, path1)
    plt.close(fig1)

    # Figure 2: Threat Engagement by Trait
    fig2, ax2 = setup_plot(
        title="Threat Engagement by Trait",
        xlabel="Trait Value",
        ylabel="Engagement",
    )
    ax2.plot(
        trait_values,
        metrics["Extraversion"]["engagement"],
        marker="o",
        label="Extraversion",
        color=PAPER_PALETTE["primary"],
    )
    ax2.plot(
        trait_values,
        metrics["Conscientiousness"]["engagement"],
        marker="s",
        label="Conscientiousness",
        color=PAPER_PALETTE["secondary"],
    )
    ax2.legend()
    path2 = output_dir / "threat_engagement_by_trait.png"
    save_paper_figure(fig2, path2)
    plt.close(fig2)

    # Figure 3: Extraversion vs Short-Term Attention
    fig3, ax3 = setup_plot(
        title="Extraversion vs Short-Term Attention",
        xlabel="Extraversion",
        ylabel="Attention Weight",
    )
    ax3.plot(
        trait_values,
        metrics["Extraversion"]["attention"][:, 10],
        marker="o",
        color=PAPER_PALETTE["primary"],
    )
    path3 = output_dir / "extraversion_vs_attention.png"
    save_paper_figure(fig3, path3)
    plt.close(fig3)

    # Figure 4: Trait vs Action Cost
    fig4, ax4 = setup_plot(
        title="Trait vs Action Cost",
        xlabel="Trait Value",
        ylabel="Action Cost",
    )
    ax4.plot(
        trait_values,
        metrics["Extraversion"]["action_cost"],
        marker="o",
        label="Extraversion",
        color=PAPER_PALETTE["accent"],
    )
    ax4.plot(
        trait_values,
        metrics["Neuroticism"]["action_cost"],
        marker="s",
        label="Neuroticism",
        color=PAPER_PALETTE["negative"],
    )
    ax4.legend()
    path4 = output_dir / "trait_vs_action_cost.png"
    save_paper_figure(fig4, path4)
    plt.close(fig4)

    compose_panel_grid(
        [path1, path2, path3, path4],
        output_dir.parent / "trait_sweeps.png",
        title="Trait Sweeps",
        columns=2,
    )

    assert path1.exists()
    assert path2.exists()
    assert path3.exists()
    assert path4.exists()
    assert path1.stat().st_size > 0
    assert path2.stat().st_size > 0
    assert path3.stat().st_size > 0
    assert path4.stat().st_size > 0
