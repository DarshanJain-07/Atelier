from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from main import map_emotions_to_sentiment, run_debug_simulation
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.plotting_utils import (
    PAPER_PALETTE,
    SENTIMENT_COLORS,
    apply_paper_style,
    compose_panel_grid,
    save_paper_figure,
    setup_plot,
)

matplotlib.use("Agg")
apply_paper_style()


def _scaled_world(direction: dict[str, float], magnitude: float):
    return build_world(
        {
            dimension_name: dimension_value * magnitude
            for dimension_name, dimension_value in direction.items()
        }
    )


def _run_boundary_worlds(config, society, worlds: dict[str, dict[str, float]], urgency: float):
    results = {}
    for label, world_values in worlds.items():
        result = run_debug_simulation(
            config,
            build_world(world_values),
            society=society,
            urgency=urgency,
        )
        sentiment = map_emotions_to_sentiment(
            result.social_state["objective_center"],
            result.social_state["acting_ratio"],
            config=config,
        )
        results[label] = {
            "mean_engagement": float(result.engagement_scores.mean().item()),
            "acting_ratio": float(result.social_state["acting_ratio"]),
            "valence": float(result.social_state["sentiment_valence"]),
            "sentiment": np.asarray(sentiment, dtype=np.float64),
        }
    return results


def test_event_magnitude_monotonically_increases_engagement_and_action(tmp_path):
    scenario = get_test_scenario("boundary_dose_response")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "boundary_dose_response",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="boundary_dose_response",
    )

    magnitudes = settings["magnitudes"]
    mean_engagement = []
    acting_ratio = []
    valence = []

    for magnitude in magnitudes:
        result = run_debug_simulation(
            config,
            _scaled_world(settings["world_direction"], magnitude),
            society=society,
            urgency=settings["urgency"],
        )
        mean_engagement.append(float(result.engagement_scores.mean().item()))
        acting_ratio.append(float(result.social_state["acting_ratio"]))
        valence.append(float(result.social_state["sentiment_valence"]))

    tol = settings["monotonic_tolerance"]
    engagement_diffs = np.diff(mean_engagement)
    acting_diffs = np.diff(acting_ratio)
    valence_diffs = np.diff(valence)

    assert (engagement_diffs >= -tol).sum() >= len(engagement_diffs) - 1
    assert (acting_diffs >= -tol).sum() >= len(acting_diffs) - 1
    assert (valence_diffs <= tol).sum() >= len(valence_diffs) - 1

    assert mean_engagement[-1] - mean_engagement[1] >= settings["min_engagement_gain"]
    assert acting_ratio[-1] - acting_ratio[1] >= settings["min_acting_gain"]
    assert valence[1] - valence[-1] >= settings["min_valence_drop"]


def test_low_salience_worlds_keep_reaction_bounded_before_escalation(tmp_path):
    scenario = get_test_scenario("boundary_low_salience")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "boundary_low_salience",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="boundary_low_salience",
    )

    results = _run_boundary_worlds(
        config,
        society,
        settings["worlds"],
        settings["urgency"],
    )

    for label in settings["low_salience_labels"]:
        assert results[label]["mean_engagement"] <= settings["max_low_engagement"]
        assert abs(results[label]["valence"]) <= settings["max_low_abs_valence"]

    salient = results[settings["salient_label"]]
    zero_case = results["Zero"]

    assert salient["mean_engagement"] >= settings["min_salient_engagement"]
    assert salient["acting_ratio"] >= settings["min_salient_acting_ratio"]
    assert zero_case["valence"] - salient["valence"] >= settings["min_salient_valence_gap"]
    assert zero_case["sentiment"][1] > zero_case["sentiment"][0]
    assert zero_case["sentiment"][1] > zero_case["sentiment"][2]
    assert all(
        salient["acting_ratio"] - results[label]["acting_ratio"]
        >= settings["min_salient_acting_gap"]
        for label in settings["low_salience_labels"]
        if label != "Zero"
    )


def test_generate_response_boundaries_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated" / "response_boundaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    dose_scenario = get_test_scenario("boundary_dose_response")
    dose_config = dose_scenario.sim_config()
    dose_settings = dose_scenario.settings()
    dose_society = prepare_scenario_society(
        "boundary_dose_response",
        tmp_path,
        enable_evolution=dose_config.enable_evolution,
        output_name="response_boundaries_dose",
    )

    magnitudes = dose_settings["magnitudes"]
    mean_engagement = []
    acting_ratio = []
    valence = []

    for magnitude in magnitudes:
        result = run_debug_simulation(
            dose_config,
            _scaled_world(dose_settings["world_direction"], magnitude),
            society=dose_society,
            urgency=dose_settings["urgency"],
        )
        mean_engagement.append(float(result.engagement_scores.mean().item()))
        acting_ratio.append(float(result.social_state["acting_ratio"]))
        valence.append(float(result.social_state["sentiment_valence"]))

    neutral_scenario = get_test_scenario("boundary_low_salience")
    neutral_config = neutral_scenario.sim_config()
    neutral_settings = neutral_scenario.settings()
    neutral_society = prepare_scenario_society(
        "boundary_low_salience",
        tmp_path,
        enable_evolution=neutral_config.enable_evolution,
        output_name="response_boundaries_low_salience",
    )
    neutral_results = _run_boundary_worlds(
        neutral_config,
        neutral_society,
        neutral_settings["worlds"],
        neutral_settings["urgency"],
    )

    # Figure 1: Dose Response - Engagement
    fig1, ax1 = setup_plot(
        title="Dose Response: Engagement",
        xlabel="Event Magnitude",
        ylabel="Mean Engagement",
    )
    ax1.plot(
        magnitudes, 
        mean_engagement, 
        marker="o", 
        color=PAPER_PALETTE["primary"],
    )
    path1 = output_dir / "dose_response_engagement.png"
    save_paper_figure(fig1, path1)
    plt.close(fig1)

    # Figure 2: Dose Response - Collective Action
    fig2, ax2 = setup_plot(
        title="Dose Response: Collective Action",
        xlabel="Event Magnitude",
        ylabel="Acting Ratio",
    )
    ax2.plot(
        magnitudes, 
        acting_ratio, 
        marker="o", 
        color=PAPER_PALETTE["negative"],
    )
    path2 = output_dir / "dose_response_action.png"
    save_paper_figure(fig2, path2)
    plt.close(fig2)

    labels = list(neutral_settings["worlds"])
    x = np.arange(len(labels))
    width = 0.38

    # Figure 3: Low-Salience Reaction Boundary
    fig3, ax3 = setup_plot(
        title="Low-Salience Reaction Boundary",
        xlabel="World",
        ylabel="Reaction Strength",
    )
    ax3.bar(
        x - width / 2,
        [neutral_results[label]["mean_engagement"] for label in labels],
        width=width,
        label="Mean engagement",
        color=PAPER_PALETTE["secondary"],
    )
    ax3.bar(
        x + width / 2,
        [neutral_results[label]["acting_ratio"] for label in labels],
        width=width,
        label="Acting ratio",
        color=PAPER_PALETTE["accent"],
    )
    ax3.set_xticks(x, labels, rotation=15)
    ax3.legend()
    path3 = output_dir / "low_salience_boundary.png"
    save_paper_figure(fig3, path3)
    plt.close(fig3)

    # Figure 4: Low-Salience Sentiment Composition
    fig4, ax4 = setup_plot(
        title="Low-Salience Sentiment Composition",
        xlabel="World",
        ylabel="Share",
    )
    sentiment_colors = SENTIMENT_COLORS
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    bottoms = np.zeros(len(labels), dtype=np.float64)

    for idx, sentiment_label in enumerate(sentiment_labels):
        heights = np.array(
            [neutral_results[label]["sentiment"][idx] for label in labels],
            dtype=np.float64,
        )
        ax4.bar(
            x,
            heights,
            bottom=bottoms,
            color=sentiment_colors[idx],
            label=sentiment_label,
        )
        bottoms += heights
    ax4.set_xticks(x, labels, rotation=15)
    ax4.set_ylim(0.0, 1.0)
    ax4.legend()
    path4 = output_dir / "low_salience_sentiment.png"
    save_paper_figure(fig4, path4)
    plt.close(fig4)

    compose_panel_grid(
        [path1, path2, path3, path4],
        output_dir.parent / "response_boundaries.png",
        title="Response Boundaries",
        columns=2,
    )

    assert path1.exists()
    assert path2.exists()
    assert path3.exists()
    assert path4.exists()
