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

matplotlib.use("Agg")


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
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "response_boundaries.png"

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

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    # x-axis: event magnitude along a fixed threat direction.
    # y-axis: mean engagement. The rise shows where weak events stop being ignored
    # and begin consistently pulling the population into the cognitive pipeline.
    axes[0].plot(magnitudes, mean_engagement, marker="o", linewidth=2, color="#1d3557")
    axes[0].set_title("Dose Response: Engagement")
    axes[0].set_xlabel("Event Magnitude")
    axes[0].set_ylabel("Mean Engagement")

    # x-axis: the same event magnitudes.
    # y-axis: acting ratio after aggregation. This marks the behavioral boundary
    # between internal attention and outward collective action.
    axes[1].plot(magnitudes, acting_ratio, marker="o", linewidth=2, color="#e76f51")
    axes[1].set_title("Dose Response: Collective Action")
    axes[1].set_xlabel("Event Magnitude")
    axes[1].set_ylabel("Acting Ratio")

    labels = list(neutral_settings["worlds"])
    x = np.arange(len(labels))
    width = 0.38

    # x-axis: low-salience and salient comparison worlds.
    # y-axis: grouped reaction metrics. Weak worlds should stay near the floor,
    # while the salient control shows where escalation begins.
    axes[2].bar(
        x - width / 2,
        [neutral_results[label]["mean_engagement"] for label in labels],
        width=width,
        label="Mean engagement",
        color="#457b9d",
    )
    axes[2].bar(
        x + width / 2,
        [neutral_results[label]["acting_ratio"] for label in labels],
        width=width,
        label="Acting ratio",
        color="#f4a261",
    )
    axes[2].set_xticks(x, labels, rotation=15)
    axes[2].set_title("Low-Salience Reaction Boundary")
    axes[2].set_ylabel("Reaction Strength")
    axes[2].legend(fontsize=8)

    sentiment_colors = ["#d62828", "#adb5bd", "#2a9d8f"]
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    bottoms = np.zeros(len(labels), dtype=np.float64)

    # x-axis: the same comparison worlds.
    # y-axis: sentiment composition of the aggregate state. This lets us inspect
    # whether "low salience" really stays centered or already leans negative/positive.
    for idx, sentiment_label in enumerate(sentiment_labels):
        heights = np.array(
            [neutral_results[label]["sentiment"][idx] for label in labels],
            dtype=np.float64,
        )
        axes[3].bar(
            x,
            heights,
            bottom=bottoms,
            color=sentiment_colors[idx],
            label=sentiment_label,
        )
        bottoms += heights
    axes[3].set_xticks(x, labels, rotation=15)
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_title("Low-Salience Sentiment Composition")
    axes[3].set_ylabel("Share")
    axes[3].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.2)

    fig.suptitle("Response Boundaries", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
