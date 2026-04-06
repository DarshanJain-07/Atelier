from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import community as community_louvain
import numpy as np
import torch

from main import (
    DIMENSION_INDICES,
    build_debug_society,
    calculate_validation_metrics,
    clone_sim_config,
    consolidate_agent_memory,
    map_emotions_to_sentiment,
    run_debug_simulation,
)
from research_paper_tests._metrics import adjacency_to_graph, average_clustering, gini
from research_paper_tests.config_schema import (
    EMOTION_INDICES,
    PERSONALITY_TRAIT_COUNT,
    SENTIMENT_INDICES,
    WORLD_DIMENSION_COUNT,
    build_world,
    get_test_scenario,
    prepare_scenario_society,
    set_dimensions,
)
from research_paper_tests.test_emotion_direction_and_bridge_diffusion import (
    _bridge_diffusion_metrics,
)
from research_paper_tests.test_response_boundaries import _run_boundary_worlds, _scaled_world
from research_paper_tests.test_viral_scaling import _viral_scaling_curve
from schema import (
    SimConfig,
    emotions_to_behavior_aware_sentiment_distribution,
    emotions_to_sentiment_distribution,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "generated"
JSON_OUTPUT_PATH = OUTPUT_DIR / "paper_values_report.json"
MARKDOWN_OUTPUT_PATH = OUTPUT_DIR / "paper_values_report.md"


def _round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _round_list(values: np.ndarray | torch.Tensor | list[float], digits: int = 6) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [_round_float(value, digits=digits) for value in array.tolist()]


def _labelled_sentiment(values: np.ndarray | torch.Tensor | list[float]) -> dict[str, float]:
    rounded = _round_list(values)
    labels = ("Negative", "Neutral", "Positive")
    return {label: rounded[idx] for idx, label in enumerate(labels)}


def _labelled_emotions(values: np.ndarray | torch.Tensor | list[float]) -> dict[str, float]:
    rounded = _round_list(values)
    return {label: rounded[idx] for label, idx in EMOTION_INDICES.items()}


def _metric_line(label: str, value: float) -> str:
    return f"- {label}: `{value:.6f}`"


def _collect_sentiment_mapping_values() -> dict[str, Any]:
    config = SimConfig()
    anger_dominant_emotion = torch.tensor(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        dtype=torch.float32,
    )

    raw_sentiment = emotions_to_sentiment_distribution(anger_dominant_emotion)
    low_activity_sentiment = emotions_to_behavior_aware_sentiment_distribution(
        anger_dominant_emotion,
        0.10,
        neutral_acting_threshold=config.sentiment_neutrality_acting_threshold,
        activation=config.sentiment_neutrality_activation,
        leaky_slope=config.sentiment_neutrality_leaky_slope,
    )
    high_activity_sentiment = emotions_to_behavior_aware_sentiment_distribution(
        anger_dominant_emotion,
        0.30,
        neutral_acting_threshold=config.sentiment_neutrality_acting_threshold,
        activation=config.sentiment_neutrality_activation,
        leaky_slope=config.sentiment_neutrality_leaky_slope,
    )

    return {
        "raw": _labelled_sentiment(raw_sentiment),
        "low_activity": _labelled_sentiment(low_activity_sentiment),
        "high_activity": _labelled_sentiment(high_activity_sentiment),
        "neutral_gain_low_vs_raw": _round_float(low_activity_sentiment[1] - raw_sentiment[1]),
        "negative_drop_low_vs_raw": _round_float(raw_sentiment[0] - low_activity_sentiment[0]),
        "neutral_gain_low_vs_high": _round_float(low_activity_sentiment[1] - high_activity_sentiment[1]),
    }


def _collect_semantic_alignment_values(tmp_path: Path) -> dict[str, Any]:
    scenario = get_test_scenario("semantic_alignment")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "semantic_alignment",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="paper_values_semantic",
    )

    positive = run_debug_simulation(
        config,
        build_world(settings["positive_world"]),
        society=society,
        urgency=settings["urgency"],
    )
    negative = run_debug_simulation(
        config,
        build_world(settings["negative_world"]),
        society=society,
        urgency=settings["urgency"],
    )

    positive_center = positive.social_state["objective_center"]
    negative_center = negative.social_state["objective_center"]
    positive_sentiment = map_emotions_to_sentiment(
        positive_center,
        positive.social_state["acting_ratio"],
        config=config,
    )
    negative_sentiment = map_emotions_to_sentiment(
        negative_center,
        negative.social_state["acting_ratio"],
        config=config,
    )

    positive_match = calculate_validation_metrics(
        positive_center,
        settings["positive_sentiment_profile"],
    )
    positive_against_negative = calculate_validation_metrics(
        positive_center,
        settings["negative_sentiment_profile"],
    )
    negative_match = calculate_validation_metrics(
        negative_center,
        settings["negative_sentiment_profile"],
    )
    negative_against_positive = calculate_validation_metrics(
        negative_center,
        settings["positive_sentiment_profile"],
    )

    return {
        "seed": config.seed,
        "positive_world": {
            "acting_ratio": _round_float(positive.social_state["acting_ratio"]),
            "sentiment_valence": _round_float(positive.social_state["sentiment_valence"]),
            "sentiment": _labelled_sentiment(positive_sentiment),
            "wasserstein_match": _round_float(positive_match["wasserstein_distance"]),
            "wasserstein_against_negative": _round_float(
                positive_against_negative["wasserstein_distance"]
            ),
        },
        "negative_world": {
            "acting_ratio": _round_float(negative.social_state["acting_ratio"]),
            "sentiment_valence": _round_float(negative.social_state["sentiment_valence"]),
            "sentiment": _labelled_sentiment(negative_sentiment),
            "wasserstein_match": _round_float(negative_match["wasserstein_distance"]),
            "wasserstein_against_positive": _round_float(
                negative_against_positive["wasserstein_distance"]
            ),
        },
        "negative_minus_positive_negative_share": _round_float(
            negative_sentiment[SENTIMENT_INDICES["Negative"]]
            - positive_sentiment[SENTIMENT_INDICES["Negative"]]
        ),
    }


def _collect_accuracy_metrics_values(tmp_path: Path) -> dict[str, Any]:
    scenario = get_test_scenario("accuracy_metrics")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "accuracy_metrics",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="paper_values_accuracy",
    )

    result = run_debug_simulation(
        config,
        build_world(settings["world"]),
        society=society,
        urgency=settings["urgency"],
        is_personal=False,
    )
    emotion_center = result.social_state["objective_center"]
    sentiment = map_emotions_to_sentiment(
        emotion_center,
        result.social_state["acting_ratio"],
        config=config,
    )
    negative_match = calculate_validation_metrics(
        emotion_center,
        settings["matching_baseline"],
    )
    positive_mismatch = calculate_validation_metrics(
        emotion_center,
        settings["mismatched_baseline"],
    )

    return {
        "seed": config.seed,
        "acting_ratio": _round_float(result.social_state["acting_ratio"]),
        "sentiment_valence": _round_float(result.social_state["sentiment_valence"]),
        "sentiment": _labelled_sentiment(sentiment),
        "matching_wasserstein": _round_float(negative_match["wasserstein_distance"]),
        "mismatched_wasserstein": _round_float(
            positive_mismatch["wasserstein_distance"]
        ),
        "wasserstein_gap": _round_float(
            positive_mismatch["wasserstein_distance"]
            - negative_match["wasserstein_distance"]
        ),
    }


def _collect_response_boundary_values(tmp_path: Path) -> dict[str, Any]:
    dose_scenario = get_test_scenario("boundary_dose_response")
    dose_config = dose_scenario.sim_config()
    dose_settings = dose_scenario.settings()
    dose_society = prepare_scenario_society(
        "boundary_dose_response",
        tmp_path,
        enable_evolution=dose_config.enable_evolution,
        output_name="paper_values_boundary_dose",
    )

    dose_response = []
    for magnitude in dose_settings["magnitudes"]:
        result = run_debug_simulation(
            dose_config,
            _scaled_world(dose_settings["world_direction"], magnitude),
            society=dose_society,
            urgency=dose_settings["urgency"],
        )
        dose_response.append(
            {
                "magnitude": _round_float(magnitude),
                "mean_engagement": _round_float(result.engagement_scores.mean().item()),
                "acting_ratio": _round_float(result.social_state["acting_ratio"]),
                "sentiment_valence": _round_float(
                    result.social_state["sentiment_valence"]
                ),
            }
        )

    low_scenario = get_test_scenario("boundary_low_salience")
    low_config = low_scenario.sim_config()
    low_settings = low_scenario.settings()
    low_society = prepare_scenario_society(
        "boundary_low_salience",
        tmp_path,
        enable_evolution=low_config.enable_evolution,
        output_name="paper_values_boundary_low",
    )
    low_results = _run_boundary_worlds(
        low_config,
        low_society,
        low_settings["worlds"],
        low_settings["urgency"],
    )

    low_salience = {
        label: {
            "mean_engagement": _round_float(values["mean_engagement"]),
            "acting_ratio": _round_float(values["acting_ratio"]),
            "sentiment_valence": _round_float(values["valence"]),
            "sentiment": _labelled_sentiment(values["sentiment"]),
        }
        for label, values in low_results.items()
    }

    return {
        "dose_response": dose_response,
        "low_salience_worlds": low_salience,
    }


def _collect_emotion_and_bridge_values(tmp_path: Path) -> dict[str, Any]:
    emotion_scenario = get_test_scenario("emotion_directionality")
    emotion_config = emotion_scenario.sim_config()
    emotion_settings = emotion_scenario.settings()
    society = prepare_scenario_society(
        "emotion_directionality",
        tmp_path,
        enable_evolution=emotion_config.enable_evolution,
        output_name="paper_values_emotion_directionality",
    )

    worlds = {}
    for label, world_values in emotion_settings["worlds"].items():
        result = run_debug_simulation(
            emotion_config,
            build_world(world_values),
            society=society,
            urgency=emotion_settings["urgency"],
        )
        center = np.asarray(result.social_state["objective_center"], dtype=np.float64)
        sentiment = map_emotions_to_sentiment(
            center,
            result.social_state["acting_ratio"],
            config=emotion_config,
        )
        worlds[label] = {
            "dominant_emotion": result.social_state["dominant_emotion"],
            "sentiment_valence": _round_float(result.social_state["sentiment_valence"]),
            "acting_ratio": _round_float(result.social_state["acting_ratio"]),
            "emotion_center": _labelled_emotions(center),
            "sentiment": _labelled_sentiment(sentiment),
        }

    bridge_scenario = get_test_scenario("bridge_diffusion")
    bridge_config = bridge_scenario.sim_config()
    bridge_settings = bridge_scenario.settings()
    bridge_metrics = _bridge_diffusion_metrics(bridge_config, bridge_settings)

    return {
        "world_directionality": worlds,
        "bridge_diffusion": {
            "acting_ratio_without_bridge": _round_float(
                bridge_metrics["without_bridge"]["acting_ratio"]
            ),
            "acting_ratio_with_bridge": _round_float(
                bridge_metrics["with_bridge"]["acting_ratio"]
            ),
            "acting_ratio_gain": _round_float(
                bridge_metrics["with_bridge"]["acting_ratio"]
                - bridge_metrics["without_bridge"]["acting_ratio"]
            ),
            "community_b_local_arousal_without_bridge": _round_float(
                bridge_metrics["without_b_local_arousal"]
            ),
            "community_b_local_arousal_with_bridge": _round_float(
                bridge_metrics["with_b_local_arousal"]
            ),
            "community_b_local_arousal_gain": _round_float(
                bridge_metrics["with_b_local_arousal"]
                - bridge_metrics["without_b_local_arousal"]
            ),
        },
    }


def _collect_wealth_gini_values(tmp_path: Path) -> dict[str, Any]:
    baseline_scenario = get_test_scenario("wealth_gini_baseline")
    baseline_config = baseline_scenario.sim_config()
    evolved_config = get_test_scenario("wealth_gini_evolved").sim_config()

    baseline_society = prepare_scenario_society(
        "wealth_gini_baseline",
        tmp_path,
        enable_evolution=baseline_config.enable_evolution,
        output_name="paper_values_wealth_baseline",
    )
    evolved_society = prepare_scenario_society(
        "wealth_gini_evolved",
        tmp_path,
        enable_evolution=evolved_config.enable_evolution,
        output_name="paper_values_wealth_evolved",
    )

    baseline_gini = gini(
        baseline_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()
    )
    evolved_gini = gini(
        evolved_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()
    )

    return {
        "baseline": _round_float(baseline_gini),
        "evolved": _round_float(evolved_gini),
        "absolute_delta": _round_float(evolved_gini - baseline_gini),
    }


def _collect_network_clustering_values() -> dict[str, Any]:
    from main import apply_triadic_closure_for_debug, create_topology_for_debug

    scenario = get_test_scenario("network_clustering_closure")
    backbone_config = get_test_scenario("network_clustering_backbone").sim_config()
    closure_config = scenario.sim_config()
    settings = scenario.settings()

    torch.manual_seed(settings["torch_seed"])
    np.random.seed(settings["numpy_seed"])
    exposures = torch.randn(backbone_config.num_agents, WORLD_DIMENSION_COUNT)
    personalities = torch.sigmoid(
        torch.randn(backbone_config.num_agents, PERSONALITY_TRAIT_COUNT)
    )
    influence = np.random.lognormal(
        mean=settings["influence_mean"],
        sigma=settings["influence_std"],
        size=backbone_config.num_agents,
    )

    backbone = create_topology_for_debug(
        backbone_config,
        exposures,
        personalities,
        influence,
    )
    refined = apply_triadic_closure_for_debug(closure_config, backbone)
    backbone_clustering = average_clustering(backbone)
    refined_clustering = average_clustering(refined)

    return {
        "backbone": _round_float(backbone_clustering),
        "with_triadic_closure": _round_float(refined_clustering),
        "absolute_gain": _round_float(refined_clustering - backbone_clustering),
    }


def _collect_louvain_modularity_values(tmp_path: Path) -> dict[str, Any]:
    low_scenario = get_test_scenario("louvain_low")
    low_config = low_scenario.sim_config()
    high_config = get_test_scenario("louvain_high").sim_config()
    settings = low_scenario.settings()

    low_society = prepare_scenario_society(
        "louvain_low",
        tmp_path,
        enable_evolution=low_config.enable_evolution,
        output_name="paper_values_louvain_low",
    )
    high_society = prepare_scenario_society(
        "louvain_high",
        tmp_path,
        enable_evolution=high_config.enable_evolution,
        output_name="paper_values_louvain_high",
    )

    low_graph = adjacency_to_graph(low_society.adjacency_matrix)
    high_graph = adjacency_to_graph(high_society.adjacency_matrix)

    low_partition = community_louvain.best_partition(
        low_graph,
        random_state=settings["partition_seed"],
    )
    high_partition = community_louvain.best_partition(
        high_graph,
        random_state=settings["partition_seed"],
    )

    low_modularity = community_louvain.modularity(low_partition, low_graph)
    high_modularity = community_louvain.modularity(high_partition, high_graph)

    return {
        "low_homophily": _round_float(low_modularity),
        "high_homophily": _round_float(high_modularity),
        "absolute_gain": _round_float(high_modularity - low_modularity),
    }


def _collect_memory_rehearsal_values() -> dict[str, Any]:
    scenario = get_test_scenario("memory_rehearsal")
    config = scenario.sim_config()
    settings = scenario.settings()

    memory = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    context = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    set_dimensions(context, settings["context"])

    isolated = consolidate_agent_memory(
        config,
        memory,
        context,
        social_rehearsal_factor=settings["isolated_rehearsal"],
    )
    rehearsed = consolidate_agent_memory(
        config,
        memory,
        context,
        social_rehearsal_factor=settings["shared_rehearsal"],
    )

    isolated_curve = [_round_float(torch.norm(isolated).item())]
    rehearsed_curve = [_round_float(torch.norm(rehearsed).item())]

    for _ in range(settings["decay_steps"]):
        isolated = consolidate_agent_memory(
            config,
            isolated,
            torch.zeros_like(context),
            social_rehearsal_factor=settings["isolated_rehearsal"],
        )
        rehearsed = consolidate_agent_memory(
            config,
            rehearsed,
            torch.zeros_like(context),
            social_rehearsal_factor=settings["shared_rehearsal"],
        )
        isolated_curve.append(_round_float(torch.norm(isolated).item()))
        rehearsed_curve.append(_round_float(torch.norm(rehearsed).item()))

    return {
        "isolated_curve": isolated_curve,
        "rehearsed_curve": rehearsed_curve,
        "final_norm_gain": _round_float(rehearsed_curve[-1] - isolated_curve[-1]),
    }


def _collect_algorithmic_filter_values(tmp_path: Path) -> dict[str, Any]:
    scenario = get_test_scenario("algorithmic_filter_bubble")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "algorithmic_filter_bubble",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="paper_values_algorithmic_bubble",
    )

    world = build_world(settings["world"])
    baseline_config = clone_sim_config(config, use_algorithmic_amplification=False)
    baseline_society = build_debug_society(
        baseline_config,
        society.exposures,
        society.personalities,
        society.affinities,
        society.metadata["Influence"].to_numpy(),
        society.adjacency_matrix,
        society.memory.clone(),
        society.metadata.copy(),
    )

    baseline = run_debug_simulation(
        baseline_config,
        world,
        society=baseline_society,
        urgency=settings["urgency"],
    )
    amplified = run_debug_simulation(
        config,
        world,
        society=society,
        urgency=settings["urgency"],
    )

    world_shift = torch.abs(amplified.final_world_tensor - world)

    return {
        "baseline_mean_engagement": _round_float(baseline.engagement_scores.mean().item()),
        "amplified_mean_engagement": _round_float(
            amplified.engagement_scores.mean().item()
        ),
        "engagement_gain": _round_float(
            amplified.engagement_scores.mean().item()
            - baseline.engagement_scores.mean().item()
        ),
        "max_world_shift": _round_float(world_shift.max().item()),
    }


def _collect_viral_scaling_values() -> dict[str, Any]:
    scenario = get_test_scenario("viral_scaling")
    config = scenario.sim_config()
    settings = scenario.settings()
    amplitudes, mean_multiplier, max_multiplier = _viral_scaling_curve(config, settings)
    slopes = np.diff(mean_multiplier) / np.diff(amplitudes)
    peak_idx = int(np.argmax(slopes))

    return {
        "amplitudes": _round_list(amplitudes),
        "mean_multiplier": _round_list(mean_multiplier),
        "max_multiplier": _round_list(max_multiplier),
        "configured_cap": _round_float(1.0 + config.max_viral_multiplier),
        "peak_slope_segment": {
            "from_amplitude": _round_float(amplitudes[peak_idx]),
            "to_amplitude": _round_float(amplitudes[peak_idx + 1]),
            "slope": _round_float(slopes[peak_idx]),
        },
    }


def _build_report(tmp_path: Path) -> dict[str, Any]:
    return {
        "report_name": "research_paper_values",
        "generation_command": ".venv/bin/pytest research_paper_tests/test_paper_values_report.py -q",
        "sections": {
            "sentiment_mapping": _collect_sentiment_mapping_values(),
            "semantic_alignment": _collect_semantic_alignment_values(tmp_path),
            "accuracy_metrics": _collect_accuracy_metrics_values(tmp_path),
            "response_boundaries": _collect_response_boundary_values(tmp_path),
            "emotion_directionality_and_bridge_diffusion": _collect_emotion_and_bridge_values(
                tmp_path
            ),
            "wealth_gini": _collect_wealth_gini_values(tmp_path),
            "network_clustering": _collect_network_clustering_values(),
            "louvain_modularity": _collect_louvain_modularity_values(tmp_path),
            "memory_rehearsal": _collect_memory_rehearsal_values(),
            "algorithmic_filter_bubble": _collect_algorithmic_filter_values(tmp_path),
            "viral_scaling": _collect_viral_scaling_values(),
        },
    }


def _render_sentiment_rows(section: dict[str, Any]) -> list[str]:
    rows = [
        "| Condition | Negative | Neutral | Positive |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for label in ("raw", "low_activity", "high_activity"):
        sentiment = section[label]
        rows.append(
            "| {label} | {negative:.6f} | {neutral:.6f} | {positive:.6f} |".format(
                label=label.replace("_", " ").title(),
                negative=sentiment["Negative"],
                neutral=sentiment["Neutral"],
                positive=sentiment["Positive"],
            )
        )
    return rows


def _render_markdown(report: dict[str, Any]) -> str:
    sections = report["sections"]
    sentiment = sections["sentiment_mapping"]
    semantic = sections["semantic_alignment"]
    accuracy = sections["accuracy_metrics"]
    boundaries = sections["response_boundaries"]
    emotions = sections["emotion_directionality_and_bridge_diffusion"]
    wealth = sections["wealth_gini"]
    clustering = sections["network_clustering"]
    modularity = sections["louvain_modularity"]
    memory = sections["memory_rehearsal"]
    algorithmic = sections["algorithmic_filter_bubble"]
    viral = sections["viral_scaling"]

    lines = [
        "# Research Paper Numeric Results",
        "",
        "This file is generated from the pytest-backed research harness.",
        "",
        f"Regenerate with `{report['generation_command']}`.",
        "",
        "## Sentiment Mapping",
        "",
        *_render_sentiment_rows(sentiment),
        "",
        _metric_line(
            "Neutral gain at low activity vs raw",
            sentiment["neutral_gain_low_vs_raw"],
        ),
        _metric_line(
            "Negative drop at low activity vs raw",
            sentiment["negative_drop_low_vs_raw"],
        ),
        _metric_line(
            "Neutral gain at low activity vs high activity",
            sentiment["neutral_gain_low_vs_high"],
        ),
        "",
        "## Semantic Alignment",
        "",
        _metric_line(
            "Positive world Wasserstein match",
            semantic["positive_world"]["wasserstein_match"],
        ),
        _metric_line(
            "Positive world mismatch against negative baseline",
            semantic["positive_world"]["wasserstein_against_negative"],
        ),
        _metric_line(
            "Negative world Wasserstein match",
            semantic["negative_world"]["wasserstein_match"],
        ),
        _metric_line(
            "Negative world mismatch against positive baseline",
            semantic["negative_world"]["wasserstein_against_positive"],
        ),
        _metric_line(
            "Negative-minus-positive negative sentiment share",
            semantic["negative_minus_positive_negative_share"],
        ),
        "",
        "## Accuracy Metrics",
        "",
        _metric_line("Matching Wasserstein distance", accuracy["matching_wasserstein"]),
        _metric_line(
            "Mismatched Wasserstein distance",
            accuracy["mismatched_wasserstein"],
        ),
        _metric_line("Wasserstein gap", accuracy["wasserstein_gap"]),
        "",
        "## Response Boundaries",
        "",
        "| Magnitude | Mean Engagement | Acting Ratio | Sentiment Valence |",
        "| :--- | ---: | ---: | ---: |",
    ]

    for row in boundaries["dose_response"]:
        lines.append(
            "| {magnitude:.6f} | {mean_engagement:.6f} | {acting_ratio:.6f} | {sentiment_valence:.6f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "### Low-Salience Worlds",
            "",
            "| World | Mean Engagement | Acting Ratio | Sentiment Valence |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )

    for label, values in boundaries["low_salience_worlds"].items():
        lines.append(
            "| {label} | {mean_engagement:.6f} | {acting_ratio:.6f} | {sentiment_valence:.6f} |".format(
                label=label,
                mean_engagement=values["mean_engagement"],
                acting_ratio=values["acting_ratio"],
                sentiment_valence=values["sentiment_valence"],
            )
        )

    lines.extend(
        [
            "",
            "## Emotion Directionality",
            "",
            "| World | Dominant Emotion | Acting Ratio | Sentiment Valence |",
            "| :--- | :--- | ---: | ---: |",
        ]
    )

    for label, values in emotions["world_directionality"].items():
        lines.append(
            "| {label} | {dominant_emotion} | {acting_ratio:.6f} | {sentiment_valence:.6f} |".format(
                label=label,
                dominant_emotion=values["dominant_emotion"],
                acting_ratio=values["acting_ratio"],
                sentiment_valence=values["sentiment_valence"],
            )
        )

    lines.extend(
        [
            "",
            "## Bridge Diffusion",
            "",
            _metric_line(
                "Acting ratio without bridge",
                emotions["bridge_diffusion"]["acting_ratio_without_bridge"],
            ),
            _metric_line(
                "Acting ratio with bridge",
                emotions["bridge_diffusion"]["acting_ratio_with_bridge"],
            ),
            _metric_line(
                "Acting ratio gain",
                emotions["bridge_diffusion"]["acting_ratio_gain"],
            ),
            _metric_line(
                "Community-B local arousal gain",
                emotions["bridge_diffusion"]["community_b_local_arousal_gain"],
            ),
            "",
            "## Inequality And Topology",
            "",
            _metric_line("Baseline wealth Gini", wealth["baseline"]),
            _metric_line("Evolved wealth Gini", wealth["evolved"]),
            _metric_line("Wealth Gini delta", wealth["absolute_delta"]),
            _metric_line("Backbone clustering", clustering["backbone"]),
            _metric_line(
                "Clustering with triadic closure",
                clustering["with_triadic_closure"],
            ),
            _metric_line("Clustering gain", clustering["absolute_gain"]),
            _metric_line("Low-homophily Louvain modularity", modularity["low_homophily"]),
            _metric_line(
                "High-homophily Louvain modularity",
                modularity["high_homophily"],
            ),
            _metric_line("Modularity gain", modularity["absolute_gain"]),
            "",
            "## Memory, Amplification, And Virality",
            "",
            _metric_line(
                "Memory final norm gain from rehearsal",
                memory["final_norm_gain"],
            ),
            _metric_line(
                "Algorithmic amplification engagement gain",
                algorithmic["engagement_gain"],
            ),
            _metric_line(
                "Algorithmic amplification max world shift",
                algorithmic["max_world_shift"],
            ),
            _metric_line("Configured viral cap", viral["configured_cap"]),
            _metric_line(
                "Peak viral slope",
                viral["peak_slope_segment"]["slope"],
            ),
            "",
        ]
    )

    return "\n".join(lines)


def test_generate_paper_values_report(tmp_path):
    OUTPUT_DIR.mkdir(exist_ok=True)

    report = _build_report(tmp_path)
    markdown = _render_markdown(report)

    JSON_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    MARKDOWN_OUTPUT_PATH.write_text(markdown + "\n", encoding="utf-8")

    assert JSON_OUTPUT_PATH.exists()
    assert MARKDOWN_OUTPUT_PATH.exists()
    assert report["sections"]["accuracy_metrics"]["wasserstein_gap"] > 0.0
    assert report["sections"]["wealth_gini"]["absolute_delta"] != 0.0
