from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import aggregate_social_state, map_emotions_to_sentiment, run_debug_simulation
from research_paper_tests.config_schema import (
    EMOTION_INDICES,
    build_world,
    get_test_scenario,
    prepare_scenario_society,
    set_emotions,
    zero_emotions,
)

matplotlib.use("Agg")


def _row_normalized_adjacency(node_count: int, undirected_edges: list[tuple[int, int]]) -> torch.Tensor:
    edge_set: set[tuple[int, int]] = set()
    for src, dst in undirected_edges:
        if src == dst:
            continue
        edge_set.add((src, dst))
        edge_set.add((dst, src))

    ordered_edges = sorted(edge_set)
    indices = torch.tensor(ordered_edges, dtype=torch.long).T
    values = torch.ones(len(ordered_edges), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(indices, values, size=(node_count, node_count)).coalesce()
    row_sums = torch.sparse.sum(adjacency, dim=1).to_dense()
    normalized_values = adjacency.values() / row_sums[adjacency.indices()[0]].clamp_min(1.0)
    return torch.sparse_coo_tensor(
        adjacency.indices(),
        normalized_values,
        size=(node_count, node_count),
    ).coalesce()


def _build_bridge_topologies(settings: dict[str, int]):
    a_size = settings["community_a_size"]
    bridge_size = settings["bridge_size"]
    b_size = settings["community_b_size"]
    total = a_size + bridge_size + b_size

    a_indices = list(range(0, a_size))
    bridge_indices = list(range(a_size, a_size + bridge_size))
    b_indices = list(range(a_size + bridge_size, total))

    def clique_edges(nodes: list[int]) -> list[tuple[int, int]]:
        return [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]

    no_bridge_edges = clique_edges(a_indices + bridge_indices) + clique_edges(b_indices)
    bridge_edges = no_bridge_edges + [
        (bridge_idx, b_idx) for bridge_idx in bridge_indices for b_idx in b_indices
    ]

    return {
        "a_indices": a_indices,
        "bridge_indices": bridge_indices,
        "b_indices": b_indices,
        "without_bridge": _row_normalized_adjacency(total, no_bridge_edges),
        "with_bridge": _row_normalized_adjacency(total, bridge_edges),
    }


def _bridge_diffusion_metrics(config, settings):
    topology = _build_bridge_topologies(settings)
    node_count = config.num_agents
    emotions = zero_emotions(node_count)
    set_emotions(
        emotions,
        settings["core_emotion"],
        rows=slice(0, settings["community_a_size"] + settings["bridge_size"]),
    )
    set_emotions(
        emotions,
        settings["marginal_emotion"],
        rows=slice(settings["community_a_size"] + settings["bridge_size"], None),
    )
    influence = torch.ones(node_count)
    engagement = torch.ones(node_count)

    without_bridge = aggregate_social_state(
        config,
        emotions,
        influence,
        engagement_scores=engagement,
        adjacency_matrix=topology["without_bridge"],
    )
    with_bridge = aggregate_social_state(
        config,
        emotions,
        influence,
        engagement_scores=engagement,
        adjacency_matrix=topology["with_bridge"],
    )

    b_slice = topology["b_indices"]
    without_local = torch.sparse.mm(topology["without_bridge"], emotions)
    with_local = torch.sparse.mm(topology["with_bridge"], emotions)
    without_b_local_arousal = float(torch.norm(without_local[b_slice], dim=1).mean().item())
    with_b_local_arousal = float(torch.norm(with_local[b_slice], dim=1).mean().item())

    return {
        "without_bridge": without_bridge,
        "with_bridge": with_bridge,
        "without_b_local_arousal": without_b_local_arousal,
        "with_b_local_arousal": with_b_local_arousal,
    }


def test_world_direction_changes_which_emotion_dominates(tmp_path):
    scenario = get_test_scenario("emotion_directionality")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "emotion_directionality",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="emotion_directionality",
    )

    results = {}
    for label, world_values in settings["worlds"].items():
        result = run_debug_simulation(
            config,
            build_world(world_values),
            society=society,
            urgency=settings["urgency"],
        )
        center = np.asarray(result.social_state["objective_center"], dtype=np.float64)
        results[label] = {
            "center": center,
            "dominant_emotion": result.social_state["dominant_emotion"],
            "valence": float(result.social_state["sentiment_valence"]),
        }

    fear_idx = EMOTION_INDICES["Fear"]
    anger_idx = EMOTION_INDICES["Anger"]
    joy_idx = EMOTION_INDICES["Joy"]

    assert results["Prosperity"]["valence"] >= settings["min_positive_valence"]
    assert results["Threat"]["valence"] <= settings["max_negative_valence"]
    assert results["Injustice"]["valence"] <= settings["max_negative_valence"]

    assert (
        results["Threat"]["center"][fear_idx] - results["Prosperity"]["center"][fear_idx]
        >= settings["min_fear_gap"]
    )
    assert (
        results["Injustice"]["center"][anger_idx] - results["Prosperity"]["center"][anger_idx]
        >= settings["min_anger_gap"]
    )
    assert results["Prosperity"]["center"][joy_idx] > results["Threat"]["center"][joy_idx]
    assert results["Threat"]["dominant_emotion"] == "Fear"
    assert (
        results["Injustice"]["dominant_emotion"] in settings["allowed_injustice_emotions"]
    )


def test_bridge_agents_expand_cross_cluster_diffusion(tmp_path):
    del tmp_path
    scenario = get_test_scenario("bridge_diffusion")
    config = scenario.sim_config()
    settings = scenario.settings()
    metrics = _bridge_diffusion_metrics(config, settings)

    assert (
        metrics["with_bridge"]["acting_ratio"] - metrics["without_bridge"]["acting_ratio"]
        >= settings["min_bridge_acting_gain"]
    )
    assert (
        metrics["with_b_local_arousal"] - metrics["without_b_local_arousal"]
        >= settings["min_bridge_local_arousal_gain"]
    )


def test_generate_emotion_direction_and_bridge_diffusion_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "emotion_direction_and_bridge_diffusion.png"

    emotion_scenario = get_test_scenario("emotion_directionality")
    emotion_config = emotion_scenario.sim_config()
    emotion_settings = emotion_scenario.settings()
    society = prepare_scenario_society(
        "emotion_directionality",
        tmp_path,
        enable_evolution=emotion_config.enable_evolution,
        output_name="emotion_directionality",
    )

    emotion_results = {}
    for label, world_values in emotion_settings["worlds"].items():
        result = run_debug_simulation(
            emotion_config,
            build_world(world_values),
            society=society,
            urgency=emotion_settings["urgency"],
        )
        center = np.asarray(result.social_state["objective_center"], dtype=np.float64)
        emotion_results[label] = {
            "center": center,
            "valence": float(result.social_state["sentiment_valence"]),
            "sentiment": np.asarray(
                map_emotions_to_sentiment(
                    result.social_state["objective_center"],
                    result.social_state["acting_ratio"],
                ),
                dtype=np.float64,
            ),
        }

    bridge_scenario = get_test_scenario("bridge_diffusion")
    bridge_config = bridge_scenario.sim_config()
    bridge_settings = bridge_scenario.settings()
    bridge_metrics = _bridge_diffusion_metrics(bridge_config, bridge_settings)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()

    labels = list(emotion_settings["worlds"])
    x = np.arange(len(labels))
    width = 0.25
    key_emotions = ["Joy", "Fear", "Anger"]
    key_colors = ["#2a9d8f", "#457b9d", "#d62828"]

    # x-axis: world framing condition.
    # y-axis: aggregate probability mass on key emotions. This shows how changing
    # the direction of the same system input shifts the dominant emotional reaction.
    for offset, emotion_name, color in zip([-width, 0.0, width], key_emotions, key_colors):
        emotion_idx = EMOTION_INDICES[emotion_name]
        axes[0].bar(
            x + offset,
            [emotion_results[label]["center"][emotion_idx] for label in labels],
            width=width,
            label=emotion_name,
            color=color,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_title("Emotion Directionality")
    axes[0].set_ylabel("Objective-Center Weight")
    axes[0].legend(fontsize=8)

    sentiment_colors = ["#d62828", "#adb5bd", "#2a9d8f"]
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    bottoms = np.zeros(len(labels), dtype=np.float64)

    # x-axis: world framing condition.
    # y-axis: sentiment composition. Compare how prosperity vs threat vs injustice
    # move the whole society toward positive or negative aggregate interpretation.
    for idx, sentiment_label in enumerate(sentiment_labels):
        heights = np.array(
            [emotion_results[label]["sentiment"][idx] for label in labels],
            dtype=np.float64,
        )
        axes[1].bar(
            x,
            heights,
            bottom=bottoms,
            color=sentiment_colors[idx],
            label=sentiment_label,
        )
        bottoms += heights
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Emotion-to-Sentiment Mapping")
    axes[1].set_ylabel("Share")
    axes[1].legend(fontsize=8)

    # x-axis: network without vs. with bridge links.
    # y-axis: average local arousal inside the remote community B. Higher values
    # mean bridge links import more emotional energy into the otherwise separate cluster.
    axes[2].bar(
        ["Without bridge", "With bridge"],
        [
            bridge_metrics["without_b_local_arousal"],
            bridge_metrics["with_b_local_arousal"],
        ],
        color=["#adb5bd", "#264653"],
    )
    axes[2].set_title("Bridge Effect on Remote Community")
    axes[2].set_ylabel("Mean Local Arousal in Community B")

    # x-axis: network without vs. with bridge links.
    # y-axis: overall acting ratio after the social-threshold step. This shows
    # whether bridges convert isolated motivation into broader collective uptake.
    axes[3].bar(
        ["Without bridge", "With bridge"],
        [
            bridge_metrics["without_bridge"]["acting_ratio"],
            bridge_metrics["with_bridge"]["acting_ratio"],
        ],
        color=["#8d99ae", "#e76f51"],
    )
    axes[3].set_title("Bridge Effect on Collective Action")
    axes[3].set_ylabel("Acting Ratio")

    for axis in axes:
        axis.grid(True, alpha=0.2)

    fig.suptitle("Emotion Direction and Bridge Diffusion", fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
