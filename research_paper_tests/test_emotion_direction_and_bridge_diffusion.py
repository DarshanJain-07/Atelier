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
from research_paper_tests.plotting_utils import (
    COMPARISON_COLORS,
    PAPER_PALETTE,
    SENTIMENT_COLORS,
    apply_paper_style,
    compose_panel_grid,
    save_paper_figure,
    setup_plot,
)

matplotlib.use("Agg")
apply_paper_style()


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

    assert results["Prosperity"]["dominant_emotion"] == "Joy"
    assert results["Prosperity"]["valence"] > 0.0
    assert results["Threat"]["valence"] < 0.0
    assert results["Injustice"]["valence"] < 0.0
    assert results["Prosperity"]["center"][joy_idx] > max(
        results["Threat"]["center"][joy_idx],
        results["Injustice"]["center"][joy_idx],
    )
    assert results["Threat"]["center"][fear_idx] > max(
        results["Prosperity"]["center"][fear_idx],
        results["Injustice"]["center"][fear_idx],
    )
    assert results["Injustice"]["center"][anger_idx] > max(
        results["Prosperity"]["center"][anger_idx],
        results["Threat"]["center"][anger_idx],
    )
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

    assert metrics["with_bridge"]["acting_ratio"] > metrics["without_bridge"]["acting_ratio"]
    assert metrics["with_b_local_arousal"] > metrics["without_b_local_arousal"]


def test_generate_emotion_direction_and_bridge_diffusion_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated" / "emotion_and_bridge"
    output_dir.mkdir(parents=True, exist_ok=True)

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
                    config=emotion_config,
                ),
                dtype=np.float64,
            ),
        }

    bridge_scenario = get_test_scenario("bridge_diffusion")
    bridge_config = bridge_scenario.sim_config()
    bridge_settings = bridge_scenario.settings()
    bridge_metrics = _bridge_diffusion_metrics(bridge_config, bridge_settings)

    labels = list(emotion_settings["worlds"])
    x = np.arange(len(labels))
    width = 0.25
    key_emotions = ["Joy", "Fear", "Anger"]
    key_colors = [
        PAPER_PALETTE["positive"],
        PAPER_PALETTE["secondary"],
        PAPER_PALETTE["negative"],
    ]

    # Figure 1: Emotion Directionality
    fig1, ax1 = setup_plot(
        title="Emotion Directionality",
        xlabel="World",
        ylabel="Objective-Center Weight",
    )
    for offset, emotion_name, color in zip([-width, 0.0, width], key_emotions, key_colors):
        emotion_idx = EMOTION_INDICES[emotion_name]
        ax1.bar(
            x + offset,
            [emotion_results[label]["center"][emotion_idx] for label in labels],
            width=width,
            label=emotion_name,
            color=color,
        )
    ax1.set_xticks(x, labels)
    ax1.legend()
    path1 = output_dir / "emotion_directionality.png"
    save_paper_figure(fig1, path1)
    plt.close(fig1)

    # Figure 2: Emotion-to-Sentiment Mapping
    fig2, ax2 = setup_plot(
        title="Emotion-to-Sentiment Mapping",
        xlabel="World",
        ylabel="Share",
    )
    sentiment_colors = SENTIMENT_COLORS
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    bottoms = np.zeros(len(labels), dtype=np.float64)

    for idx, sentiment_label in enumerate(sentiment_labels):
        heights = np.array(
            [emotion_results[label]["sentiment"][idx] for label in labels],
            dtype=np.float64,
        )
        ax2.bar(
            x,
            heights,
            bottom=bottoms,
            color=sentiment_colors[idx],
            label=sentiment_label,
        )
        bottoms += heights
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0.0, 1.0)
    ax2.legend()
    path2 = output_dir / "sentiment_mapping.png"
    save_paper_figure(fig2, path2)
    plt.close(fig2)

    # Figure 3: Bridge Effect on Remote Community
    fig3, ax3 = setup_plot(
        title="Bridge Effect on Remote Community",
        xlabel="Topology",
        ylabel="Mean Local Arousal in Community B",
    )
    ax3.bar(
        ["Without bridge", "With bridge"],
        [
            bridge_metrics["without_b_local_arousal"],
            bridge_metrics["with_b_local_arousal"],
        ],
        color=COMPARISON_COLORS,
    )
    path3 = output_dir / "bridge_arousal_effect.png"
    save_paper_figure(fig3, path3)
    plt.close(fig3)

    # Figure 4: Bridge Effect on Collective Action
    fig4, ax4 = setup_plot(
        title="Bridge Effect on Collective Action",
        xlabel="Topology",
        ylabel="Acting Ratio",
    )
    ax4.bar(
        ["Without bridge", "With bridge"],
        [
            bridge_metrics["without_bridge"]["acting_ratio"],
            bridge_metrics["with_bridge"]["acting_ratio"],
        ],
        color=COMPARISON_COLORS,
    )
    path4 = output_dir / "bridge_action_effect.png"
    save_paper_figure(fig4, path4)
    plt.close(fig4)

    compose_panel_grid(
        [path1, path2, path3, path4],
        output_dir.parent / "emotion_direction_and_bridge_diffusion.png",
        title="Emotion Direction and Bridge Diffusion",
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
