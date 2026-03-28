from pathlib import Path

import community as community_louvain
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import (
    DIMENSION_INDICES,
    aggregate_social_state,
    build_debug_society,
    clone_sim_config,
    consolidate_agent_memory,
    distort_world_signal,
    prepare_society_for_debug,
    run_debug_simulation,
)
from research_paper_tests._metrics import (
    adjacency_to_graph,
    average_neighbor_distance,
    gini,
    mean_edge_cosine_similarity,
)
from research_paper_tests.config_schema import (
    EMOTION_INDICES,
    PERSONALITY_INDICES,
    WORLD_DIMENSION_COUNT,
    build_world,
    fraction_count,
    get_test_scenario,
    set_dimensions,
    set_emotions,
    set_traits,
    zero_emotions,
    zero_personalities,
)

matplotlib.use("Agg")


def _line_of_best_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ys = slope * xs + intercept
    return xs, ys


def _build_gate_society(config, settings):
    exposures = torch.zeros(config.num_agents, WORLD_DIMENSION_COUNT)
    personalities = zero_personalities(config.num_agents, fill=settings["trait_fill"])
    personalities[:, PERSONALITY_INDICES["Openness"]] = torch.linspace(
        settings["openness_start"],
        settings["openness_end"],
        config.num_agents,
    )
    worldview_scale = torch.linspace(
        settings["worldview_min_scale"],
        settings["worldview_max_scale"],
        config.num_agents,
    )

    for dimension_name, dimension_value in settings["aligned_worldview"].items():
        exposures[:, DIMENSION_INDICES[dimension_name]] = -dimension_value * worldview_scale

    return build_debug_society(config, exposures, personalities)


def test_generate_research_paper_summary_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "research_paper_summary.png"

    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    axes = axes.flatten()

    # 1. Signal distortion scatter
    distortion_scenario = get_test_scenario("figure_signal_distortion")
    distortion_config = distortion_scenario.sim_config()
    distortion_settings = distortion_scenario.settings()
    distortion_society = prepare_society_for_debug(
        distortion_config, output_dir=str(tmp_path / "distortion"), evolve=False
    )
    distortion_world = build_world(distortion_settings["world"])
    perceived = distort_world_signal(
        distortion_config, distortion_world, distortion_society.personalities
    )
    neuroticism = distortion_society.personalities[
        :, PERSONALITY_INDICES["Neuroticism"]
    ].numpy()
    distortion = np.abs(
        perceived[:, DIMENSION_INDICES["Physical_Safety"]].numpy()
        - distortion_world[0, DIMENSION_INDICES["Physical_Safety"]].item()
    )
    xs, ys = _line_of_best_fit(neuroticism, distortion)
    axes[0].scatter(neuroticism, distortion, s=10, alpha=0.3, color="#457b9d")
    axes[0].plot(xs, ys, color="#e63946", linewidth=2)
    axes[0].set_title("Signal Distortion")
    axes[0].set_xlabel("Neuroticism")
    axes[0].set_ylabel("Threat Exaggeration")

    # 2. Memory rehearsal decay
    memory_scenario = get_test_scenario("figure_memory_rehearsal")
    memory_config = memory_scenario.sim_config()
    memory_settings = memory_scenario.settings()
    memory = torch.zeros(memory_config.num_agents, WORLD_DIMENSION_COUNT)
    context = torch.zeros(memory_config.num_agents, WORLD_DIMENSION_COUNT)
    set_dimensions(context, memory_settings["context"])
    isolated = consolidate_agent_memory(
        memory_config,
        memory,
        context,
        memory_settings["isolated_rehearsal"],
    )
    rehearsed = consolidate_agent_memory(
        memory_config,
        memory,
        context,
        memory_settings["shared_rehearsal"],
    )
    isolated_curve = [torch.norm(isolated).item()]
    rehearsed_curve = [torch.norm(rehearsed).item()]
    for _ in range(memory_settings["decay_steps"]):
        isolated = consolidate_agent_memory(
            memory_config,
            isolated,
            torch.zeros_like(context),
            memory_settings["isolated_rehearsal"],
        )
        rehearsed = consolidate_agent_memory(
            memory_config,
            rehearsed,
            torch.zeros_like(context),
            memory_settings["shared_rehearsal"],
        )
        isolated_curve.append(torch.norm(isolated).item())
        rehearsed_curve.append(torch.norm(rehearsed).item())
    steps = np.arange(len(isolated_curve))
    axes[1].plot(steps, isolated_curve, marker="o", label="Isolated")
    axes[1].plot(steps, rehearsed_curve, marker="s", label="Rehearsed")
    axes[1].set_title("Memory Rehearsal")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Memory Norm")
    axes[1].legend()

    # 3. Cognitive gate distributions
    gate_scenario = get_test_scenario("figure_cognitive_gate")
    gate_config = gate_scenario.sim_config()
    gate_settings = gate_scenario.settings()
    gate_society = _build_gate_society(gate_config, gate_settings)
    gate_world = build_world(gate_settings["world"])
    gate_result = run_debug_simulation(
        gate_config,
        gate_world,
        society=gate_society,
        urgency=gate_settings["urgency"],
    )
    gate_engagement = gate_result.engagement_scores.numpy()
    gate_openness = gate_society.personalities[:, PERSONALITY_INDICES["Openness"]].numpy()
    xs, ys = _line_of_best_fit(gate_openness, gate_engagement)
    axes[2].scatter(
        gate_openness,
        gate_engagement,
        s=10,
        alpha=0.35,
        color="#457b9d",
        label="Agent outcomes",
    )
    axes[2].plot(xs, ys, color="#f4a261", linewidth=2, label="Trend line")
    axes[2].set_title("Cognitive Gate")
    axes[2].set_xlabel("Openness")
    axes[2].set_ylabel("Engagement")
    axes[2].legend()

    # 4. Algorithmic amplification
    algo_scenario = get_test_scenario("figure_algorithmic_filter_bubble")
    algo_config = algo_scenario.sim_config()
    algo_settings = algo_scenario.settings()
    algo_society = prepare_society_for_debug(
        algo_config, output_dir=str(tmp_path / "algo"), evolve=False
    )
    boring_world = build_world(algo_settings["world"])
    baseline_config = clone_sim_config(algo_config, use_algorithmic_amplification=False)
    baseline_society = build_debug_society(
        baseline_config,
        algo_society.exposures,
        algo_society.personalities,
        algo_society.affinities,
        algo_society.metadata["Influence"].to_numpy(),
        algo_society.adjacency_matrix,
        algo_society.memory.clone(),
        algo_society.metadata.copy(),
    )
    base_result = run_debug_simulation(
        baseline_config,
        boring_world,
        society=baseline_society,
        urgency=algo_settings["urgency"],
    )
    algo_result = run_debug_simulation(
        algo_config,
        boring_world,
        society=algo_society,
        urgency=algo_settings["urgency"],
    )
    axes[3].bar(
        ["Baseline", "Amplified"],
        [
            base_result.engagement_scores.mean().item(),
            algo_result.engagement_scores.mean().item(),
        ],
        color=["#a8dadc", "#1d3557"],
    )
    axes[3].set_title("Algorithmic Amplification")
    axes[3].set_ylabel("Mean Engagement")

    # 5. Social consensus
    consensus_scenario = get_test_scenario("figure_social_consensus")
    consensus_config = consensus_scenario.sim_config()
    consensus_settings = consensus_scenario.settings()
    consensus_society = prepare_society_for_debug(
        consensus_config, output_dir=str(tmp_path / "consensus"), evolve=False
    )
    assert consensus_society.adjacency_matrix is not None
    consensus_world = build_world(consensus_settings["world"])
    baseline_consensus_config = get_test_scenario(
        "perception_social_consensus_baseline"
    ).sim_config(
        num_agents=consensus_config.num_agents
    )
    baseline_perceived = distort_world_signal(
        baseline_consensus_config,
        consensus_world,
        consensus_society.personalities,
        adjacency_matrix=consensus_society.adjacency_matrix,
    )
    consensus_perceived = (
        (1.0 - consensus_config.perception_social_consensus_gain) * baseline_perceived
        + consensus_config.perception_social_consensus_gain
        * torch.sparse.mm(consensus_society.adjacency_matrix, baseline_perceived)
    )
    axes[4].bar(
        ["Baseline", "Consensus"],
        [
            average_neighbor_distance(
                baseline_perceived, consensus_society.adjacency_matrix
            ),
            average_neighbor_distance(
                consensus_perceived, consensus_society.adjacency_matrix
            ),
        ],
        color=["#f1fa8c", "#2a9d8f"],
    )
    axes[4].set_title("Perception Consensus")
    axes[4].set_ylabel("Neighbor Distance")

    # 6. Granovetter cascade
    granovetter_scenario = get_test_scenario("figure_granovetter_cascade")
    granovetter_config = granovetter_scenario.sim_config()
    granovetter_settings = granovetter_scenario.settings()
    granovetter_society = prepare_society_for_debug(
        granovetter_config, output_dir=str(tmp_path / "gran"), evolve=False
    )
    emotions = zero_emotions(granovetter_config.num_agents)
    instigators = fraction_count(
        granovetter_config.num_agents,
        granovetter_settings["instigator_share"],
    )
    sympathizers = fraction_count(
        granovetter_config.num_agents,
        granovetter_settings["sympathizer_share"],
    )
    set_emotions(
        emotions,
        granovetter_settings["instigator_emotion"],
        rows=slice(None, instigators),
    )
    set_emotions(
        emotions,
        granovetter_settings["sympathizer_emotion"],
        rows=slice(instigators, instigators + sympathizers),
    )
    baseline_granovetter = aggregate_social_state(
        clone_sim_config(granovetter_config, use_granovetter_thresholds=False),
        emotions,
        granovetter_society.metadata["Influence"].to_numpy(),
        engagement_scores=torch.ones(granovetter_config.num_agents),
        adjacency_matrix=granovetter_society.adjacency_matrix,
        personalities=granovetter_society.personalities,
    )
    cascade_granovetter = aggregate_social_state(
        granovetter_config,
        emotions,
        granovetter_society.metadata["Influence"].to_numpy(),
        engagement_scores=torch.ones(granovetter_config.num_agents),
        adjacency_matrix=granovetter_society.adjacency_matrix,
        personalities=granovetter_society.personalities,
    )
    axes[5].bar(
        ["Baseline", "Cascade"],
        [baseline_granovetter["acting_ratio"], cascade_granovetter["acting_ratio"]],
        color=["#bde0fe", "#ef476f"],
    )
    axes[5].set_title("Granovetter Cascade")
    axes[5].set_ylabel("Acting Ratio")

    # 7-8. Homophily and modularity share the same societies
    low_homophily_scenario = get_test_scenario("figure_echo_chambers_low")
    low_homophily = low_homophily_scenario.sim_config()
    high_homophily = get_test_scenario("figure_echo_chambers_high").sim_config()
    homophily_settings = low_homophily_scenario.settings()
    low_society = prepare_society_for_debug(
        low_homophily, output_dir=str(tmp_path / "low_h"), evolve=False
    )
    high_society = prepare_society_for_debug(
        high_homophily, output_dir=str(tmp_path / "high_h"), evolve=False
    )
    assert low_society.adjacency_matrix is not None
    assert high_society.adjacency_matrix is not None
    axes[6].bar(
        ["Low", "High"],
        [
            mean_edge_cosine_similarity(low_society.exposures, low_society.adjacency_matrix),
            mean_edge_cosine_similarity(high_society.exposures, high_society.adjacency_matrix),
        ],
        color=["#ced4da", "#6d597a"],
    )
    axes[6].set_title("Echo Chambers")
    axes[6].set_ylabel("Edge Similarity")

    low_graph = adjacency_to_graph(low_society.adjacency_matrix)
    high_graph = adjacency_to_graph(high_society.adjacency_matrix)
    low_modularity = community_louvain.modularity(
        community_louvain.best_partition(
            low_graph,
            random_state=homophily_settings["partition_seed"],
        ),
        low_graph,
    )
    high_modularity = community_louvain.modularity(
        community_louvain.best_partition(
            high_graph,
            random_state=homophily_settings["partition_seed"],
        ),
        high_graph,
    )
    axes[7].bar(
        ["Low", "High"],
        [low_modularity, high_modularity],
        color=["#adb5bd", "#264653"],
    )
    axes[7].set_title("Louvain Modularity")
    axes[7].set_ylabel("Q")

    # 9. Personality correlation heatmap
    corr_config = get_test_scenario("figure_personality_correlations").sim_config()
    corr_society = prepare_society_for_debug(
        corr_config, output_dir=str(tmp_path / "corr"), evolve=False
    )
    observed_corr = np.corrcoef(corr_society.personalities.numpy().T)
    im = axes[8].imshow(observed_corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[8].set_title("Personality Correlations")
    axes[8].set_xticks(range(5), ["O", "C", "E", "A", "N"])
    axes[8].set_yticks(range(5), ["O", "C", "E", "A", "N"])
    fig.colorbar(im, ax=axes[8], fraction=0.046, pad=0.04)

    # 10. Wealth inequality
    base_wealth_config = get_test_scenario("figure_wealth_baseline").sim_config()
    evolved_wealth_config = get_test_scenario("figure_wealth_evolved").sim_config()
    base_wealth = prepare_society_for_debug(
        base_wealth_config, output_dir=str(tmp_path / "base_wealth"), evolve=False
    )
    evolved_wealth = prepare_society_for_debug(
        evolved_wealth_config, output_dir=str(tmp_path / "evolved_wealth"), evolve=True
    )
    axes[9].bar(
        ["Baseline", "Evolved"],
        [
            gini(base_wealth.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()),
            gini(evolved_wealth.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()),
        ],
        color=["#8ecae6", "#fb8500"],
    )
    axes[9].set_title("Wealth Gini")
    axes[9].set_ylabel("Gini")

    # 11. Relative deprivation
    relative_scenario = get_test_scenario("relative_deprivation")
    relative_config = relative_scenario.sim_config()
    relative_settings = relative_scenario.settings()
    exposures_marginalized = torch.zeros(
        relative_settings["group_size"],
        WORLD_DIMENSION_COUNT,
    )
    set_dimensions(exposures_marginalized, relative_settings["marginalized_exposures"])
    personalities_marginalized = zero_personalities(
        relative_settings["group_size"],
        fill=relative_settings["trait_fill"],
    )
    set_traits(personalities_marginalized, relative_settings["marginalized_traits"])
    exposures_elites = torch.zeros(
        relative_settings["group_size"],
        WORLD_DIMENSION_COUNT,
    )
    set_dimensions(exposures_elites, relative_settings["elite_exposures"])
    personalities_elites = zero_personalities(
        relative_settings["group_size"],
        fill=relative_settings["trait_fill"],
    )
    set_traits(personalities_elites, relative_settings["elite_traits"])
    relative_society = build_debug_society(
        relative_config,
        torch.cat([exposures_marginalized, exposures_elites], dim=0),
        torch.cat([personalities_marginalized, personalities_elites], dim=0),
    )
    relative_world = build_world(relative_settings["world"])
    relative_result = run_debug_simulation(
        relative_config,
        relative_world,
        society=relative_society,
        urgency=relative_settings["urgency"],
    )
    anger = relative_result.final_emotions[:, EMOTION_INDICES["Anger"]]
    axes[10].bar(
        ["Marginalized", "Elites"],
        [
            anger[: relative_settings["group_size"]].mean().item(),
            anger[relative_settings["group_size"] :].mean().item(),
        ],
        color=["#d62828", "#577590"],
    )
    axes[10].set_title("Relative Deprivation")
    axes[10].set_ylabel("Mean Anger")

    # 12. Sentiment profile comparison
    semantic_scenario = get_test_scenario("figure_semantic_alignment")
    semantic_config = semantic_scenario.sim_config()
    semantic_settings = semantic_scenario.settings()
    semantic_society = prepare_society_for_debug(
        semantic_config, output_dir=str(tmp_path / "semantic"), evolve=False
    )
    positive_world = build_world(semantic_settings["positive_world"])
    negative_world = build_world(semantic_settings["negative_world"])
    semantic_positive = run_debug_simulation(
        semantic_config,
        positive_world,
        society=semantic_society,
        urgency=semantic_settings["urgency"],
    )
    semantic_negative = run_debug_simulation(
        semantic_config,
        negative_world,
        society=semantic_society,
        urgency=semantic_settings["urgency"],
    )
    positive_sentiment = semantic_positive.social_state["objective_center"]
    negative_sentiment = semantic_negative.social_state["objective_center"]
    from main import map_emotions_to_sentiment

    pos = map_emotions_to_sentiment(positive_sentiment)
    neg = map_emotions_to_sentiment(negative_sentiment)
    x = np.arange(3)
    width = 0.35
    axes[11].bar(x - width / 2, pos, width=width, label="Prosperity")
    axes[11].bar(x + width / 2, neg, width=width, label="Threat")
    axes[11].set_xticks(x, ["Negative", "Neutral", "Positive"])
    axes[11].set_title("Semantic Sentiment Profile")
    axes[11].legend()

    for axis in axes:
        axis.grid(True, alpha=0.2)

    fig.suptitle("Research Paper Summary Panels", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
