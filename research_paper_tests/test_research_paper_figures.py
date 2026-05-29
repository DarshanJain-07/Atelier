from pathlib import Path

import community as community_louvain
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import (
    aggregate_social_state,
    apply_triadic_closure_for_debug,
    build_debug_society,
    clone_sim_config,
    consolidate_agent_memory,
    create_topology_for_debug,
    distort_world_signal,
    map_emotions_to_sentiment,
    run_cognitive_cycle,
    run_debug_simulation,
)
from research_paper_tests._metrics import (
    adjacency_to_graph,
    average_clustering,
    average_neighbor_distance,
    bimodality_coefficient,
    gini,
    mean_edge_topology_similarity,
)
from research_paper_tests.config_schema import (
    DIMENSION_INDICES,
    EMOTION_INDICES,
    PERSONALITY_INDICES,
    PERSONALITY_TRAIT_COUNT,
    WORLD_DIMENSION_COUNT,
    build_world,
    fraction_count,
    get_test_scenario,
    prepare_scenario_society,
    set_dimensions,
    set_emotions,
    set_traits,
    zero_emotions,
    zero_personalities,
)
from research_paper_tests.plotting_utils import (
    CATEGORICAL_COLORS,
    COMPARISON_COLORS,
    PAPER_DIVERGING_CMAP,
    PAPER_PALETTE,
    SENTIMENT_COLORS,
    apply_paper_style,
    compose_panel_grid,
    place_legend_outside,
    save_paper_figure,
    setup_plot,
)

matplotlib.use("Agg")
apply_paper_style()


def _line_of_best_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ys = slope * xs + intercept
    return xs, ys


def _prepare_seeded_society(
    scenario_name: str,
    tmp_path,
    seed: int,
    *,
    output_prefix: str,
    **config_overrides,
):
    scenario = get_test_scenario(scenario_name)
    config = scenario.sim_config(seed=seed, **config_overrides)
    settings = scenario.settings()
    society = prepare_scenario_society(
        scenario_name,
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name=f"{output_prefix}_seed_{seed}",
        seed=seed,
        **config_overrides,
    )
    return config, settings, society


def _stacked_share_bars(
    ax,
    group_labels: list[str],
    values: list[np.ndarray | list[float]],
    stack_labels: list[str],
    colors: list[str],
):
    # Each row is normalized to sum to 1.0, so the x-axis shows categories/groups
    # and the y-axis always means "share of that group" rather than raw count.
    # Read each bar from bottom to top to see how the composition changes.
    matrix = np.asarray(values, dtype=np.float64)
    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-9, None)
    bottoms = np.zeros(normalized.shape[0], dtype=np.float64)
    x = np.arange(normalized.shape[0])

    for idx, stack_label in enumerate(stack_labels):
        heights = normalized[:, idx]
        ax.bar(
            x,
            heights,
            bottom=bottoms,
            color=colors[idx],
            label=stack_label,
        )
        bottoms += heights

    ax.set_xticks(x, group_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share")


def _plot_seed_lines(
    ax,
    seeds: list[int],
    series: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
):
    # The x-axis is the random seed used to regenerate the full scenario, not time.
    # Each point is one independent rerun. If lines stay flat across seeds, the
    # result is robust; if they swing a lot, the metric is seed-sensitive.
    markers = ["o", "s", "^", "D", "P"]
    for idx, (label, values) in enumerate(series.items()):
        ax.plot(
            seeds,
            values,
            marker=markers[idx % len(markers)],
            color=CATEGORICAL_COLORS[idx % len(CATEGORICAL_COLORS)],
            linewidth=2,
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.set_xticks(seeds)
    if len(series) > 1:
        ax.legend(fontsize=8)


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
    output_dir = Path(__file__).resolve().parent / "generated" / "summary_panels"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Signal distortion scatter
    distortion_scenario = get_test_scenario("figure_signal_distortion")
    distortion_config = distortion_scenario.sim_config()
    distortion_settings = distortion_scenario.settings()
    distortion_society = prepare_scenario_society(
        "figure_signal_distortion",
        tmp_path,
        enable_evolution=distortion_config.enable_evolution,
        output_name="distortion",
    )
    distortion_world = build_world(distortion_settings["world"])
    perceived = distort_world_signal(
        distortion_config, distortion_world, distortion_society.personalities,
    )
    neuroticism = distortion_society.personalities[
        :, PERSONALITY_INDICES["Neuroticism"],
    ].numpy()
    distortion = np.abs(
        perceived[:, DIMENSION_INDICES["Physical_Safety"]].numpy()
        - distortion_world[0, DIMENSION_INDICES["Physical_Safety"]].item(),
    )
    xs, ys = _line_of_best_fit(neuroticism, distortion)

    fig, ax = setup_plot(
        title="Signal Distortion",
        xlabel="Neuroticism",
        ylabel="Threat Exaggeration",
    )
    ax.scatter(neuroticism, distortion, s=10, alpha=0.3, color=PAPER_PALETTE["primary"])
    ax.plot(xs, ys, color=PAPER_PALETTE["secondary"], linewidth=2)
    save_paper_figure(fig, output_dir / "01_signal_distortion.png")
    plt.close(fig)

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

    fig, ax = setup_plot(
        title="Memory Rehearsal",
        xlabel="Step",
        ylabel="Memory Norm",
    )
    ax.plot(
        steps,
        isolated_curve,
        marker="o",
        color=PAPER_PALETTE["primary"],
        label="Isolated",
    )
    ax.plot(
        steps,
        rehearsed_curve,
        marker="s",
        color=PAPER_PALETTE["secondary"],
        label="Rehearsed",
    )
    ax.legend()
    save_paper_figure(fig, output_dir / "02_memory_rehearsal.png")
    plt.close(fig)

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

    fig, ax = setup_plot(
        title="Cognitive Gate",
        xlabel="Openness",
        ylabel="Engagement",
    )
    ax.scatter(
        gate_openness,
        gate_engagement,
        s=10,
        alpha=0.35,
        color=PAPER_PALETTE["primary"],
        label="Agent outcomes",
    )
    ax.plot(xs, ys, color=PAPER_PALETTE["secondary"], linewidth=2, label="Trend line")
    ax.legend()
    save_paper_figure(fig, output_dir / "03_cognitive_gate.png")
    plt.close(fig)

    # 4. Algorithmic amplification
    algo_scenario = get_test_scenario("figure_algorithmic_filter_bubble")
    algo_config = algo_scenario.sim_config()
    algo_settings = algo_scenario.settings()
    algo_society = prepare_scenario_society(
        "figure_algorithmic_filter_bubble",
        tmp_path,
        enable_evolution=algo_config.enable_evolution,
        output_name="algo",
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

    fig, ax = setup_plot(
        title="Algorithmic Amplification",
        xlabel="Condition",
        ylabel="Mean Engagement",
    )
    ax.bar(
        ["Baseline", "Amplified"],
        [
            base_result.engagement_scores.mean().item(),
            algo_result.engagement_scores.mean().item(),
        ],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "04_algorithmic_amplification.png")
    plt.close(fig)

    # 5. Social consensus
    consensus_scenario = get_test_scenario("figure_social_consensus")
    consensus_config = consensus_scenario.sim_config()
    consensus_settings = consensus_scenario.settings()
    consensus_society = prepare_scenario_society(
        "figure_social_consensus",
        tmp_path,
        enable_evolution=consensus_config.enable_evolution,
        output_name="consensus",
    )
    assert consensus_society.adjacency_matrix is not None
    consensus_world = build_world(consensus_settings["world"])
    baseline_consensus_config = get_test_scenario(
        "perception_social_consensus_baseline",
    ).sim_config(
        num_agents=consensus_config.num_agents,
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

    fig, ax = setup_plot(
        title="Perception Consensus",
        xlabel="Condition",
        ylabel="Neighbor Distance",
    )
    ax.bar(
        ["Baseline", "Consensus"],
        [
            average_neighbor_distance(
                baseline_perceived, consensus_society.adjacency_matrix,
            ),
            average_neighbor_distance(
                consensus_perceived, consensus_society.adjacency_matrix,
            ),
        ],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "05_perception_consensus.png")
    plt.close(fig)

    # 6. Granovetter cascade
    granovetter_scenario = get_test_scenario("figure_granovetter_cascade")
    granovetter_config = granovetter_scenario.sim_config()
    granovetter_settings = granovetter_scenario.settings()
    granovetter_society = prepare_scenario_society(
        "figure_granovetter_cascade",
        tmp_path,
        enable_evolution=granovetter_config.enable_evolution,
        output_name="gran",
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

    fig, ax = setup_plot(
        title="Granovetter Cascade",
        xlabel="Condition",
        ylabel="Acting Ratio",
    )
    ax.bar(
        ["Baseline", "Cascade"],
        [baseline_granovetter["acting_ratio"], cascade_granovetter["acting_ratio"]],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "06_granovetter_cascade.png")
    plt.close(fig)

    # 7-8. Homophily and modularity share the same societies
    low_homophily_scenario = get_test_scenario("figure_echo_chambers_low")
    low_homophily = low_homophily_scenario.sim_config()
    high_homophily = get_test_scenario("figure_echo_chambers_high").sim_config()
    homophily_settings = low_homophily_scenario.settings()
    low_society = prepare_scenario_society(
        "figure_echo_chambers_low",
        tmp_path,
        enable_evolution=low_homophily.enable_evolution,
        output_name="low_h",
    )
    high_society = prepare_scenario_society(
        "figure_echo_chambers_high",
        tmp_path,
        enable_evolution=high_homophily.enable_evolution,
        output_name="high_h",
    )
    assert low_society.adjacency_matrix is not None
    assert high_society.adjacency_matrix is not None

    fig, ax = setup_plot(
        title="Echo Chambers",
        xlabel="Homophily",
        ylabel="Topology Similarity",
    )
    ax.bar(
        ["Low", "High"],
        [
            mean_edge_topology_similarity(
                low_society.exposures,
                low_society.personalities,
                low_society.adjacency_matrix,
            ),
            mean_edge_topology_similarity(
                high_society.exposures,
                high_society.personalities,
                high_society.adjacency_matrix,
            ),
        ],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "07_echo_chambers.png")
    plt.close(fig)

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

    fig, ax = setup_plot(
        title="Louvain Modularity",
        xlabel="Homophily",
        ylabel="Q",
    )
    ax.bar(
        ["Low", "High"],
        [low_modularity, high_modularity],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "08_louvain_modularity.png")
    plt.close(fig)

    # 9. Personality correlation heatmap
    corr_config = get_test_scenario("figure_personality_correlations").sim_config()
    corr_society = prepare_scenario_society(
        "figure_personality_correlations",
        tmp_path,
        enable_evolution=corr_config.enable_evolution,
        output_name="corr",
    )
    observed_corr = np.corrcoef(corr_society.personalities.numpy().T)

    fig, ax = setup_plot(title="Personality Correlations")
    im = ax.imshow(observed_corr, vmin=-1.0, vmax=1.0, cmap=PAPER_DIVERGING_CMAP)
    ax.grid(False)
    ax.set_xticks(range(5), ["O", "C", "E", "A", "N"])
    ax.set_yticks(range(5), ["O", "C", "E", "A", "N"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_paper_figure(fig, output_dir / "09_personality_correlations.png")
    plt.close(fig)

    # 10. Wealth inequality
    base_wealth_config = get_test_scenario("figure_wealth_baseline").sim_config()
    evolved_wealth_config = get_test_scenario("figure_wealth_evolved").sim_config()
    base_wealth = prepare_scenario_society(
        "figure_wealth_baseline",
        tmp_path,
        enable_evolution=base_wealth_config.enable_evolution,
        output_name="base_wealth",
    )
    evolved_wealth = prepare_scenario_society(
        "figure_wealth_evolved",
        tmp_path,
        enable_evolution=evolved_wealth_config.enable_evolution,
        output_name="evolved_wealth",
    )

    fig, ax = setup_plot(
        title="Wealth Gini",
        xlabel="Condition",
        ylabel="Gini",
    )
    ax.bar(
        ["Baseline", "Evolved"],
        [
            gini(base_wealth.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()),
            gini(evolved_wealth.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()),
        ],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "10_wealth_gini.png")
    plt.close(fig)

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

    fig, ax = setup_plot(
        title="Relative Deprivation",
        xlabel="Group",
        ylabel="Mean Anger",
    )
    ax.bar(
        ["Marginalized", "Elites"],
        [
            anger[: relative_settings["group_size"]].mean().item(),
            anger[relative_settings["group_size"] :].mean().item(),
        ],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "11_relative_deprivation.png")
    plt.close(fig)

    # 12. Sentiment profile comparison
    semantic_scenario = get_test_scenario("figure_semantic_alignment")
    semantic_config = semantic_scenario.sim_config()
    semantic_settings = semantic_scenario.settings()
    semantic_society = prepare_scenario_society(
        "figure_semantic_alignment",
        tmp_path,
        enable_evolution=semantic_config.enable_evolution,
        output_name="semantic",
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

    pos = map_emotions_to_sentiment(
        positive_sentiment,
        semantic_positive.social_state["acting_ratio"],
        config=semantic_config,
    )
    neg = map_emotions_to_sentiment(
        negative_sentiment,
        semantic_negative.social_state["acting_ratio"],
        config=semantic_config,
    )
    x = np.arange(3)
    width = 0.35

    fig, ax = setup_plot(
        title="Semantic Sentiment Profile",
        xlabel="Sentiment Class",
        ylabel="Share",
    )
    ax.bar(
        x - width / 2,
        pos,
        width=width,
        color=PAPER_PALETTE["primary"],
        label="Prosperity",
    )
    ax.bar(
        x + width / 2,
        neg,
        width=width,
        color=PAPER_PALETTE["secondary"],
        label="Threat",
    )
    ax.set_xticks(x, ["Negative", "Neutral", "Positive"])
    place_legend_outside(ax, ncol=2)
    save_paper_figure(fig, output_dir / "12_semantic_sentiment.png")
    plt.close(fig)

    # 13. Triadic closure increases clustering
    clustering_scenario = get_test_scenario("network_clustering_closure")
    backbone_config = get_test_scenario("network_clustering_backbone").sim_config()
    clustering_config = clustering_scenario.sim_config()
    clustering_settings = clustering_scenario.settings()
    torch_state = torch.get_rng_state()
    numpy_state = np.random.get_state()
    try:
        torch.manual_seed(clustering_settings["torch_seed"])
        np.random.seed(clustering_settings["numpy_seed"])
        clustering_exposures = torch.randn(
            backbone_config.num_agents,
            WORLD_DIMENSION_COUNT,
        )
        clustering_personalities = torch.sigmoid(
            torch.randn(backbone_config.num_agents, PERSONALITY_TRAIT_COUNT),
        )
        clustering_influence = np.random.lognormal(
            mean=clustering_settings["influence_mean"],
            sigma=clustering_settings["influence_std"],
            size=backbone_config.num_agents,
        )
        backbone = create_topology_for_debug(
            backbone_config,
            clustering_exposures,
            clustering_personalities,
            clustering_influence,
        )
        refined = apply_triadic_closure_for_debug(clustering_config, backbone)
    finally:
        torch.set_rng_state(torch_state)
        np.random.set_state(numpy_state)

    fig, ax = setup_plot(
        title="Network Clustering",
        xlabel="Network",
        ylabel="Average Clustering",
    )
    ax.bar(
        ["Backbone", "Closure"],
        [average_clustering(backbone), average_clustering(refined)],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "13_network_clustering.png")
    plt.close(fig)

    # 14. Personality socialization
    base_social_config = get_test_scenario("personality_socialization_base").sim_config()
    socialized_config = get_test_scenario(
        "personality_socialization_socialized",
    ).sim_config()
    base_social_society = prepare_scenario_society(
        "personality_socialization_base",
        tmp_path,
        enable_evolution=base_social_config.enable_evolution,
        output_name="social_base",
    )
    socialized_society = prepare_scenario_society(
        "personality_socialization_socialized",
        tmp_path,
        enable_evolution=socialized_config.enable_evolution,
        output_name="socialized",
    )

    fig, ax = setup_plot(
        title="Personality Socialization",
        xlabel="Condition",
        ylabel="Neighbor Trait Distance",
    )
    ax.bar(
        ["Base", "Socialized"],
        [
            average_neighbor_distance(
                base_social_society.personalities,
                base_social_society.adjacency_matrix,
            ),
            average_neighbor_distance(
                socialized_society.personalities,
                socialized_society.adjacency_matrix,
            ),
        ],
        color=COMPARISON_COLORS,
    )
    save_paper_figure(fig, output_dir / "14_personality_socialization.png")
    plt.close(fig)

    # 15. Influence concentration
    flat_influence_config = get_test_scenario("cascade_power_law_flat").sim_config()
    power_influence_config = get_test_scenario("cascade_power_law_power").sim_config()
    flat_influence_society = prepare_scenario_society(
        "cascade_power_law_flat",
        tmp_path,
        enable_evolution=flat_influence_config.enable_evolution,
        output_name="flat_influence",
    )
    power_influence_society = prepare_scenario_society(
        "cascade_power_law_power",
        tmp_path,
        enable_evolution=power_influence_config.enable_evolution,
        output_name="power_influence",
    )
    flat_influence = flat_influence_society.metadata["Influence"].to_numpy()
    power_influence = power_influence_society.metadata["Influence"].to_numpy()

    fig, ax = setup_plot(
        title="Influence Tail",
        xlabel="Influence Score",
        ylabel="CCDF",
    )
    
    # Flat influence CCDF
    flat_sorted = np.sort(flat_influence)
    flat_ccdf = 1.0 - np.linspace(0, 1, len(flat_sorted), endpoint=False)
    ax.plot(
        flat_sorted,
        flat_ccdf,
        linewidth=2,
        color=PAPER_PALETTE["primary"],
        label=f"Flat (G={gini(flat_influence):.2f})",
    )
    
    # Power law influence CCDF
    power_sorted = np.sort(power_influence)
    power_ccdf = 1.0 - np.linspace(0, 1, len(power_sorted), endpoint=False)
    ax.plot(
        power_sorted,
        power_ccdf,
        linewidth=2,
        color=PAPER_PALETTE["secondary"],
        label=f"Power law (G={gini(power_influence):.2f})",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend()
    save_paper_figure(fig, output_dir / "15_influence_tail.png")
    plt.close(fig)

    # 16. Structural influence and realized reach
    reach_scenario = get_test_scenario("influence_susceptibility")
    reach_config = reach_scenario.sim_config()
    reach_settings = reach_scenario.settings()
    reach_society = prepare_scenario_society(
        "influence_susceptibility",
        tmp_path,
        enable_evolution=reach_config.enable_evolution,
        output_name="reach",
    )
    reach_influence = reach_society.metadata["Influence"].to_numpy()
    mean_reach_influence = reach_influence.mean()
    reach_rng = np.random.default_rng(reach_settings["rng_seed"])
    reach_indices = reach_rng.choice(
        reach_config.num_agents,
        size=reach_settings["sample_size"],
        replace=False,
    )
    realized_reach = []
    for idx in reach_indices:
        thought = reach_society.exposures[idx].unsqueeze(0)
        reach_result = run_debug_simulation(
            reach_config,
            thought,
            society=reach_society,
            urgency=reach_settings["urgency"],
        )
        reach_probability = min(
            1.0,
            reach_settings["reach_probability_base"]
            + (reach_influence[idx] / mean_reach_influence)
            * reach_settings["reach_probability_gain"],
        )
        sees_post = (
            reach_rng.random(reach_config.num_agents) < reach_probability
        )
        authority_bonus = 1.0 + np.log1p(reach_influence[idx] / mean_reach_influence)
        engaged = (
            reach_result.engagement_scores.detach().cpu().numpy() * authority_bonus
        )
        realized_reach.append(
            float(
                (
                    (engaged > reach_settings["engagement_threshold"])
                    & sees_post
                ).sum(),
            ),
        )
    realized_reach = np.asarray(realized_reach, dtype=np.float64)
    sampled_influence = reach_influence[reach_indices]
    xs, ys = _line_of_best_fit(sampled_influence, realized_reach)

    fig, ax = setup_plot(
        title="Influence vs. Reach",
        xlabel="Structural Influence",
        ylabel="Realized Reach",
    )
    ax.scatter(
        sampled_influence,
        realized_reach,
        s=18,
        alpha=0.6,
        color=PAPER_PALETTE["primary"],
    )
    ax.plot(xs, ys, color=PAPER_PALETTE["secondary"], linewidth=2)
    save_paper_figure(fig, output_dir / "16_influence_vs_reach.png")
    plt.close(fig)

    # 17. Fairness polarization
    polarization_scenario = get_test_scenario("bimodality_polarization")
    polarization_config = polarization_scenario.sim_config()
    polarization_society = prepare_scenario_society(
        "bimodality_polarization",
        tmp_path,
        enable_evolution=polarization_config.enable_evolution,
        output_name="bimodality",
    )
    fairness = polarization_society.exposures[:, DIMENSION_INDICES["Fairness"]].numpy()
    fairness_bc = bimodality_coefficient(fairness)

    fig, ax = setup_plot(
        title="Fairness Polarization",
        xlabel="Fairness Exposure",
        ylabel="Agent Count",
    )
    ax.hist(
        fairness,
        bins=24,
        color=PAPER_PALETTE["primary"],
        alpha=0.8,
        edgecolor="white",
    )
    ax.axvline(
        fairness.mean(),
        color=PAPER_PALETTE["secondary"],
        linestyle="--",
        linewidth=2,
    )
    ax.text(
        0.04,
        0.95,
        f"BC = {fairness_bc:.2f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    save_paper_figure(fig, output_dir / "17_fairness_polarization.png")
    plt.close(fig)

    # 18. Truth refinement attention split
    truth_scenario = get_test_scenario("truth_refinement")
    truth_config = truth_scenario.sim_config()
    truth_settings = truth_scenario.settings()
    truth_config.skepticism_gain = truth_settings["skepticism_gain"]
    truth_config.logic_gap_threshold = truth_settings["logic_gap_threshold"]
    truth_society = build_debug_society(
        truth_config,
        torch.zeros(truth_config.num_agents, WORLD_DIMENSION_COUNT),
        torch.tensor(truth_settings["personalities"], dtype=torch.float32),
    )
    truth_world = build_world(truth_settings["world"])
    _, truth_attention, _ = run_cognitive_cycle(
        truth_config,
        truth_world,
        urgency=0.0,
        is_personal=False,
        exposures=truth_society.exposures,
        personalities=truth_society.personalities,
        affinities=truth_society.affinities,
    )
    truth_x = np.arange(truth_config.num_agents)
    width = 0.35

    fig, ax = setup_plot(
        title="Truth Refinement",
        xlabel="Agent Archetype",
        ylabel="Attention Weight",
    )
    ax.bar(
        truth_x - width / 2,
        truth_attention[:, 10].numpy(),
        width=width,
        label="Short term",
        color=PAPER_PALETTE["primary"],
    )
    ax.bar(
        truth_x + width / 2,
        truth_attention[:, 11].numpy(),
        width=width,
        label="Long term",
        color=PAPER_PALETTE["secondary"],
    )
    ax.set_xticks(truth_x, ["Populist", "Skeptical"])
    ax.legend()
    save_paper_figure(fig, output_dir / "18_truth_refinement.png")
    plt.close(fig)

    # 19. Agent memory stacking
    stack_scenario = get_test_scenario("agent_memory")
    stack_config = stack_scenario.sim_config()
    stack_settings = stack_scenario.settings()
    memory_society = prepare_scenario_society(
        "agent_memory",
        tmp_path,
        enable_evolution=stack_config.enable_evolution,
        output_name="stacking_memory",
    )
    repeated_threat = build_world(stack_settings["repeat_threat"])
    repeated_curve = []
    for _ in range(stack_settings["repeat_count"]):
        repeated_result = run_debug_simulation(
            stack_config,
            repeated_threat,
            society=memory_society,
            urgency=stack_settings["urgency"],
        )
        repeated_curve.append(repeated_result.engagement_scores.mean().item())

    new_threat = build_world(stack_settings["new_threat"])
    stacked_result = run_debug_simulation(
        stack_config,
        new_threat,
        society=memory_society,
        urgency=stack_settings["urgency"],
    )
    fresh_memory_society = prepare_scenario_society(
        "agent_memory",
        tmp_path,
        enable_evolution=stack_config.enable_evolution,
        output_name="fresh_memory",
    )
    fresh_result = run_debug_simulation(
        stack_config,
        new_threat,
        society=fresh_memory_society,
        urgency=stack_settings["urgency"],
    )
    memory_steps = np.arange(1, len(repeated_curve) + 1)

    fig, ax = setup_plot(
        title="Agent Memory",
        xlabel="Repeat Exposure",
        ylabel="Mean Engagement",
    )
    ax.plot(
        memory_steps,
        repeated_curve,
        marker="o",
        color=PAPER_PALETTE["primary"],
        label="Repeated threat",
    )
    ax.axhline(
        stacked_result.engagement_scores.mean().item(),
        color=PAPER_PALETTE["secondary"],
        linestyle="--",
        linewidth=2,
        label="Stacked new threat",
    )
    ax.axhline(
        fresh_result.engagement_scores.mean().item(),
        color=PAPER_PALETTE["neutral"],
        linestyle=":",
        linewidth=2,
        label="Fresh new threat",
    )
    ax.legend()
    save_paper_figure(fig, output_dir / "19_agent_memory.png")
    plt.close(fig)

    # 20. Virality stays bounded
    virality_scenario = get_test_scenario("maximum_virality")
    virality_config = virality_scenario.sim_config()
    virality_settings = virality_scenario.settings()
    virality_influence = torch.ones(virality_config.num_agents)

    consensus_emotions = zero_emotions(virality_config.num_agents)
    set_emotions(consensus_emotions, virality_settings["consensus_emotion"])

    outlier_emotions = zero_emotions(virality_config.num_agents)
    mainstream_count = virality_settings["outlier_mainstream_count"]
    set_emotions(
        outlier_emotions,
        virality_settings["outlier_mainstream_emotion"],
        rows=slice(None, mainstream_count),
    )
    set_emotions(
        outlier_emotions,
        virality_settings["outlier_emotions"],
        rows=slice(mainstream_count, None),
    )

    consensus_state = aggregate_social_state(
        virality_config,
        consensus_emotions,
        virality_influence,
        engagement_scores=torch.ones(virality_config.num_agents),
    )
    outlier_state = aggregate_social_state(
        virality_config,
        outlier_emotions,
        virality_influence,
        engagement_scores=torch.cat(
            [
                torch.ones(mainstream_count),
                torch.full(
                    (virality_settings["outlier_count"],),
                    virality_settings["boosted_engagement"],
                ),
            ],
        ),
    )
    virality_x = np.arange(2)

    fig, ax = setup_plot(
        title="Virality Bounds",
        xlabel="Emotion Scenario",
        ylabel="Outrage Multiplier",
    )
    ax.bar(
        virality_x - width / 2,
        [
            consensus_state["mean_outrage_multiplier"],
            outlier_state["mean_outrage_multiplier"],
        ],
        width=width,
        label="Mean",
        color=PAPER_PALETTE["primary"],
    )
    ax.bar(
        virality_x + width / 2,
        [
            consensus_state["max_outrage_multiplier"],
            outlier_state["max_outrage_multiplier"],
        ],
        width=width,
        label="Max",
        color=PAPER_PALETTE["secondary"],
    )
    ax.axhline(
        1.0 + virality_config.max_viral_multiplier,
        color=PAPER_PALETTE["neutral"],
        linestyle="--",
        linewidth=2,
        label="Configured cap",
    )
    ax.set_xticks(virality_x, ["Consensus", "Outliers"])
    ax.legend()
    save_paper_figure(fig, output_dir / "20_virality_bounds.png")
    plt.close(fig)




def test_generate_research_paper_advanced_visualizations(tmp_path):
    import umap
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    output_dir = Path(__file__).resolve().parent / "generated" / "advanced_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1-3. Cluster landscape via UMAP, centroids, and neuroticism spread
    cluster_scenario = get_test_scenario("clusters")
    cluster_config = cluster_scenario.sim_config()
    cluster_settings = cluster_scenario.settings()
    cluster_society = prepare_scenario_society(
        "clusters",
        tmp_path,
        enable_evolution=cluster_config.enable_evolution,
        output_name="advanced_clusters",
    )
    cluster_personalities = cluster_society.personalities.numpy()
    scaled_personalities = StandardScaler().fit_transform(cluster_personalities)
    cluster_model = KMeans(
        n_clusters=cluster_settings["cluster_count"],
        random_state=cluster_settings["cluster_seed"],
        n_init=cluster_settings["cluster_initializations"],
    )
    raw_cluster_labels = cluster_model.fit_predict(scaled_personalities)
    neuroticism_idx = PERSONALITY_INDICES["Neuroticism"]
    cluster_order = sorted(
        range(cluster_settings["cluster_count"]),
        key=lambda cluster_idx: float(
            cluster_personalities[
                raw_cluster_labels == cluster_idx,
                neuroticism_idx,
            ].mean(),
        ),
    )
    cluster_labels = np.empty_like(raw_cluster_labels)
    for new_cluster_idx, old_cluster_idx in enumerate(cluster_order):
        cluster_labels[raw_cluster_labels == old_cluster_idx] = new_cluster_idx

    reducer = umap.UMAP(
        random_state=cluster_settings["cluster_seed"],
        n_neighbors=30,
        min_dist=0.15,
    )
    embedding = reducer.fit_transform(scaled_personalities)
    influence = cluster_society.metadata["Influence"].to_numpy(dtype=np.float64)
    influence_sizes = 2.0 + 8.0 * (
        (influence - influence.min()) / max(np.ptp(influence), 1e-6)
    )
    # Subsample if too many individuals to avoid cluttered visualization
    max_plot_points = 2000
    if len(embedding) > max_plot_points:
        rng = np.random.default_rng(cluster_settings.get("cluster_seed", 42))
        plot_idx = rng.choice(len(embedding), max_plot_points, replace=False)
    else:
        plot_idx = np.arange(len(embedding))

    # Figure 1: Personality Cluster UMAP
    fig1, ax1 = setup_plot(
        title="Personality Cluster UMAP",
        xlabel="UMAP 1",
        ylabel="UMAP 2",
    )
    for cluster_idx in range(cluster_settings["cluster_count"]):
        mask = (cluster_labels == cluster_idx)
        # Intersect mask with subsampled indices
        mask_idx = np.where(mask)[0]
        plot_mask_idx = np.intersect1d(mask_idx, plot_idx)
        
        if len(plot_mask_idx) == 0:
            continue
            
        ax1.scatter(
            embedding[plot_mask_idx, 0],
            embedding[plot_mask_idx, 1],
            s=influence_sizes[plot_mask_idx],
            alpha=0.35,
            color=CATEGORICAL_COLORS[cluster_idx % len(CATEGORICAL_COLORS)],
            label=f"C{cluster_idx + 1}",
        )
    ax1.legend(ncol=2)
    save_paper_figure(fig1, output_dir / "01_cluster_umap.png")
    plt.close(fig1)

    # Figure 2: Cluster Trait Profiles
    cluster_centers = np.vstack(
        [
            cluster_personalities[cluster_labels == cluster_idx].mean(axis=0)
            for cluster_idx in range(cluster_settings["cluster_count"])
        ],
    )
    fig2, ax2 = setup_plot(title="Cluster Trait Profiles")
    im = ax2.imshow(
        cluster_centers,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap=PAPER_DIVERGING_CMAP,
    )
    ax2.grid(False)
    ax2.set_xticks(range(5), ["O", "C", "E", "A", "N"])
    ax2.set_yticks(
        range(cluster_settings["cluster_count"]),
        [f"C{i + 1}" for i in range(cluster_settings["cluster_count"])],
    )
    fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    save_paper_figure(fig2, output_dir / "02_cluster_profiles.png")
    plt.close(fig2)

    # Figure 3: Cluster Neuroticism Spread
    neuroticism_by_cluster = [
        cluster_personalities[cluster_labels == cluster_idx, neuroticism_idx]
        for cluster_idx in range(cluster_settings["cluster_count"])
    ]
    fig3, ax3 = setup_plot(
        title="Cluster Neuroticism Spread",
        xlabel="Cluster",
        ylabel="Neuroticism",
    )
    neuroticism_boxplot = ax3.boxplot(
        neuroticism_by_cluster,
        tick_labels=[f"C{i + 1}" for i in range(cluster_settings["cluster_count"])],
        patch_artist=True,
        showfliers=False,
    )
    for idx, patch in enumerate(neuroticism_boxplot["boxes"]):
        patch.set_facecolor(CATEGORICAL_COLORS[idx % len(CATEGORICAL_COLORS)])
        patch.set_alpha(0.7)
    save_paper_figure(fig3, output_dir / "03_neuroticism_spread.png")
    plt.close(fig3)

    # 4. Polarization as a distribution rather than a single scalar
    polarization_config = get_test_scenario("bimodality_polarization").sim_config()
    polarization_society = prepare_scenario_society(
        "bimodality_polarization",
        tmp_path,
        enable_evolution=polarization_config.enable_evolution,
        output_name="advanced_bimodality",
    )
    fairness = polarization_society.exposures[:, DIMENSION_INDICES["Fairness"]].numpy()
    fairness_mean = fairness.mean()
    fairness_bc = bimodality_coefficient(fairness)

    fig4, ax4 = setup_plot(
        title="Fairness Polarization",
        xlabel="Fairness Exposure",
        ylabel="Agent Count",
    )
    ax4.hist(
        [fairness[fairness < fairness_mean], fairness[fairness >= fairness_mean]],
        bins=26,
        stacked=True,
        color=COMPARISON_COLORS,
        alpha=0.85,
        label=["Lower fairness pole", "Upper fairness pole"],
    )
    ax4.axvline(
        fairness_mean,
        color=PAPER_PALETTE["neutral"],
        linestyle="--",
        linewidth=2,
    )
    ax4.legend()
    ax4.text(
        0.03,
        0.95,
        f"BC = {fairness_bc:.2f}",
        transform=ax4.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    save_paper_figure(fig4, output_dir / "04_fairness_polarization.png")
    plt.close(fig4)

    # 5. Class composition stacked by wealth quartile
    evolved_wealth_config = get_test_scenario("figure_wealth_evolved").sim_config()
    evolved_wealth_society = prepare_scenario_society(
        "figure_wealth_evolved",
        tmp_path,
        enable_evolution=evolved_wealth_config.enable_evolution,
        output_name="advanced_wealth_evolved",
    )
    wealth_values = evolved_wealth_society.metadata["Raw_Wealth"].to_numpy(dtype=np.float64)
    class_labels = evolved_wealth_society.metadata["Class"].astype(str).to_numpy()
    wealth_quantile_ids = np.digitize(
        wealth_values,
        np.quantile(wealth_values, [0.25, 0.5, 0.75]),
        right=True,
    )
    ordered_classes = [
        label
        for label in [
            "Underclass",
            "Working Class",
            "Middle Class",
            "Upper Middle",
            "Elite",
        ]
        if label in set(class_labels)
    ]
    class_mix = []
    for quartile_idx in range(4):
        quartile_classes = class_labels[wealth_quantile_ids == quartile_idx]
        class_mix.append(
            np.array(
                [(quartile_classes == class_name).sum() for class_name in ordered_classes],
                dtype=np.float64,
            ),
        )

    fig5, ax5 = setup_plot(
        title="Class Mix by Wealth Quartile",
        xlabel="Wealth Quartile",
        ylabel="Share",
    )
    _stacked_share_bars(
        ax5,
        ["Q1", "Q2", "Q3", "Q4"],
        class_mix,
        ordered_classes,
        CATEGORICAL_COLORS[: len(ordered_classes)],
    )
    ax5.legend(ncol=2)
    save_paper_figure(fig5, output_dir / "05_class_mix_by_wealth.png")
    plt.close(fig5)

    # 6. Sentiment shown as stacked composition
    semantic_scenario = get_test_scenario("figure_semantic_alignment")
    semantic_config = semantic_scenario.sim_config()
    semantic_settings = semantic_scenario.settings()
    semantic_society = prepare_scenario_society(
        "figure_semantic_alignment",
        tmp_path,
        enable_evolution=semantic_config.enable_evolution,
        output_name="advanced_semantic",
    )
    prosperity_result = run_debug_simulation(
        semantic_config,
        build_world(semantic_settings["positive_world"]),
        society=semantic_society,
        urgency=semantic_settings["urgency"],
    )
    threat_result = run_debug_simulation(
        semantic_config,
        build_world(semantic_settings["negative_world"]),
        society=semantic_society,
        urgency=semantic_settings["urgency"],
    )

    fig6, ax6 = setup_plot(
        title="Semantic Sentiment Composition",
        xlabel="World",
        ylabel="Share",
    )
    _stacked_share_bars(
        ax6,
        ["Prosperity", "Threat"],
        [
            map_emotions_to_sentiment(
                prosperity_result.social_state["objective_center"],
                prosperity_result.social_state["acting_ratio"],
                config=semantic_config,
            ),
            map_emotions_to_sentiment(
                threat_result.social_state["objective_center"],
                threat_result.social_state["acting_ratio"],
                config=semantic_config,
            ),
        ],
        ["Negative", "Neutral", "Positive"],
        SENTIMENT_COLORS,
    )
    ax6.legend()
    save_paper_figure(fig6, output_dir / "06_sentiment_composition.png")
    plt.close(fig6)

    # 7. Endogenous events as stacked social-state sentiment
    endogenous_scenario = get_test_scenario("endogenous_events")
    endogenous_config = endogenous_scenario.sim_config()
    endogenous_settings = endogenous_scenario.settings()
    endogenous_influence = torch.ones(endogenous_config.num_agents)
    stable_emotions = zero_emotions(endogenous_config.num_agents)
    set_emotions(stable_emotions, endogenous_settings["stable_emotion"])
    stable_state = aggregate_social_state(
        endogenous_config,
        stable_emotions,
        endogenous_influence,
    )
    polarized_emotions = zero_emotions(endogenous_config.num_agents)
    midpoint = endogenous_config.num_agents // 2
    set_emotions(
        polarized_emotions,
        endogenous_settings["polarized_group_a"],
        rows=slice(None, midpoint),
    )
    set_emotions(
        polarized_emotions,
        endogenous_settings["polarized_group_b"],
        rows=slice(midpoint, None),
    )
    polarized_state = aggregate_social_state(
        endogenous_config,
        polarized_emotions,
        endogenous_influence,
    )

    fig7, ax7 = setup_plot(
        title="Endogenous Event Trigger",
        xlabel="Social State",
        ylabel="Share",
    )
    _stacked_share_bars(
        ax7,
        ["Stable", "Polarized"],
        [
            map_emotions_to_sentiment(
                stable_state["objective_center"],
                stable_state["acting_ratio"],
                config=endogenous_config,
            ),
            map_emotions_to_sentiment(
                polarized_state["objective_center"],
                polarized_state["acting_ratio"],
                config=endogenous_config,
            ),
        ],
        ["Negative", "Neutral", "Positive"],
        SENTIMENT_COLORS,
    )
    ax7.legend()
    ax7.text(
        0.03,
        0.95,
        f"Action: {polarized_state.get('action_name') or 'None'}",
        transform=ax7.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    save_paper_figure(fig7, output_dir / "07_endogenous_event_trigger.png")
    plt.close(fig7)

    # 8. Personal shocks stay more local
    personal_scenario = get_test_scenario("personal")
    personal_config = personal_scenario.sim_config()
    personal_settings = personal_scenario.settings()
    personal_society = prepare_scenario_society(
        "personal",
        tmp_path,
        enable_evolution=personal_config.enable_evolution,
        output_name="advanced_personal",
    )
    personal_world = build_world(personal_settings["world"])
    general_result = run_debug_simulation(
        personal_config,
        personal_world,
        society=personal_society,
        urgency=personal_settings["urgency"],
        is_personal=False,
    )
    personal_result = run_debug_simulation(
        personal_config,
        personal_world,
        society=personal_society,
        urgency=personal_settings["urgency"],
        is_personal=True,
    )

    fig8, ax8 = setup_plot(
        title="Event Scope Localization",
        xlabel="Event Scope",
        ylabel="Engagement",
    )
    scope_boxplot = ax8.boxplot(
        [
            general_result.engagement_scores.numpy(),
            personal_result.engagement_scores.numpy(),
        ],
        tick_labels=["General", "Personal"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(scope_boxplot["boxes"], COMPARISON_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    save_paper_figure(fig8, output_dir / "08_event_scope_localization.png")
    plt.close(fig8)

    # 9. Cascade-size distribution from sampled seeds
    r0_scenario = get_test_scenario("r0_basic_reproduction")
    r0_config = r0_scenario.sim_config()
    r0_settings = r0_scenario.settings()
    r0_society = prepare_scenario_society(
        "r0_basic_reproduction",
        tmp_path,
        enable_evolution=r0_config.enable_evolution,
        output_name="advanced_r0",
    )
    r0_rng = np.random.default_rng(r0_settings["rng_seed"])
    r0_seed_indices = r0_rng.choice(
        r0_config.num_agents,
        size=r0_settings["seed_sample_count"],
        replace=False,
    )
    cascade_sizes = []
    for idx in r0_seed_indices:
        thought = r0_society.exposures[idx].unsqueeze(0)
        r0_result = run_debug_simulation(
            r0_config,
            thought,
            society=r0_society,
            urgency=r0_settings["urgency"],
        )
        engaged = (r0_result.engagement_scores > r0_config.cascade_threshold).sum().item() - 1
        cascade_sizes.append(max(0, engaged))
    cascade_sizes = np.asarray(cascade_sizes, dtype=np.int64)
    cascade_bins = np.arange(cascade_sizes.max() + 2) - 0.5

    fig9, ax9 = setup_plot(
        title="Cascade Size Distribution",
        xlabel="Secondary Engagement Count",
        ylabel="Sample Count",
    )
    ax9.hist(
        cascade_sizes,
        bins=cascade_bins,
        color=PAPER_PALETTE["primary"],
        alpha=0.85,
        edgecolor="white",
    )
    ax9.axvline(
        cascade_sizes.mean(),
        color=PAPER_PALETTE["secondary"],
        linestyle="--",
        linewidth=2,
    )
    save_paper_figure(fig9, output_dir / "09_cascade_size_distribution.png")
    plt.close(fig9)




def test_generate_research_paper_multiseed_debug_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated" / "multiseed_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [7, 21, 42, 84]
    wealth_baseline_gini = []
    wealth_evolved_gini = []
    echo_low_similarity = []
    echo_high_similarity = []
    influence_flat_gini = []
    influence_power_gini = []
    consensus_baseline_distance = []
    consensus_distance = []
    social_base_distance = []
    socialized_distance = []
    fairness_bimodality = []

    for seed in seeds:
        _, _, wealth_baseline_society = _prepare_seeded_society(
            "figure_wealth_baseline",
            tmp_path,
            seed,
            output_prefix="multiseed_wealth_baseline",
        )
        _, _, wealth_evolved_society = _prepare_seeded_society(
            "figure_wealth_evolved",
            tmp_path,
            seed,
            output_prefix="multiseed_wealth_evolved",
        )
        wealth_baseline_gini.append(
            gini(wealth_baseline_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()),
        )
        wealth_evolved_gini.append(
            gini(wealth_evolved_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy()),
        )

        _, _, echo_low_society = _prepare_seeded_society(
            "figure_echo_chambers_low",
            tmp_path,
            seed,
            output_prefix="multiseed_echo_low",
        )
        _, _, echo_high_society = _prepare_seeded_society(
            "figure_echo_chambers_high",
            tmp_path,
            seed,
            output_prefix="multiseed_echo_high",
        )
        echo_low_similarity.append(
            mean_edge_topology_similarity(
                echo_low_society.exposures,
                echo_low_society.personalities,
                echo_low_society.adjacency_matrix,
            ),
        )
        echo_high_similarity.append(
            mean_edge_topology_similarity(
                echo_high_society.exposures,
                echo_high_society.personalities,
                echo_high_society.adjacency_matrix,
            ),
        )

        _, _, influence_flat_society = _prepare_seeded_society(
            "cascade_power_law_flat",
            tmp_path,
            seed,
            output_prefix="multiseed_influence_flat",
        )
        _, _, influence_power_society = _prepare_seeded_society(
            "cascade_power_law_power",
            tmp_path,
            seed,
            output_prefix="multiseed_influence_power",
        )
        influence_flat_gini.append(gini(influence_flat_society.metadata["Influence"].to_numpy()))
        influence_power_gini.append(gini(influence_power_society.metadata["Influence"].to_numpy()))

        consensus_config, consensus_settings, consensus_society = _prepare_seeded_society(
            "figure_social_consensus",
            tmp_path,
            seed,
            output_prefix="multiseed_consensus",
        )
        assert consensus_society.adjacency_matrix is not None
        consensus_world = build_world(consensus_settings["world"])
        baseline_consensus_config = get_test_scenario(
            "perception_social_consensus_baseline",
        ).sim_config(
            num_agents=consensus_config.num_agents,
            seed=seed,
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
        consensus_baseline_distance.append(
            average_neighbor_distance(
                baseline_perceived,
                consensus_society.adjacency_matrix,
            ),
        )
        consensus_distance.append(
            average_neighbor_distance(
                consensus_perceived,
                consensus_society.adjacency_matrix,
            ),
        )

        _, _, social_base_society = _prepare_seeded_society(
            "personality_socialization_base",
            tmp_path,
            seed,
            output_prefix="multiseed_social_base",
        )
        _, _, socialized_society = _prepare_seeded_society(
            "personality_socialization_socialized",
            tmp_path,
            seed,
            output_prefix="multiseed_socialized",
        )
        social_base_distance.append(
            average_neighbor_distance(
                social_base_society.personalities,
                social_base_society.adjacency_matrix,
            ),
        )
        socialized_distance.append(
            average_neighbor_distance(
                socialized_society.personalities,
                socialized_society.adjacency_matrix,
            ),
        )

        _, _, polarization_society = _prepare_seeded_society(
            "bimodality_polarization",
            tmp_path,
            seed,
            output_prefix="multiseed_bimodality",
        )
        fairness_bimodality.append(
            bimodality_coefficient(
                polarization_society.exposures[:, DIMENSION_INDICES["Fairness"]].numpy(),
            ),
        )

    # Figure 1: Wealth Gini by Seed
    fig1, ax1 = setup_plot()
    _plot_seed_lines(
        ax1,
        seeds,
        {
            "Baseline": wealth_baseline_gini,
            "Evolved": wealth_evolved_gini,
        },
        title="Wealth Gini by Seed",
        ylabel="Gini",
    )
    save_paper_figure(fig1, output_dir / "01_wealth_gini_by_seed.png")
    plt.close(fig1)

    # Figure 2: Echo Similarity by Seed
    fig2, ax2 = setup_plot()
    _plot_seed_lines(
        ax2,
        seeds,
        {
            "Low homophily": echo_low_similarity,
            "High homophily": echo_high_similarity,
        },
        title="Echo Similarity by Seed",
        ylabel="Topology Similarity",
    )
    save_paper_figure(fig2, output_dir / "02_echo_similarity_by_seed.png")
    plt.close(fig2)

    # Figure 3: Influence Inequality by Seed
    fig3, ax3 = setup_plot()
    _plot_seed_lines(
        ax3,
        seeds,
        {
            "Flat influence": influence_flat_gini,
            "Power law": influence_power_gini,
        },
        title="Influence Inequality by Seed",
        ylabel="Influence Gini",
    )
    save_paper_figure(fig3, output_dir / "03_influence_inequality_by_seed.png")
    plt.close(fig3)

    # Figure 4: Consensus Distance by Seed
    fig4, ax4 = setup_plot()
    _plot_seed_lines(
        ax4,
        seeds,
        {
            "Baseline": consensus_baseline_distance,
            "Consensus": consensus_distance,
        },
        title="Consensus Distance by Seed",
        ylabel="Neighbor Distance",
    )
    save_paper_figure(fig4, output_dir / "04_consensus_distance_by_seed.png")
    plt.close(fig4)

    # Figure 5: Trait Friction by Seed
    fig5, ax5 = setup_plot()
    _plot_seed_lines(
        ax5,
        seeds,
        {
            "Base": social_base_distance,
            "Socialized": socialized_distance,
        },
        title="Trait Friction by Seed",
        ylabel="Neighbor Trait Distance",
    )
    save_paper_figure(fig5, output_dir / "05_trait_friction_by_seed.png")
    plt.close(fig5)

    # Figure 6: Polarization by Seed
    fig6, ax6 = setup_plot()
    _plot_seed_lines(
        ax6,
        seeds,
        {"Fairness BC": fairness_bimodality},
        title="Polarization by Seed",
        ylabel="Bimodality Coefficient",
    )
    ax6.axhline(
        0.555,
        color=PAPER_PALETTE["secondary"],
        linestyle="--",
        label="Polarized threshold",
    )
    ax6.legend()
    save_paper_figure(fig6, output_dir / "06_polarization_by_seed.png")
    plt.close(fig6)

    compose_panel_grid(
        sorted(output_dir.glob("*.png")),
        output_dir.parent / "research_paper_multiseed_debug.png",
        title="Multi-Seed Robustness Checks",
        columns=3,
    )
