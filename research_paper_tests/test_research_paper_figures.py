from pathlib import Path

import community as community_louvain
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from main import (
    DIMENSION_INDICES,
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

matplotlib.use("Agg")


def _line_of_best_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    ys = slope * xs + intercept
    return xs, ys


def _lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64).flatten()
    if array.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    if np.min(array) < 0.0:
        array = array - np.min(array)
    array = np.sort(array + 1e-9)
    cumulative = np.concatenate([[0.0], np.cumsum(array) / np.sum(array)])
    population = np.linspace(0.0, 1.0, cumulative.size)
    return population, cumulative


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
    for label, values in series.items():
        ax.plot(seeds, values, marker="o", linewidth=2, label=label)
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
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "research_paper_summary.png"

    fig, axes = plt.subplots(5, 4, figsize=(24, 28))
    axes = axes.flatten()

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
    # x-axis: each agent's Neuroticism trait score.
    # y-axis: how much the agent's perceived Physical_Safety signal deviates from
    # the true world signal. Higher points mean stronger threat exaggeration.
    # The red line summarizes whether more neurotic agents distort the signal more.
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
    # x-axis: consolidation/decay step number after the initial memory is formed.
    # y-axis: overall memory strength (vector norm). Higher values mean the memory
    # trace remains stronger. A slower drop means the memory is retained longer.
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
    # x-axis: Openness trait score for each agent.
    # y-axis: final engagement score under this cognitive-gate scenario.
    # Each dot is one agent. The trend line shows whether openness is associated
    # with greater or lower willingness to engage with the incoming signal.
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
    # x-axis: experimental condition with and without algorithmic amplification.
    # y-axis: mean engagement across the whole population.
    # Taller bars indicate that the recommender/amplification logic keeps more
    # users engaged on average.
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
    consensus_society = prepare_scenario_society(
        "figure_social_consensus",
        tmp_path,
        enable_evolution=consensus_config.enable_evolution,
        output_name="consensus",
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
    # x-axis: perception model without vs. with local social consensus.
    # y-axis: average distance between neighbors' perceived world states.
    # Lower bars mean neighbors think more similarly, so consensus pulls local
    # perceptions toward one another.
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
    # x-axis: baseline aggregation vs. Granovetter threshold dynamics enabled.
    # y-axis: acting ratio, i.e. the share of agents who end up taking action.
    # A taller Cascade bar means a small committed minority triggers wider uptake.
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
    # x-axis: low-homophily vs. high-homophily network generation settings.
    # y-axis: similarity across connected pairs in the same combined feature space
    # that the topology builder uses (non-wealth exposures + personality traits).
    # Higher values mean neighbors already resemble each other more according to
    # the network-construction logic, which is the structural signature of echo chambers.
    axes[6].bar(
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
        color=["#ced4da", "#6d597a"],
    )
    axes[6].set_title("Echo Chambers")
    axes[6].set_ylabel("Topology Similarity")

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
    # x-axis: the same low/high homophily comparison as the previous panel.
    # y-axis: Louvain modularity Q, where larger values mean cleaner separation
    # into densely connected communities with fewer cross-community links.
    axes[7].bar(
        ["Low", "High"],
        [low_modularity, high_modularity],
        color=["#adb5bd", "#264653"],
    )
    axes[7].set_title("Louvain Modularity")
    axes[7].set_ylabel("Q")

    # 9. Personality correlation heatmap
    corr_config = get_test_scenario("figure_personality_correlations").sim_config()
    corr_society = prepare_scenario_society(
        "figure_personality_correlations",
        tmp_path,
        enable_evolution=corr_config.enable_evolution,
        output_name="corr",
    )
    observed_corr = np.corrcoef(corr_society.personalities.numpy().T)
    # x-axis and y-axis both list the Big Five traits: O, C, E, A, N.
    # Each cell shows the correlation between a pair of traits across agents.
    # Warm colors mean the traits rise together, cool colors mean trade-offs, and
    # the diagonal is always 1 because each trait is perfectly correlated with itself.
    im = axes[8].imshow(observed_corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axes[8].set_title("Personality Correlations")
    axes[8].set_xticks(range(5), ["O", "C", "E", "A", "N"])
    axes[8].set_yticks(range(5), ["O", "C", "E", "A", "N"])
    fig.colorbar(im, ax=axes[8], fraction=0.046, pad=0.04)

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
    # x-axis: baseline society vs. evolved society.
    # y-axis: Gini coefficient of the Wealth dimension, where 0 is perfect equality
    # and larger values mean more concentration of wealth among fewer agents.
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
    # x-axis: social group identity in the deprivation experiment.
    # y-axis: mean Anger after exposure to the same world state.
    # A higher marginalized bar indicates the setup produces asymmetric emotional
    # burden rather than a uniform reaction across groups.
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
    )
    neg = map_emotions_to_sentiment(
        negative_sentiment,
        semantic_negative.social_state["acting_ratio"],
    )
    x = np.arange(3)
    width = 0.35
    # x-axis: sentiment buckets produced by mapping the social emotional state into
    # Negative / Neutral / Positive components.
    # y-axis: strength of each sentiment component for the prosperity vs. threat
    # worlds. Whichever bar is taller dominates the emotional interpretation.
    axes[11].bar(x - width / 2, pos, width=width, label="Prosperity")
    axes[11].bar(x + width / 2, neg, width=width, label="Threat")
    axes[11].set_xticks(x, ["Negative", "Neutral", "Positive"])
    axes[11].set_title("Semantic Sentiment Profile")
    axes[11].legend()

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
            torch.randn(backbone_config.num_agents, PERSONALITY_TRAIT_COUNT)
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

    # x-axis: network before and after triadic closure is applied.
    # y-axis: average clustering coefficient, which measures how often a node's
    # neighbors are also connected to each other. Higher means more triangles.
    axes[12].bar(
        ["Backbone", "Closure"],
        [average_clustering(backbone), average_clustering(refined)],
        color=["#adb5bd", "#2a9d8f"],
    )
    axes[12].set_title("Network Clustering")
    axes[12].set_ylabel("Average Clustering")

    # 14. Personality socialization
    base_social_config = get_test_scenario("personality_socialization_base").sim_config()
    socialized_config = get_test_scenario(
        "personality_socialization_socialized"
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
    # x-axis: unsocialized baseline vs. socialized personality dynamics.
    # y-axis: average personality distance between connected neighbors.
    # Lower values mean network neighbors have become more alike in trait space.
    axes[13].bar(
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
        color=["#f4a261", "#2a9d8f"],
    )
    axes[13].set_title("Personality Socialization")
    axes[13].set_ylabel("Neighbor Trait Distance")

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
    flat_population, flat_cumulative = _lorenz_curve(flat_influence)
    power_population, power_cumulative = _lorenz_curve(power_influence)
    # x-axis: cumulative population share after sorting agents from least to most influential.
    # y-axis: cumulative share of total influence held by that bottom share.
    # The dashed diagonal is perfect equality; curves farther below it indicate
    # stronger concentration, and the legend's Gini gives the same idea numerically.
    axes[14].plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="#adb5bd",
        linewidth=1.5,
        label="Perfect equality",
    )
    axes[14].plot(
        flat_population,
        flat_cumulative,
        linewidth=2,
        color="#457b9d",
        label=f"Flat (G={gini(flat_influence):.2f})",
    )
    axes[14].plot(
        power_population,
        power_cumulative,
        linewidth=2,
        color="#e76f51",
        label=f"Power law (G={gini(power_influence):.2f})",
    )
    axes[14].set_title("Influence Tail")
    axes[14].set_xlabel("Population Share")
    axes[14].set_ylabel("Influence Share")
    axes[14].legend(fontsize=8)

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
                ).sum()
            )
        )
    realized_reach = np.asarray(realized_reach, dtype=np.float64)
    sampled_influence = reach_influence[reach_indices]
    xs, ys = _line_of_best_fit(sampled_influence, realized_reach)
    # x-axis: each sampled agent's structural influence score from the network.
    # y-axis: realized reach, meaning how many agents actually end up engaged.
    # Upward slope means central/high-status agents translate network position into
    # wider practical spread rather than just nominal influence.
    axes[15].scatter(
        sampled_influence,
        realized_reach,
        s=18,
        alpha=0.6,
        color="#457b9d",
    )
    axes[15].plot(xs, ys, color="#e63946", linewidth=2)
    axes[15].set_title("Influence vs. Reach")
    axes[15].set_xlabel("Structural Influence")
    axes[15].set_ylabel("Realized Reach")

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
    # x-axis: agents' Fairness exposure values.
    # y-axis: how many agents fall into each bin.
    # Multiple separated peaks suggest polarization into camps, and a higher BC
    # annotation indicates the distribution is more strongly bimodal.
    axes[16].hist(
        fairness,
        bins=24,
        color="#6d597a",
        alpha=0.8,
        edgecolor="white",
    )
    axes[16].axvline(fairness.mean(), color="#f4a261", linestyle="--", linewidth=2)
    axes[16].text(
        0.04,
        0.95,
        f"BC = {fairness_bc:.2f}",
        transform=axes[16].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    axes[16].set_title("Fairness Polarization")
    axes[16].set_xlabel("Fairness Exposure")
    axes[16].set_ylabel("Agent Count")

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
    # x-axis: the two hand-crafted agent archetypes in this truth-refinement test.
    # y-axis: attention weight assigned to short-term vs. long-term considerations.
    # Within each pair, compare which timescale dominates; across pairs, compare
    # how different personality styles allocate cognitive attention.
    axes[17].bar(
        truth_x - width / 2,
        truth_attention[:, 10].numpy(),
        width=width,
        label="Short term",
        color="#8ecae6",
    )
    axes[17].bar(
        truth_x + width / 2,
        truth_attention[:, 11].numpy(),
        width=width,
        label="Long term",
        color="#ffb703",
    )
    axes[17].set_xticks(truth_x, ["Populist", "Skeptical"])
    axes[17].set_title("Truth Refinement")
    axes[17].set_ylabel("Attention Weight")
    axes[17].legend(fontsize=8)

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
    # x-axis: how many times the same threat has been repeated.
    # y-axis: mean engagement level. The line shows sensitization to repetition,
    # while the horizontal references compare a new threat with prior memory
    # already built ("stacked") versus a truly fresh population.
    axes[18].plot(
        memory_steps,
        repeated_curve,
        marker="o",
        color="#264653",
        label="Repeated threat",
    )
    axes[18].axhline(
        stacked_result.engagement_scores.mean().item(),
        color="#e76f51",
        linestyle="--",
        linewidth=2,
        label="Stacked new threat",
    )
    axes[18].axhline(
        fresh_result.engagement_scores.mean().item(),
        color="#8d99ae",
        linestyle=":",
        linewidth=2,
        label="Fresh new threat",
    )
    axes[18].set_title("Agent Memory")
    axes[18].set_xlabel("Repeat Exposure")
    axes[18].set_ylabel("Mean Engagement")
    axes[18].legend(fontsize=8)

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
            ]
        ),
    )
    virality_x = np.arange(2)
    # x-axis: a broad consensus emotion profile vs. an outlier-heavy profile.
    # y-axis: outrage multiplier produced by the aggregation logic.
    # Compare mean and max bars to the dashed configured cap to verify that even
    # extreme cases stay bounded instead of exploding without limit.
    axes[19].bar(
        virality_x - width / 2,
        [
            consensus_state["mean_outrage_multiplier"],
            outlier_state["mean_outrage_multiplier"],
        ],
        width=width,
        label="Mean",
        color="#a8dadc",
    )
    axes[19].bar(
        virality_x + width / 2,
        [
            consensus_state["max_outrage_multiplier"],
            outlier_state["max_outrage_multiplier"],
        ],
        width=width,
        label="Max",
        color="#1d3557",
    )
    axes[19].axhline(
        1.0 + virality_config.max_viral_multiplier,
        color="#e63946",
        linestyle="--",
        linewidth=2,
        label="Configured cap",
    )
    axes[19].set_xticks(virality_x, ["Consensus", "Outliers"])
    axes[19].set_title("Virality Bounds")
    axes[19].set_ylabel("Outrage Multiplier")
    axes[19].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.2)

    fig.suptitle("Expanded Research Paper Summary Panels", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_research_paper_advanced_visualizations(tmp_path):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import umap

    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "research_paper_advanced_visualizations.png"

    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    axes = axes.flatten()

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
            ].mean()
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
    influence_sizes = 16.0 + 48.0 * (
        (influence - influence.min()) / max(np.ptp(influence), 1e-6)
    )
    cluster_colors = plt.get_cmap("tab10")(
        np.linspace(0.0, 1.0, cluster_settings["cluster_count"])
    )

    for cluster_idx in range(cluster_settings["cluster_count"]):
        mask = cluster_labels == cluster_idx
        # x-axis/y-axis: UMAP coordinates, which are learned embedding dimensions
        # rather than original traits. Their absolute values do not matter; local
        # distance and separation do. Nearby points have similar personalities.
        # Point size reflects influence, so large points are influential agents.
        axes[0].scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=influence_sizes[mask],
            alpha=0.65,
            color=cluster_colors[cluster_idx],
            label=f"C{cluster_idx + 1}",
        )
    axes[0].set_title("Personality Cluster UMAP")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].legend(fontsize=8, ncol=2)

    cluster_centers = np.vstack(
        [
            cluster_personalities[cluster_labels == cluster_idx].mean(axis=0)
            for cluster_idx in range(cluster_settings["cluster_count"])
        ]
    )
    # x-axis: Big Five traits O, C, E, A, N.
    # y-axis: cluster identity after sorting clusters by average neuroticism.
    # Cell color is the mean trait level of that cluster, so each row acts like a
    # compact personality fingerprint for one discovered subgroup.
    im = axes[1].imshow(
        cluster_centers,
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
        cmap="coolwarm",
    )
    axes[1].set_title("Cluster Trait Profiles")
    axes[1].set_xticks(range(5), ["O", "C", "E", "A", "N"])
    axes[1].set_yticks(
        range(cluster_settings["cluster_count"]),
        [f"C{i + 1}" for i in range(cluster_settings["cluster_count"])],
    )
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    neuroticism_by_cluster = [
        cluster_personalities[cluster_labels == cluster_idx, neuroticism_idx]
        for cluster_idx in range(cluster_settings["cluster_count"])
    ]
    neuroticism_boxplot = axes[2].boxplot(
        neuroticism_by_cluster,
        tick_labels=[f"C{i + 1}" for i in range(cluster_settings["cluster_count"])],
        patch_artist=True,
        showfliers=False,
    )
    # x-axis: cluster identity.
    # y-axis: Neuroticism scores within that cluster.
    # The median line shows the typical value, the box shows the middle 50%, and
    # taller boxes/whiskers mean more within-cluster spread.
    for patch, color in zip(neuroticism_boxplot["boxes"], cluster_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[2].set_title("Cluster Neuroticism Spread")
    axes[2].set_ylabel("Neuroticism")

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
    # x-axis: Fairness exposure.
    # y-axis: agent count per bin.
    # The stacked colors split the population below vs. above the mean so the two
    # poles are visually separated. Clear mass on both sides indicates polarization.
    axes[3].hist(
        [fairness[fairness < fairness_mean], fairness[fairness >= fairness_mean]],
        bins=26,
        stacked=True,
        color=["#8ecae6", "#ef476f"],
        alpha=0.85,
        label=["Lower fairness pole", "Upper fairness pole"],
    )
    axes[3].axvline(fairness_mean, color="#1d3557", linestyle="--", linewidth=2)
    axes[3].set_title("Fairness Polarization")
    axes[3].set_xlabel("Fairness Exposure")
    axes[3].set_ylabel("Agent Count")
    axes[3].legend(fontsize=8)
    axes[3].text(
        0.03,
        0.95,
        f"BC = {fairness_bc:.2f}",
        transform=axes[3].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

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
            )
        )
    # x-axis: wealth quartiles from Q1 (poorest) to Q4 (richest).
    # y-axis: within-quartile share, so every bar sums to 100%.
    # Each colored segment shows class composition inside that quartile. This helps
    # separate absolute wealth ranking from the class labels assigned to agents.
    _stacked_share_bars(
        axes[4],
        ["Q1", "Q2", "Q3", "Q4"],
        class_mix,
        ordered_classes,
        ["#8d99ae", "#adb5bd", "#90be6d", "#f4a261", "#e76f51"][: len(ordered_classes)],
    )
    axes[4].set_title("Class Mix by Wealth Quartile")
    axes[4].legend(fontsize=8, ncol=2)

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
    # x-axis: world framing condition, prosperity vs. threat.
    # y-axis: share of the resulting sentiment mix in Negative / Neutral / Positive.
    # Because bars are normalized, compare color proportions rather than total height.
    _stacked_share_bars(
        axes[5],
        ["Prosperity", "Threat"],
        [
            map_emotions_to_sentiment(
                prosperity_result.social_state["objective_center"],
                prosperity_result.social_state["acting_ratio"],
            ),
            map_emotions_to_sentiment(
                threat_result.social_state["objective_center"],
                threat_result.social_state["acting_ratio"],
            ),
        ],
        ["Negative", "Neutral", "Positive"],
        ["#d62828", "#adb5bd", "#2a9d8f"],
    )
    axes[5].set_title("Semantic Sentiment Composition")
    axes[5].legend(fontsize=8)

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
    # x-axis: stable emotions vs. polarized emotions before the endogenous-event check.
    # y-axis: share of Negative / Neutral / Positive sentiment in the aggregate state.
    # The annotation reports which social action/event was triggered under that state,
    # so the panel links emotional composition to discrete collective outcomes.
    _stacked_share_bars(
        axes[6],
        ["Stable", "Polarized"],
        [
            map_emotions_to_sentiment(
                stable_state["objective_center"],
                stable_state["acting_ratio"],
            ),
            map_emotions_to_sentiment(
                polarized_state["objective_center"],
                polarized_state["acting_ratio"],
            ),
        ],
        ["Negative", "Neutral", "Positive"],
        ["#d62828", "#adb5bd", "#2a9d8f"],
    )
    axes[6].set_title("Endogenous Event Trigger")
    axes[6].legend(fontsize=8)
    axes[6].text(
        0.03,
        0.95,
        f"Action: {polarized_state.get('action_name') or 'None'}",
        transform=axes[6].transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

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
    scope_boxplot = axes[7].boxplot(
        [
            general_result.engagement_scores.numpy(),
            personal_result.engagement_scores.numpy(),
        ],
        tick_labels=["General", "Personal"],
        patch_artist=True,
        showfliers=False,
    )
    # x-axis: whether the same event is treated as general/public or personal/local.
    # y-axis: engagement score distribution across agents.
    # Compare medians and spread: a lower/tighter Personal box means the shock stays
    # more localized instead of diffusing broadly through the population.
    for patch, color in zip(scope_boxplot["boxes"], ["#457b9d", "#f4a261"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[7].set_title("Event Scope Localization")
    axes[7].set_ylabel("Engagement")

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
    # x-axis: number of secondary agents engaged by a sampled seed post.
    # y-axis: how many sampled seeds produced that cascade size.
    # A long right tail means most seeds stay small but a few create much larger
    # cascades, which is typical of heavy-tailed diffusion processes.
    axes[8].hist(
        cascade_sizes,
        bins=cascade_bins,
        color="#6d597a",
        alpha=0.85,
        edgecolor="white",
    )
    axes[8].axvline(cascade_sizes.mean(), color="#e76f51", linestyle="--", linewidth=2)
    axes[8].set_title("Cascade Size Distribution")
    axes[8].set_xlabel("Secondary Engagement Count")
    axes[8].set_ylabel("Sample Count")

    for axis in axes:
        axis.grid(True, alpha=0.2)

    fig.suptitle("Research Paper Advanced Visualizations", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_generate_research_paper_multiseed_debug_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "research_paper_multiseed_debug.png"

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
            gini(wealth_baseline_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy())
        )
        wealth_evolved_gini.append(
            gini(wealth_evolved_society.exposures[:, DIMENSION_INDICES["Wealth"]].numpy())
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
            )
        )
        echo_high_similarity.append(
            mean_edge_topology_similarity(
                echo_high_society.exposures,
                echo_high_society.personalities,
                echo_high_society.adjacency_matrix,
            )
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
            "perception_social_consensus_baseline"
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
            )
        )
        consensus_distance.append(
            average_neighbor_distance(
                consensus_perceived,
                consensus_society.adjacency_matrix,
            )
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
            )
        )
        socialized_distance.append(
            average_neighbor_distance(
                socialized_society.personalities,
                socialized_society.adjacency_matrix,
            )
        )

        _, _, polarization_society = _prepare_seeded_society(
            "bimodality_polarization",
            tmp_path,
            seed,
            output_prefix="multiseed_bimodality",
        )
        fairness_bimodality.append(
            bimodality_coefficient(
                polarization_society.exposures[:, DIMENSION_INDICES["Fairness"]].numpy()
            )
        )

    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    axes = axes.flatten()

    # x-axis: random seed used to regenerate the baseline/evolved wealth scenarios.
    # y-axis: wealth inequality (Gini). Small gaps across seeds mean the inequality
    # result is reproducible instead of being an artifact of one initialization.
    _plot_seed_lines(
        axes[0],
        seeds,
        {
            "Baseline": wealth_baseline_gini,
            "Evolved": wealth_evolved_gini,
        },
        title="Wealth Gini by Seed",
        ylabel="Gini",
    )
    # x-axis: random seed.
    # y-axis: topology-space similarity between connected agents using the same
    # feature bundle the network generator uses. Consistently higher values for
    # the high-homophily line mean stronger echo chambers across reruns.
    _plot_seed_lines(
        axes[1],
        seeds,
        {
            "Low homophily": echo_low_similarity,
            "High homophily": echo_high_similarity,
        },
        title="Echo Similarity by Seed",
        ylabel="Topology Similarity",
    )
    # x-axis: random seed.
    # y-axis: Gini of influence scores. The comparison checks whether the power-law
    # setup keeps producing a more unequal influence distribution across reruns.
    _plot_seed_lines(
        axes[2],
        seeds,
        {
            "Flat influence": influence_flat_gini,
            "Power law": influence_power_gini,
        },
        title="Influence Inequality by Seed",
        ylabel="Influence Gini",
    )
    # x-axis: random seed.
    # y-axis: average neighbor perception distance. Lower consensus values across
    # seeds indicate the consensus mechanism reliably aligns local perceptions.
    _plot_seed_lines(
        axes[3],
        seeds,
        {
            "Baseline": consensus_baseline_distance,
            "Consensus": consensus_distance,
        },
        title="Consensus Distance by Seed",
        ylabel="Neighbor Distance",
    )
    # x-axis: random seed.
    # y-axis: neighbor personality distance. If the socialized line stays below the
    # base line, socialization consistently reduces trait friction between neighbors.
    _plot_seed_lines(
        axes[4],
        seeds,
        {
            "Base": social_base_distance,
            "Socialized": socialized_distance,
        },
        title="Trait Friction by Seed",
        ylabel="Neighbor Trait Distance",
    )
    # x-axis: random seed.
    # y-axis: bimodality coefficient of Fairness exposure. Values above the dashed
    # threshold suggest a clearly polarized distribution rather than a single mound.
    _plot_seed_lines(
        axes[5],
        seeds,
        {"Fairness BC": fairness_bimodality},
        title="Polarization by Seed",
        ylabel="Bimodality Coefficient",
    )
    axes[5].axhline(
        0.555,
        color="#e63946",
        linestyle="--",
        linewidth=2,
        label="Polarized threshold",
    )
    axes[5].legend(fontsize=8)

    for axis in axes:
        axis.grid(True, alpha=0.2)

    fig.suptitle("Research Paper Multi-Seed Debug Comparison", fontsize=20)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
