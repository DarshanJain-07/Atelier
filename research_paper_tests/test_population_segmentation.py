from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from main import run_debug_simulation
from research_paper_tests.config_schema import (
    DIMENSIONS,
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from research_paper_tests.plotting_utils import (
    CATEGORICAL_COLORS,
    apply_paper_style,
    compose_panel_grid,
    save_paper_figure,
    setup_plot,
)
from research_paper_tests.stats_utils import assert_statistically_greater, run_monte_carlo

matplotlib.use("Agg")
apply_paper_style()


def _reconstruct_action_ready_mask(config, society, result) -> np.ndarray:
    """Mirror the non-topology action gating in physics_engine.aggregate_society().

    The debug result exposes per-agent emotions and engagement but not the final
    boolean action mask. Rebuilding the same thresholds here lets us compare
    class-level readiness to act while keeping the test aligned with the real
    social-physics logic.
    """
    emotions = result.final_emotions.detach()
    engagement = result.engagement_scores.detach()
    influence = torch.tensor(
        society.metadata["Influence"].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )

    structural_weights = torch.log1p(influence)
    normalized_engagement = engagement / (engagement.mean() + 1e-9)
    weights = torch.clamp(structural_weights * normalized_engagement, min=1e-8)
    weights = weights / weights.sum()

    center_of_gravity = (emotions * weights.unsqueeze(1)).sum(dim=0)
    local_centers = center_of_gravity.unsqueeze(0).expand(emotions.shape[0], -1)

    final_arousal = torch.norm(emotions, dim=1)
    norm_emotion = emotions / (final_arousal.unsqueeze(1) + 1e-9)
    local_arousal = torch.norm(local_centers, dim=1)
    norm_local = local_centers / (local_arousal.unsqueeze(1) + 1e-9)
    alignment = (norm_emotion * norm_local).sum(dim=1)
    social_validation = 1.0 + alignment

    personalities = society.personalities
    action_cost = torch.clamp(
        config.base_action_cost
        - 0.1 * personalities[:, 2]
        - 0.1 * personalities[:, 4]
        - 0.05 * torch.log1p(influence),
        min=0.05,
    )

    individual_motivation = (final_arousal * social_validation) - action_cost
    max_emotion_strength, _ = torch.max(emotions, dim=1)
    is_motivated = individual_motivation > 0.1
    is_emotional = max_emotion_strength >= config.dominant_emotion_threshold
    engaged_mask = engagement > (engagement.mean() * 0.1)

    return (is_motivated & is_emotional & engaged_mask).detach().cpu().numpy()


def _class_stress_profiles(tmp_path: Path, output_name: str = "population_segmentation"):
    scenario = get_test_scenario("population_segmentation")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "population_segmentation",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name=output_name,
    )

    class_order = [
        class_name
        for class_name in settings["class_order"]
        if class_name in set(society.metadata["Class"].tolist())
    ]

    rows: list[dict[str, object]] = []
    for dimension_name in DIMENSIONS:
        for magnitude in settings["magnitudes"]:
            world = {} if magnitude == 0.0 else {dimension_name: magnitude}
            result = run_debug_simulation(
                config,
                build_world(world),
                society=society,
                urgency=settings["urgency"],
            )

            profiles = society.metadata.copy()
            profiles["engagement"] = result.engagement_scores.detach().cpu().numpy()
            profiles["action_ready_rate"] = _reconstruct_action_ready_mask(
                config,
                society,
                result,
            ).astype(np.float32)

            grouped = (
                profiles.groupby("Class", observed=False)[
                    ["engagement", "action_ready_rate"]
                ]
                .mean()
                .reindex(class_order)
            )

            for class_name, values in grouped.iterrows():
                rows.append(
                    {
                        "dimension": dimension_name,
                        "magnitude": float(magnitude),
                        "class_name": str(class_name),
                        "engagement": float(values["engagement"]),
                        "action_ready_rate": float(values["action_ready_rate"]),
                        "dominant_emotion": str(result.social_state["dominant_emotion"]),
                    },
                )

    return settings, pd.DataFrame(rows), class_order


def test_same_event_produces_distinct_subgroup_response_profiles(tmp_path):
    def run_segmentation_analysis():
        seed = torch.initial_seed()
        settings, profiles, _ = _class_stress_profiles(
            tmp_path,
            output_name=f"segmentation_{seed}",
        )

        grouped = profiles.groupby(["dimension", "magnitude"], observed=False)
        gap_table = grouped.agg(
            engagement_gap=(
                "engagement",
                lambda series: float(series.max() - series.min()),
            ),
            action_gap=(
                "action_ready_rate",
                lambda series: float(series.max() - series.min()),
            ),
        ).reset_index()

        zero_magnitude = min(settings["magnitudes"])
        max_magnitude = max(settings["magnitudes"])

        zero_gap = gap_table[gap_table["magnitude"] == zero_magnitude].set_index(
            "dimension",
        )
        max_gap = gap_table[gap_table["magnitude"] == max_magnitude].set_index(
            "dimension",
        )

        return {
            "max_engagement_gaps": max_gap["engagement_gap"].to_list(),
            "zero_engagement_gaps": zero_gap["engagement_gap"].to_list(),
            "max_action_gaps": max_gap["action_gap"].to_list(),
            "zero_action_gaps": zero_gap["action_gap"].to_list(),
        }

    results = run_monte_carlo(run_segmentation_analysis)

    # Aggregate across Monte Carlo runs
    max_eng = [np.mean(r["max_engagement_gaps"]) for r in results]
    zero_eng = [np.mean(r["zero_engagement_gaps"]) for r in results]
    max_act = [np.mean(r["max_action_gaps"]) for r in results]
    zero_act = [np.mean(r["zero_action_gaps"]) for r in results]

    assert_statistically_greater(max_eng, zero_eng)
    assert_statistically_greater(max_act, zero_act)


def test_generate_population_segmentation_figure(tmp_path):
    output_dir = (
        Path(__file__).resolve().parent / "generated" / "population_segmentation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    settings, profiles, class_order = _class_stress_profiles(tmp_path)
    panel_paths = []
    magnitudes = np.sort(profiles["magnitude"].unique())
    x_min = float(magnitudes.min())
    x_max = float(magnitudes.max())
    y_max = float(profiles["engagement"].max())
    y_upper = y_max * 1.05 if y_max > 0.0 else 1.0
    y_ticks = np.linspace(0.0, y_upper, 5)

    for dimension_name in DIMENSIONS:
        fig, ax = setup_plot(
            title=f"Class Engagement: {dimension_name}",
            xlabel="Dimension Magnitude",
            ylabel="Mean Engagement",
        )
        dimension_rows = profiles[profiles["dimension"] == dimension_name]

        for color, class_name in zip(CATEGORICAL_COLORS, class_order, strict=False):
            class_rows = dimension_rows[
                dimension_rows["class_name"] == class_name
            ].sort_values("magnitude")
            ax.plot(
                class_rows["magnitude"].to_numpy(),
                class_rows["engagement"].to_numpy(),
                marker="o",
                label=class_name,
                color=color,
            )

        ax.set_xlim(x_min - 0.05, x_max + 0.05)
        ax.set_xticks(magnitudes)
        ax.set_ylim(0.0, y_upper)
        ax.set_yticks(y_ticks)

        safe_name = dimension_name.lower().replace(" ", "_")
        path = output_dir / f"segmentation_{safe_name}.png"
        save_paper_figure(fig, path)
        plt.close(fig)
        panel_paths.append(path)
        assert path.exists()
        assert path.stat().st_size > 0

    compose_panel_grid(
        panel_paths,
        output_dir.parent / "population_segmentation.png",
        title="Population Segmentation",
        columns=4,
        legend_labels=list(class_order),
        legend_colors=CATEGORICAL_COLORS[: len(class_order)],
    )
