from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from main import run_debug_simulation
from research_paper_tests.config_schema import (
    build_world,
    get_test_scenario,
    prepare_scenario_society,
)
from schema import DIMENSIONS

matplotlib.use("Agg")


def _reconstruct_action_ready_mask(config, society, result) -> np.ndarray:
    """
    Mirror the non-topology action gating in physics_engine.aggregate_society().

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


def _class_stress_profiles(tmp_path: Path):
    scenario = get_test_scenario("population_segmentation")
    config = scenario.sim_config()
    settings = scenario.settings()
    society = prepare_scenario_society(
        "population_segmentation",
        tmp_path,
        enable_evolution=config.enable_evolution,
        output_name="population_segmentation",
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
                    }
                )

    return settings, pd.DataFrame(rows), class_order


def test_same_event_produces_distinct_subgroup_response_profiles(tmp_path):
    settings, profiles, _ = _class_stress_profiles(tmp_path)

    grouped = profiles.groupby(["dimension", "magnitude"], observed=False)
    gap_table = grouped.agg(
        engagement_gap=("engagement", lambda series: float(series.max() - series.min())),
        action_gap=(
            "action_ready_rate",
            lambda series: float(series.max() - series.min()),
        ),
    ).reset_index()

    peak_gap_by_dimension = gap_table.groupby("dimension", observed=False).max(
        numeric_only=True
    )

    assert (
        int(
            (
                peak_gap_by_dimension["engagement_gap"]
                >= settings["engagement_gap_floor"]
            ).sum()
        )
        >= settings["min_dimensions_with_engagement_separation"]
    )
    assert (
        int(
            (peak_gap_by_dimension["action_gap"] >= settings["action_gap_floor"]).sum()
        )
        >= settings["min_dimensions_with_action_separation"]
    )


def test_generate_population_segmentation_figure(tmp_path):
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "population_segmentation.png"

    settings, profiles, class_order = _class_stress_profiles(tmp_path)

    fig, axes = plt.subplots(3, 4, figsize=(22, 16), sharex=True, sharey=True)
    palette = ["#d62828", "#f77f00", "#457b9d", "#2a9d8f", "#6d597a"]

    for axis, dimension_name in zip(axes.flatten(), DIMENSIONS, strict=True):
        dimension_rows = profiles[profiles["dimension"] == dimension_name]

        # x-axis: magnitude of one world dimension with every other dimension fixed at 0.
        # y-axis: mean engagement for one social class. Higher curves mean that class
        # is more reactive to that isolated dimension as its intensity increases.
        # Comparing the five curves shows whether the dimension amplifies class
        # differences or keeps the society relatively uniform.
        for color, class_name in zip(palette, class_order, strict=False):
            class_rows = (
                dimension_rows[dimension_rows["class_name"] == class_name]
                .sort_values("magnitude")
            )
            axis.plot(
                class_rows["magnitude"].to_numpy(),
                class_rows["engagement"].to_numpy(),
                marker="o",
                linewidth=2,
                label=class_name,
                color=color,
            )

        axis.set_title(dimension_name)
        axis.grid(True, alpha=0.2)

    for axis in axes[-1]:
        axis.set_xlabel("Dimension Magnitude")
    for axis in axes[:, 0]:
        axis.set_ylabel("Mean Engagement")

    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        (
            "Population Segmentation Stress Test\n"
            f"Per-dimension sweep across magnitudes {settings['magnitudes']} for the five social classes"
        ),
        fontsize=18,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
