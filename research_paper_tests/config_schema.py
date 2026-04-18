from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from schema import (
    DIMENSION_INDICES,
    DIMENSIONS,
    EMOTION_LABELS,
    RUN_PROFILE_FIELDS,
    SIM_CONFIG_DEFAULTS,
    SIM_CONFIG_FIELDS,
    SimConfig,
)

PERSONALITY_LABELS = (
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
)
SENTIMENT_LABELS = ("Negative", "Neutral", "Positive")
PERSONALITY_INDICES = {
    label: index for index, label in enumerate(PERSONALITY_LABELS)
}
EMOTION_INDICES = {label: index for index, label in enumerate(EMOTION_LABELS)}
SENTIMENT_INDICES = {label: index for index, label in enumerate(SENTIMENT_LABELS)}

WORLD_DIMENSION_COUNT = len(DIMENSIONS)
EMOTION_DIMENSION_COUNT = len(EMOTION_LABELS)
PERSONALITY_TRAIT_COUNT = len(PERSONALITY_LABELS)


def _merge(*mappings: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for mapping in mappings:
        merged.update(mapping)
    return merged


def live_sim_config_defaults() -> dict[str, Any]:
    return deepcopy(SIM_CONFIG_DEFAULTS)


DEFAULT_SMOKE_NUM_AGENTS = 128
DEFAULT_SMOKE_EVOLUTION_GENERATIONS = 2
SESSION_EVOLUTION_MODE_ENV = "RESEARCH_TEST_EVOLUTION_MODE"
EVOLUTION_MATRIX_MODE_ENV = "RESEARCH_TEST_EVOLUTION_MATRIX"
EVOLUTION_VARIANT_LABELS = {
    False: "without_evolution",
    True: "with_evolution",
}


def live_run_profile_defaults() -> dict[str, Any]:
    from main import RunProfile

    return RunProfile().model_dump()


def apply_config_attrs(
    config: SimConfig, extra_attrs: Mapping[str, Any] | None = None,
) -> SimConfig:
    for attr_name, attr_value in (extra_attrs or {}).items():
        setattr(config, attr_name, deepcopy(attr_value))
    return config


def fraction_count(total: int, fraction: float) -> int:
    return int(total * fraction)


def zero_world(rows: int = 1) -> torch.Tensor:
    return torch.zeros(rows, WORLD_DIMENSION_COUNT, dtype=torch.float32)


def zero_emotions(rows: int) -> torch.Tensor:
    return torch.zeros(rows, EMOTION_DIMENSION_COUNT, dtype=torch.float32)


def zero_personalities(rows: int, fill: float = 0.0) -> torch.Tensor:
    return torch.full(
        (rows, PERSONALITY_TRAIT_COUNT),
        float(fill),
        dtype=torch.float32,
    )


def set_dimensions(
    tensor: torch.Tensor,
    values: Mapping[str, float],
    rows: slice | int | None = None,
) -> torch.Tensor:
    target_rows: slice | int = slice(None) if rows is None else rows
    for dimension_name, dimension_value in values.items():
        tensor[target_rows, DIMENSION_INDICES[dimension_name]] = dimension_value
    return tensor


def set_emotions(
    tensor: torch.Tensor,
    values: Mapping[str, float],
    rows: slice | int | None = None,
) -> torch.Tensor:
    target_rows: slice | int = slice(None) if rows is None else rows
    for emotion_name, emotion_value in values.items():
        tensor[target_rows, EMOTION_INDICES[emotion_name]] = emotion_value
    return tensor


def set_traits(
    tensor: torch.Tensor,
    values: Mapping[str, float],
    rows: slice | int | None = None,
) -> torch.Tensor:
    target_rows: slice | int = slice(None) if rows is None else rows
    for trait_name, trait_value in values.items():
        tensor[target_rows, PERSONALITY_INDICES[trait_name]] = trait_value
    return tensor


def build_world(values: Mapping[str, float]) -> torch.Tensor:
    return set_dimensions(zero_world(), values)


def requested_evolution_override() -> bool | None:
    raw_mode = os.getenv(SESSION_EVOLUTION_MODE_ENV, "").strip()
    if not raw_mode:
        raw_mode = os.getenv(EVOLUTION_MATRIX_MODE_ENV, "").strip()
    if not raw_mode:
        return None

    normalized = raw_mode.lower()
    if normalized == "both":
        return None
    if normalized in {"with", "on", "true"}:
        return True
    if normalized in {"without", "off", "false"}:
        return False
    raise ValueError(
        "evolution mode must be one of: both, with, without, on, off, true, false",
    )


@dataclass(frozen=True)
class ResearchPaperTestScenario:
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    run_profile_overrides: Mapping[str, Any] = field(default_factory=dict)
    extra_config_attrs: Mapping[str, Any] = field(default_factory=dict)
    values: Mapping[str, Any] = field(default_factory=dict)
    allow_session_evolution_override: bool = True

    def _resolved_config_overrides(
        self,
        *,
        enable_evolution: bool | None = None,
        smoke: bool = False,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved = _merge(self.config_overrides, overrides or {})
        if smoke:
            resolved["num_agents"] = min(
                int(resolved.get("num_agents", SIM_CONFIG_DEFAULTS["num_agents"])),
                DEFAULT_SMOKE_NUM_AGENTS,
            )
            resolved["evolution_generations"] = min(
                int(
                    resolved.get(
                        "evolution_generations",
                        SIM_CONFIG_DEFAULTS["evolution_generations"],
                    ),
                ),
                DEFAULT_SMOKE_EVOLUTION_GENERATIONS,
            )
        selected_evolution = enable_evolution
        if selected_evolution is None and self.allow_session_evolution_override:
            selected_evolution = requested_evolution_override()
        if selected_evolution is not None:
            resolved["enable_evolution"] = selected_evolution
        return resolved

    def sim_config(
        self,
        *,
        enable_evolution: bool | None = None,
        smoke: bool = False,
        **overrides: Any,
    ) -> SimConfig:
        from main import create_sim_config

        config = create_sim_config(
            **self._resolved_config_overrides(
                enable_evolution=enable_evolution,
                smoke=smoke,
                overrides=overrides,
            ),
        )
        return apply_config_attrs(config, self.extra_config_attrs)

    def run_profile(
        self,
        *,
        enable_evolution: bool | None = None,
        smoke: bool = False,
        **overrides: Any,
    ):
        from main import RunProfile

        profile_kwargs = live_run_profile_defaults()
        profile_kwargs.update(deepcopy(dict(self.run_profile_overrides)))
        if smoke:
            profile_kwargs["agent_count"] = min(
                int(profile_kwargs.get("agent_count", SIM_CONFIG_DEFAULTS["num_agents"])),
                DEFAULT_SMOKE_NUM_AGENTS,
            )
            profile_kwargs["evolution_generations"] = min(
                int(
                    profile_kwargs.get(
                        "evolution_generations",
                        SIM_CONFIG_DEFAULTS["evolution_generations"],
                    ),
                ),
                DEFAULT_SMOKE_EVOLUTION_GENERATIONS,
            )
        selected_evolution = enable_evolution
        if selected_evolution is None and self.allow_session_evolution_override:
            selected_evolution = requested_evolution_override()
        if selected_evolution is not None:
            profile_kwargs["enable_evolution"] = selected_evolution
        profile_kwargs.update(overrides)
        return RunProfile(**profile_kwargs)

    def settings(self) -> dict[str, Any]:
        return deepcopy(dict(self.values))


NO_NETWORK_NO_EVOLUTION = {
    "use_network_topology": False,
    "enable_evolution": False,
}
NETWORK_NO_EVOLUTION = {
    "use_network_topology": True,
    "enable_evolution": False,
}
CALM_NO_NETWORK = _merge(
    NO_NETWORK_NO_EVOLUTION,
    {"use_signal_distortion": False, "use_time_pressure": False},
)


TEST_SCENARIOS: dict[str, ResearchPaperTestScenario] = {
    "accuracy_metrics": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 256, "use_signal_distortion": False},
        ),
        values={
            "world": {"Physical_Safety": -0.8},
            "urgency": 0.5,
            "matching_baseline": [0.9, 0.05, 0.05],
            "mismatched_baseline": [0.05, 0.05, 0.9],
        },
    ),
    "agent_memory": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 300,
                "use_agent_memory": True,
                "memory_desensitization_gain": 5.0,
                "memory_trigger_stacking_gain": 15.0,
            },
        ),
        values={
            "repeat_count": 10,
            "repeat_threat": {"Wealth": -0.8},
            "new_threat": {"Physical_Safety": -0.2},
            "urgency": 0.5,
        },
    ),
    "algorithmic_filter_bubble": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 500,
                "use_algorithmic_amplification": True,
                "algo_sample_size": 0.1,
                "algo_exaggeration_factor": 2.0,
            },
        ),
        values={
            "world": {"Innovation": 0.2, "Freedom": -0.1},
            "urgency": 0.5,
        },
    ),
    "bimodality_polarization": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 1500},
        ),
        values={
            "rng_seed": SIM_CONFIG_DEFAULTS["seed"],
            "polarized_mean_negative": -0.8,
            "polarized_mean_positive": 0.8,
            "polarized_std": 0.1,
            "polarized_count_per_mode": 750,
            "normal_mean": 0.0,
            "normal_std": 1.0,
            "min_polarized_bc": 0.555,
        },
    ),
    "cascade_power_law_flat": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 2000, "use_power_law_influence": False},
        ),
    ),
    "cascade_power_law_power": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 2000, "use_power_law_influence": True},
        ),
    ),
    "clusters": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 1200},
        ),
        values={
            "cluster_count": 8,
            "cluster_seed": SIM_CONFIG_DEFAULTS["seed"],
            "cluster_initializations": 10,
        },
    ),
    "cognitive_gate": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 400, "use_signal_distortion": False},
        ),
        extra_config_attrs={
            "use_selective_exposure": True,
            "selective_exposure_base_tolerance": -0.3,
            "selective_exposure_openness_factor": 0.4,
        },
        values={
            "trait_fill": 0.5,
            "low_openness": 0.1,
            "high_openness": 0.9,
            "aligned_worldview": {
                "Innovation": 1.0,
                "Fairness": 1.0,
                "Sanctity": -1.0,
                "In_Group": -1.0,
            },
            "world": {
                "Innovation": 0.8,
                "Fairness": 0.7,
                "Sanctity": -0.9,
                "In_Group": -0.5,
            },
            "urgency": 0.2,
        },
    ),
    "divergence": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 2, "use_signal_distortion": False},
        ),
        values={
            "personalities": [
                [0.5, 0.5, 0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5, 0.5, 0.95],
            ],
            "world": {"Physical_Safety": -0.4},
            "urgency": 0.5,
        },
    ),
    "boundary_dose_response": ResearchPaperTestScenario(
        config_overrides=_merge(
            CALM_NO_NETWORK,
            {"num_agents": 256},
        ),
        values={
            "world_direction": {
                "Physical_Safety": -1.0,
                "Fairness": -0.7,
            },
            "magnitudes": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
            "urgency": 0.2,
            "monotonic_tolerance": 1e-6,
        },
    ),
    "boundary_low_salience": ResearchPaperTestScenario(
        config_overrides=_merge(
            CALM_NO_NETWORK,
            {"num_agents": 256},
        ),
        values={
            "worlds": {
                "Zero": {},
                "Faint Threat": {"Physical_Safety": -0.03},
                "Mixed Weak": {
                    "Wealth": 0.03,
                    "Physical_Safety": -0.03,
                    "Innovation": 0.03,
                    "Fairness": -0.03,
                },
                "Salient Threat": {
                    "Physical_Safety": -0.6,
                    "Fairness": -0.35,
                },
            },
            "low_salience_labels": ["Zero", "Faint Threat", "Mixed Weak"],
            "salient_label": "Salient Threat",
            "urgency": 0.2,
        },
    ),
    "emotion_directionality": ResearchPaperTestScenario(
        config_overrides=_merge(
            CALM_NO_NETWORK,
            {"num_agents": 256},
        ),
        values={
            "worlds": {
                "Prosperity": {
                    "Wealth": 0.8,
                    "Freedom": 0.6,
                    "Innovation": 0.5,
                },
                "Threat": {
                    "Physical_Safety": -0.9,
                    "Stability": -0.7,
                },
                "Injustice": {
                    "Fairness": -0.9,
                    "Care": -0.4,
                },
            },
            "urgency": 0.2,
            "allowed_injustice_emotions": ["Anger", "Disgust"],
        },
    ),
    "bridge_diffusion": ResearchPaperTestScenario(
        config_overrides=_merge(
            CALM_NO_NETWORK,
            {
                "num_agents": 10,
                "use_granovetter_thresholds": True,
                "granovetter_threshold_mean": 0.25,
                "granovetter_threshold_std": 0.0,
                "dominant_emotion_threshold": 0.1,
                "base_action_cost": 0.42,
                "stewing_ticks": 1,
            },
        ),
        values={
            "community_a_size": 4,
            "bridge_size": 2,
            "community_b_size": 4,
            "core_emotion": {"Anger": 0.35},
            "marginal_emotion": {"Anger": 0.18},
        },
    ),
    "viral_scaling": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 128,
                "stewing_ticks": 1,
            },
        ),
        values={
            "amplitudes": [0.0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.36, 0.6, 0.9],
            "emotion_name": "Anger",
            "near_cap_tolerance": 0.05,
        },
    ),
    "trait_sweeps": ResearchPaperTestScenario(
        config_overrides=_merge(
            CALM_NO_NETWORK,
            {
                "num_agents": 9,
                "use_selective_exposure": True,
                "selective_exposure_base_tolerance": -0.3,
                "selective_exposure_openness_factor": 0.4,
                "base_action_cost": 0.55,
            },
        ),
        values={
            "trait_values": [0.05, 0.16, 0.27, 0.39, 0.5, 0.61, 0.73, 0.84, 0.95],
            "baseline_fill": 0.5,
            "openness_world": {
                "Innovation": 0.8,
                "Fairness": 0.7,
                "Sanctity": -0.9,
                "In_Group": -0.5,
            },
            "openness_exposure": {
                "Innovation": -0.8,
                "Fairness": -0.7,
                "Sanctity": 0.9,
                "In_Group": 0.5,
            },
            "threat_world": {
                "Physical_Safety": -0.85,
                "Stability": -0.65,
            },
            "urgency": 0.2,
            "monotonic_tolerance": 1e-6,
        },
    ),
    "population_segmentation": ResearchPaperTestScenario(
        config_overrides=_merge(
            CALM_NO_NETWORK,
            {
                "num_agents": 400,
            },
        ),
        values={
            "urgency": 0.2,
            "magnitudes": [0.0, 0.25, 0.5, 0.75, 1.0],
            "class_order": [
                "Underclass",
                "Working Class",
                "Middle Class",
                "Upper Middle",
                "Elite",
            ],
        },
    ),
    "echo_chambers_high": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 500,
                "homophily_strength": 8.0,
                "influence_bias_exp": 0.1,
                "personality_socialization_gain": 0.0,
            },
        ),
    ),
    "echo_chambers_low": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 500,
                "homophily_strength": 0.1,
                "influence_bias_exp": 1.0,
                "personality_socialization_gain": 0.0,
            },
        ),
    ),
    "endogenous_events": ResearchPaperTestScenario(
        config_overrides={
            "num_agents": 1000,
            "elite_divergence_threshold": 0.3,
            "polarization_threshold": 0.3,
            "stewing_ticks": 1,
        },
        values={
            "acting_population": 10,
            "stable_emotion": {"Joy": 0.5},
            "polarized_group_a": {"Anger": 1.0},
            "polarized_group_b": {"Joy": 1.0},
            "allowed_actions": {
                "Civil Protest",
                "Populist Uprising",
                "Elite Policy Shift",
            },
        },
    ),
    "figure_algorithmic_filter_bubble": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 400,
                "use_algorithmic_amplification": True,
                "algo_sample_size": 0.1,
                "algo_exaggeration_factor": 2.0,
            },
        ),
        values={
            "world": {"Innovation": 0.2, "Freedom": -0.1},
            "urgency": 0.5,
        },
    ),
    "figure_cognitive_gate": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 400, "use_signal_distortion": False},
        ),
        extra_config_attrs={
            "use_selective_exposure": True,
            "selective_exposure_base_tolerance": -0.3,
            "selective_exposure_openness_factor": 0.4,
        },
        values={
            "trait_fill": 0.5,
            "aligned_worldview": {
                "Innovation": 1.0,
                "Fairness": 1.0,
                "Sanctity": -1.0,
                "In_Group": -1.0,
            },
            "world": {
                "Innovation": 0.8,
                "Fairness": 0.7,
                "Sanctity": -0.9,
                "In_Group": -0.5,
            },
            "urgency": 0.2,
            "openness_start": 0.02,
            "openness_end": 0.98,
            "worldview_min_scale": 0.85,
            "worldview_max_scale": 1.15,
        },
    ),
    "figure_echo_chambers_high": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 400,
                "homophily_strength": 8.0,
                "influence_bias_exp": 0.0,
                "base_connections": 2,
                "triadic_closure_prob": 0.8,
            },
        ),
        values={"partition_seed": SIM_CONFIG_DEFAULTS["seed"]},
    ),
    "figure_echo_chambers_low": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {"num_agents": 400, "homophily_strength": 1.0},
        ),
        values={"partition_seed": SIM_CONFIG_DEFAULTS["seed"]},
    ),
    "figure_granovetter_cascade": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 400,
                "use_granovetter_thresholds": True,
                "granovetter_threshold_mean": 0.2,
                "dominant_emotion_threshold": 0.1,
            },
        ),
        values={
            "instigator_share": 0.05,
            "sympathizer_share": 0.4,
            "instigator_emotion": {"Anger": 0.8},
            "sympathizer_emotion": {"Anger": 0.2},
        },
    ),
    "figure_memory_rehearsal": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 80,
                "use_agent_memory": True,
                "memory_decay_rate": 0.5,
                "memory_social_rehearsal_gain": 0.8,
            },
        ),
        values={
            "context": {"Physical_Safety": -1.0},
            "decay_steps": 5,
            "isolated_rehearsal": 0.0,
            "shared_rehearsal": 1.0,
        },
    ),
    "figure_personality_correlations": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 1200, "mutation_temperature": 0.0},
        ),
    ),
    "figure_semantic_alignment": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 256, "use_signal_distortion": False},
        ),
        values={
            "positive_world": {"Wealth": 0.8, "Innovation": 0.6},
            "negative_world": {"Physical_Safety": -0.8, "Fairness": -0.6},
            "urgency": 0.5,
            "positive_sentiment_profile": [0.05, 0.1, 0.85],
            "negative_sentiment_profile": [0.85, 0.1, 0.05],
        },
    ),
    "figure_signal_distortion": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 400,
                "use_signal_distortion": True,
                "distortion_max_noise": 0.8,
                "distortion_neurotic_gain": 1.5,
            },
        ),
        values={
            "world": {"Physical_Safety": -0.4},
        },
    ),
    "figure_social_consensus": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 400,
                "use_signal_distortion": True,
                "distortion_max_noise": 0.6,
                "distortion_neurotic_gain": 1.0,
                "perception_social_consensus_gain": 0.3,
            },
        ),
        values={"world": {"Physical_Safety": -0.5}},
    ),
    "figure_wealth_baseline": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 800},
        ),
        allow_session_evolution_override=False,
    ),
    "figure_wealth_evolved": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 800, "evolution_generations": 20, "enable_evolution": True},
        ),
        allow_session_evolution_override=False,
    ),
    "granovetter_cascade": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 500,
                "use_granovetter_thresholds": True,
                "granovetter_threshold_mean": 0.2,
                "dominant_emotion_threshold": 0.1,
            },
        ),
        values={
            "instigator_share": 0.05,
            "sympathizer_share": 0.4,
            "instigator_emotion": {"Anger": 0.8},
            "sympathizer_emotion": {"Anger": 0.2},
        },
    ),
    "ideological_influence_power": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 3000, "use_power_law_influence": True},
        ),
    ),
    "ideological_influence_standard": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 3000, "use_power_law_influence": False},
        ),
    ),
    "influence_susceptibility": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 800, "use_power_law_influence": True},
        ),
        values={
            "rng_seed": SIM_CONFIG_DEFAULTS["seed"],
            "sample_size": 120,
            "urgency": 0.5,
            "reach_probability_base": 0.10,
            "reach_probability_gain": 0.10,
            "engagement_threshold": 0.18,
            "reach_top_percentile": 75,
        },
    ),
    "louvain_high": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 500,
                "homophily_strength": 8.0,
                "influence_bias_exp": 0.0,
                "base_connections": 2,
                "triadic_closure_prob": 0.8,
            },
        ),
        values={"partition_seed": SIM_CONFIG_DEFAULTS["seed"]},
    ),
    "louvain_low": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {"num_agents": 500, "homophily_strength": 1.0},
        ),
        values={"partition_seed": SIM_CONFIG_DEFAULTS["seed"]},
    ),
    "maximum_virality": ResearchPaperTestScenario(
        config_overrides={"num_agents": 1000},
        values={
            "consensus_emotion": {"Joy": 1.0},
            "outlier_mainstream_count": 950,
            "outlier_mainstream_emotion": {"Joy": 0.2},
            "outlier_count": 50,
            "outlier_emotions": {"Anger": 1.0, "Disgust": 1.0},
            "boosted_engagement": 2.0,
            "max_multiplier_tolerance": 1e-3,
        },
    ),
    "memory_rehearsal": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 100,
                "use_agent_memory": True,
                "memory_decay_rate": 0.5,
                "memory_social_rehearsal_gain": 0.8,
            },
        ),
        values={
            "context": {"Physical_Safety": -1.0},
            "decay_steps": 5,
            "isolated_rehearsal": 0.0,
            "shared_rehearsal": 1.0,
        },
    ),
    "network_clustering_backbone": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 400,
                "base_connections": 10,
                "triadic_closure_prob": 0.0,
            },
        ),
    ),
    "network_clustering_closure": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 400,
                "base_connections": 10,
                "triadic_closure_prob": 0.3,
            },
        ),
        values={
            "torch_seed": SIM_CONFIG_DEFAULTS["seed"],
            "numpy_seed": SIM_CONFIG_DEFAULTS["seed"],
            "influence_mean": 1.0,
            "influence_std": 0.5,
        },
    ),
    "network_topology": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {"num_agents": 400, "base_connections": 20, "homophily_strength": 3.0},
        ),
        values={"row_sum_tolerance": 1e-5},
    ),
    "perception_social_consensus": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 500,
                "use_signal_distortion": True,
                "distortion_max_noise": 0.6,
                "distortion_neurotic_gain": 1.0,
                "perception_social_consensus_gain": 0.3,
            },
        ),
        values={"world": {"Physical_Safety": -0.5}},
    ),
    "perception_social_consensus_baseline": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {
                "num_agents": 500,
                "use_signal_distortion": True,
                "distortion_max_noise": 0.6,
                "distortion_neurotic_gain": 1.0,
                "perception_social_consensus_gain": 0.0,
            },
        ),
    ),
    "personal": ResearchPaperTestScenario(
        config_overrides=_merge(CALM_NO_NETWORK, {"num_agents": 200}),
        values={"world": {"Care": -0.8}, "urgency": 0.2},
    ),
    "personalities_for_clustering": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 3000},
        ),
        values={
            "high_threshold": 0.8,
            "low_threshold": 0.2,
        },
    ),
    "personality_correlations": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 4000, "mutation_temperature": 0.0},
        ),
    ),
    "personality_socialization_base": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {"num_agents": 500, "personality_socialization_gain": 0.0},
        ),
    ),
    "personality_socialization_socialized": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {"num_agents": 500, "personality_socialization_gain": 0.4},
        ),
    ),
    "r0_basic_reproduction": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 400},
        ),
        values={
            "rng_seed": SIM_CONFIG_DEFAULTS["seed"],
            "seed_sample_count": 20,
            "urgency": 0.5,
        },
    ),
    "ram_usage": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {"num_agents": 300, "use_agent_memory": True},
        ),
        values={
            "world": {"Wealth": -0.6, "Fairness": -0.4},
            "urgency": 0.5,
            "max_memory_bytes": 256 * 1024 * 1024,
        },
    ),
    "relative_deprivation": ResearchPaperTestScenario(
        config_overrides=_merge(
            CALM_NO_NETWORK,
            {"num_agents": 200},
        ),
        values={
            "group_size": 100,
            "trait_fill": 0.5,
            "marginalized_exposures": {"Wealth": -0.8, "Fairness": -0.8},
            "marginalized_traits": {"Agreeableness": 0.1, "Neuroticism": 0.9},
            "elite_exposures": {"Wealth": 0.8, "Fairness": 0.8},
            "elite_traits": {"Agreeableness": 0.9, "Neuroticism": 0.1},
            "world": {"Wealth": 0.5, "Fairness": -1.0},
            "urgency": 0.0,
        },
    ),
    "runtime_acting_ratio": ResearchPaperTestScenario(
        config_overrides={
            "num_agents": 1000,
            "use_granovetter_thresholds": False,
            "dominant_emotion_threshold": 0.1,
        },
        values={
            "acting_population": 10,
            "acting_emotion": {"Anger": 1.0},
            "expected_acting_ratio": 1.0,
        },
    ),
    "runtime_profile_evolution": ResearchPaperTestScenario(
        run_profile_overrides={
            "seed": 7,
            "temperature": 0.4,
            "agent_count": 96,
            "use_power_law": False,
            "use_network_topology": False,
            "enable_evolution": True,
            "evolution_generations": 2,
        },
        values={"baseline_enable_evolution": False},
        allow_session_evolution_override=False,
    ),
    "runtime_profile_memory": ResearchPaperTestScenario(
        run_profile_overrides={
            "seed": 99,
            "temperature": 0.1,
            "agent_count": 64,
            "use_power_law": False,
            "use_network_topology": False,
            "enable_evolution": False,
            "use_agent_memory": True,
        },
        values={"memory_marker": 123.0, "empty_memory_value": 0.0},
        allow_session_evolution_override=False,
    ),
    "runtime_profile_topology": ResearchPaperTestScenario(
        run_profile_overrides={
            "seed": SIM_CONFIG_DEFAULTS["seed"],
            "temperature": SIM_CONFIG_DEFAULTS["mutation_temperature"],
            "agent_count": 128,
            "use_power_law": False,
            "use_network_topology": True,
            "enable_evolution": False,
        },
        values={"flat_use_network_topology": False},
        allow_session_evolution_override=False,
    ),
    "runtime_small_populations": ResearchPaperTestScenario(
        config_overrides=_merge(
            NETWORK_NO_EVOLUTION,
            {"use_network_topology": True},
        ),
        values={
            "population_sizes": [10, 50, 200],
            "row_sum_tolerance": 1e-5,
        },
    ),
    "semantic_alignment": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 256, "use_signal_distortion": False},
        ),
        values={
            "positive_world": {"Wealth": 0.8, "Innovation": 0.6},
            "negative_world": {"Physical_Safety": -0.8, "Fairness": -0.6},
            "urgency": 0.5,
            "positive_sentiment_profile": [0.05, 0.1, 0.85],
            "negative_sentiment_profile": [0.85, 0.1, 0.05],
        },
    ),
    "signal_distortion": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 500,
                "use_signal_distortion": True,
                "distortion_max_noise": 0.8,
                "distortion_neurotic_gain": 1.5,
            },
        ),
        values={
            "world": {"Physical_Safety": -0.4},
        },
    ),
    "temp_wealth_analysis": ResearchPaperTestScenario(
        config_overrides={"num_agents": 5000},
        values={
            "wealth_threshold": 10000,
            "scatter_alpha": 0.5,
            "hist_bins": 50,
            "x_axis_percentile": 99.5,
            "output_template": "network_synergy_verification_seed_{seed}.png",
        },
    ),
    "trait_distribution": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 1200},
        ),
    ),
    "truth_refinement": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {
                "num_agents": 2,
                "use_signal_distortion": False,
                "use_time_pressure": False,
            },
        ),
        values={
            "skepticism_gain": 4.0,
            "logic_gap_threshold": 0.4,
            "world": {
                "Wealth": 0.8,
                "Stability": -0.9,
                "Short_Term": 0.9,
                "Long_Term": -0.9,
            },
            "personalities": [
                [0.1, 0.1, 0.5, 0.5, 0.5],
                [0.9, 0.9, 0.5, 0.5, 0.5],
            ],
        },
    ),
    "wealth_gini_baseline": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 1500, "evolution_generations": 20, "enable_evolution": False},
        ),
        allow_session_evolution_override=False,
    ),
    "wealth_gini_evolved": ResearchPaperTestScenario(
        config_overrides=_merge(
            NO_NETWORK_NO_EVOLUTION,
            {"num_agents": 1500, "evolution_generations": 20, "enable_evolution": True},
        ),
        allow_session_evolution_override=False,
    ),
}

SOCIETY_EVOLUTION_CASES: tuple[str, ...] = (
    "accuracy_metrics",
    "agent_memory",
    "algorithmic_filter_bubble",
    "bimodality_polarization",
    "cascade_power_law_flat",
    "cascade_power_law_power",
    "clusters",
    "echo_chambers_high",
    "echo_chambers_low",
    "figure_algorithmic_filter_bubble",
    "figure_echo_chambers_high",
    "figure_echo_chambers_low",
    "figure_granovetter_cascade",
    "figure_personality_correlations",
    "figure_semantic_alignment",
    "figure_signal_distortion",
    "figure_social_consensus",
    "figure_wealth_baseline",
    "figure_wealth_evolved",
    "granovetter_cascade",
    "ideological_influence_power",
    "ideological_influence_standard",
    "influence_susceptibility",
    "louvain_high",
    "louvain_low",
    "network_topology",
    "perception_social_consensus",
    "personal",
    "personalities_for_clustering",
    "personality_correlations",
    "personality_socialization_base",
    "personality_socialization_socialized",
    "r0_basic_reproduction",
    "ram_usage",
    "runtime_small_populations",
    "semantic_alignment",
    "signal_distortion",
    "trait_distribution",
    "wealth_gini_evolved",
)


def get_test_scenario(name: str) -> ResearchPaperTestScenario:
    return TEST_SCENARIOS[name]


def evolution_variants(mode: str = "both") -> tuple[bool, ...]:
    normalized = mode.strip().lower()
    if normalized == "both":
        return (False, True)
    if normalized in {"without", "off", "false"}:
        return (False,)
    if normalized in {"with", "on", "true"}:
        return (True,)
    raise ValueError(
        "evolution mode must be one of: both, with, without, on, off, true, false",
    )


def prepare_scenario_society(
    scenario_name: str,
    root_dir: str | Path,
    *,
    enable_evolution: bool,
    smoke: bool = False,
    output_name: str | None = None,
    **config_overrides: Any,
):
    from main import prepare_society_for_debug

    scenario = get_test_scenario(scenario_name)
    config = scenario.sim_config(
        enable_evolution=enable_evolution,
        smoke=smoke,
        **config_overrides,
    )
    output_root = Path(root_dir)
    case_name = output_name or scenario_name
    variant_label = EVOLUTION_VARIANT_LABELS[enable_evolution]
    output_dir = output_root / f"{case_name}_{variant_label}"
    return prepare_society_for_debug(
        config,
        output_dir=str(output_dir),
        evolve=enable_evolution,
    )


def _validate_schema() -> None:
    invalid_overrides: dict[str, list[str]] = {}
    invalid_run_profile_overrides: dict[str, list[str]] = {}
    for scenario_name, scenario in TEST_SCENARIOS.items():
        unknown_fields = sorted(set(scenario.config_overrides) - SIM_CONFIG_FIELDS)
        if unknown_fields:
            invalid_overrides[scenario_name] = unknown_fields
        unknown_run_fields = sorted(
            set(scenario.run_profile_overrides) - RUN_PROFILE_FIELDS,
        )
        if unknown_run_fields:
            invalid_run_profile_overrides[scenario_name] = unknown_run_fields
    if invalid_overrides:
        raise ValueError(f"Invalid SimConfig overrides: {invalid_overrides}")
    if invalid_run_profile_overrides:
        raise ValueError(
            f"Invalid RunProfile overrides: {invalid_run_profile_overrides}",
        )
    missing_society_cases = sorted(
        scenario_name
        for scenario_name in SOCIETY_EVOLUTION_CASES
        if scenario_name not in TEST_SCENARIOS
    )
    if missing_society_cases:
        raise ValueError(
            f"Unknown society evolution cases declared: {missing_society_cases}",
        )


_validate_schema()
