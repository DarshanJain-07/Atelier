import asyncio
import hashlib
import html
import json
import os
import re
import shutil
import time
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, create_model

from cognitive_engine import CognitiveEngine
from docs_router import router as docs_router
from explainability import ExplainabilityEngine
from generate_society import (
    apply_triadic_closure,
    create_topology,
    finalize_social_structure,
    generate_society,
)
from input_layer import get_world_state
from physics_engine import SocialPhysicsEngine

# Import our Core Logic
from schema import (
    DIMENSION_INDICES,
    EMOTION_LABELS,
    RUN_PROFILE_INTERNAL_ONLY_FIELDS,
    RUN_PROFILE_TO_SIM_CONFIG_FIELD_MAP,
    SIM_CONFIG_FIELDS,
    SimConfig,
    emotions_to_behavior_aware_sentiment_distribution,
    sim_config_default,
)
from society_evolution import SocietyEvolution
from validation import Validator

try:
    import markdown as markdown_lib
except ImportError:
    markdown_lib = None

load_dotenv()

# Check API Key
if not os.getenv("GEMINI_API_KEY"):
    print("❌ ERROR: GEMINI_API_KEY not set. Simulation will fail.")

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DOCS_HTML_PATH = FRONTEND_DIR / "docs.html"
PRIMARY_DOC_PATHS = (
    ("index", PROJECT_ROOT / "docs" / "index.md"),
    ("readme", PROJECT_ROOT / "README.md"),
    ("development", PROJECT_ROOT / "docs" / "development.md"),
    ("api-reference", PROJECT_ROOT / "docs" / "api-reference.md"),
    ("testing", PROJECT_ROOT / "docs" / "testing.md"),
)

app = FastAPI(
    docs_url="/api/docs",
)
validator = Validator()

# Enable CORS - Restrict this in production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("✅ Server Ready.")

app.include_router(docs_router)

# Persistent LRU cache for societies (seed + count + temp -> data)
MAX_CACHE_SIZE = 7
SOCIETY_CACHE: OrderedDict[str, Any] = OrderedDict()
SOCIETY_CACHE_LOCK = Lock()

def _extract_doc_title(markdown_text: str, fallback: str) -> str:
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def _collect_docs_registry() -> tuple[list[dict[str, str]], dict[str, Path], dict[Path, str]]:
    pages: list[dict[str, str]] = []
    slug_to_path: dict[str, Path] = {}
    path_to_slug: dict[Path, str] = {}
    seen_paths: set[Path] = set()

    def register(slug: str, path: Path) -> None:
        resolved = path.resolve()
        if not path.exists() or slug in slug_to_path or resolved in seen_paths:
            return

        markdown_text = path.read_text(encoding="utf-8")
        slug_to_path[slug] = path
        path_to_slug[resolved] = slug
        seen_paths.add(resolved)
        pages.append(
            {
                "slug": slug,
                "title": _extract_doc_title(
                    markdown_text,
                    slug.replace("-", " ").title(),
                ),
                "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
            },
        )

    for slug, path in PRIMARY_DOC_PATHS:
        register(slug, path)

    for path in sorted((PROJECT_ROOT / "docs").glob("*.md")):
        register(path.stem, path)

    return pages, slug_to_path, path_to_slug


_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _rewrite_markdown_links(
    markdown_text: str,
    source_path: Path,
    path_to_slug: dict[Path, str],
) -> str:
    def replace_link(match: re.Match[str]) -> str:
        label, target = match.groups()

        if (
            "://" in target
            or target.startswith("#")
            or target.startswith("mailto:")
        ):
            return match.group(0)

        path_part, sep, anchor = target.partition("#")
        if not path_part.endswith(".md"):
            return match.group(0)

        resolved_target = (source_path.parent / path_part).resolve()
        slug = path_to_slug.get(resolved_target)
        if slug is None:
            return match.group(0)

        doc_href = "/docs" if slug == "index" else f"/docs/{slug}"
        if sep:
            doc_href = f"{doc_href}#{anchor}"
        return f"[{label}]({doc_href})"

    return _MARKDOWN_LINK_PATTERN.sub(replace_link, markdown_text)


def _render_markdown(markdown_text: str) -> str:
    if markdown_lib is None:
        return f"<pre>{html.escape(markdown_text)}</pre>"

    return markdown_lib.markdown(
        markdown_text,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "sane_lists",
        ],
    )


def _sim_config_default(field_name: str) -> Any:
    return sim_config_default(field_name)


def _sim_config_field(
    sim_field_name: str,
    *,
    alias: str | None = None,
    **field_kwargs: Any,
) -> Any:
    if alias is not None:
        field_kwargs["alias"] = alias
    return Field(
        default_factory=lambda sim_field_name=sim_field_name: _sim_config_default(
            sim_field_name,
        ),
        **field_kwargs,
    )


class _RunProfileBaseModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


_RUN_PROFILE_FIELD_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "agent_count": {"gt": 0},
    "emotion_temperature": {"ge": 0.0, "le": 1.0},
    "sentiment_neutrality_acting_threshold": {"ge": 0.0},
    "sentiment_neutrality_leaky_slope": {"ge": 0.0},
    "temperature": {"ge": 0.0, "le": 1.0},
}


def _build_run_profile_model() -> type[BaseModel]:
    sim_to_run_field_map = {
        sim_field_name: run_field_name
        for run_field_name, sim_field_name in RUN_PROFILE_TO_SIM_CONFIG_FIELD_MAP.items()
    }
    field_definitions: dict[str, tuple[Any, Any]] = {
        "social_class": (str, "All"),
    }

    for config_field in dataclass_fields(SimConfig):
        if not config_field.init or config_field.name in RUN_PROFILE_INTERNAL_ONLY_FIELDS:
            continue

        run_field_name = sim_to_run_field_map.get(config_field.name, config_field.name)
        alias = config_field.name if run_field_name != config_field.name else None
        field_definitions[run_field_name] = (
            config_field.type,
            _sim_config_field(
                config_field.name,
                alias=alias,
                **_RUN_PROFILE_FIELD_CONSTRAINTS.get(run_field_name, {}),
            ),
        )

    return create_model(
        "RunProfile",
        __base__=_RunProfileBaseModel,
        __module__=__name__,
        **field_definitions,
    )


RunProfile = cast("type[BaseModel]", _build_run_profile_model())


class SimulationRequest(BaseModel):
    news_text: str
    runs: list[RunProfile]


@dataclass
class PreparedSociety:
    config: SimConfig
    metadata: pd.DataFrame
    exposures: torch.Tensor
    personalities: torch.Tensor
    affinities: torch.Tensor
    memory: torch.Tensor
    adjacency_matrix: Any | None


@dataclass
class DebugSimulationResult:
    society: PreparedSociety
    input_world_tensor: torch.Tensor
    final_world_tensor: torch.Tensor
    context_vector: torch.Tensor
    attention_weights: torch.Tensor
    engagement_scores: torch.Tensor
    final_emotions: torch.Tensor
    social_state: dict[str, Any]
    validation_result: dict[str, Any] | None = None
    official_world_tensor: torch.Tensor | None = None
    skeptical_world_tensor: torch.Tensor | None = None
    backlash_potential: float = 0.0
    narrative_frame: str = "official"
    backlash_diagnostics: dict[str, Any] | None = None
    followup_context_vector: torch.Tensor | None = None
    followup_attention_weights: torch.Tensor | None = None
    followup_engagement_scores: torch.Tensor | None = None
    followup_emotions: torch.Tensor | None = None
    followup_social_state: dict[str, Any] | None = None


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_sim_config(**overrides: Any) -> SimConfig:
    config = SimConfig(**overrides)
    config.wealth_dim_idx = DIMENSION_INDICES["Wealth"]
    return config


def clone_sim_config(config: SimConfig, **overrides: Any) -> SimConfig:
    cloned = replace(config, **overrides)
    cloned.wealth_dim_idx = DIMENSION_INDICES["Wealth"]
    return cloned


def run_profile_to_sim_config_kwargs(
    run: RunProfile,
    **overrides: Any,
) -> dict[str, Any]:
    config_kwargs: dict[str, Any] = {}
    for field_name, field_value in run.model_dump().items():
        sim_field_name = RUN_PROFILE_TO_SIM_CONFIG_FIELD_MAP.get(
            field_name,
            field_name,
        )
        if (
            sim_field_name in SIM_CONFIG_FIELDS
            and sim_field_name not in RUN_PROFILE_INTERNAL_ONLY_FIELDS
        ):
            config_kwargs[sim_field_name] = deepcopy(field_value)
    config_kwargs.update(overrides)
    return config_kwargs


def build_debug_society(
    config: SimConfig,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    affinities: torch.Tensor | None = None,
    influence_scores: torch.Tensor | np.ndarray | list[float] | None = None,
    adjacency_matrix: Any | None = None,
    memory: torch.Tensor | None = None,
    metadata: pd.DataFrame | None = None,
) -> PreparedSociety:
    config = clone_sim_config(config)
    count = int(exposures.shape[0])

    if affinities is None:
        affinities = torch.ones_like(exposures)
    if memory is None:
        memory = torch.zeros_like(exposures)

    if influence_scores is None:
        influence_np = np.ones(count, dtype=np.float32)
    elif isinstance(influence_scores, torch.Tensor):
        influence_np = influence_scores.detach().cpu().numpy().astype(np.float32)
    else:
        influence_np = np.asarray(influence_scores, dtype=np.float32)

    if metadata is None:
        metadata = pd.DataFrame(
            {
                "Agent_ID": np.arange(count),
                "Class": ["Agent"] * count,
                "Region": ["Global"] * count,
                "Influence": influence_np,
            },
        )
    elif "Influence" not in metadata.columns:
        metadata = metadata.copy()
        metadata["Influence"] = influence_np

    return PreparedSociety(
        config=config,
        metadata=metadata,
        exposures=exposures,
        personalities=personalities,
        affinities=affinities,
        memory=memory,
        adjacency_matrix=adjacency_matrix,
    )


def materialize_prepared_society(
    config: SimConfig,
    metadata: pd.DataFrame,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    affinities: torch.Tensor,
    memory: torch.Tensor,
    adjacency_matrix: Any | None,
) -> PreparedSociety:
    return build_debug_society(
        config,
        exposures=exposures,
        personalities=personalities,
        affinities=affinities,
        influence_scores=metadata["Influence"].to_numpy(dtype=np.float32),
        adjacency_matrix=adjacency_matrix,
        memory=memory,
        metadata=metadata,
    )


def slice_prepared_society(
    society: PreparedSociety,
    *,
    social_class: str = "All",
    agent_limit: int | None = None,
) -> PreparedSociety | None:
    if social_class == "All":
        selected_indices_np = np.arange(len(society.metadata), dtype=np.int64)
    else:
        class_mask = society.metadata["Class"] == social_class
        selected_indices_np = np.flatnonzero(class_mask.to_numpy())

    if selected_indices_np.size == 0:
        return None

    if agent_limit is not None:
        limit = min(int(agent_limit), int(selected_indices_np.size))
        if limit <= 0:
            return None
        selected_indices_np = selected_indices_np[:limit]

    selected_indices = torch.as_tensor(
        selected_indices_np,
        dtype=torch.long,
        device=society.exposures.device,
    )
    subset_metadata = society.metadata.iloc[selected_indices_np].reset_index(drop=True)

    return build_debug_society(
        clone_sim_config(society.config, num_agents=int(selected_indices.numel())),
        exposures=society.exposures.index_select(0, selected_indices).clone(),
        personalities=society.personalities.index_select(0, selected_indices).clone(),
        affinities=society.affinities.index_select(0, selected_indices).clone(),
        adjacency_matrix=subset_adjacency(society.adjacency_matrix, selected_indices),
        memory=society.memory.index_select(0, selected_indices).clone(),
        metadata=subset_metadata,
    )


def prepare_society_for_debug(
    config: SimConfig,
    *,
    output_dir: str | None = None,
    evolve: bool | None = None,
) -> PreparedSociety:
    effective_config = clone_sim_config(
        config,
        output_dir=output_dir or getattr(config, "output_dir", "society_data"),
    )
    if evolve is not None:
        effective_config.enable_evolution = evolve

    seed_everything(effective_config.seed)

    metadata, exposures, personalities, affinities, adjacency_matrix = generate_society(
        effective_config,
        defer_structure=getattr(effective_config, "enable_evolution", True),
    )

    if getattr(effective_config, "enable_evolution", True):
        evolver = SocietyEvolution(
            effective_config, metadata, exposures, personalities,
        )
        metadata, exposures, personalities = evolver.evolve()
        metadata, personalities, adjacency_matrix = finalize_social_structure(
            effective_config,
            metadata,
            exposures,
            personalities,
        )

    return build_debug_society(
        effective_config,
        exposures=exposures,
        personalities=personalities,
        affinities=affinities,
        influence_scores=metadata["Influence"].to_numpy(dtype=np.float32),
        adjacency_matrix=adjacency_matrix,
        metadata=metadata,
    )


def evolve_society_for_debug(
    config: SimConfig,
    metadata: pd.DataFrame,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    effective_config = clone_sim_config(config)
    evolver = SocietyEvolution(effective_config, metadata, exposures, personalities)
    return evolver.evolve()


def distort_world_signal(
    config: SimConfig,
    world_tensor_raw: torch.Tensor,
    personalities: torch.Tensor,
    adjacency_matrix: Any | None = None,
) -> torch.Tensor:
    engine = CognitiveEngine(clone_sim_config(config))
    return engine.perceive_world(
        world_tensor_raw,
        personalities,
        adjacency_matrix=adjacency_matrix,
    )


def run_cognitive_cycle(
    config: SimConfig,
    world_tensor_raw: torch.Tensor,
    urgency: float,
    is_personal: bool,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    affinities: torch.Tensor,
    memory: torch.Tensor | None = None,
    adjacency_matrix: Any | None = None,
    *,
    world_tensor_skp: torch.Tensor | None = None,
    backlash_potential: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    engine = CognitiveEngine(clone_sim_config(config))
    chosen_world_tensor = world_tensor_raw
    if world_tensor_skp is not None and getattr(config, "use_backlash_ab_testing", False):
        decision = engine.run_backlash_ab_test(
            world_tensor_off=world_tensor_raw,
            world_tensor_skp=world_tensor_skp,
            backlash_potential=backlash_potential,
            urgency=urgency,
            is_personal=is_personal,
            exposures=exposures,
            personalities=personalities,
            agent_affinities=affinities,
            agent_memory=memory,
            adjacency_matrix=adjacency_matrix,
        )
        chosen_world_tensor = decision.world_tensor
    return engine.run(
        world_tensor_raw=chosen_world_tensor,
        urgency=urgency,
        is_personal=is_personal,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=memory,
        adjacency_matrix=adjacency_matrix,
    )


def consolidate_agent_memory(
    config: SimConfig,
    agent_memory: torch.Tensor,
    context_vector: torch.Tensor,
    social_rehearsal_factor: float = 0.0,
) -> torch.Tensor:
    engine = CognitiveEngine(clone_sim_config(config))
    return engine.consolidate_memory(
        agent_memory=agent_memory,
        context_vector=context_vector,
        social_rehearsal_factor=social_rehearsal_factor,
    )


def project_emotions(
    config: SimConfig,
    context_vector: torch.Tensor,
) -> torch.Tensor:
    engine = CognitiveEngine(clone_sim_config(config))
    return engine.project_emotions(context_vector)


def aggregate_social_state(
    config: SimConfig,
    emotion_tensor: torch.Tensor,
    influence_scores: torch.Tensor | np.ndarray | list[float],
    engagement_scores: torch.Tensor | None = None,
    adjacency_matrix: Any | None = None,
    personalities: torch.Tensor | None = None,
    is_personal: bool = False,
) -> dict[str, Any]:
    engine = SocialPhysicsEngine(clone_sim_config(config))
    return engine.aggregate_society(
        emotion_tensor,
        influence_scores,
        engagement_scores=engagement_scores,
        adjacency_matrix=adjacency_matrix,
        personalities=personalities,
        is_personal=is_personal,
    )


def map_emotions_to_sentiment(
    emotion_probs_8dim: torch.Tensor | np.ndarray,
    acting_ratio: float | torch.Tensor | np.ndarray | None = None,
    *,
    config: SimConfig | None = None,
    activation: str | None = None,
    leaky_slope: float | None = None,
) -> np.ndarray:
    if acting_ratio is None:
        return validator.map_plutchik_to_sentiment(emotion_probs_8dim)

    active_config = config if config is not None else SimConfig()
    activation_name = activation or active_config.sentiment_neutrality_activation
    leaky_slope_value = (
        active_config.sentiment_neutrality_leaky_slope
        if leaky_slope is None
        else leaky_slope
    )

    return (
        emotions_to_behavior_aware_sentiment_distribution(
            emotion_probs_8dim,
            acting_ratio,
            neutral_acting_threshold=active_config.sentiment_neutrality_acting_threshold,
            activation=activation_name,
            leaky_slope=leaky_slope_value,
        )
        .detach()
        .cpu()
        .numpy()
    )


def calculate_validation_metrics(
    system_probs_8dim: torch.Tensor | np.ndarray,
    baseline_probs: torch.Tensor | np.ndarray | list[float],
) -> dict[str, Any]:
    return validator.calculate_divergence(system_probs_8dim, baseline_probs)


def build_validation_result(
    config: SimConfig,
    social_state: dict[str, Any],
    baseline_probs: torch.Tensor | np.ndarray | list[float],
) -> dict[str, Any]:
    validation_result = calculate_validation_metrics(
        social_state["objective_center"],
        baseline_probs,
    )
    validation_result["stewing_interpretation"] = validator.validate_stewing(
        float(social_state.get("negative_integral") or 0.0),
        config.stewing_ticks,
    )
    return validation_result


def describe_agent_emotions(
    config: SimConfig,
    final_emotions: torch.Tensor,
) -> list[str]:
    emotion_indices = torch.argmax(final_emotions, dim=1).tolist()
    max_vals, _ = torch.max(final_emotions, dim=1)
    max_vals_list = max_vals.tolist()
    current_agent_emotions: list[str] = []
    for idx, max_val in zip(emotion_indices, max_vals_list):
        if max_val < config.dominant_emotion_threshold:
            current_agent_emotions.append("Neutral")
        else:
            current_agent_emotions.append(EMOTION_LABELS[idx])
    return current_agent_emotions


def serialize_agent_metadata(
    metadata: pd.DataFrame,
    personalities: torch.Tensor,
) -> list[dict[str, Any]]:
    agent_data: list[dict[str, Any]] = []
    for index, meta_row in enumerate(metadata.to_dict("records")):
        agent_data.append(
            {
                "id": int(meta_row["Agent_ID"]),
                "social_class": meta_row.get("Class", "Agent"),
                "region": meta_row.get("Region", "Global"),
                "big5": personalities[index].tolist(),
            },
        )
    return agent_data


def create_topology_for_debug(
    config: SimConfig,
    exposures: torch.Tensor,
    personalities: torch.Tensor,
    influence_scores: torch.Tensor | np.ndarray | list[float],
) -> Any | None:
    if isinstance(influence_scores, torch.Tensor):
        influence_np = influence_scores.detach().cpu().numpy()
    else:
        influence_np = np.asarray(influence_scores)
    return create_topology(clone_sim_config(config), exposures, personalities, influence_np)


def apply_triadic_closure_for_debug(config: SimConfig, adjacency_matrix: torch.Tensor):
    return apply_triadic_closure(clone_sim_config(config), adjacency_matrix, adjacency_matrix.device)


def execute_simulation_cycle(
    society: PreparedSociety,
    world_tensor_raw: torch.Tensor,
    *,
    urgency: float = 0.5,
    is_personal: bool = False,
    baseline_probs: torch.Tensor | np.ndarray | list[float] | None = None,
    world_tensor_skp: torch.Tensor | None = None,
    backlash_potential: float = 0.0,
) -> DebugSimulationResult:
    active_society = society
    seed_everything(active_society.config.seed)

    cog_engine = CognitiveEngine(clone_sim_config(active_society.config))
    phys_engine = SocialPhysicsEngine(clone_sim_config(active_society.config))
    memory = active_society.memory.clone()
    input_world_tensor = world_tensor_raw.clone()
    skeptical_world_tensor = (
        world_tensor_skp.clone() if world_tensor_skp is not None else None
    )

    backlash_decision = None
    if world_tensor_skp is not None and getattr(
        active_society.config, "use_backlash_ab_testing", False,
    ):
        backlash_decision = cog_engine.run_backlash_ab_test(
            world_tensor_off=input_world_tensor,
            world_tensor_skp=world_tensor_skp,
            backlash_potential=backlash_potential,
            urgency=urgency,
            is_personal=is_personal,
            exposures=active_society.exposures,
            personalities=active_society.personalities,
            agent_affinities=active_society.affinities,
            agent_memory=memory,
            adjacency_matrix=active_society.adjacency_matrix,
        )
        final_world_tensor = backlash_decision.world_tensor.clone()
        if active_society.config.use_agent_memory:
            if (
                backlash_decision.sample_indices is not None
                and backlash_decision.sample_context is not None
                and backlash_decision.sample_indices.numel() > 0
            ):
                updated_sample_memory = cog_engine.consolidate_memory(
                    agent_memory=memory.index_select(0, backlash_decision.sample_indices),
                    context_vector=backlash_decision.sample_context,
                    social_rehearsal_factor=min(0.5, backlash_potential),
                )
                memory.index_copy_(
                    0,
                    backlash_decision.sample_indices,
                    updated_sample_memory,
                )
    else:
        final_world_tensor = input_world_tensor.clone()

    context_vector, attention_weights, engagement_scores = cog_engine.run(
        world_tensor_raw=final_world_tensor,
        urgency=urgency,
        is_personal=is_personal,
        exposures=active_society.exposures,
        personalities=active_society.personalities,
        agent_affinities=active_society.affinities,
        agent_memory=memory,
        adjacency_matrix=active_society.adjacency_matrix,
    )

    if active_society.config.use_algorithmic_amplification:
        sample_size = max(
            1,
            int(len(active_society.exposures) * active_society.config.algo_sample_size),
        )
        _, ab_attention, ab_engagement = cog_engine.run(
            world_tensor_raw=final_world_tensor,
            urgency=urgency,
            is_personal=is_personal,
            exposures=active_society.exposures[:sample_size],
            personalities=active_society.personalities[:sample_size],
            agent_affinities=active_society.affinities[:sample_size],
            agent_memory=memory[:sample_size],
            adjacency_matrix=subset_adjacency(
                active_society.adjacency_matrix,
                torch.arange(sample_size, device=active_society.exposures.device),
            ),
        )

        engagement_weighted_attention = ab_attention * ab_engagement.unsqueeze(1)
        avg_attention_per_dim = engagement_weighted_attention.mean(dim=0)
        top_dims = torch.topk(avg_attention_per_dim, k=2).indices

        final_world_tensor = final_world_tensor.clone()
        exaggeration = active_society.config.algo_exaggeration_factor
        signal_floor = 0.05
        for dim_idx in top_dims:
            current_val = final_world_tensor[0, dim_idx].item()
            # Amplification should only strengthen existing signal, not invent
            # a negative narrative for neutral or near-zero inputs.
            if abs(current_val) <= signal_floor:
                continue
            final_world_tensor[0, dim_idx] *= exaggeration

        final_world_tensor = torch.clamp(final_world_tensor, -1.0, 1.0)
        context_vector, attention_weights, engagement_scores = cog_engine.run(
            world_tensor_raw=final_world_tensor,
            urgency=urgency,
            is_personal=is_personal,
            exposures=active_society.exposures,
            personalities=active_society.personalities,
            agent_affinities=active_society.affinities,
            agent_memory=memory,
            adjacency_matrix=active_society.adjacency_matrix,
        )

    final_emotions = cog_engine.project_emotions(context_vector)
    social_state = phys_engine.aggregate_society(
        final_emotions,
        active_society.metadata["Influence"].to_numpy(dtype=np.float32),
        engagement_scores=engagement_scores,
        adjacency_matrix=active_society.adjacency_matrix,
        personalities=active_society.personalities,
        is_personal=is_personal,
    )
    primary_context_vector = context_vector
    primary_attention_weights = attention_weights
    primary_engagement_scores = engagement_scores
    primary_final_emotions = final_emotions
    primary_social_state = deepcopy(social_state)

    followup_context_vector = None
    followup_attention_weights = None
    followup_engagement_scores = None
    followup_emotions = None
    followup_social_state = None

    action_vector = social_state.get("action_vector")
    action_name = social_state.get("action_name")
    if action_vector is not None:
        action_tensor = torch.tensor(
            [action_vector],
            dtype=torch.float32,
            device=final_world_tensor.device,
        )
        (
            followup_context_vector,
            followup_attention_weights,
            followup_engagement_scores,
        ) = cog_engine.run(
            world_tensor_raw=action_tensor,
            urgency=0.8,
            is_personal=True,
            exposures=active_society.exposures,
            personalities=active_society.personalities,
            agent_affinities=active_society.affinities,
            agent_memory=memory,
            adjacency_matrix=active_society.adjacency_matrix,
        )
        followup_emotions = cog_engine.project_emotions(followup_context_vector)
        followup_social_state = phys_engine.aggregate_society(
            followup_emotions,
            active_society.metadata["Influence"].to_numpy(dtype=np.float32),
            engagement_scores=followup_engagement_scores,
            adjacency_matrix=active_society.adjacency_matrix,
            personalities=active_society.personalities,
            is_personal=True,
        )
        primary_social_state["endogenous_event"] = action_name

    if active_society.config.use_agent_memory:
        memory_context_vector = (
            followup_context_vector
            if followup_context_vector is not None
            else primary_context_vector
        )
        memory_social_state = (
            followup_social_state
            if followup_social_state is not None
            else primary_social_state
        )
        conf = float(memory_social_state.get("confidence", 0.0))
        act_ratio = float(memory_social_state.get("acting_ratio", 0.0))
        rehearsal_factor = (conf + act_ratio) / 2.0
        active_society.memory = cog_engine.consolidate_memory(
            agent_memory=memory,
            context_vector=memory_context_vector,
            social_rehearsal_factor=rehearsal_factor,
        )

    validation_result = None
    if baseline_probs is not None:
        validation_result = build_validation_result(
            active_society.config,
            primary_social_state,
            baseline_probs,
        )

    return DebugSimulationResult(
        society=active_society,
        input_world_tensor=input_world_tensor,
        final_world_tensor=final_world_tensor,
        context_vector=primary_context_vector,
        attention_weights=primary_attention_weights,
        engagement_scores=primary_engagement_scores,
        final_emotions=primary_final_emotions,
        social_state=primary_social_state,
        validation_result=validation_result,
        official_world_tensor=input_world_tensor,
        skeptical_world_tensor=skeptical_world_tensor,
        backlash_potential=backlash_potential,
        narrative_frame=(
            backlash_decision.chosen_frame if backlash_decision is not None else "official"
        ),
        backlash_diagnostics=(
            backlash_decision.as_dict() if backlash_decision is not None else None
        ),
        followup_context_vector=followup_context_vector,
        followup_attention_weights=followup_attention_weights,
        followup_engagement_scores=followup_engagement_scores,
        followup_emotions=followup_emotions,
        followup_social_state=followup_social_state,
    )


def run_debug_simulation(
    config: SimConfig,
    world_tensor_raw: torch.Tensor,
    *,
    society: PreparedSociety | None = None,
    urgency: float = 0.5,
    is_personal: bool = False,
    baseline_probs: torch.Tensor | np.ndarray | list[float] | None = None,
    world_tensor_skp: torch.Tensor | None = None,
    backlash_potential: float = 0.0,
) -> DebugSimulationResult:
    effective_config = clone_sim_config(config)
    active_society = society or prepare_society_for_debug(effective_config)
    return execute_simulation_cycle(
        active_society,
        world_tensor_raw,
        urgency=urgency,
        is_personal=is_personal,
        baseline_probs=baseline_probs,
        world_tensor_skp=world_tensor_skp,
        backlash_potential=backlash_potential,
    )


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def build_society_cache_key(config: SimConfig) -> str:
    cache_payload = asdict(config)
    cache_payload.pop("output_dir", None)
    cache_payload.pop("wealth_dim_idx", None)
    cache_payload["_cache_version"] = 2

    raw_key = json.dumps(cache_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def prepare_society_sync(run: RunProfile, run_output_dir: str):
    """Generates and evolves society synchronously"""
    config = create_sim_config(
        **run_profile_to_sim_config_kwargs(
            run,
            output_dir=run_output_dir,
        ),
    )
    cache_key = build_society_cache_key(config)

    with SOCIETY_CACHE_LOCK:
        cached_entry = SOCIETY_CACHE.get(cache_key)
        if cached_entry is not None:
            print(f"Cache Hit for {cache_key[:12]}")
            SOCIETY_CACHE.move_to_end(cache_key)

    if cached_entry is not None:
        (
            metadata_full,
            exposures_full,
            personalities_full,
            affinities_full,
            adjacency_matrix,
            cached_warnings,
        ) = cached_entry
        memory_full = torch.zeros_like(exposures_full)
        return (
            config,
            metadata_full,
            exposures_full,
            personalities_full,
            affinities_full,
            memory_full,
            adjacency_matrix,
            list(cached_warnings),
        )

    print(f"Cache Miss. Generating & Caching {cache_key[:12]}")
    (
        metadata_full,
        exposures_full,
        personalities_full,
        affinities_full,
        adjacency_matrix,
    ) = generate_society(
        config,
        defer_structure=getattr(config, "enable_evolution", True),
    )

    generation_warnings: list[str] = []

    # Evolution phase
    if getattr(config, "enable_evolution", True):
        try:
            evolver = SocietyEvolution(
                config, metadata_full, exposures_full, personalities_full,
            )
            metadata_full, exposures_full, personalities_full = evolver.evolve()
            metadata_full, personalities_full, adjacency_matrix = finalize_social_structure(
                config,
                metadata_full,
                exposures_full,
                personalities_full,
            )
        except Exception as e:
            warning = f"Evolution failed; using base society instead: {e}"
            generation_warnings.append(warning)
            print(warning)

    memory_full = torch.zeros_like(exposures_full)

    with SOCIETY_CACHE_LOCK:
        SOCIETY_CACHE[cache_key] = (
            metadata_full,
            exposures_full,
            personalities_full,
            affinities_full,
            adjacency_matrix,
            tuple(generation_warnings),
        )

        if len(SOCIETY_CACHE) > MAX_CACHE_SIZE:
            evicted_key, _ = SOCIETY_CACHE.popitem(last=False)
            print(f"🧹 LRU Evicted {evicted_key[:12]} from RAM cache.")

    return (
        config,
        metadata_full,
        exposures_full,
        personalities_full,
        affinities_full,
        memory_full,
        adjacency_matrix,
        generation_warnings,
    )


def subset_adjacency(adj, keep_indices):
    """Slices a sparse adjacency matrix to keep only rows/cols specified in 'keep_indices'.
    Remaps indices to 0..len(keep_indices).
    Renormalizes rows to sum to 1.
    """
    if adj is None:
        return None

    device = adj.device

    # 1. Create a mask of nodes to keep
    N = adj.shape[0]

    # Check if keep_indices is a tensor or list
    if not isinstance(keep_indices, torch.Tensor):
        keep_indices = torch.tensor(keep_indices, device=device, dtype=torch.long)
    else:
        keep_indices = keep_indices.to(device)

    M = keep_indices.numel()
    if M == 0:
        return None

    # Mapping from old_index -> new_index (0..M)
    # Initialize with -1
    mapping = torch.full((N,), -1, dtype=torch.long, device=device)
    mapping[keep_indices] = torch.arange(M, device=device)

    # 2. Filter edges
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()

    row, col = indices[0], indices[1]

    # Keep edge only if BOTH src and dst are in the subset
    # We use the mapping to check: if mapping[idx] != -1, it's kept
    new_row = mapping[row]
    new_col = mapping[col]

    # Filter where both are valid (>= 0)
    mask = (new_row >= 0) & (new_col >= 0)

    if not mask.any():
        return None

    final_row = new_row[mask]
    final_col = new_col[mask]
    final_values = values[mask]

    # 3. Create new sparse tensor
    new_indices = torch.stack([final_row, final_col])

    # 4. Renormalize rows
    # Construct temp to get row sums
    temp_adj = torch.sparse_coo_tensor(new_indices, final_values, size=(M, M))
    row_sums = torch.sparse.sum(temp_adj, dim=1).to_dense()

    # Avoid division by zero
    row_sums[row_sums == 0] = 1.0

    # Normalize
    final_values = final_values / row_sums[final_row]

    return torch.sparse_coo_tensor(new_indices, final_values, size=(M, M)).coalesce()


def cleanup_memory():
    """Forces garbage collection and clears PyTorch cache."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory Cleaned.")


def cleanup_old_files():
    shutil.rmtree("society_data", ignore_errors=True)
    current_time = time.time()
    for folder in os.listdir("."):
        if os.path.isdir(folder) and folder.startswith("temp_sim_"):
            try:
                # Only delete if older than 5 minutes to avoid race conditions with concurrent requests
                if current_time - os.path.getctime(folder) > 300:
                    shutil.rmtree(folder)
            except Exception as e:
                print(f"Cleanup skipped for {folder}: {e}")


@app.post("/simulate")
async def run_simulation(req: SimulationRequest, background_tasks: BackgroundTasks):
    # 0. Global Memory Management - Start with a clean slate
    print("\n--- Initializing New Simulation Request ---")

    # Offload sweeping cleanup to background to prevent blocking
    background_tasks.add_task(cleanup_old_files)

    cleanup_memory()

    request_id = str(uuid.uuid4())[:8]
    temp_dir = f"temp_sim_{request_id}"
    os.makedirs(temp_dir, exist_ok=True)

    print(
        f"[{request_id}] Received Batch Request: {req.news_text[:50]}... ({len(req.runs)} runs)",
    )

    try:
        # Start LLM Task
        print(f"[{request_id}] Analyzing News with LLM...")
        llm_task = asyncio.create_task(get_world_state(req.news_text))

        # Start Baseline Tasks
        print(f"[{request_id}] Analyzing News with Baseline AIs...")
        baseline_task = asyncio.to_thread(validator.get_baseline_prob, req.news_text)

        # Start Society Generation/Evolution Tasks
        society_tasks = []
        for i, run in enumerate(req.runs):
            run_output_dir = os.path.join(temp_dir, f"run_{i}")
            os.makedirs(run_output_dir, exist_ok=True)
            society_tasks.append(
                asyncio.to_thread(prepare_society_sync, run, run_output_dir),
            )

        # Wait for all tasks to complete concurrently
        results = await asyncio.gather(
            llm_task,
            baseline_task,
            *society_tasks,
            return_exceptions=True,
        )

        # Parse LLM result
        llm_result = results[0]
        if isinstance(llm_result, Exception):
            raise llm_result

        llm_result = cast(
            "tuple[torch.Tensor, torch.Tensor, float, bool, list[str], str, float]",
            llm_result,
        )
        (
            world_tensor,
            world_tensor_skp,
            urgency,
            is_personal,
            detected_biases,
            reasoning,
            backlash_potential,
        ) = llm_result

        # Parse Baseline results
        baseline_result = results[1]
        if isinstance(baseline_result, Exception):
            raise baseline_result

        # Process each run
        all_results: list[dict[str, Any]] = []
        for i, (run, society_result) in enumerate(zip(req.runs, results[2:])):
            if isinstance(society_result, Exception):
                print(f"[{request_id}] Run {i} Error: {society_result}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Simulation Run {i} failed: {society_result!s}",
                )

            society_result = cast(
                "tuple[SimConfig, pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, list[str]]",
                society_result,
            )

            (
                config,
                metadata_full,
                exposures_full,
                personalities_full,
                affinities_full,
                memory_full,
                adjacency_matrix_full,
                generation_warnings,
            ) = society_result

            available_agents = (
                len(metadata_full)
                if run.social_class == "All"
                else int((metadata_full["Class"] == run.social_class).sum())
            )
            if available_agents == 0:
                all_results.append(
                    {"error": f"No agents found for class: {run.social_class}"},
                )
                continue

            base_society = materialize_prepared_society(
                config=config,
                metadata=metadata_full,
                exposures=exposures_full,
                personalities=personalities_full,
                affinities=affinities_full,
                memory=memory_full,
                adjacency_matrix=adjacency_matrix_full,
            )
            active_society = slice_prepared_society(
                base_society,
                social_class=run.social_class,
                agent_limit=run.agent_count,
            )
            if active_society is None:
                all_results.append(
                    {"error": f"No agents found for class: {run.social_class}"},
                )
                continue

            if run.social_class != "All":
                print(
                    f"[{request_id}] Filtered Run {i} to Class: {run.social_class} ({len(active_society.metadata)} agents)",
                )

            debug_result = run_debug_simulation(
                config,
                world_tensor,
                society=active_society,
                urgency=urgency,
                is_personal=is_personal,
                baseline_probs=baseline_result,
                world_tensor_skp=world_tensor_skp,
                backlash_potential=backlash_potential,
            )

            if debug_result.social_state.get("endogenous_event") is not None:
                print(
                    f"[{request_id}] Autopoietic Trigger: {debug_result.social_state['endogenous_event']} generated.",
                )

            metadata = debug_result.society.metadata
            personalities = debug_result.society.personalities
            influence = metadata["Influence"].to_numpy(dtype=np.float32)
            validation_result = debug_result.validation_result or build_validation_result(
                config,
                debug_result.social_state,
                baseline_result,
            )

            explain_engine = ExplainabilityEngine()
            explainability_data = explain_engine.generate_explanation(
                social_state=debug_result.social_state,
                metadata=metadata,
                personalities=personalities,
                final_emotions=debug_result.final_emotions,
                attention_weights=debug_result.attention_weights,
                narrative_frame=debug_result.narrative_frame,
                backlash_potential=debug_result.backlash_potential,
                backlash_diagnostics=debug_result.backlash_diagnostics,
                official_world_tensor=debug_result.official_world_tensor,
                skeptical_world_tensor=debug_result.skeptical_world_tensor,
            )

            current_agent_emotions = describe_agent_emotions(
                config,
                debug_result.final_emotions,
            )
            agent_data = serialize_agent_metadata(metadata, personalities)

            all_results.append(
                {
                    "run_index": i,
                    "config": run.model_dump(),
                    "dominant_emotion": debug_result.social_state.get(
                        "dominant_emotion", "Neutral",
                    ),
                    "polarization": round(
                        debug_result.social_state.get("polarization") or 0.0,
                        3,
                    ),
                    "elite_divergence": round(
                        debug_result.social_state.get("elite_divergence") or 0.0,
                        4,
                    ),
                    "divergence": validation_result[
                        "wasserstein_distance"
                    ],  # Keep key for UI compatibility
                    "wasserstein_distance": validation_result["wasserstein_distance"],
                    "kl_divergence": validation_result["kl_divergence"],
                    "validation_details": validation_result,
                    "explainability": explainability_data,
                    "agent_states": current_agent_emotions,
                    "agent_influence": influence.tolist(),
                    "agent_metadata": agent_data,
                    "endogenous_event": debug_result.social_state.get(
                        "endogenous_event",
                    ),
                    "detected_biases": detected_biases,
                    "reasoning": reasoning,
                    "negative_integral": debug_result.social_state.get(
                        "negative_integral",
                    )
                    or 0.0,
                    "acting_ratio": debug_result.social_state.get("acting_ratio"),
                    "total_eligible": debug_result.social_state.get("total_eligible"),
                    "population_size": debug_result.social_state.get("population_size"),
                    "warnings": generation_warnings,
                },
            )

    except Exception as e:
        print(f"[{request_id}] Processing Error: {e}")
        raise HTTPException(status_code=502, detail=f"Simulation Error: {e!s}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

    return {"status": "success", "results": all_results}

# Mount the generated test figures directory
generated_dir = os.path.join(os.path.dirname(__file__), "research_paper_tests", "generated")
if os.path.exists(generated_dir):
    app.mount("/generated", StaticFiles(directory=generated_dir), name="generated")

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
