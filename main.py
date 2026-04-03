import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from threading import Lock
from typing import Any, List, Tuple, cast

import numpy as np
import pandas as pd
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cognitive_engine import CognitiveEngine
from explainability import ExplainabilityEngine
from generate_society import apply_triadic_closure, create_topology, generate_society
from input_layer import get_world_state
from physics_engine import SocialPhysicsEngine

# Import our Core Logic
from schema import (
    DIMENSIONS,
    DIMENSION_INDICES,
    EMOTION_LABELS,
    PERSONALITY_CORRELATIONS,
    SimConfig,
)
from society_evolution import SocietyEvolution
from validation import Validator

load_dotenv()

# Check API Key
if not os.getenv("GEMINI_API_KEY"):
    print("❌ ERROR: GEMINI_API_KEY not set. Simulation will fail.")

app = FastAPI()
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

# Persistent LRU cache for societies (seed + count + temp -> data)
MAX_CACHE_SIZE = 7
SOCIETY_CACHE: OrderedDict[str, Any] = OrderedDict()
SOCIETY_CACHE_LOCK = Lock()


class RunProfile(BaseModel):
    seed: int = 42
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    social_class: str = "All"
    agent_count: int = Field(default=1000, gt=0)
    use_distortion: bool = True
    use_pressure: bool = True
    use_maslow: bool = True
    use_power_law: bool = False
    emotion_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    panic_threshold: float = Field(default=-1.2, le=0.0)

    # New Features
    stewing_ticks: int = 5
    stewing_self_retention: float = 0.6
    stewing_local_influence: float = 0.3
    stewing_viral_influence: float = 0.1

    use_algorithmic_amplification: bool = False
    algo_sample_size: float = 0.1
    algo_exaggeration_factor: float = 1.5

    use_selective_exposure: bool = True
    selective_exposure_base_tolerance: float = -0.3
    selective_exposure_openness_factor: float = 0.4
    selective_exposure_gain: float = 8.0
    selective_exposure_max_suppression: float = 0.85

    use_agent_memory: bool = False
    memory_decay_rate: float = 0.7
    memory_desensitization_gain: float = 0.5
    memory_trigger_stacking_gain: float = 1.2
    memory_social_rehearsal_gain: float = 0.4

    use_network_topology: bool = True
    homophily_strength: float = 6.0
    influence_bias_exp: float = 0.4
    triadic_closure_prob: float = 0.2
    triadic_closure_iterations: int = 2
    triadic_closure_homophily_threshold: float = 0.5
    use_granovetter_thresholds: bool = True
    granovetter_threshold_mean: float = 0.25
    granovetter_threshold_std: float = 0.15
    personality_socialization_gain: float = 0.2
    enable_evolution: bool = True

    # Researcher (Cognitive)
    cross_dim_interaction_strength: float = 0.3
    threat_sensitivity_gain: float = 1.5
    k_processing_tanh_gain: float = 1.5
    attention_residual_gain: float = 0.35
    attention_modulated_gain: float = 1.0
    relevance_importance_weight: float = 0.7
    relevance_base_weight: float = 0.3
    threat_amplifier_gain: float = 1.5
    stress_neurotic_amplification: float = 1.5
    stress_openness_reduction: float = 0.5
    stress_extraversion_boost: float = 0.7

    # Researcher (Physics)
    outrage_gain: float = 5.0
    max_viral_multiplier: float = 10.0
    saturation_midpoint: float = 0.5

    # Researcher (Distortion)
    distortion_max_noise: float = 0.4
    distortion_neurotic_gain: float = 0.6
    perception_social_consensus_gain: float = 0.25  # New 2-Stage Perception
    affinity_min_strength: float = 0.01
    normalize_affinities_by_mean: bool = True

    # Researcher (Evolution)
    evolution_generations: int = 10
    inheritance_fraction: float = 0.7
    shock_frequency: float = 0.1
    shock_magnitude: float = 0.2


class SimulationRequest(BaseModel):
    news_text: str
    runs: List[RunProfile]


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
            }
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
        effective_config
    )

    if getattr(effective_config, "enable_evolution", True):
        evolver = SocietyEvolution(
            effective_config, metadata, exposures, personalities
        )
        metadata, exposures, personalities = evolver.evolve()

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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    engine = CognitiveEngine(clone_sim_config(config))
    return engine.run(
        world_tensor_raw=world_tensor_raw,
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


def map_emotions_to_sentiment(emotion_probs_8dim: torch.Tensor | np.ndarray) -> np.ndarray:
    return validator.map_plutchik_to_sentiment(emotion_probs_8dim)


def calculate_validation_metrics(
    system_probs_8dim: torch.Tensor | np.ndarray,
    baseline_probs: torch.Tensor | np.ndarray | list[float],
) -> dict[str, Any]:
    return validator.calculate_divergence(system_probs_8dim, baseline_probs)


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


def run_debug_simulation(
    config: SimConfig,
    world_tensor_raw: torch.Tensor,
    *,
    society: PreparedSociety | None = None,
    urgency: float = 0.5,
    is_personal: bool = False,
    baseline_probs: torch.Tensor | np.ndarray | list[float] | None = None,
) -> DebugSimulationResult:
    effective_config = clone_sim_config(config)
    active_society = society or prepare_society_for_debug(effective_config)
    cog_engine = CognitiveEngine(clone_sim_config(active_society.config))
    phys_engine = SocialPhysicsEngine(clone_sim_config(active_society.config))
    memory = active_society.memory.clone()
    input_world_tensor = world_tensor_raw.clone()

    final_world_tensor = input_world_tensor
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

    if getattr(active_society.config, "use_algorithmic_amplification", False):
        sample_size = max(
            1,
            int(
                len(active_society.exposures)
                * getattr(active_society.config, "algo_sample_size", 0.1)
            ),
        )
        _, ab_attention, ab_engagement = cog_engine.run(
            world_tensor_raw=input_world_tensor,
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

        final_world_tensor = input_world_tensor.clone()
        exaggeration = getattr(active_society.config, "algo_exaggeration_factor", 1.5)
        for dim_idx in top_dims:
            current_val = final_world_tensor[0, dim_idx].item()
            if abs(current_val) > 0.05:
                final_world_tensor[0, dim_idx] *= exaggeration
            else:
                final_world_tensor[0, dim_idx] = -0.3

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

    if getattr(active_society.config, "use_agent_memory", False):
        conf = float(social_state.get("confidence", 0.0))
        act_ratio = float(social_state.get("acting_ratio", 0.0))
        rehearsal_factor = (conf + act_ratio) / 2.0
        active_society.memory = cog_engine.consolidate_memory(
            agent_memory=memory,
            context_vector=context_vector,
            social_rehearsal_factor=rehearsal_factor,
        )

    validation_result = None
    if baseline_probs is not None:
        validation_result = calculate_validation_metrics(
            social_state["objective_center"], baseline_probs
        )

    return DebugSimulationResult(
        society=active_society,
        input_world_tensor=input_world_tensor,
        final_world_tensor=final_world_tensor,
        context_vector=context_vector,
        attention_weights=attention_weights,
        engagement_scores=engagement_scores,
        final_emotions=final_emotions,
        social_state=social_state,
        validation_result=validation_result,
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
    config = SimConfig(
        num_agents=run.agent_count,
        seed=run.seed,
        use_signal_distortion=run.use_distortion,
        use_time_pressure=run.use_pressure,
        use_maslow_gating=run.use_maslow,
        use_power_law_influence=run.use_power_law,
        mutation_temperature=run.temperature,
        emotion_temperature=run.emotion_temperature,
        panic_threshold=run.panic_threshold,
        output_dir=run_output_dir,
        cross_dim_interaction_strength=run.cross_dim_interaction_strength,
        threat_sensitivity_gain=run.threat_sensitivity_gain,
        k_processing_tanh_gain=run.k_processing_tanh_gain,
        attention_residual_gain=run.attention_residual_gain,
        attention_modulated_gain=run.attention_modulated_gain,
        relevance_importance_weight=run.relevance_importance_weight,
        relevance_base_weight=run.relevance_base_weight,
        threat_amplifier_gain=run.threat_amplifier_gain,
        stress_neurotic_amplification=run.stress_neurotic_amplification,
        stress_openness_reduction=run.stress_openness_reduction,
        stress_extraversion_boost=run.stress_extraversion_boost,
        outrage_gain=run.outrage_gain,
        max_viral_multiplier=run.max_viral_multiplier,
        saturation_midpoint=run.saturation_midpoint,
        distortion_max_noise=run.distortion_max_noise,
        distortion_neurotic_gain=run.distortion_neurotic_gain,
        perception_social_consensus_gain=run.perception_social_consensus_gain,
        affinity_min_strength=run.affinity_min_strength,
        normalize_affinities_by_mean=run.normalize_affinities_by_mean,
        evolution_generations=run.evolution_generations,
        inheritance_fraction=run.inheritance_fraction,
        shock_frequency=run.shock_frequency,
        shock_magnitude=run.shock_magnitude,
        use_algorithmic_amplification=run.use_algorithmic_amplification,
        algo_sample_size=run.algo_sample_size,
        algo_exaggeration_factor=run.algo_exaggeration_factor,
        use_selective_exposure=run.use_selective_exposure,
        selective_exposure_base_tolerance=run.selective_exposure_base_tolerance,
        selective_exposure_openness_factor=run.selective_exposure_openness_factor,
        selective_exposure_gain=run.selective_exposure_gain,
        selective_exposure_max_suppression=run.selective_exposure_max_suppression,
        use_agent_memory=run.use_agent_memory,
        memory_decay_rate=run.memory_decay_rate,
        memory_desensitization_gain=run.memory_desensitization_gain,
        memory_trigger_stacking_gain=run.memory_trigger_stacking_gain,
        memory_social_rehearsal_gain=run.memory_social_rehearsal_gain,
        stewing_ticks=run.stewing_ticks,
        stewing_self_retention=run.stewing_self_retention,
        stewing_local_influence=run.stewing_local_influence,
        stewing_viral_influence=run.stewing_viral_influence,
        use_network_topology=run.use_network_topology,
        homophily_strength=run.homophily_strength,
        influence_bias_exp=run.influence_bias_exp,
        triadic_closure_prob=run.triadic_closure_prob,
        triadic_closure_iterations=run.triadic_closure_iterations,
        triadic_closure_homophily_threshold=run.triadic_closure_homophily_threshold,
        use_granovetter_thresholds=run.use_granovetter_thresholds,
        granovetter_threshold_mean=run.granovetter_threshold_mean,
        granovetter_threshold_std=run.granovetter_threshold_std,
        personality_socialization_gain=run.personality_socialization_gain,
        enable_evolution=run.enable_evolution,
    )
    config.wealth_dim_idx = DIMENSION_INDICES["Wealth"]
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
    ) = generate_society(config)

    generation_warnings: list[str] = []

    # Evolution phase
    if getattr(config, "enable_evolution", True):
        try:
            evolver = SocietyEvolution(
                config, metadata_full, exposures_full, personalities_full
            )
            metadata_full, exposures_full, personalities_full = evolver.evolve()
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
    """
    Slices a sparse adjacency matrix to keep only rows/cols specified in 'keep_indices'.
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
        f"[{request_id}] Received Batch Request: {req.news_text[:50]}... ({len(req.runs)} runs)"
    )

    try:
        # Start LLM Task
        print(f"[{request_id}] Analyzing News with LLM...")
        llm_task = asyncio.to_thread(get_world_state, req.news_text)

        # Start Baseline Tasks
        print(f"[{request_id}] Analyzing News with Baseline AIs...")
        baseline_task = asyncio.to_thread(validator.get_baseline_prob, req.news_text)

        # Start Society Generation/Evolution Tasks
        society_tasks = []
        for i, run in enumerate(req.runs):
            run_output_dir = os.path.join(temp_dir, f"run_{i}")
            os.makedirs(run_output_dir, exist_ok=True)
            society_tasks.append(
                asyncio.to_thread(prepare_society_sync, run, run_output_dir)
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

        llm_result = cast(Tuple[torch.Tensor, float, bool, list[str], str], llm_result)
        world_tensor, urgency, is_personal, detected_biases, reasoning = llm_result

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
                    detail=f"Simulation Run {i} failed: {str(society_result)}",
                )

            society_result = cast(
                Tuple[
                    SimConfig,
                    pd.DataFrame,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    Any,
                    list[str],
                ],
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

            # --- ENFORCE DETERMINISM ---
            # Reset RNG state before simulation logic (e.g. CognitiveEngine noise) executes
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

            # --- CLASS FILTERING ---
            indices_torch = []
            if run.social_class != "All":
                mask = metadata_full["Class"] == run.social_class
                indices_np = np.where(mask.to_numpy())[0]
                indices_torch = torch.tensor(indices_np, dtype=torch.long)

                metadata = metadata_full.iloc[indices_np].reset_index(drop=True)
                exposures = exposures_full[indices_np]
                personalities = personalities_full[indices_np]
                affinities = affinities_full[indices_np]
                memory = memory_full[indices_np]

                # Slicing sparse adjacency matrix to preserve physics within the subgroup
                if adjacency_matrix_full is not None:
                    adjacency_matrix = subset_adjacency(
                        adjacency_matrix_full, indices_torch
                    )
                    if adjacency_matrix is not None:
                        print(
                            f"[{request_id}] Resliced Adjacency Matrix for Class: {run.social_class} (Nodes: {len(indices_np)})"
                        )
                    else:
                        print(
                            f"[{request_id}] Adjacency Slice Empty/Failed for Class: {run.social_class}"
                        )
                else:
                    adjacency_matrix = None

                print(
                    f"[{request_id}] Filtered Run {i} to Class: {run.social_class} ({len(metadata)} agents)"
                )
            else:
                metadata = metadata_full.copy()
                exposures = exposures_full
                personalities = personalities_full
                affinities = affinities_full
                memory = memory_full
                adjacency_matrix = adjacency_matrix_full

            limit = min(run.agent_count, len(metadata))
            if limit == 0:
                all_results.append(
                    {"error": f"No agents found for class: {run.social_class}"}
                )
                continue

            current_count = len(metadata)
            metadata = metadata.iloc[:limit]
            exposures = exposures[:limit]
            personalities = personalities[:limit]
            affinities = affinities[:limit]
            memory = memory[:limit]

            if adjacency_matrix is not None and limit < current_count:
                # Further subsetting if limit < filtered_population
                limit_indices = torch.arange(limit, dtype=torch.long)
                adjacency_matrix = subset_adjacency(adjacency_matrix, limit_indices)

            try:
                influence = metadata["Influence"].to_numpy(dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Type conversion failed for metadata columns: {e}")

            # 5. Cognitive Engine & Algorithmic Amplification
            cog_engine = CognitiveEngine(config)

            if getattr(config, "use_algorithmic_amplification", False):
                # --- PASS 1: The A/B Test ---
                sample_size = int(limit * getattr(config, "algo_sample_size", 0.1))
                sample_size = max(1, sample_size)

                # We don't want to update global memory during the A/B test pass, so we clone it
                ab_memory = memory[:sample_size].clone() if memory is not None else None

                # We need a subset of the adjacency matrix for the A/B test sample
                ab_adj = subset_adjacency(
                    adjacency_matrix, torch.arange(sample_size, device=exposures.device)
                )

                _, ab_attention, ab_engagement = cog_engine.run(
                    world_tensor_raw=world_tensor,
                    urgency=urgency,
                    is_personal=is_personal,
                    exposures=exposures[:sample_size],
                    personalities=personalities[:sample_size],
                    agent_affinities=affinities[:sample_size],
                    agent_memory=ab_memory,
                    adjacency_matrix=ab_adj,
                )

                # --- The Algorithm's Intervention ---
                # Which dimensions received the highest attention * weighted by how engaged the user was?
                engagement_weighted_attention = ab_attention * ab_engagement.unsqueeze(
                    1
                )
                avg_attention_per_dim = engagement_weighted_attention.mean(dim=0)

                # Find the top 2 dimensions that caused the most engagement
                top_dims = torch.topk(avg_attention_per_dim, k=2).indices

                # Mutate the world tensor to exaggerate those specific dimensions
                mutated_world_tensor = world_tensor.clone()
                exaggeration = getattr(config, "algo_exaggeration_factor", 1.5)

                for dim_idx in top_dims:
                    current_val = mutated_world_tensor[0, dim_idx].item()

                    # If the dimension is already active, exaggerate it
                    if abs(current_val) > 0.05:
                        mutated_world_tensor[0, dim_idx] *= exaggeration
                    else:
                        # The algorithm "hallucinates" or injects a threat/benefit
                        # to manufacture engagement where none existed.
                        # We inject a moderate threat (-0.3) because fear drives engagement.
                        mutated_world_tensor[0, dim_idx] = -0.3

                # Clamp the mutated tensor to realistic boundaries
                mutated_world_tensor = torch.clamp(mutated_world_tensor, -1.0, 1.0)

                print(
                    f"[{request_id}] Algorithmic Pass 1 Complete. Mutated Dimensions {top_dims.tolist()} by {exaggeration}x"
                )

                # Use the mutated tensor for the real broadcast
                final_world_tensor = mutated_world_tensor
            else:
                final_world_tensor = world_tensor

            # --- PASS 2: The Viral Broadcast ---
            context_vector, attention_weights, engagement_scores = cog_engine.run(
                world_tensor_raw=final_world_tensor,
                urgency=urgency,
                is_personal=is_personal,
                exposures=exposures,
                personalities=personalities,
                agent_affinities=affinities,
                agent_memory=memory,
                adjacency_matrix=adjacency_matrix,
            )

            final_emotions = cog_engine.project_emotions(context_vector)

            # 6. Social Physics
            phys_engine = SocialPhysicsEngine(config)
            social_state = phys_engine.aggregate_society(
                final_emotions,
                influence,
                engagement_scores,
                adjacency_matrix,
                personalities=personalities,
                is_personal=is_personal,
            )

            # --- ENDOGENOUS EVENT FEEDBACK LOOP (Autopoietic Simulation) ---
            action_vector = social_state.get("action_vector")
            action_name = social_state.get("action_name")

            if action_vector is not None:
                print(f"[{request_id}] Autopoietic Trigger: {action_name} generated.")
                action_tensor = torch.tensor(
                    [action_vector],
                    dtype=torch.float32,
                    device=final_world_tensor.device,
                )

                # Feedback loop into cognitive engine without user input
                (
                    context_vector_2,
                    attention_weights_2,
                    engagement_scores_2,
                ) = cog_engine.run(
                    world_tensor_raw=action_tensor,
                    urgency=0.8,  # High urgency for endogenous events
                    is_personal=True,  # Protests/uprisings are personal
                    exposures=exposures,
                    personalities=personalities,
                    agent_affinities=affinities,
                    agent_memory=memory,  # Use original memory for the 2nd pass imprint
                    adjacency_matrix=adjacency_matrix,
                )

                final_emotions_2 = cog_engine.project_emotions(context_vector_2)

                # Re-aggregate society with the new emotional state
                social_state = phys_engine.aggregate_society(
                    final_emotions_2,
                    influence,
                    engagement_scores_2,
                    adjacency_matrix,
                    personalities=personalities,
                    is_personal=True,
                )

                # Update loop variables for consolidation
                final_emotions = final_emotions_2
                attention_weights = attention_weights_2
                engagement_scores = engagement_scores_2
                context_vector = context_vector_2  # The final internalized thing
                social_state["endogenous_event"] = action_name

            # --- 2-STAGE MEMORY CONSOLIDATION ---
            # Update the global memory array for this specific run
            if getattr(config, "use_agent_memory", False):
                # Calculate Social Rehearsal Factor (Salience)
                # Proxy: Combination of Confidence (intensity) and Acting Ratio (volume)
                conf = social_state.get("confidence", 0.0)
                act_ratio = social_state.get("acting_ratio", 0.0)
                rehearsal_factor = (conf + act_ratio) / 2.0

                updated_memory = cog_engine.consolidate_memory(
                    agent_memory=memory,
                    context_vector=context_vector,
                    social_rehearsal_factor=rehearsal_factor,
                )

                if run.social_class != "All":
                    memory_full[indices_torch[:limit]] = updated_memory.to(
                        memory_full.device
                    )
                else:
                    memory_full[:limit] = updated_memory.to(memory_full.device)

            # 7. Validation
            validation_result = validator.calculate_divergence(
                social_state["objective_center"], baseline_result
            )
            validation_result["stewing_interpretation"] = validator.validate_stewing(
                social_state.get("negative_integral") or 0.0,
                getattr(config, "stewing_ticks", 5),
            )

            # 8. Explainability
            explain_engine = ExplainabilityEngine()
            explainability_data = explain_engine.generate_explanation(
                social_state=social_state,
                metadata=metadata.iloc[:limit],
                personalities=personalities,
                final_emotions=final_emotions,
                attention_weights=attention_weights,
            )

            # Prepare emotions for UI
            emotion_indices = torch.argmax(final_emotions, dim=1).tolist()
            max_vals, _ = torch.max(final_emotions, dim=1)
            max_vals_list = max_vals.tolist()
            current_agent_emotions = []
            for k, idx in enumerate(emotion_indices):
                if max_vals_list[k] < config.dominant_emotion_threshold:
                    current_agent_emotions.append("Neutral")
                else:
                    current_agent_emotions.append(EMOTION_LABELS[idx])

            # Prepare Agent Metadata
            agent_data = []
            metadata_dicts = metadata.iloc[:limit].to_dict("records")
            for j, meta_row in enumerate(metadata_dicts):
                agent_data.append(
                    {
                        "id": int(meta_row["Agent_ID"]),
                        "social_class": meta_row.get("Class", "Agent"),
                        "region": meta_row.get("Region", "Global"),
                        "big5": personalities[j].tolist(),
                    }
                )

            all_results.append(
                {
                    "run_index": i,
                    "config": run.model_dump(),
                    "dominant_emotion": social_state.get("dominant_emotion", "Neutral"),
                    "polarization": round(social_state.get("polarization") or 0.0, 3),
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
                    "endogenous_event": social_state.get("endogenous_event"),
                    "detected_biases": detected_biases,
                    "reasoning": reasoning,
                    "negative_integral": social_state.get("negative_integral") or 0.0,
                    "acting_ratio": social_state.get("acting_ratio"),
                    "total_eligible": social_state.get("total_eligible"),
                    "population_size": social_state.get("population_size"),
                    "warnings": generation_warnings,
                }
            )

    except Exception as e:
        print(f"[{request_id}] Processing Error: {e}")
        raise HTTPException(status_code=502, detail=f"Simulation Error: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            background_tasks.add_task(shutil.rmtree, temp_dir, ignore_errors=True)

    return {"status": "success", "results": all_results}


# app.mount("/docs/", StaticFiles(directory="site", html=True), name="docs")


# @app.get("/docs")
# async def docs_redirect():
#     from fastapi.responses import RedirectResponse

#     return RedirectResponse(url="/docs/")


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
