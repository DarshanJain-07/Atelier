import asyncio
import os
import shutil
import uuid
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from input_layer import get_world_state
from physics_engine import SocialPhysicsEngine
from explainability import ExplainabilityEngine

# Import our Core Logic
from schema import DIMENSIONS, EMOTION_LABELS, PSYCH_PROJECTION, SimConfig
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

# Persistent cache for societies (seed + count + temp -> data)
SOCIETY_CACHE: Dict[str, Any] = {}


class RunProfile(BaseModel):
    seed: int = 42
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    region: str = "All"
    role: str = "All"
    agent_count: int = Field(default=1000, gt=0)
    use_distortion: bool = True
    use_pressure: bool = True
    use_maslow: bool = True
    use_power_law: bool = False
    emotion_temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    panic_threshold: float = Field(default=-1.2, le=0.0)

    # Researcher (Cognitive)
    cross_dim_interaction_strength: float = 0.3
    threat_sensitivity_gain: float = 1.5
    k_processing_tanh_gain: float = 1.5
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

    # Researcher (Evolution)
    evolution_generations: int = 10
    inheritance_fraction: float = 0.7
    shock_frequency: float = 0.1
    shock_magnitude: float = 0.2


class SimulationRequest(BaseModel):
    news_text: str
    runs: List[RunProfile]


@app.get("/health")
async def health_check():
    return {"status": "ok"}


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
        evolution_generations=run.evolution_generations,
        inheritance_fraction=run.inheritance_fraction,
        shock_frequency=run.shock_frequency,
        shock_magnitude=run.shock_magnitude,
    )
    config.wealth_dim_idx = DIMENSIONS.index("Wealth")

    cache_key = f"{run.seed}_{run.agent_count}_{run.temperature}_{run.use_power_law}"

    if cache_key in SOCIETY_CACHE:
        print(f"Cache Hit for {cache_key}")
        metadata_full, exposures_full, personalities_full, affinities_full, memory_full, adjacency_matrix = (
            SOCIETY_CACHE[cache_key]
        )
        return (
            config,
            metadata_full,
            exposures_full,
            personalities_full,
            affinities_full,
            memory_full,
            adjacency_matrix,
        )

    print(f"Cache Miss. Generating & Caching {cache_key}")
    metadata_full, exposures_full, personalities_full, affinities_full, adjacency_matrix = (
        generate_society(config)
    )

    # Evolution phase
    if getattr(config, "enable_evolution", True):
        try:
            evolver = SocietyEvolution(
                config, metadata_full, exposures_full, personalities_full
            )
            metadata_full, exposures_full, personalities_full = evolver.evolve()
        except Exception as e:
            print(f"Evolution failed or skipped: {e}")
            pass

    memory_full = torch.zeros_like(exposures_full)

    SOCIETY_CACHE[cache_key] = (
        metadata_full,
        exposures_full,
        personalities_full,
        affinities_full,
        memory_full,
        adjacency_matrix,
    )
    return config, metadata_full, exposures_full, personalities_full, affinities_full, memory_full, adjacency_matrix


def cleanup_memory():
    """Forces garbage collection and clears PyTorch cache."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Memory Cleaned.")


@app.post("/simulate")
async def run_simulation(req: SimulationRequest):
    global SOCIETY_CACHE

    # 0. Global Memory Management - Start with a clean slate
    print("\n--- Initializing New Simulation Request ---")

    needed_keys = {
        f"{run.seed}_{run.agent_count}_{run.temperature}_{run.use_power_law}"
        for run in req.runs
    }
    keys_to_evict = [k for k in SOCIETY_CACHE if k not in needed_keys]

    for k in keys_to_evict:
        del SOCIETY_CACHE[k]

    if keys_to_evict:
        print(f"🧹 Evicted {len(keys_to_evict)} unused societies from RAM cache.")

    shutil.rmtree("society_data", ignore_errors=True)
    for folder in os.listdir("."):
        if os.path.isdir(folder) and folder.startswith("temp_sim_"):
            try:
                shutil.rmtree(folder)
            except Exception:
                pass

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

        llm_result = cast(Tuple[torch.Tensor, float, bool], llm_result)
        world_tensor, urgency, is_personal = llm_result

        # Parse Baseline results
        baseline_result = results[1]
        if isinstance(baseline_result, Exception):
            raise baseline_result

        # Process each run
        all_results = []
        for i, (run, society_result) in enumerate(zip(req.runs, results[2:])):
            if isinstance(society_result, Exception):
                print(f"[{request_id}] Run {i} Error: {society_result}")
                all_results.append({"error": str(society_result)})
                continue

            society_result = cast(
                Tuple[
                    SimConfig, pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any
                ],
                society_result,
            )

            config, metadata_full, exposures_full, personalities_full, affinities_full, memory_full, adjacency_matrix_full = society_result

            # --- ROLE FILTERING ---
            if run.role != "All":
                mask = metadata_full["Role"] == run.role
                metadata = metadata_full[mask].reset_index(drop=True)
                indices = np.where(mask.to_numpy())[0]
                exposures = exposures_full[indices]
                personalities = personalities_full[indices]
                affinities = affinities_full[indices]
                memory = memory_full[indices]
                
                # Topological subsetting is complex for sparse matrices without breaking normalizations.
                # For role-filtered sub-populations, we disable network topology to preserve accuracy.
                adjacency_matrix = None
                
                print(
                    f"[{request_id}] Filtered Run {i} to Role: {run.role} ({len(metadata)} agents)"
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
                all_results.append({"error": f"No agents found for role: {run.role}"})
                continue

            metadata = metadata.iloc[:limit]
            exposures = exposures[:limit]
            personalities = personalities[:limit]
            affinities = affinities[:limit]
            memory = memory[:limit]
            
            if adjacency_matrix is not None and limit < len(metadata_full):
                 # Same logic: if we truncated the population, topology breaks.
                 adjacency_matrix = None

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
                
                _, ab_attention, ab_engagement, _ = cog_engine.run(
                    world_tensor_raw=world_tensor,
                    urgency=urgency,
                    is_personal=is_personal,
                    exposures=exposures[:sample_size],
                    personalities=personalities[:sample_size],
                    agent_affinities=affinities[:sample_size],
                    agent_memory=ab_memory,
                )
                
                # --- The Algorithm's Intervention ---
                # Which dimensions received the highest attention * weighted by how engaged the user was?
                engagement_weighted_attention = ab_attention * ab_engagement.unsqueeze(1)
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
                
                print(f"[{request_id}] Algorithmic Pass 1 Complete. Mutated Dimensions {top_dims.tolist()} by {exaggeration}x")
                
                # Use the mutated tensor for the real broadcast
                final_world_tensor = mutated_world_tensor
            else:
                final_world_tensor = world_tensor

            # --- PASS 2: The Viral Broadcast ---
            context_vector, attention_weights, engagement_scores, updated_memory = cog_engine.run(
                world_tensor_raw=final_world_tensor,
                urgency=urgency,
                is_personal=is_personal,
                exposures=exposures,
                personalities=personalities,
                agent_affinities=affinities,
                agent_memory=memory,
            )

            # --- MEMORY UPDATE ---
            # Update the global memory array for this specific run
            if getattr(config, "use_agent_memory", False):
                if run.role != "All":
                    memory_full[indices[:limit]] = updated_memory.to(memory_full.device)
                else:
                    memory_full[:limit] = updated_memory.to(memory_full.device)

            device = context_vector.device
            projection_matrix = PSYCH_PROJECTION.to(device)
            final_emotions = torch.matmul(context_vector, projection_matrix)

            # --- SHARPENING ---
            # Use Softmax with temperature to amplify the primary emotion of each agent.
            # This ensures the Physics Engine detects a clear 'Dominant Emotion'.
            final_emotions = F.softmax(
                final_emotions / max(0.01, config.emotion_temperature), dim=1
            )

            # 6. Social Physics
            phys_engine = SocialPhysicsEngine(config)
            social_state = phys_engine.aggregate_society(
                final_emotions, influence, engagement_scores, adjacency_matrix
            )

            # --- ENDOGENOUS EVENT FEEDBACK LOOP (Autopoietic Simulation) ---
            action_vector = social_state.get("action_vector")
            action_name = social_state.get("action_name")
            
            if action_vector is not None:
                print(f"[{request_id}] ⚠️ Autopoietic Trigger: {action_name} generated.")
                action_tensor = torch.tensor([action_vector], dtype=torch.float32, device=final_world_tensor.device)
                
                # Feedback loop into cognitive engine without user input
                context_vector_2, attention_weights_2, engagement_scores_2, updated_memory_2 = cog_engine.run(
                    world_tensor_raw=action_tensor,
                    urgency=0.8, # High urgency for endogenous events
                    is_personal=True, # Protests/uprisings are personal
                    exposures=exposures,
                    personalities=personalities,
                    agent_affinities=affinities,
                    agent_memory=updated_memory,
                )
                
                if getattr(config, "use_agent_memory", False):
                    if run.role != "All":
                        memory_full[indices[:limit]] = updated_memory_2.to(memory_full.device)
                    else:
                        memory_full[:limit] = updated_memory_2.to(memory_full.device)

                final_emotions_2 = torch.matmul(context_vector_2, projection_matrix)
                final_emotions_2 = F.softmax(
                    final_emotions_2 / max(0.01, config.emotion_temperature), dim=1
                )
                
                # Re-aggregate society with the new emotional state
                social_state = phys_engine.aggregate_society(
                    final_emotions_2, influence, engagement_scores_2, adjacency_matrix
                )
                final_emotions = final_emotions_2
                attention_weights = attention_weights_2
                social_state["endogenous_event"] = action_name

            # 7. Validation
            validation_result = validator.calculate_divergence(
                social_state["objective_center"], baseline_result
            )

            # 8. Explainability
            explain_engine = ExplainabilityEngine()
            explainability_data = explain_engine.generate_explanation(
                social_state=social_state,
                metadata=metadata.iloc[:limit],
                personalities=personalities,
                final_emotions=final_emotions,
                attention_weights=attention_weights
            )

            # Prepare emotions for UI
            emotion_indices = torch.argmax(final_emotions, dim=1).tolist()
            current_agent_emotions = [EMOTION_LABELS[idx] for idx in emotion_indices]

            # Prepare Agent Metadata
            agent_data = []
            metadata_dicts = metadata.iloc[:limit].to_dict("records")
            for j, meta_row in enumerate(metadata_dicts):
                agent_data.append(
                    {
                        "id": int(meta_row["Agent_ID"]),
                        "role": meta_row.get("Role", "Agent"),
                        "region": meta_row.get("Region", "Global"),
                        "big5": personalities[j].tolist(),
                    }
                )

            all_results.append(
                {
                    "run_index": i,
                    "config": run.model_dump(),
                    "dominant_emotion": social_state["dominant_emotion"],
                    "polarization": round(social_state["polarization"], 3),
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
                }
            )

    except Exception as e:
        print(f"[{request_id}] Processing Error: {e}")
        raise HTTPException(status_code=502, detail=f"Simulation Error: {str(e)}")
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    return {"status": "success", "results": all_results}


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
