import sys
import os
import tracemalloc
import time
import gc
import torch
import asyncio

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig, PSYCH_PROJECTION
from generate_society import generate_society
from society_evolution import SocietyEvolution
from cognitive_engine import CognitiveEngine
from physics_engine import SocialPhysicsEngine
from input_layer import get_world_state
from validation import Validator
from explainability import ExplainabilityEngine
import torch.nn.functional as F

def get_process_memory_mb():
    """Returns the maximum resident set size (RSS) in MB."""
    try:
        import resource
        # On Linux ru_maxrss is in kilobytes, on macOS it's in bytes
        if sys.platform == "darwin":
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024.0)
        else:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024.0)
        except ImportError:
            return 0.0

async def run_full_pipeline(count, prompt):
    print(f"\n[ Running Full Pipeline Memory Test for {count} Agents ]")
    start_time = time.time()
    
    # 1. Configuration
    config = SimConfig(
        num_agents=count,
        use_agent_memory=True,
        use_network_topology=True,
        enable_evolution=True
    )
    
    # 2. LLM and Baseline (concurrent)
    print("-> Calling LLM and Baseline API for Prompt Extraction...")
    validator = Validator()
    
    llm_task = asyncio.to_thread(get_world_state, prompt)
    baseline_task = asyncio.to_thread(validator.get_baseline_prob, prompt)
    
    world_tensor, urgency, is_personal, detected_biases, reasoning = await llm_task
    baseline_result = await baseline_task
    
    # 3. Society Generation
    print("-> Generating Society...")
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    
    if config.enable_evolution:
        print("-> Evolving Society...")
        evolver = SocietyEvolution(config, df_meta, exposures, personalities)
        df_meta, exposures, personalities = evolver.evolve()

    agent_memory = torch.zeros_like(exposures)
    
    # 4. Cognitive Engine Run
    print("-> Running Cognitive Engine...")
    cog_engine = CognitiveEngine(config)
    
    ctx, att, eng, agent_memory = cog_engine.run(
        world_tensor_raw=world_tensor,
        urgency=urgency,
        is_personal=is_personal,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=agent_memory,
    )
    
    # 5. Social Physics Engine Run
    print("-> Running Social Physics Engine...")
    phys_engine = SocialPhysicsEngine(config)
    
    device = ctx.device
    projection_matrix = PSYCH_PROJECTION.to(device)
    final_emotions = torch.matmul(ctx, projection_matrix)
    final_emotions = F.softmax(final_emotions / max(0.01, config.emotion_temperature), dim=1)
    
    influence = df_meta["Influence"].to_numpy()
    
    social_state = phys_engine.aggregate_society(
        final_emotions, influence, eng, adjacency_matrix
    )
    
    # 6. Validation & Metrics
    print("-> Running Validation & Metrics...")
    validation_result = validator.calculate_divergence(
        social_state["objective_center"], baseline_result
    )

    # 7. Explainability

    print("-> Running Explainability Engine...")
    explain_engine = ExplainabilityEngine()
    explainability_data = explain_engine.generate_explanation(
        social_state=social_state,
        metadata=df_meta,
        personalities=personalities,
        final_emotions=final_emotions,
        attention_weights=att,
    )
    
    end_time = time.time()
    
    # Measure Memory usage
    current, peak = tracemalloc.get_traced_memory()
    os_max_rss = get_process_memory_mb()
    gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    
    print(f"\n--- Results for {count} Agents ---")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print(f"Tracemalloc Peak Memory: {peak / (1024 * 1024):.2f} MB")
    print(f"Tracemalloc Current Memory: {current / (1024 * 1024):.2f} MB")
    print(f"OS Reported Max RSS Memory: {os_max_rss:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU Max Memory Allocated: {gpu_mem:.2f} MB")
        
    return (df_meta, exposures, personalities, affinities, adjacency_matrix, 
            cog_engine, phys_engine, ctx, att, eng, agent_memory, social_state,
            validation_result, explainability_data)

def test_ram_usage():
    print("--- Testing System RAM Usage for FULL Simulation Pipeline ---")
    prompt = "The central bank has just announced a surprise 2% interest rate hike to combat inflation."
    print(f"Test Prompt: '{prompt}'")
    
    # Start tracing memory allocations
    tracemalloc.start()
    
    agent_counts = [1000, 5000, 10000]
    
    for count in agent_counts:
        # Run async pipeline
        objects_to_del = asyncio.run(run_full_pipeline(count, prompt))
        
        # Clean up explicitly to see independent runs
        for obj in objects_to_del:
            del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tracemalloc.stop()
    print("\n--- RAM Usage Test Complete ---")

if __name__ == "__main__":
    test_ram_usage()
