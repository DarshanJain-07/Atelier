import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from input_layer import get_world_state
from physics_engine import SocialPhysicsEngine
from schema import PSYCH_PROJECTION, SimConfig
from validation import Validator

def test_semantic_alignment():
    print("--- Testing Semantic Alignment and Mathematical Validation ---")

    prompts = [
        "The weather is sunny with a high of 75 degrees.",
        "The city council approved a new budget for road maintenance.",
        "The company reported a 5% increase in Q3 revenue.",
        "The local sports team won their game last night 3-2.",
        "New tax regulations will come into effect next month.",
        "The controversial politician gave a speech claiming economic victory.",
        "The CEO insists the mass layoffs are purely for 'strategic realignment'.",
        "The company is optimizing its workforce by letting go of 10,000 employees.",
        "Radical activists are trying to destroy our way of life with their new policies.",
        "The corrupted elite are stealing your hard-earned money and giving it to their friends!"
    ]

    # Initialize Validator
    validator = Validator()
    
    # Generate Society once
    config = SimConfig(num_agents=1000, emotion_temperature=0.01)
    print("Generating Society...")
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    
    distances = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Analyzing Prompt: '{prompt}'")
        
        try:
            # 1. LLM
            import time
            time.sleep(4)  # Avoid rate limits
            world_tensor, urgency, is_personal, detected_biases, reasoning = get_world_state(prompt)
            print(f"  Biases Detected: {detected_biases}")
            
            # 2. Baseline
            baseline_result = validator.get_baseline_prob(prompt)
            print(f"  Baseline Probs (Neg, Neu, Pos): {[round(x, 3) for x in baseline_result]}")
            
            # 3. Cognitive Engine
            agent_memory = torch.zeros_like(exposures)
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
            
            # 4. Social Physics Engine
            phys_engine = SocialPhysicsEngine(config)
            device = ctx.device
            projection_matrix = PSYCH_PROJECTION.to(device)
            final_emotions = torch.matmul(ctx, projection_matrix)
            final_emotions = F.softmax(final_emotions / max(0.01, config.emotion_temperature), dim=1)
            
            influence = df_meta["Influence"].to_numpy()
            social_state = phys_engine.aggregate_society(
                final_emotions, influence, eng, adjacency_matrix
            )
            
            # 5. Validation
            sys_sentiment = validator.map_plutchik_to_sentiment(social_state["objective_center"])
            print(f"  System Probs (Neg, Neu, Pos): {[round(x, 3) for x in sys_sentiment]}")
            
            validation_result = validator.calculate_divergence(
                social_state["objective_center"], baseline_result
            )
            w_dist = validation_result["wasserstein_distance"]
            print(f"  Wasserstein Distance: {w_dist}")
            distances.append(w_dist)
            
        except Exception as e:
            print(f"  Error processing prompt: {e}")
            distances.append(0.0)

    # 6. Visualization
    plt.figure(figsize=(12, 6))
    
    # X-axis is the index 1 to 10
    x_indices = np.arange(1, len(prompts) + 1)
    
    plt.plot(x_indices, distances, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    # Annotate points
    for i, txt in enumerate(distances):
        plt.annotate(f"{txt:.2f}", (x_indices[i], distances[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Background color shades to show objective vs deceptive zones
    plt.axvspan(0.5, 4.5, color='green', alpha=0.1, label='Objective Zone')
    plt.axvspan(4.5, 7.5, color='yellow', alpha=0.1, label='Spin/Mild Zone')
    plt.axvspan(7.5, 10.5, color='red', alpha=0.1, label='Deceptive/Polarized Zone')
    
    plt.title("Semantic Alignment vs Baseline (RoBERTa)\nWasserstein Distance over increasingly Polarized Prompts", fontsize=14, fontweight="bold")
    plt.xlabel("Prompt Index (1=Objective, 10=Deceptive/Polarized)", fontsize=12)
    plt.ylabel("Wasserstein Distance", fontsize=12)
    plt.xticks(x_indices)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    output_path = os.path.join(
        os.path.dirname(__file__), "semantic_alignment_wasserstein.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    test_semantic_alignment()
