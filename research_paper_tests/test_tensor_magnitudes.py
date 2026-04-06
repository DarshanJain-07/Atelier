import torch
from schema import SimConfig, PSYCH_PROJECTION
from cognitive_engine import CognitiveEngine
from physics_engine import SocialPhysicsEngine
import numpy as np

def run_dry_test():
    config = SimConfig()
    cog_engine = CognitiveEngine(config)
    phys_engine = SocialPhysicsEngine(config)

    # Consider 3 example agents
    N = 3
    world_tensor_raw = torch.randn(1, 12) * 1.5  # Random event
    personalities = torch.rand(N, 5)  # Big Five
    exposures = torch.rand(N, 12) * 2 - 1
    agent_affinities = torch.ones(N, 12)
    urgency = 0.5
    is_personal = False

    print("=== COGNITIVE ENGINE DRY RUN ===")
    print(f"1. Original World Tensor Max: {world_tensor_raw.max().item():.4f}, Mean: {world_tensor_raw.mean().item():.4f}, Norm: {torch.norm(world_tensor_raw).item():.4f}")

    # Run cognitive engine
    context_vector, attention_weights, engagement_scores = cog_engine.run(
        world_tensor_raw=world_tensor_raw,
        urgency=urgency,
        is_personal=is_personal,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=agent_affinities,
    )
    
    print(f"2. Context Vector Max: {context_vector.max().item():.4f}, Mean: {context_vector.mean().item():.4f}, Norm (avg per agent): {torch.norm(context_vector, dim=1).mean().item():.4f}")

    # Project emotions
    emotions = cog_engine.project_emotions(context_vector)
    print(f"3. Projected Emotions Max: {emotions.max().item():.4f}, Mean: {emotions.mean().item():.4f}, Norm (avg per agent): {torch.norm(emotions, dim=1).mean().item():.4f}")
    print(f"   Note: Emotions are softmaxed, so they sum to {emotions.sum(dim=1).mean().item():.4f} per agent.")

    print("\n=== PHYSICS ENGINE DRY RUN ===")
    influence_scores = torch.tensor([1.0, 1.0, 1.0])
    
    # We will instrument physics engine to trace what happens during stewing
    emotion_tensor = emotions.clone()
    structural_weights = influence_scores.float()
    weights = structural_weights / structural_weights.sum()
    
    current_emotions = emotion_tensor.clone()
    print(f"Initial Arousal (norm) avg: {torch.norm(current_emotions, dim=1).mean().item():.4f}")
    print(f"Initial Peak Emotion avg: {current_emotions.max(dim=1)[0].mean().item():.4f}")

    num_ticks = getattr(config, "stewing_ticks", 5)
    for tick in range(num_ticks):
        center_of_gravity = (current_emotions * weights.unsqueeze(1)).sum(dim=0)
        local_centers = center_of_gravity.unsqueeze(0).expand(N, -1)
        
        arousal = torch.norm(current_emotions, dim=1)
        viral_energy = arousal
        
        norm_emotion = current_emotions / (arousal.unsqueeze(1) + 1e-9)
        local_arousal = torch.norm(local_centers, dim=1)
        norm_local = local_centers / (local_arousal.unsqueeze(1) + 1e-9)
        alignment = (norm_emotion * norm_local).sum(dim=1)
        
        validation_multiplier = 1.0 + alignment
        viral_energy = viral_energy * validation_multiplier
        
        outrage_gain = config.outrage_gain
        max_multiplier = config.max_viral_multiplier
        midpoint = config.saturation_midpoint
        outrage_boost = 1.0 + max_multiplier * torch.sigmoid(outrage_gain * (viral_energy - midpoint))
        
        viral_weights = weights * outrage_boost
        viral_weights = viral_weights / viral_weights.sum()
        viral_center = (current_emotions * viral_weights.unsqueeze(1)).sum(dim=0)
        
        if tick < num_ticks - 1:
            self_retention = getattr(config, "stewing_self_retention", 0.6)
            local_influence = getattr(config, "stewing_local_influence", 0.3)
            viral_influence = getattr(config, "stewing_viral_influence", 0.1)

            new_emotions = (
                self_retention * current_emotions
                + local_influence * local_centers
                + viral_influence * viral_center.unsqueeze(0).expand(N, -1)
            )
            
            # Print state BEFORE tick ends
            print(f"Tick {tick}: Arousal avg: {torch.norm(current_emotions, dim=1).mean().item():.4f} -> {torch.norm(new_emotions, dim=1).mean().item():.4f} (after blend)")
            print(f"Tick {tick}: Peak emotion avg: {current_emotions.max(dim=1)[0].mean().item():.4f} -> {new_emotions.max(dim=1)[0].mean().item():.4f} (after blend)")
            current_emotions = new_emotions

    print("\nObservation:")
    print("Notice how the blending operations (self_retention, local_influence, viral_influence) shrink the values without renormalizing.")

if __name__ == "__main__":
    run_dry_test()
