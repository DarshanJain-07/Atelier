import torch
from cognitive_engine import CognitiveEngine
from schema import SimConfig, PSYCH_PROJECTION, EMOTION_LABELS

def test_emotional_divergence():
    config = SimConfig(num_agents=2, seed=42)
    # Agent 0: Average (0.5 Neuroticism)
    # Agent 1: Extreme (0.95 Neuroticism)
    personalities = torch.tensor([
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.95]
    ])
    exposures = torch.zeros((2, 12))
    affinities = torch.ones((2, 12)) * 0.5
    
    # Event: Moderate Safety Threat (-0.4)
    world_tensor = torch.zeros(12)
    world_tensor[1] = -0.4 
    
    engine = CognitiveEngine(config)
    
    context, attention, engagement = engine.run(
        world_tensor_raw=world_tensor,
        urgency=0.5,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities
    )
    
    # Project to emotions
    emotions = torch.matmul(context, PSYCH_PROJECTION)
    
    print("\n--- Emotional Divergence Test (Safety Threat -0.4) ---")
    for i in range(2):
        print(f"\nAgent {i} (Neuroticism={personalities[i, 4]:.2f}):")
        for j, label in enumerate(EMOTION_LABELS):
            val = emotions[i, j].item()
            if abs(val) > 0.01:
                print(f"  {label}: {val:.3f}")

if __name__ == "__main__":
    test_emotional_divergence()
