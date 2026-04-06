import torch
from schema import SimConfig
from cognitive_engine import CognitiveEngine
from physics_engine import SocialPhysicsEngine

def run_zero_test():
    config = SimConfig()
    cog_engine = CognitiveEngine(config)
    phys_engine = SocialPhysicsEngine(config)

    N = 3
    # Use a pure zero tensor
    world_tensor_raw = torch.zeros(1, 12)
    personalities = torch.rand(N, 5)
    exposures = torch.zeros(N, 12)
    agent_affinities = torch.ones(N, 12)
    urgency = 0.0
    is_personal = False

    print("=== COGNITIVE ENGINE ZERO TEST ===")
    print(f"1. Original World Tensor: {world_tensor_raw.tolist()}")

    context_vector, attention_weights, engagement_scores = cog_engine.run(
        world_tensor_raw=world_tensor_raw,
        urgency=urgency,
        is_personal=is_personal,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=agent_affinities,
    )
    
    print(f"2. Context Vector: {context_vector.tolist()[0]}")

    emotions = cog_engine.project_emotions(context_vector)
    print(f"3. Projected Emotions (Logits -> Softmax): {emotions.tolist()[0]}")

    print("\n=== PHYSICS ENGINE ZERO TEST ===")
    influence_scores = torch.tensor([1.0, 1.0, 1.0])
    
    result = phys_engine.aggregate_society(emotions, influence_scores, engagement_scores)
    
    print(f"Dominant Emotion: {result['dominant_emotion']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Valence: {result['sentiment_valence']}")
    print(f"Action Potential Vector: {result['action_vector']}")

if __name__ == "__main__":
    run_zero_test()
