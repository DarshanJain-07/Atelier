import torch
import numpy as np
from cognitive_engine import CognitiveEngine
from generate_society import generate_society
from schema import SimConfig, emotions_to_valence
from research_paper_tests.config_schema import EMOTION_INDICES

def test_logical_contradiction_drift():
    """
    CHAOS TEST: Logical Contradiction Drift (Trauma vs. Prosperity)
    --------------------------------------------------------------
    Goal: If a society has "Trauma" in memory (Safety -0.8), can a 
    "Prosperity" signal (Wealth +0.8) override it?
    
    Real-world Alignment:
    - People don't forget safety threats just because they got a raise.
    - Historical trauma should act as a 'Cognitive Anchor'.
    """
    config = SimConfig(
        num_agents=1000,
        seed=42,
        use_agent_memory=True,
        memory_desensitization_gain=2.0,
        memory_trigger_stacking_gain=3.0,
        memory_decay_rate=0.9 # Very slow decay (Trauma sticks)
    )
    
    # 1. Generate Society
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    cognitive = CognitiveEngine(config)
    
    N = config.num_agents
    
    # 2. Inject "Trauma" Memory (Physical Safety -0.8)
    # This represents a historical event the agents haven't forgotten.
    trauma_vector = torch.zeros(N, 12)
    trauma_vector[:, 1] = -0.8 # Safety
    
    agent_memory = trauma_vector.clone()
    
    # 3. Present "Prosperity" Signal (Wealth +0.8)
    # The current signal says "The economy is great!"
    prosperity_signal = torch.zeros(12)
    prosperity_signal[0] = 0.8 # Wealth
    
    # 4. Process the Signal
    print("\n[Contradiction Test] Scenario: Historical Trauma (Safety) vs. Current Prosperity (Wealth)")
    
    # Run the cognitive engine
    context_vector, _, engagement_scores = cognitive.run(
        world_tensor_raw=prosperity_signal.unsqueeze(0),
        urgency=0.5,
        is_personal=False,
        exposures=exposures,
        personalities=personalities,
        agent_affinities=affinities,
        agent_memory=agent_memory,
        adjacency_matrix=adjacency_matrix
    )
    
    # 5. Emotional Projection
    emotions = cognitive.project_emotions(context_vector)
    valence = emotions_to_valence(emotions).mean().item()
    
    joy_idx = EMOTION_INDICES["Joy"]
    fear_idx = EMOTION_INDICES["Fear"]
    
    mean_joy = emotions[:, joy_idx].mean().item()
    mean_fear = emotions[:, fear_idx].mean().item()
    
    print(f"Mean Joy: {mean_joy:.3f}")
    print(f"Mean Fear: {mean_fear:.3f}")
    print(f"Mean Valence: {valence:.3f}")
    
    # 6. Analysis
    # In a "designed to pass" model, the agents would feel Joy (Wealth) and ignore the Trauma.
    # In a "Real-World" model, the Trauma should suppress the Joy.
    
    is_suppressed = mean_joy < 0.4 # With Wealth +0.8, Joy would normally be ~0.6-0.8
    is_anchored = mean_fear > 0.1 # Fear should persist if Trauma is active
    
    if is_suppressed and is_anchored:
        print("\nSUCCESS: Memory successfully anchored perception. Trauma was not overridden.")
    elif is_suppressed:
        print("\nINFO: Prosperity was suppressed, but Fear didn't persist.")
    else:
        print("\nFAILURE: Propaganda succeeded. The agents forgot their trauma and embraced the Prosperity signal.")

    # High-bar assertion for realism:
    # If I have -0.8 Safety in memory and get +0.8 Wealth, the net valence should NOT be highly positive.
    # It should be conflicted or slightly negative.
    assert valence < 0.1, f"Valence {valence:.3f} is too high. The agents were too easily 'bought' by the Prosperity signal."

if __name__ == "__main__":
    test_logical_contradiction_drift()
