import torch
import pytest
from physics_engine import SocialPhysicsEngine
from cognitive_engine import CognitiveEngine
from schema import SimConfig, DIMENSIONS

def test_truth_decay_rate():
    """
    Test 1: Misinformation Resilience (Truth Decay Rate)
    Goal: Measure how many "stewing ticks" it takes for viral arousal to override the logic gap.
    """
    config = SimConfig()
    config.stewing_ticks = 20  # Increase ticks to observe decay over time
    config.outrage_gain = 12.0 # High contagion
    config.logic_gap_threshold = 0.5
    config.use_signal_distortion = False # Keep it clean for testing
    
    physics = SocialPhysicsEngine(config)
    cognitive = CognitiveEngine(config)
    
    N = 100
    # A completely false signal: Positive Wealth (0) but negative Safety (1)
    # The world state usually implies Wealth and Safety are correlated.
    false_signal = torch.zeros(12)
    false_signal[0] = 1.0  # High Wealth
    false_signal[1] = -4.0 # EXTREME Danger (Contradiction)
    
    personalities = torch.rand(N, 5)
    
    # 1. Distort signal into agent perceptions
    distorted_signals = cognitive.distort_signal(false_signal, personalities)
    
    # 2. Project to emotions
    # PSYCH_PROJECTION is (12, 8). distorted_signals is (N, 12)
    from schema import PSYCH_PROJECTION
    raw_emotions = torch.matmul(distorted_signals, PSYCH_PROJECTION)
    # Clamp to ensure valid probabilities for the engine (though aggregate handles it)
    raw_emotions = torch.clamp(raw_emotions, min=0.0)
    
    # 3. Simulate Stewing and track Logic Gap vs Viral Energy
    influence = torch.ones(N)
    
    # We need to manually run ticks if we want to observe the *internal* state per tick,
    # OR we can compare the final result with a "Truthful" signal.
    # For "Truth Decay Rate", let's measure the magnitude of the dominant false emotion 
    # relative to the center of gravity over time.
    
    # Let's run the aggregate society which now has stewing logic.
    result = physics.aggregate_society(raw_emotions, influence)
    
    # Validation: In a high-outrage society, the 'negative' integral should be high
    # despite the initial "Joy" from Wealth, because "Fear" from Safety is more viral.
    assert result["negative_integral"] > 0
    assert result["mean_outrage_multiplier"] > 1.0
    
    print(f"Truth Decay - Negative Integral: {result['negative_integral']}")
    print(f"Truth Decay - Max Outrage: {result['max_outrage_multiplier']}")

if __name__ == "__main__":
    test_truth_decay_rate()
