import torch
import pytest
from physics_engine import SocialPhysicsEngine
from schema import SimConfig

def test_information_fatigue_refractory_period():
    """
    Test 3: Information Fatigue (The "Burnout" Test)
    Goal: Validate that sustained high arousal triggers a refractory period (News Avoidance).
    """
    config = SimConfig()
    config.use_refractory_period = True
    config.refractory_arousal_threshold = 0.5 # Lower for easier triggering
    config.refractory_threshold_duration = 3   # 3 steps to trigger
    config.refractory_engagement_drop = 0.9    # 90% drop
    config.stewing_ticks = 1 # We want to control the external steps
    
    physics = SocialPhysicsEngine(config)
    
    N = 100
    # High arousal emotions (e.g., pure Anger)
    # Emotions: [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation]
    high_arousal_emotions = torch.zeros(N, 8)
    high_arousal_emotions[:, 6] = 1.0 # Pure Anger
    
    influence = torch.ones(N)
    
    # 1. First event: High arousal, but no refractory yet
    result1 = physics.aggregate_society(high_arousal_emotions, influence)
    # In tick 1, arousal is 1.0, counters become 1
    
    # 2. Second event: High arousal
    result2 = physics.aggregate_society(high_arousal_emotions, influence)
    # In tick 1, counters become 2
    
    # 3. Third event: High arousal
    result3 = physics.aggregate_society(high_arousal_emotions, influence)
    # In tick 1, counters become 3. Still not triggered *during* this step 
    # but will be active for the NEXT step because duration is 3.
    
    # 4. Fourth event: Should trigger refractory period
    # Note: Our implementation in physics_engine updates counters *during* stewing ticks.
    # If stewing_ticks=1, the counter is updated, and then applied in the NEXT aggregate_society call.
    result4 = physics.aggregate_society(high_arousal_emotions, influence)
    
    print(f"Arousal 1: {result1['confidence']}, Engagement Multiplier: {result1['mean_outrage_multiplier']}")
    print(f"Arousal 4: {result4['confidence']}, Engagement Multiplier: {result4['mean_outrage_multiplier']}")
    
    # Validation: In result4, agents should be in refractory mode.
    # This means their engagement is dropped, which should lower the viral center magnitude
    # and the outrage multipliers compared to result1.
    assert result4['mean_outrage_multiplier'] < result1['mean_outrage_multiplier']
    
    # Check internal counter (via private attribute access for test verification)
    assert torch.all(physics._refractory_counters >= 3)

if __name__ == "__main__":
    test_information_fatigue_refractory_period()
