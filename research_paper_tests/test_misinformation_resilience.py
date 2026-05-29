import torch
import numpy as np
from physics_engine import SocialPhysicsEngine
from cognitive_engine import CognitiveEngine
from research_paper_tests.config_schema import PSYCH_PROJECTION, SimConfig
from research_paper_tests.stats_utils import run_monte_carlo, assert_statistically_greater

def run_misinfo_simulation(is_false_signal=True):
    config = SimConfig()
    config.stewing_ticks = 20
    config.outrage_gain = 12.0
    config.logic_gap_threshold = 0.5
    config.use_signal_distortion = False
    
    physics = SocialPhysicsEngine(config)
    cognitive = CognitiveEngine(config)
    
    N = 100
    signal = torch.zeros(12)
    if is_false_signal:
        # Contradictory signal: High Wealth, Extreme Danger
        signal[0] = 1.0  # Wealth
        signal[1] = -4.0 # Danger
    else:
        # Consistent signal: High Wealth, High Safety
        signal[0] = 1.0
        signal[1] = 1.0
    
    personalities = torch.rand(N, 5)
    distorted_signals = cognitive.distort_signal(signal, personalities)
    
    raw_emotions = torch.matmul(distorted_signals, PSYCH_PROJECTION)
    raw_emotions = torch.clamp(raw_emotions, min=0.0)
    
    influence = torch.ones(N)
    result = physics.aggregate_society(raw_emotions, influence)
    
    return result

def test_misinformation_resilience_statistical():
    """
    Goal: Statistically verify that contradictory (false) signals 
    trigger significantly higher outrage than consistent (true) signals.
    """
    print("\nRunning Monte Carlo for Misinformation Resilience...")
    
    true_results = run_monte_carlo(lambda: run_misinfo_simulation(is_false_signal=False))
    false_results = run_monte_carlo(lambda: run_misinfo_simulation(is_false_signal=True))
    
    true_outrage = [r["mean_outrage_multiplier"] for r in true_results]
    false_outrage = [r["mean_outrage_multiplier"] for r in false_results]
    
    true_negative = [r["negative_integral"] for r in true_results]
    false_negative = [r["negative_integral"] for r in false_results]
    
    print(f"True Signal Mean Outrage: {np.mean(true_outrage):.3f}")
    print(f"False Signal Mean Outrage: {np.mean(false_outrage):.3f}")
    
    # False signals should trigger significantly higher outrage and negative integral
    assert_statistically_greater(false_outrage, true_outrage)
    assert_statistically_greater(false_negative, true_negative)
    
    # Absolute check for the false signal
    assert np.mean(false_negative) > 0
    assert np.mean(false_outrage) > 1.0

if __name__ == "__main__":
    test_misinformation_resilience_statistical()
