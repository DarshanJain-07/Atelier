import numpy as np
import torch
import sys
import os

# Add parent directory to path to import schema, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema import SimConfig
from physics_engine import SocialPhysicsEngine

def test_maximum_virality():
    print("--- Running Maximum Virality Analysis ---")
    
    # Initialize physics engine
    config = SimConfig()
    phys_engine = SocialPhysicsEngine(config)
    
    N = 1000
    
    print("\n[ Scenario 1: Total Consensus (Everyone agrees) ]")
    # Everyone has exact same emotion (e.g., Joy)
    emotions_consensus = torch.zeros(N, 8)
    emotions_consensus[:, 0] = 1.0 # Max Joy
    
    influence = torch.ones(N)
    state_consensus = phys_engine.aggregate_society(emotions_consensus, influence)
    print(f"Mean Outrage Multiplier (Virality): {state_consensus['mean_outrage_multiplier']}x")
    print(f"Max Outrage Multiplier: {state_consensus['max_outrage_multiplier']}x")
    
    print("\n[ Scenario 2: High Polarization (50% vs 50%) ]")
    emotions_polarized = torch.zeros(N, 8)
    emotions_polarized[:500, 0] = 1.0 # 50% Joy
    emotions_polarized[500:, 4] = 1.0 # 50% Anger
    
    state_polarized = phys_engine.aggregate_society(emotions_polarized, influence)
    print(f"Mean Outrage Multiplier (Virality): {state_polarized['mean_outrage_multiplier']}x")
    print(f"Max Outrage Multiplier: {state_polarized['max_outrage_multiplier']}x")
    
    print("\n[ Scenario 3: Extreme Outlier Rebellion ]")
    # 95% of people feel moderate calmness, 5% are ABSOLUTELY OUTRAGED (max Anger + Disgust)
    emotions_outlier = torch.zeros(N, 8)
    emotions_outlier[:950, 0] = 0.2  # Moderate Joy/Calm
    emotions_outlier[950:, 4] = 1.0  # Max Anger
    emotions_outlier[950:, 5] = 1.0  # Max Disgust
    
    state_outlier = phys_engine.aggregate_society(emotions_outlier, influence)
    print(f"Mean Outrage Multiplier (Virality): {state_outlier['mean_outrage_multiplier']}x")
    print(f"Max Outrage Multiplier: {state_outlier['max_outrage_multiplier']}x")
    
    print("\n[ Scenario 4: Maximum Theoretical Virality ]")
    # To maximize distance from center, we need the center to be 0, but outliers to be massive.
    # Since emotions are softly capped in reality, we simulate a center far from an extreme emotion.
    emotions_extreme = torch.zeros(N, 8)
    emotions_extreme[:990] = torch.ones(8) * 0.1 # Very bland society
    emotions_extreme[990:] = torch.ones(8) * 10.0 # Mathematically impossible but proves the sigmoid bounds
    
    state_extreme = phys_engine.aggregate_society(emotions_extreme, influence)
    print(f"Mean Outrage Multiplier (Virality): {state_extreme['mean_outrage_multiplier']}x")
    print(f"Max Outrage Multiplier: {state_extreme['max_outrage_multiplier']}x")
    
    print("\n--- Why is your interaction always near 1.1x? ---")
    print("In your 'outrage_boost' formula: 1.0 + max_multiplier * torch.sigmoid(outrage_gain * (distances - midpoint))")
    print(f"Midpoint = {config.saturation_midpoint}")
    print(f"If agent emotional distance from the societal center is much less than {config.saturation_midpoint},")
    print("the sigmoid outputs a very small number. In normal simulations without extreme polarization,")
    print("most agents feel similarly, keeping the 'distances' low, so the outrage boost is minimal (~1.0 - 1.2x).")

if __name__ == "__main__":
    test_maximum_virality()
