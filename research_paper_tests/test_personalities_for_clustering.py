import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from generate_society import generate_society
from schema import SimConfig

def analyze_personality_distribution():
    config = SimConfig(num_agents=5000, seed=42)
    df_meta, exposures, personalities, affinities, adj = generate_society(config)
    
    pers_np = personalities.numpy()
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    
    print("\n--- Personality Distribution Analysis ---")
    for i, trait in enumerate(traits):
        data = pers_np[:, i]
        print(f"{trait}: Mean={data.mean():.3f}, Std={data.std():.3f}, Min={data.min():.3f}, Max={data.max():.3f}")
        
    # Check for "High Neuroticism" agents (e.g. > 0.8)
    high_n = (pers_np[:, 4] > 0.8).sum()
    print(f"\nAgents with Neuroticism > 0.8: {high_n} ({high_n/5000*100:.2f}%)")
    
    # Check for "Low Neuroticism" agents (e.g. < 0.2)
    low_n = (pers_np[:, 4] < 0.2).sum()
    print(f"\nAgents with Neuroticism < 0.2: {low_n} ({low_n/5000*100:.2f}%)")

if __name__ == "__main__":
    analyze_personality_distribution()
