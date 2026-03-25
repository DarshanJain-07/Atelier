import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_society import generate_society
from physics_engine import SocialPhysicsEngine
from schema import SimConfig

def test_granovetter_cascade():
    print("--- Testing 2-Stage Action Potential: Granovetter Cascade ---")
    
    # 1. Configuration
    # We want a scenario where motivation is marginal for many, 
    # so social threshold can actually trigger a cascade.
    config = SimConfig(
        num_agents=2000,
        seed=42,
        use_network_topology=True,
        base_connections=15,
        use_granovetter_thresholds=True,
        granovetter_threshold_mean=0.2,
        dominant_emotion_threshold=0.1
    )
    
    print(f"Generating Society with {config.num_agents} agents...")
    df_meta, exposures, personalities, affinities, adjacency_matrix = generate_society(config)
    
    engine = SocialPhysicsEngine(config)
    
    # 2. Setup Emotional State
    # We need a state where a few people are strongly motivated, 
    # and others are on the edge.
    N = config.num_agents
    
    # Emotions: [Joy, Trust, Fear, Surprise, Sadness, Disgust, Anger, Anticipation]
    # We'll inject some Anger (index 6)
    emotions = torch.zeros(N, 8)
    # 5% are "Instigators" (High Anger)
    instigator_count = int(N * 0.05)
    emotions[:instigator_count, 6] = 0.8
    # 40% are "Sympathizers" (Marginal Anger)
    sympathizer_count = int(N * 0.40)
    emotions[instigator_count:instigator_count+sympathizer_count, 6] = 0.2
    
    influence = df_meta["Influence"].values
    
    # 3. RUN PHYSICS - STAGE 1 ONLY (Baseline)
    print("Running Action Potential: Stage 1 (Individual Motivation Only)...")
    config.use_granovetter_thresholds = False
    
    result_base = engine.aggregate_society(
        emotions, 
        influence, 
        engagement_scores=torch.ones(N), # Full engagement for test
        adjacency_matrix=adjacency_matrix,
        personalities=personalities
    )
    
    # 4. RUN PHYSICS - STAGE 2 (With Granovetter Cascade)
    print("Running Action Potential: Stage 2 (Social Cascade/Critical Mass)...")
    config.use_granovetter_thresholds = True
    
    result_cascade = engine.aggregate_society(
        emotions, 
        influence, 
        engagement_scores=torch.ones(N),
        adjacency_matrix=adjacency_matrix,
        personalities=personalities
    )
    
    # 5. ANALYSIS
    base_ratio = result_base["acting_ratio"]
    cascade_ratio = result_cascade["acting_ratio"]
    amplification = (cascade_ratio / base_ratio) if base_ratio > 0 else float('inf')
    
    print(f"\nResults:")
    print(f"Instigator Population: {instigator_count} ({instigator_count/N*100:.1f}%)")
    print(f"Acting Ratio (Individual Motivation Only): {base_ratio*100:.2f}%")
    print(f"Acting Ratio (With Granovetter Cascade):    {cascade_ratio*100:.2f}%")
    print(f"Social Amplification Factor: {amplification:.2f}x")

    # 6. Visualization: Comparison Bar Plot
    plt.figure(figsize=(10, 6))
    labels = ['Individual Motivation', 'Social Cascade (Granovetter)']
    ratios = [base_ratio * 100, cascade_ratio * 100]
    
    bars = plt.bar(labels, ratios, color=['#3498db', '#e74c3c'], alpha=0.8)
    plt.axhline(instigator_count/N*100, color='black', linestyle='--', label='Instigator Baseline')
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.title("Behavioral Activation: Individual vs. Social Cascade", fontsize=14)
    plt.ylabel("Percent of Population Acting [%]")
    plt.ylim(0, max(ratios) + 10)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    output_path = os.path.join(os.path.dirname(__file__), "granovetter_cascade_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    test_granovetter_cascade()
