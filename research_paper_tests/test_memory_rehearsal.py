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

from cognitive_engine import CognitiveEngine
from schema import SimConfig

def test_memory_rehearsal():
    print("--- Testing 2-Stage Memory: Social Rehearsal Consolidation ---")
    
    # 1. Configuration
    config = SimConfig(
        num_agents=100,
        use_agent_memory=True,
        memory_decay_rate=0.5, # Fast decay by default
        memory_social_rehearsal_gain=0.8 # Strong rehearsal effect
    )
    
    engine = CognitiveEngine(config)
    
    # 2. Setup initial state
    N = config.num_agents
    initial_memory = torch.zeros(N, 12)
    
    # A high-impact event context (Stage 1 Imprint)
    # Let's say it's a major threat on Safety (-1.0)
    context_vector = torch.zeros(N, 12)
    context_vector[:, 1] = -1.0
    
    # 3. SCENARIO A: Isolated Event (No Social Rehearsal)
    print("Scenario A: Isolated Event (Rehearsal = 0.0)...")
    memory_isolated = engine.consolidate_memory(
        initial_memory, 
        context_vector, 
        social_rehearsal_factor=0.0
    )
    
    # 4. SCENARIO B: Viral/Rehearsed Event (High Social Rehearsal)
    print("Scenario B: Viral Event (Rehearsal = 1.0)...")
    memory_rehearsed = engine.consolidate_memory(
        initial_memory, 
        context_vector, 
        social_rehearsal_factor=1.0
    )
    
    # 5. Long-term Retention Check (simulating more time steps)
    # In Scenario B, the memory should persist much longer.
    # Let's run 5 more "empty" consolidation steps (just decay)
    
    mem_a = memory_isolated
    mem_b = memory_rehearsed
    empty_context = torch.zeros(N, 12)
    
    history_a = [torch.norm(mem_a).item()]
    history_b = [torch.norm(mem_b).item()]
    
    for _ in range(5):
        # Scenario A continues with no rehearsal
        mem_a = engine.consolidate_memory(mem_a, empty_context, social_rehearsal_factor=0.0)
        # Scenario B continues to be "rehearsed" (e.g. news cycle / social media)
        mem_b = engine.consolidate_memory(mem_b, empty_context, social_rehearsal_factor=1.0)
        
        history_a.append(torch.norm(mem_a).item())
        history_b.append(torch.norm(mem_b).item())
        
    # 6. RESULTS
    print(f"\nResults (Norm of Memory Vector):")
    print(f"Step 0 (Imprint) -> Isolated: {history_a[0]:.4f}, Rehearsed: {history_b[0]:.4f}")
    print(f"Step 5 (Decay)   -> Isolated: {history_a[5]:.4f}, Rehearsed: {history_b[5]:.4f}")
    
    retention_a = (history_a[5] / history_a[0]) * 100
    retention_b = (history_b[5] / history_b[0]) * 100
    print(f"Retention Rate (Isolated):  {retention_a:.1f}%")
    print(f"Retention Rate (Rehearsed): {retention_b:.1f}%")
    
    # 7. Visualization
    plt.figure(figsize=(10, 6))
    steps = np.arange(6)
    plt.plot(steps, history_a, marker='o', color='blue', label='Isolated (No Rehearsal)')
    plt.plot(steps, history_b, marker='s', color='red', label='Rehearsed (High Salience)')
    
    plt.title("Memory Decay Curve: The Effect of Social Rehearsal", fontsize=14)
    plt.xlabel("Time Steps (Post-Event)")
    plt.ylabel("Memory Strength (Vector Norm)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_path = os.path.join(os.path.dirname(__file__), "memory_rehearsal_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved visualization to: {output_path}")

if __name__ == "__main__":
    test_memory_rehearsal()
